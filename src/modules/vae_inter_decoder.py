#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import AtomEncoder
from modules.common_nn import MLP
from utils.chem_utils import smiles2molecule, valence_check, cnt_atom, cycle_check, molecule2smiles, rec
#from src.utils.chem_utils import del_N_positive
#from src.utils.logger import print_log
from src.data.group_graph_dataset import atom_features
import numpy as np
#from modules.transformer import TransformerModel


def get_label(y_pred, k):
    datas = torch.zeros(y_pred.size())
    _, a = y_pred.topk(k=k, dim=-1)
    # print(a)
    a_tensor = torch.ones(a.size())
    datas.scatter_add_(1, a.long(), a_tensor)
    return datas


class FocalLoss(nn.Module):
    def __init__(self, gamma=2,weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)  # 使用交叉熵损失函数计算基础损失
        pt = torch.exp(-ce_loss)  # 计算预测的概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 根据Focal Loss公式计算Focal Loss
        return focal_loss




class VAEPieceDecoder(nn.Module):
    '''use variational auto encoder'''
    def __init__(self, device,tokenizer, piece_in_dim, num_pos_type, dim_hidden, t=4):
        super(VAEPieceDecoder, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_data = tokenizer.get_vocab_data().to(self.device)


        self.chem_vocab = tokenizer.chem_vocab
        self.piece_embedding = nn.Embedding(tokenizer.num_piece(), dim_hidden)
        self.latent_to_rnn_hidden = nn.Linear(dim_hidden, dim_hidden)
        #self.transformer = TransformerModel(dim_hidden, tokenizer)
        self.rnn = nn.GRU(dim_hidden, dim_hidden, batch_first=True)
        self.to_vocab = nn.Linear(dim_hidden, tokenizer.num_piece())
        self.dim_hidden = dim_hidden
        # num_atom_type, piece_in_dim, num_pos_type, num_edge_type, dim_hidden, t
        self.atom_gnn = AtomEncoder(self.chem_vocab, piece_in_dim,  num_pos_type, dim_hidden, t)

        self.pieces_type_embedding = nn.Embedding(tokenizer.num_piece_type(), dim_hidden)
        #self.pieces_inter_embedding = nn.Embedding(tokenizer.num_inter_type(), dim_hidden)

        self.piece_embedding_predictor = nn.Sequential(
            MLP(
                dim_in =  dim_hidden + piece_in_dim + dim_hidden + 1,
                dim_hidden=dim_hidden + dim_hidden,
                dim_out=dim_hidden,
                act_func=nn.ReLU,
                num_layers=3
            ),
        )

        #pieces_embed, piece_z_repeat, pieces_descriptor, pieces_type, pieces_inter
        self.piece_edge_predictor = nn.Linear(dim_hidden, 2)

        # vae

        # loss
        self.piece_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx())
        #self.piece_edge_loss = torch.nn.L1Loss()
        weight0 = torch.tensor([1, 10]).float().to(self.device)
        self.piece_edge_loss = nn.CrossEntropyLoss(weight=weight0)
        #node_embedding, x_pieces_embed, latent_repeat, x_type, x_pos
        self.atom_embedding_predictor = nn.Sequential(
            MLP(
                dim_in=dim_hidden + piece_in_dim + dim_hidden + dim_hidden + 1,
                dim_hidden=dim_hidden + dim_hidden,
                dim_out=dim_hidden,
                act_func=nn.ReLU,
                num_layers=3
            ),
        )
        self.edge_predictor = nn.Linear(dim_hidden, 2)

        weight = torch.tensor([1, 10]).float().to(self.device)
        #self.edge_loss  = nn.BCEWithLogitsLoss(reduction='none')
        self.edge_loss = FocalLoss(weight=weight)

        #self.gamma = 3

        #self.graph_loss = nn.MSELoss()



    def forward(self, z_vecs, piece_encoder, batch):
        # x: [batch_size, N, dim_in]
        # edge_index and edge_attr do not include the edges to be predicted
        # pieces: [batch_size, seq_len] in idx form
        # conds: [batch_size, cond_dim] 原子图embed
        # edge_select: [batch_size, N, N] indicate which edges are to be predicted
        # golden_egdes: [E], only include edges to be predicted


        atom_batch, group_batch = batch
        #z_vecs, kl_loss = self.rsample(conds)  # [batch_size, latent_dim]
        #print(z_vecs.size()) # batch_size,56
        # embed pieces
        batch_size =  atom_batch.num_graphs
        pieces = group_batch.pieces
        pieces = pieces.view(batch_size, -1)


        gold_piece = pieces[:,1:].flatten()
        #print(self.tokenizer.num_piece_type())
        #piece_embed = self.vocab_data.index_select(index=pieces.view(-1), dim=0)
        #piece_embed = piece_embed.view(pieces.shape[0], pieces.shape[1], -1)
        #pieces = self.piece_embedding(piece_embed)
        pieces = self.piece_embedding(pieces) # [batch_size, seq_len, embedding_dim]
        # pieces prediction
        # [1 (num_layers), batch_size, rnn_hidden_size]
        init_hidden = self.latent_to_rnn_hidden(z_vecs).unsqueeze(0) #1,batch_size, -1
        # init_hidden = torch.zeros(self.rnn.num_layers, z_vecs.shape[0], self.rnn.hidden_size, device=z_vecs.device)
        pieces_in = pieces[:,:-1]  # 不要最后一位 [sos] [batch_size, seq_len, embedding_dim]
        rnn_in = pieces_in
        #print(init_hidden.size(), rnn_in.size())
        pieces_embed, _ = self.rnn(rnn_in, init_hidden)  # [batch_size, seq_len, hidden_size]

        #pieces_embed = self.transformer(pieces[:,:-1], pieces[:,1:])

        output = self.to_vocab(pieces_embed)  # [batch_size, seq_len, vocab_size]
        output = output.view(-1, output.shape[-1])
        # loss
        piece_loss = self.piece_loss(output, gold_piece)

        '''
        pieces_embed = pieces_embed.reshape(-1, pieces_embed.shape[-1])
        pieces_select = group_batch.pieces_select.view(batch_size, -1)
        pieces_select = pieces_select[:,1:].flatten().bool()
        pieces_embed = pieces_embed[pieces_select,:]
        

        pieces_type = group_batch.frags_type
        pieces_inter = group_batch.frags_inter
        pieces_type = self.pieces_type_embedding(pieces_type)
        #pieces_inter = self.pieces_inter_embedding(pieces_inter)

        pieces_z_repeat = z_vecs.index_select(index=group_batch.batch, dim=0)
        pieces_descriptor = self.vocab_data.index_select(index=group_batch.x.squeeze(), dim=0)


        pieces_inter = pieces_inter.reshape(-1,1)
        pieces_embed = torch.cat([pieces_z_repeat, pieces_descriptor, pieces_type, pieces_inter], dim=-1)

        pieces_embed = self.piece_embedding_predictor(pieces_embed)  # MLP预测卷积后的piece嵌入


        piece_edge_index = group_batch.edge_index
        sample_len = piece_edge_index.shape[1]
        neg_pieces_link = group_batch.neg_edge_index
        sample = torch.randint(neg_pieces_link.shape[1], size=(sample_len * 10,))
        neg_pieces_link = neg_pieces_link[:, sample]
        
        all_edge_index = torch.cat([neg_pieces_link, piece_edge_index], dim=-1)
        piece_link_logits = pieces_embed[all_edge_index[0]] * pieces_embed[all_edge_index[1]]
        piece_link_logits = self.piece_edge_predictor(piece_link_logits)

        E = all_edge_index.shape[1]
        link_labels = torch.zeros(E, 2, dtype=torch.float, device=piece_edge_index.device)
        link_labels[:sample_len * 10, 0] = 1.
        link_labels[sample_len * 10: sample_len * 10 + sample_len, 1] = 1.
        
        # link_labels[sample_len + len(pieces_link_index) +  len(pieces_in_index):, 3] = 1.
        # link_labels[(neg_1_len + pos_len):, 2] = 1.  # 无边2

        piece_link_loss = self.piece_edge_loss(piece_link_logits, link_labels)

        
        piece_link_logits = torch.mm(pieces_embed, pieces_embed.t()).view(pieces_len, pieces_len, -1)
        piece_matrix_pred = self.piece_edge_predictor(piece_link_logits) # (piece_len, piece_len, 2)
        piece_matrix = torch.zeros(piece_len,piece_len,2).to(pieces_embed.device)

        piece_matrix[:,:,0] = 1 # 初始化假设所有的piece之间都没边
        for a, [i, j] in enumerate(piece_edge_index.t().tolist()):
            #print(i, j, piece_edge_attr[a,0])
            piece_matrix[i, j, 0] = 0
            piece_matrix[j, i, 0] = 0
            piece_matrix[i, j, 1] = 1
            piece_matrix[j, i, 1] = 1
        piece_link_loss = self.piece_edge_loss(piece_matrix_pred.view(-1,2), piece_matrix.view(-1,2))
        '''
        piece_link_loss = torch.zeros_like(piece_loss)

        #pieces_embed = self.piece_embedding_predictor(pieces_embed)  # MLP预测卷积后的piece嵌入
        pieces_descriptor = self.vocab_data.index_select(index=group_batch.x.squeeze(), dim=0)
        x, x_pieces, x_pos = atom_batch.x, atom_batch.x_piece, atom_batch.x_pos
        x_type = atom_batch.x_type
        #x_inter = atom_batch.x_inter
        x_pieces_embed = []
        for i in range(atom_batch.num_graphs):
            atom_i = torch.where(atom_batch.batch==i)[0]
            #print(atom_i)
            atom_group = x_pieces.index_select(index=atom_i,dim=0)
            group_i = torch.where(group_batch.batch==i)[0]
            #print(group_i)
            group_i_embed = pieces_descriptor.index_select(index=group_i,dim=0)
            atom_i_embed = group_i_embed.index_select(index=atom_group,dim=0)
            x_pieces_embed.append(atom_i_embed)

        x_pieces_embed = torch.cat(x_pieces_embed, dim=0)



        edge_index, edge_attr, edge_mask = atom_batch.edge_index, atom_batch.edge_attr, atom_batch.mask_edge
        #print(edge_index.size(), edge_attr.size(), edge_mask.size())
        pieces_in_index = edge_index[:,edge_mask.bool()]   # in_piece_edge_idx 作为训练集的边
        piece_in_attr = edge_attr[edge_mask.bool(), :]


        node_embedding = self.atom_gnn.encoder_node(x, pieces_in_index, piece_in_attr)
        #x_type = self.pieces_type_embedding(x_type)
        #x_inter = self.pieces_inter_embedding(x_inter)
        latent_repeat = z_vecs.index_select(index=atom_batch.batch, dim=0)
        # print(latent_repeat.size())
        x_type = self.pieces_type_embedding(x_type)
        x_pos = x_pos.reshape(-1, 1)
        node_embedding = torch.cat([node_embedding, x_pieces_embed, latent_repeat, x_type, x_pos], dim=-1)
        #node_embedding = torch.cat([node_embedding, x_pieces_embed], dim=-1)

        '''
        inter_embedding = node_embedding[x_pos.bool(),:]
        inter_embedding = self.atom_embedding_predictor(inter_embedding)
        inter_len = inter_embedding.shape[0]
        atom_link_logits = torch.mm(inter_embedding, inter_embedding.t()).view(inter_len, inter_len, -1)
        atom_matrix_pred = self.edge_predictor(atom_link_logits)

        atom_matrix = torch.zeros(inter_len, inter_len, 2).to(pieces_embed.device)
        atom_matrix[:,:,0] = 1 # 初始化假设所有的piece之间都没边
        pieces_link_index = edge_index[:, ~edge_mask.bool()]  # inter_pair 需要预测的结果
        for a, [i, j] in enumerate(pieces_link_index.t().tolist()):
            #print(i, j, piece_edge_attr[a,0])
            atom_matrix[i, j, 0] = 0
            atom_matrix[j, i, 0] = 0
            atom_matrix[i, j, 1] = 1
            atom_matrix[j, i, 1] = 1

        edge_loss = self.edge_loss(atom_matrix_pred.view(-1, 2), atom_matrix.view(-1, 2))
        #print(node_embedding.size())
        #latent_repeat = torch.repeat_interleave(z_vecs, node_num, dim=0).view(node_num, -1)
 
        '''
        node_embedding = self.atom_embedding_predictor(node_embedding)

        pieces_link_index = edge_index[:, ~edge_mask.bool()]  # inter_pair 需要预测的结果

        neg  = atom_batch.neg_edge_index # neg_1表示的 piece之间有边，原子之间无边
        sample_len = pieces_link_index.shape[1]
        
        sample = torch.randint(neg.shape[1], size=(sample_len*10,))
        neg_edge_index = neg[:, sample]

        all_edge_index = torch.cat([neg_edge_index, pieces_link_index], dim=-1)

        link_logits = node_embedding[all_edge_index[0]] * node_embedding[all_edge_index[1]]

        link_logits = self.edge_predictor(link_logits)

        E = all_edge_index.shape[1]
        link_labels = torch.zeros(E,2, dtype=torch.float, device=edge_index.device)
        link_labels[:sample_len*10, 0] = 1.
        link_labels[sample_len*10 : sample_len*10 + sample_len, 1] = 1.
        #link_labels[sample_len + len(pieces_link_index) +  len(pieces_in_index):, 3] = 1.
        #link_labels[(neg_1_len + pos_len):, 2] = 1.  # 无边2
        
        edge_loss = self.edge_loss(link_logits, link_labels)



        return piece_loss, piece_link_loss, edge_loss



