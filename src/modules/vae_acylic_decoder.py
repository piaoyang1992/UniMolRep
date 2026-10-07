#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import AtomEncoder,GroupEncoder
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


from torch.nn import TransformerDecoder, TransformerDecoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        d_model: 嵌入维度
        max_len: 最大序列长度
        """
        super(PositionalEncoding, self).__init__()

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))  # (d_model/2,)

        # 计算位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度

        # 将位置编码矩阵注册为缓冲区（不参与训练）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """
        x: 输入序列 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]  # 添加位置编码
        return x


class TransformerDecoderWithZ(nn.Module):
    def __init__(self, d_model, nhead, num_layers, pad_id, vocab_size, max_seq_len):
        super(TransformerDecoderWithZ, self).__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)

        # 将隐变量 z 映射到解码器的初始隐藏状态
        self.z_projection = nn.Linear(d_model, d_model)

        # Transformer 解码器
        decoder_layer = TransformerDecoderLayer(d_model, nhead)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, z, tgt):
        """
        z: 隐变量 (batch_size, dim_z)
        tgt: 目标序列 (batch_size, seq_len)
        """
        batch_size, seq_len = tgt.size()

        # 将目标序列嵌入
        tgt_embed = self.embedding(tgt)  # (batch_size, seq_len, d_model)

        # 添加位置编码
        tgt_embed = self.positional_encoding(tgt_embed)  # (batch_size, seq_len, d_model)

        # 将隐变量 z 映射到解码器的初始隐藏状态
        #z_projected = self.z_projection(z).unsqueeze(0)  # (1, batch_size, d_model)

        # 解码器输入
        tgt_mask = self.generate_square_subsequent_mask(seq_len).to(tgt.device)
        tgt_padding_mask = (tgt == self.pad_id)  # 假设 padding 的 token id 为 0

        # Transformer 解码
        output = self.transformer_decoder(
            tgt_embed.transpose(0, 1),  # (seq_len, batch_size, d_model)
            z,  # (1, batch_size, d_model)
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )  # (seq_len, batch_size, d_model)

        # 输出层
        output = self.fc_out(output.transpose(0, 1))  # (batch_size, seq_len, vocab_size)
        return output

    def generate_square_subsequent_mask(self, sz):
        """生成因果掩码，防止解码器看到未来信息"""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



# 定义 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, dim_hidden):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(dim_hidden, dim_hidden,  batch_first=True)

    def forward(self, x, h):
        # 前向传播
        out, h_next = self.rnn(x, h)
        return out, h_next


# 定义整体模型
class DualRNN(nn.Module):
    def __init__(self, input_size_a, input_size_b, hidden_size):
        super(DualRNN, self).__init__()
        self.hidden_size = hidden_size

        # 定义 r1 和 r2
        self.piece_embedding = nn.Embedding(input_size_a, hidden_size)
        self.inter_embedding = nn.Embedding(input_size_b, hidden_size)
        self.piece_embedding_cat = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_dim),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            torch.nn.Linear(hidden_size + hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.r1 = RNNModel(hidden_size)
        self.r2 = RNNModel(hidden_size)
        self.to_piece_vocab = nn.Linear(hidden_size, input_size_a)
        self.to_inter_vocab = nn.Linear(hidden_size, input_size_b)

    def forward(self, pieces, inters, h):
        # 初始化输出
        pieces_embedding = self.piece_embedding(pieces)
        inter_embedding = self.inter_embedding(inters)
        outputs_pieces = []
        outputs_inters = []

        # 遍历每个时间步
        h1, h2 = h.clone(), h.clone()
        for t in range(pieces.size(1) - 1):
            # r1 的前向传播
            x1_t = inter_embedding[:, t:t + 1, :]  # 当前时间步的输入
            out_a, h1 = self.r1(x1_t, h1)

            outputs_inters.append(out_a)

            # r2 的前向传播
            x2_t = torch.cat([inter_embedding[:, t + 1:t + 2, :], pieces_embedding[:, t:t + 1, :]], dim=-1)  # 拼接输入
            #x2_t = torch.cat([out_a, pieces_embedding[:, t:t + 1, :]], dim=-1)  # 拼接输入

            x2_t = self.piece_embedding_cat(x2_t)
            out_b, h2 = self.r2(x2_t, h2)
            outputs_pieces.append(out_b)

            # 合并隐藏状态
            #h1 = (h1 + h2) / 2  # 例如，取平均值

        # 将输出堆叠成一个张量
        outputs_pieces = torch.cat(outputs_pieces, dim=1)
        r_pieces = self.to_piece_vocab(outputs_pieces)
        outputs_inters = torch.cat(outputs_inters, dim=1)
        r_inters = self.to_inter_vocab(outputs_inters)

        return r_pieces, r_inters, h1, h2

    def decode(self,piece, inter, h1, h2):
        h1, h2 = h1.reshape(1,-1,self.hidden_size), h2.reshape(1,-1,self.hidden_size)
        inter_embedding = self.inter_embedding(inter).reshape(1,-1,self.hidden_size)
        out_a, h1  = self.r1(inter_embedding, h1)

        out_inter = self.to_inter_vocab(out_a)

        piece_embedding = self.piece_embedding(piece).reshape(1,-1,self.hidden_size)

        x2_t = torch.cat([inter_embedding[:, 1, :].reshape(1,-1,self.hidden_size), piece_embedding], dim=-1)  # 拼接输入
        x2_t = self.piece_embedding_cat(x2_t)
        out_piece, h2 = self.r2(x2_t, h2)
        #h1 = (h1 + h2) / 2
        out_piece = self.to_piece_vocab(out_piece)
        return out_piece, out_inter, h1, h2




class VAEPieceDecoder(nn.Module):
    '''use variational auto encoder'''
    def __init__(self, device,max_atom_index,tokenizer, piece_in_dim, piece_link_dim, num_pos_type, dim_hidden,
                 graph_pooling='mean',t=4):
        super(VAEPieceDecoder, self).__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.vocab_data = tokenizer.get_vocab_data().to(self.device)
        self.max_atom_index = max_atom_index
        self.piece_in_dim = piece_in_dim
        self.chem_vocab = tokenizer.chem_vocab
        self.piece_embedding = nn.Embedding(tokenizer.num_frag_type(), dim_hidden)
        self.inter_embedding = nn.Embedding(tokenizer.num_inter(), dim_hidden)


        self.num_layer = 3
        self.z_to_hidden = nn.Linear(dim_hidden, dim_hidden)
        #self.z_to_cell = nn.Linear(dim_hidden, dim_hidden * self.num_layer)
        self.inter_rnn = nn.GRU(dim_hidden, dim_hidden,  batch_first=True)
        self.index_rnn = nn.GRU(dim_hidden, dim_hidden, batch_first=True)
        #self.transformer = TransformerDecoderWithZ(dim_hidden, 8, num_layers=6, pad_id=tokenizer.pad_idx(), vocab_size=tokenizer.num_piece(), max_seq_len=200)
        self.DRNN = DualRNN(tokenizer.num_frag_type(), tokenizer.num_inter(), dim_hidden)

        self.to_piece_vocab = nn.Linear(dim_hidden, tokenizer.num_frag_type())
        self.to_inter_vocab = nn.Linear(dim_hidden, tokenizer.num_inter())

        self.dim_hidden = dim_hidden
        # num_atom_type, piece_in_dim, num_pos_type, num_edge_type, dim_hidden, t
        self.atom_gnn = AtomEncoder(self.chem_vocab, piece_in_dim,  num_pos_type, dim_hidden, t)
        self.group_gnn = GroupEncoder(piece_in_dim, piece_link_dim,dim_hidden, t, graph_pooling)



        self.piece_embedding_predictor = nn.Sequential(
            MLP(
                dim_in =  piece_in_dim + dim_hidden,
                dim_hidden=dim_hidden,
                dim_out=dim_hidden,
                act_func=nn.ReLU,
                num_layers=1
            ),
        )

        #pieces_embed, piece_z_repeat, pieces_descriptor, pieces_type, pieces_inter
        self.piece_edge_predictor = nn.Linear(dim_hidden, 2)

        # vae

        # loss
        self.piece_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.piece_pad_idx())
        self.inter_loss = nn.CrossEntropyLoss(ignore_index=tokenizer.inter_pad_idx())
        #self.piece_edge_loss = torch.nn.L1Loss()
        weight0 = torch.tensor([1, 10]).float().to(self.device)
        self.piece_edge_loss = nn.CrossEntropyLoss(weight=weight0)
        #node_embedding, x_pieces_embed, latent_repeat, x_type, x_pos
        self.atom_embedding_predictor = nn.Sequential(
            MLP(
                dim_in=dim_hidden + dim_hidden + dim_hidden,
                dim_hidden=dim_hidden + dim_hidden,
                dim_out=dim_hidden,
                act_func=nn.ReLU,
                num_layers=3
            ),
        )
        #self.edge_predictor = nn.Linear(dim_hidden, 3)
        self.atom_pieces_pred = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_dim),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            torch.nn.Linear(dim_hidden+167, dim_hidden),
            nn.ReLU(),
            torch.nn.BatchNorm1d(dim_hidden),
            torch.nn.Linear(dim_hidden, 4),
            #nn.Tanh()
        )
        self.edge_predictor = nn.Linear(dim_hidden, 1)

        weight = torch.tensor([1, 10,1,1]).float().to(self.device)
        #self.edge_link_loss  = torch.nn.L1Loss()
        self.edge_link_loss = FocalLoss(weight=weight)
        #self.edge_link_loss = nn.CrossEntropyLoss(weight=weight)

        self.edge_loss  = nn.CrossEntropyLoss()


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
        batch_size = atom_batch.num_graphs
        hidden = self.z_to_hidden(z_vecs).unsqueeze(0)


        inters = group_batch.inters
        inters = inters.view(batch_size, -1)
        gold_inter = inters[:, 1:].flatten()

        '''
        inters_in = self.inter_embedding(inters[:, :-1])
        inter_embed, _ = self.inter_rnn(inters_in, hidden)  # [batch_size, seq_len, hidden_size]
        inters_output = self.to_inter_vocab(inter_embed)
        inters_output = inters_output.view(-1, inters_output.shape[-1])

        '''
        pieces = group_batch.pieces
        pieces = pieces.view(batch_size, -1)
        gold_piece = pieces[:, 1:].flatten()


        #print(self.tokenizer.num_piece_type())
        #piece_embed = self.vocab_data.index_select(index=pieces.view(-1), dim=0)
        #piece_embed = piece_embed.view(batch_size, -1, self.piece_in_dim)

        #pieces_in = self.piece_embedding(pieces[:,:-1])  # [batch_size, seq_len, embedding_dim]

        #pieces = torch.cat([piece_inter,piece_embed],dim=-1)
        #pieces = self.piece_embedding_predictor(pieces)
        # pieces prediction
        # [1 (num_layers), batch_size, rnn_hidden_size]

        # init_hidden = torch.zeros(self.rnn.num_layers, z_vecs.shape[0], self.rnn.hidden_size, device=z_vecs.device)

        #pieces_in = torch.cat([inters_in, pieces_in])

        #hidden = hidden.view(self.num_layer, batch_size, self.dim_hidden)
        #cell = cell.view(self.num_layer, batch_size, self.dim_hidden)
        #pieces_embed, _ = self.rnn(rnn_in, hidden)
        '''
        inters_out = self.inter_embedding(inters[:, 1:])
        pieces_cat_in =torch.cat([inters_out, pieces_in], dim=-1)

        #print(inters_in.size(), pieces_in.size(), pieces_cat_in.size())
        pieces_cat_in = self.piece_embedding_cat(pieces_cat_in)

        pieces_embed, pieces_hidden = self.index_rnn(pieces_cat_in, hidden)
        pieces_output = self.to_piece_vocab(pieces_embed)  # [batch_size, seq_len, vocab_size]
        '''

        pieces_output, inters_output, _,_ = self.DRNN(pieces,inters,hidden)

        mask_tensor = self.tokenizer.get_mask_tensor().T.to(pieces_output.device)  # 为mask的部分 inter_num * piece_num
        mask_select = mask_tensor.index_select(index=inters[:, 1:].reshape(-1), dim=0)  #

        pieces_output = pieces_output.view(-1, pieces_output.shape[-1])
        pieces_output[mask_select==1] = -torch.inf

        # loss
        inters_output = inters_output.view(-1, inters_output.shape[-1])


        indices = (gold_inter != self.tokenizer.inter_pad_idx()).nonzero()
        #print('gold inter',gold_inter[indices].squeeze())

        inters_output = torch.argmax(inters_output, dim=-1)
        inters_output = inters_output[indices].squeeze()

        gold_inter = gold_inter[indices].squeeze()
        comparison = inters_output == gold_inter
        print('inters accuracy',comparison.sum().item()/len(gold_inter))
        inter_loss = self.inter_loss(inters_output, gold_inter)

        piece_loss = self.piece_loss(pieces_output, gold_piece)
        #print(piece_loss)
        piece_loss = inter_loss

        # 在inter确认的情况下，要将填充的部分变为torch.-inf



        groups_embed = self.group_gnn.encoder_node(self.vocab_data, group_batch,edge_attr=False)

        x, x_pieces = atom_batch.x, atom_batch.x_piece

        edge_index, edge_attr, edge_mask = atom_batch.edge_index, atom_batch.edge_attr, atom_batch.mask_edge
        #print(edge_index.size(), edge_attr.size(), edge_mask.size())
        pieces_in_index = edge_index[:,edge_mask.bool()]   # in_piece_edge_idx 作为训练集的边
        piece_in_attr = edge_attr[edge_mask.bool(), :]

        x_pieces_embed = []
        for i in range(atom_batch.num_graphs):
            atom_i = torch.where(atom_batch.batch==i)[0]
            #print(atom_i)
            atom_group = x_pieces.index_select(index=atom_i,dim=0) # x_pieces是一个分子里的序号
            group_i = torch.where(group_batch.batch==i)[0] # 一个分子里的group
            #print(group_i)
            group_i_embed = groups_embed.index_select(index=group_i,dim=0)
            atom_i_embed = group_i_embed.index_select(index=atom_group,dim=0)
            x_pieces_embed.append(atom_i_embed)

        x_pieces_embed = torch.cat(x_pieces_embed, dim=0)

        node_embedding = self.atom_gnn.encoder_node(x, pieces_in_index, piece_in_attr)

        atom_mol_embed = z_vecs.index_select(index=atom_batch.batch,dim=0)
        atom_embed = torch.cat([node_embedding,x_pieces_embed,atom_mol_embed],dim=-1)
        atom_embed = self.atom_embedding_predictor(atom_embed)


        atom_index = atom_batch.atom_index
        piece_edge_id = atom_batch.piece_edge_id
        true_atom_index = atom_batch.true_atom_id


        #neg  = atom_index[:,~true_atom_index.bool()]
        #pieces_link_index = atom_index[:, true_atom_index.bool()]

        #sample_len = pieces_link_index.shape[1]
        #sample = torch.randint(neg.shape[1], size=(sample_len * 10,))
        #neg_edge_index = neg[:, sample]

        #all_edge_index = torch.cat([neg_edge_index, pieces_link_index], dim=-1)

        #link_logits = atom_embed[atom_index[0]] * atom_embed[atom_index[1]]
        
        #link_logits = self.edge_predictor(link_logits) # 预测连接概率
        
        
        #E = all_edge_index.shape[1]
        #link_labels = torch.zeros(E, 2, dtype=torch.float, device=edge_index.device)
        #link_labels[:sample_len * 10, 0] = 1.
        #link_labels[sample_len * 10:, 1] = 1.
        # link_labels[sample_len + len(pieces_link_index) +  len(pieces_in_index):, 3] = 1.
        # link_labels[(neg_1_len + pos_len):, 2] = 1.  # 无边2

        #edge_link_loss = self.edge_link_loss(link_logits, link_labels)


        edge_link_loss = 0
        all_link_logits = atom_embed[atom_index[0]] * atom_embed[atom_index[1]]
        link_fp = atom_batch.link_fp

        all_link_logits = torch.cat([link_fp, all_link_logits], dim=-1)
        all_link_logits = self.atom_pieces_pred(all_link_logits)  # 预测连接概率
        edge_link_loss = self.edge_link_loss(all_link_logits, true_atom_index)

        all_link_logits = all_link_logits[:,1].squeeze()  # 为1的概率

        #all_link_logits = torch.where(all_link_logits < 0.5, torch.tensor(0.0), all_link_logits)

        piece_link_logits = []
        piece_link_truth = []

        # Temporary groups for current sequence
        current_a_group = [all_link_logits[0]]
        current_b_group = [true_atom_index[0]]

        # Iterate over the list to group adjacent elements
        for i in range(1, len(piece_edge_id)):
            if piece_edge_id[i] == piece_edge_id[i - 1]:  # If the current element matches the previous
                current_a_group.append(all_link_logits[i])
                current_b_group.append(true_atom_index[i])
            else:  # Otherwise, finalize the current group and start a new one
                piece_link_logits.append(current_a_group)
                piece_link_truth.append(current_b_group)
                current_a_group = [all_link_logits[i]]
                current_b_group = [true_atom_index[i]]

        # Add the last group
        piece_link_logits.append(current_a_group)
        piece_link_truth.append(current_b_group)

        # Output the results
        #print(piece_link_logits)
        padded_pred = [F.pad(torch.tensor(sequence), (0, self.max_atom_index - len(sequence)),value=-torch.inf) for sequence in piece_link_logits]
        padded_pred = torch.stack(padded_pred, dim=0).to(atom_index.device)


        piece_link_truth = [p.index(torch.tensor(1)) for p in piece_link_truth]
        piece_link_truth = torch.tensor(piece_link_truth).to(atom_index.device)



        edge_loss = self.edge_loss(padded_pred, piece_link_truth)


        return piece_loss, edge_link_loss, edge_loss



