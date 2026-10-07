#!/usr/bin/python
# -*- coding:utf-8 -*-
from collections import defaultdict
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.data import Batch, Data

from modules.common_nn import MLP
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from modules.graph_embed import GNN_mol

atom_types = 120 #including the extra mask tokens
num_chirality_tag = 3

bond_types = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3



class GroupEncoder(nn.Module):
    def __init__(self, piece_in_dim, piece_link_dim,dim_hidden, t, graph_pooling):
        super(GroupEncoder, self).__init__()
        self.piece_in_dim = piece_in_dim
        self.dim_hidden = dim_hidden
        self.piece_link_dim = piece_link_dim

        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(piece_in_dim, dim_hidden),
            nn.ReLU(),
            torch.nn.BatchNorm1d(dim_hidden),
            torch.nn.Dropout(0),

        )
        self.attr_embed = torch.nn.Sequential(
            torch.nn.Linear(piece_link_dim, dim_hidden),
            nn.ReLU(),
            torch.nn.BatchNorm1d(dim_hidden),
            torch.nn.Dropout(0.1),

        )

        self.graph_gnn = GNN_mol(t, dim_hidden, 'gin')
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        self.W_mean = nn.Linear(dim_hidden, dim_hidden)
        self.W_log_var = nn.Linear(dim_hidden, dim_hidden)


    def encoder_node(self, vocab_datas, batch_data, edge_attr=True):
        x, edge_index, batch = batch_data.x, batch_data.edge_index, batch_data.batch

        # print(vocab_datas.size(), x)
        x_input_1 = vocab_datas.index_select(index=x.squeeze(), dim=0)
        if not edge_attr:
            edge_attr = torch.zeros(edge_index.size(1), self.piece_link_dim).to(x.device)

        else:
            edge_attr = batch_data.edge_attr
            inter_tensor = batch_data.inter_tensor
            edge_attr_0 = torch.cat([inter_tensor[:, 0, :], edge_attr[:, 0:2]], dim=1).to(torch.float)
            edge_attr_1 = torch.cat([inter_tensor[:, 1, :], edge_attr[:, 2:4]], dim=1).to(torch.float)
            edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)
            edge_attr_trans = torch.cat([edge_attr_1, edge_attr_0], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr_trans], dim=0)  # 增加反向边信息

            edge_index_trans = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
            edge_index = torch.cat([edge_index, edge_index_trans], dim=1)  # 增加反向边
            edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

            self_loop_attr = torch.zeros(x.size(0), 270)
            self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        x_tensor = self.node_embed(x_input_1)
        edge_attr = self.attr_embed(edge_attr)

        node_representation = self.graph_gnn(x_tensor, edge_index, edge_attr)

        return node_representation

    def rsample(self, conds):
        batch_size = conds.size(0)
        z_mean = self.W_mean(conds)
        z_log_var = -torch.abs(self.W_log_var(conds)) #Following Mueller et al.
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean)
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        #print(conds)
        #print(z_mean)
        #print(torch.any(torch.isnan(conds)), torch.any(torch.isnan(z_mean)), torch.any(torch.isnan(z_vecs)))
        return z_vecs, kl_loss

    def forward(self, vocab_data, group_batch):

        all_x = self.encoder_node(vocab_data, group_batch)
        # print(x.size()) #[1663, 200])
        # print(all_x.size())
        graph_embedding = self.pool(all_x, group_batch.batch)
        # print(graph_embedding.size())
        z_vecs, kl_loss = self.rsample(graph_embedding)

        return z_vecs, kl_loss



class AtomEncoder(nn.Module):
    def __init__(self, chem_vocab, piece_in_dim,  num_pos_type,  dim_hidden, t):
        super(AtomEncoder, self).__init__()


        self.piece_in_dim = piece_in_dim
        #self.edge_trans = nn.Embedding(num_edge_type, dim_hidden)

        self.piece_hidden = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            torch.nn.Dropout(0),

        )

        #self.atom_embedding = nn.Embedding(num_atom_type, dim_hidden)
        '''
        self.atom_hidden = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            torch.nn.Dropout(0),
        )
        '''
        self.pos_embedding = nn.Embedding(num_pos_type, dim_hidden)  # max position idx is 99, 0 is padding
        '''
        self.pos_hidden = torch.nn.Sequential(
            torch.nn.Linear(dim_hidden, dim_hidden),
            torch.nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            torch.nn.Dropout(0),

        )
        '''

        node_in = dim_hidden  + dim_hidden
        self.node_hidden = torch.nn.Sequential(
            torch.nn.Linear(node_in, dim_hidden),
            torch.nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            torch.nn.Dropout(),

        )
        self.gnn = GNN_mol(t,dim_hidden,'gin')


        self.x_embedding1 = torch.nn.Embedding(chem_vocab.num_atom_type(),dim_hidden)
        self.x_embedding2 = torch.nn.Embedding(chem_vocab.num_atom_chirality(), dim_hidden)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.edge_embedding1 = torch.nn.Embedding(chem_vocab.num_bond_type(), dim_hidden)
        self.edge_embedding2 = torch.nn.Embedding(chem_vocab.num_bond_direction(), dim_hidden)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def embed_atom(self, x, piece_ids, x_pos):
        atom_embed  = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        #atom_embed = self.atom_hidden(atom_embed)
        piece_embed = self.piece_hidden(piece_ids)
        #piece_embed = self.piece_hidden(piece_ids.view(-1,self.piece_in_dim))
        #pos_embed = self.pos_embedding(x_pos)
        #pos_embed = self.pos_hidden(pos_embed)
        node_embed = torch.cat([atom_embed, piece_embed], dim=-1)
        #node_embed = torch.cat([atom_embed, piece_embed,pos_embed], dim=-1)
        node_embed = self.node_hidden(node_embed)
        # torch.Size([128, 147, 50]) torch.Size([128, 147, 100]) torch.Size([128, 147, 50])
        return node_embed


    def encoder_node(self,x,edge_index, edge_attr):
        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        edge_attr = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])
        all_x  = self.gnn(x, edge_index, edge_attr)
        return all_x


    def link_predict(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def link_all_predict(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        E = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(E, dtype=torch.float, device=pos_edge_index.device)
        link_labels[:pos_edge_index.size(1)] = 1.
        return link_labels


