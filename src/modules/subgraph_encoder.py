#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2022/9/22 9:11 
# @Author : piaoyang cao 
# @File : subgraph_encoder.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops

import os
import numpy as np
import pandas as pd

from src.modules.graph_embed import GNN_mol
#from torch_geometric.nn import GCNConv, ChebConv, global_mean_pool
#from pcqm4mv2_mol_split import SmilesDataset, DrugDataset
#from frag2graph import VocabDataset
from tqdm import tqdm
# 得到整个图的隐变量表示




class Encoder(nn.Module):
    def __init__(self, vocab_size,num_edge_attr,embed_size,drop_ratio,graph_pooling):  # atom_sum 表示frag的最大原子数
        super(Encoder, self).__init__()
        #print(args)
        self.vocab = vocab_size
        self.num_edge_attr = num_edge_attr
        self.device = 'cpu'
        self.emb_dim = embed_size
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling



        self.graph_gnn = GNN_mol(3, self.emb_dim, 'gin', drop_ratio=0)


        if self.graph_pooling == "sum":
            self.pool = global_add_pool
        elif self.graph_pooling == "mean":
            self.pool = global_mean_pool
        elif self.graph_pooling == "max":
            self.pool = global_max_pool



        self.attr_embed = torch.nn.Sequential(
            torch.nn.Linear(num_edge_attr, self.emb_dim),
            torch.nn.ReLU(),

            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(0.1),
        )

        self.node_embed = torch.nn.Sequential(
            torch.nn.Linear(self.vocab[1] + self.emb_dim, self.emb_dim),
            #torch.nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            #torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim)
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Dropout(0),

        )

        self.piece_embedding = nn.Embedding(vocab_size[0], self.emb_dim)

    def forward(self, vocab_datas, group_batch):

        '''
        x, edge_index, edge_attr, inter_tensor,  batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, \
                                                                   batch_data.inter_tensor,  batch_data.batch

        '''
        x, edge_index, edge_attr, inter_tensor, batch = group_batch

        x = x.to('cpu')


        x_input_1 = vocab_datas.index_select(index=x.squeeze(), dim=0).to(self.device)
        x_embed = self.piece_embedding(x.squeeze())
        x_tensor = self.node_embed(torch.cat([x_input_1,x_embed],dim=1))


        edge_attr_0 = torch.cat([inter_tensor[:,0,:], edge_attr[:, 0:2]], dim=1).to(torch.float)
        edge_attr_1 = torch.cat([inter_tensor[:,1,:], edge_attr[:, 2:4]], dim=1).to(torch.float)

        edge_attr = torch.cat([edge_attr_0, edge_attr_1], dim=1)

        edge_attr_trans = torch.cat([edge_attr_1, edge_attr_0], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr_trans], dim=0)  #增加反向边信息


        edge_index_trans = torch.stack([edge_index[1, :], edge_index[0, :]], dim=0)
        #print(edge_index_trans.size())
        edge_index = torch.cat([edge_index, edge_index_trans], dim=1) # 增加反向边
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0),270)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_attr = self.attr_embed(edge_attr)
        #print(x_tensor.size(), edge_index.size(), edge_attr.size())
        node_representation = self.graph_gnn(x_tensor, edge_index, edge_attr)
        #print(node_representation.size())
        gat_out =  global_add_pool(node_representation, batch)
        return node_representation,gat_out



class Predictor(nn.Module):
    def __init__(self, encoder, latent_dim=128, num_tasks=1):
        super().__init__()
        self.encoder = encoder
        self.emb_dim = latent_dim
        self.predictor = torch.nn.Sequential(
            # torch.nn.BatchNorm1d(emb_dim),
            # torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim),
            torch.nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            torch.nn.BatchNorm1d(self.emb_dim),
            torch.nn.Linear(self.emb_dim, num_tasks)
        )

    def forward(self, vocab_datas, batch_data1):
        emb = self.encoder(vocab_datas, batch_data1)
        out = self.predictor(emb)

        return out

    def from_pretrained(self, model_path, device):
        pre_model = torch.load(model_path, map_location=device)['model_state_dict']
        #print(pre_model)
        self.encoder.load_state_dict(pre_model)




class SiameseNetwork(nn.Module):
    def __init__(self, encoder, latent_dim=300, dropout=0):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(latent_dim*4, latent_dim)
        )

    def forward(self, vocab_datas,x):
        _,emb = self.encoder(vocab_datas,x)
        emb = self.projector(emb)
        return emb

