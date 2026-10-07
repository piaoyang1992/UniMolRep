#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/8/21 14:14 
# @Author : piaoyang cao 
# @File : link_predict.py

import networkx as nx
#from deepsnap.graph import Graph
import torch
import torch.nn.functional as F
#from sklearn.metrics import roc_auc_score
#from torch_geometric.utils import negative_sampling
from  torch.utils.data.sampler import SubsetRandomSampler
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split, DataLoader
#from data.bpe_dataset import *
from data.group_graph_dataset import get_dataloader, MolDataset
#from data.mol_bpe import Tokenizer
from data.mol_group_graph import Tokenizer, get_vocab
from torch_geometric.utils import train_test_split_edges
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv
from modules.common_nn import MLP
import numpy as np

class Net(torch.nn.Module):
    def __init__(self,num_piece_type,max_pos=200, atom_embedding_dim=50, piece_embedding_dim=100,
                 pos_embedding_dim=100,dim_hidden=100,t=4):
        super(Net, self).__init__()
        self.t = t
        atom_dim = 133
        self.atom_embedding  = torch.nn.Sequential(
            torch.nn.Linear(atom_dim, atom_embedding_dim),
            #torch.nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            #torch.nn.GroupNorm(num_groups=4, num_channels=emb_dim)
            #torch.nn.BatchNorm1d(atom_embedding_dim),
            torch.nn.Dropout(0),

        )
        #self.atom_embedding = nn.Embedding(atom_dim, atom_embedding_dim)
        self.piece_embedding = nn.Embedding(num_piece_type, piece_embedding_dim)
        self.pos_embedding = nn.Embedding(max_pos, pos_embedding_dim)

        dim_in = atom_embedding_dim + piece_embedding_dim + pos_embedding_dim
        self.node_trans = nn.Linear(dim_in, dim_hidden)
        #self.edge_trans = nn.Embedding(num_edge_type, dim_hidden)
        #self.edge_trans = nn.Linear(num_edge_type, dim_hidden)
        self.edge_trans = nn.Linear(1, dim_hidden)
        self.conv = GINEConv(MLP(dim_hidden, dim_hidden, dim_hidden, nn.ReLU, 2))



    def embed_atom(self, atom_ids, piece_ids, x_pos):
        atom_embed = self.atom_embedding(atom_ids)
        piece_embed = self.piece_embedding(piece_ids)
        pos_embed = self.pos_embedding(x_pos)
        # torch.Size([128, 147, 50]) torch.Size([128, 147, 100]) torch.Size([128, 147, 50])
        return torch.cat([atom_embed, piece_embed, pos_embed], dim=-1)


    def encode_atom(self,x,edge_index,edge_attr):
        x = self.node_trans(x.float())                  # [total_num_nodes, dim_hidden]
        edge_attr = self.edge_trans(edge_attr.float()).squeeze(1)  # [total_num_edges, dim_hidden]
        all_x = []
        for _ in range(self.t):
            #print(x)
            x = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attr)
            #print(x)
            all_x.append(x)
        all_x = all_x[-1]
        #all_x = torch.cat(all_x, dim=-1)  # [total_num_nodes, dim_hidden * t]
        return all_x

    def link_predict(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def link_all_predict(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels




def train(model,train_loader,device):

    model.train()
    loss_accum = 0
    for batch in train_loader:
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']  # 原子图
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']  # 每个原子对应的piece ID
        in_piece_edge_idx = torch.tensor(batch['in_piece_edge_idx'])
        #none_id = torch.nonzero(in_piece_edge_idx == 0).squeeze()
        x = model.embed_atom(x, x_pieces, x_pos)
        batch_size, node_num, node_dim = x.shape

        edge_in_index = edge_index[:,in_piece_edge_idx]   # in_piece_edge_idx 作为训练集的边， inter_edge 作为测试集的边
        edge_attr = edge_attr.view(-1, 1)
        edge_in_attr = edge_attr[in_piece_edge_idx, ]


        neg = batch['neg_edge_index']

        sample = np.random.choice(range(neg.shape[1]), edge_index.shape[1])
        neg_edge_index = neg[:,sample]


        '''
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
            force_undirected=True,
        )
        '''
        model.to(device)
        optimizer.zero_grad()
        z = model.encode_atom(x.view(-1, node_dim), edge_in_index, edge_in_attr) # 得到所有节点的表示
        link_logits = model.link_predict(z, edge_index, neg_edge_index) # 正负边合并
        link_labels = get_link_labels(edge_index, neg_edge_index) # 计算边存在的概率
        loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
        print(loss)
        loss_accum += loss.detach().cpu().item()
        loss.backward()
        optimizer.step()

    return loss_accum / len(train_loader)



@torch.no_grad()
def test(model,loader):
    model.eval()
    y_pred_list = []
    y_true_list = []
    for batch in loader:
        x, edge_index, edge_attr = batch['x'], batch['edge_index'], batch['edge_attr']  # 原子图
        x_pieces, x_pos = batch['x_pieces'], batch['x_pos']  # 每个原子对应的piece ID
        in_piece_edge_idx = torch.tensor(batch['in_piece_edge_idx'])
        x = model.embed_atom(x, x_pieces, x_pos)
        edge_attr = edge_attr.view(-1, 1)
        batch_size, node_num, node_dim = x.shape

        edge_in_index = edge_index[:,in_piece_edge_idx]   # in_piece_edge_idx 作为训练集的边， inter_edge 作为测试集的边
        edge_in_attr = edge_attr[in_piece_edge_idx, ]

        neg = batch['neg_edge_index']
        sample = np.random.choice(range(neg.shape[1]), edge_index.shape[1])
        neg_edge_index = neg[:,sample]
        '''
        neg_edge_index = negative_sampling(
            edge_index=data.edge_index, num_nodes=data.num_nodes,
            num_neg_samples=data.edge_index.size(1),
            force_undirected=True,
        )
        '''
        model.to(device)
        optimizer.zero_grad()
        z = model.encode_atom(x.view(-1, node_dim), edge_in_index, edge_in_attr)  # 得到所有节点的表示
        y_pred = model.link_predict(z, edge_index, neg_edge_index)  # 正负边合并
        y_true = get_link_labels(edge_index, neg_edge_index)  # 计算边存在的概率

        y_pred_list.append(y_pred.detach().cpu())
        y_true_list.append(y_true.view(y_pred.shape).detach().cpu())
    y_pred = torch.cat(y_pred_list, dim=0)
    y_true = torch.cat(y_true_list, dim=0)
    y_pred_list = torch.sigmoid(y_pred)
    y_pred_label = torch.where(y_pred_list >= 0.5, 1, 0)
    accuracy = torch.eq(y_pred_label,y_true).float().mean()
    return accuracy






if __name__ == '__main__':
    '''
    G = nx.karate_club_graph()
    data = Graph(G)
    print(data)
    '''
    path = '/home/pycao/PS-VAE-main-1/data/np'
    train_set = '/home/pycao/PS-VAE-main-1/data/np/np_sample.csv'
    check_path = '/home/pycao/PS-VAE-main-1/data/np/link_predict_checkpoint.pt'
    tokenizer = Tokenizer(path)
    vocab = tokenizer.chem_vocab
    '''
    train_loader = get_dataloader(train_set, tokenizer, batch_size=False,
                                  shuffle=True, num_workers=0)
    '''

    dataset = MolDataset(train_set, tokenizer)
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[int(0.8 * len(dataset)), int(0.2 * len(dataset))],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,
                    collate_fn=dataset.collate_fn, num_workers=0)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True,
                    collate_fn=dataset.collate_fn, num_workers=0)


    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cuda'
    model = Net(tokenizer.num_piece_type())
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    test_perf = 0
    for epoch in range(1, 2):
        train_loss = train(model, train_loader, device)

        test_acc = test(model,test_loader)
        if test_acc > test_perf:
            test_perf = test_acc
            torch.save(model.state_dict(), check_path)
        '''
        val_perf, tmp_test_perf = test(model,batch)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        '''
        log = 'Epoch: {:03d}, Loss: {:.4f}, test_accuracy: {:.2f}'
        print(log.format(epoch, train_loss,test_acc))

    #z = model.encode()
    #final_edge_index = model.decode_all(z)

