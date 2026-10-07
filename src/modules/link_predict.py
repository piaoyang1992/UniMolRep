#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/8/21 14:14 
# @Author : piaoyang cao 
# @File : link_predict.py

import networkx as nx
from deepsnap.graph import Graph
import torch
import torch.nn.functional as F
#from sklearn.metrics import roc_auc_score
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
#from data.bpe_dataset import *
from data.group_graph_dataset import get_dataloader
#from data.mol_bpe import Tokenizer
from data.mol_group_graph import Tokenizer, get_vocab



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 128)
        self.conv2 = GCNConv(128, 64)

    def encode(self):
        x = self.conv1(data.x, data.train_pos_edge_index)
        x = x.relu()
        return self.conv2(x, data.train_pos_edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
        return logits

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()


def get_link_labels(pos_edge_index, neg_edge_index):
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float, device=device)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels




def train(model,data):
    model.train()
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index, num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1),
        force_undirected=True,
    )
    optimizer.zero_grad()
    z = model.encode()
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss


@torch.no_grad()
def test(model,data):
    model.eval()
    perfs = []
    for prefix in ["val", "test"]:
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']
        z = model.encode()
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = link_logits.sigmoid()
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        perfs.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return perfs






if __name__ == '__main__':
    G = nx.karate_club_graph()
    data = Graph(G)
    print(data)

    path = '/home/pycao/PS-VAE-main-1/data/np'
    train_set = '/home/pycao/PS-VAE-main-1/data/np/np_sample.csv'
    tokenizer = Tokenizer(path)
    vocab = tokenizer.chem_vocab
    train_loader = get_dataloader(train_set, tokenizer, batch_size=args.batch_size,
                                  shuffle=args.shuffle, num_workers=args.num_workers)


    data.num_features = 3
    data.x = torch.ones((data.num_nodes, data.num_features), dtype=torch.float32)
    data.edge_attr = data.weight
    data = train_test_split_edges(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data = Net().to(device), data.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)

    best_val_perf = test_perf = 0
    for epoch in range(1, 11):
        train_loss = train(model,data)
        val_perf, tmp_test_perf = test(model,data)
        if val_perf > best_val_perf:
            best_val_perf = val_perf
            test_perf = tmp_test_perf
        log = 'Epoch: {:03d}, Loss: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(epoch, train_loss, best_val_perf, test_perf))

    z = model.encode()
    final_edge_index = model.decode_all(z)

