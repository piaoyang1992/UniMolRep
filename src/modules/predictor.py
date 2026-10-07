#!/usr/bin/python
# -*- coding:utf-8 -*-
import torch.nn as nn


class Predictor(nn.Module):
    """given graph-level embedding, predict all properties"""
    def __init__(self, dim_hidden, num_property, encoder):
        """2-layer MLP"""
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.mlp = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU()
        )
        self.output = nn.Linear(dim_hidden, num_property)

    def get_z(self,x):
        hidden = self.mlp(x)
        return hidden

    def predict_z(self,x):
        hidden = self.mlp(x)
        return self.output(hidden)

    def forward(self, vocab_data, batch):
        x,_ = self.encoder(vocab_data, batch)
        #print(x.size())
        hidden = self.mlp(x)
        return self.output(hidden)