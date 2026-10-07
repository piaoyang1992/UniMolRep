#!/usr/bin/env python
# -*- coding: utf-8 -*- 
# @Time : 2024/11/18 16:15 
# @Author : piaoyang cao 
# @File : transformer.py.py

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2024/3/28 11:15
# @Author : piaoyang cao
# @File : rnn_rxn.py
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torch.nn.utils.rnn import pad_sequence
import math






def get_key_padding_mask(tokens, pad): #将'<pad>'掩蔽
    key_padding_mask = torch.zeros(tokens.size())
    key_padding_mask[tokens == pad] = -torch.inf
    return key_padding_mask


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, hidden_size,tokenizer):
        super(TransformerModel, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位小数。
        self.vocab_size = tokenizer.num_piece_type()
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=hidden_size)
        # 定义Transformer。超参是我拍脑袋想的
        self.transformer = nn.Transformer(d_model=hidden_size, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=512,
                                          batch_first=True)

        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(hidden_size, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.pad = tokenizer.pad_idx()

    def forward(self, src, tgt):
        # 自生成 src == tgt
        src = src.to(torch.long)
        tgt = tgt.to(torch.long)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size()[-1]).to(src.device)
        #print(tgt_mask)
        #self.pad = self.pad.to(src.device)
        src_key_padding_mask = TransformerModel.get_key_padding_mask(src, self.pad)
        #print(src_key_padding_mask)
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(tgt, self.pad)
        #print(tgt_key_padding_mask)
        #src_key_padding_mask = torch.zeros(src.size())
        #src_key_padding_mask[src == -torch.inf] = -torch.inf
        #src_key_padding_mask = torch.sum(src_key_padding_mask,dim=2)
        #src_key_padding_mask[src > 0] = -torch.inf

        #src[src == -torch.inf] = 0
        #print(src.size())
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        #print(tgt.size())
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_key_padding_mask,
                               tgt_key_padding_mask=tgt_key_padding_mask)

        return out

    @staticmethod
    def get_key_padding_mask(tokens, pad):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size()).to(tokens.device)
        key_padding_mask[tokens == pad] = 1
        return key_padding_mask.bool()







