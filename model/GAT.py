#!/usr/bin/env python
# -*- coding: utf-8

# transformer: copy from https://github.com/JayParks/transformer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init


class AttentionUnit(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha):

        super(AttentionUnit, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        
        b = input.shape[0]
        h = torch.bmm(input, self.W.repeat([b,1]).view(b,self.W.shape[0],self.W.shape[1])) # [batch_size, N, out_features]
        N = h.shape[1]
        
        e = self.leakyrelu(torch.matmul(h, self.a).squeeze(2)) # [batch_size,N]

        attention = torch.softmax(e, dim=1)# [batch_size,N]
        attention = torch.dropout(attention, self.dropout, train=self.training)
        h_prime = torch.bmm(attention.view(b,1,-1), h).squeeze(1) # [batch_size, out_features]

        return torch.nn.functional.elu(h_prime)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'



class EncoderLayer(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim, n_heads, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attentions = [AttentionUnit(in_features, hidden_dim, dropout=dropout, alpha=0.2) for _ in range(n_heads)]
        for j,attn in enumerate(self.attentions):
            self.add_module('attention_%d'%(j), attn)
        self.out_attention = AttentionUnit(hidden_dim, out_features, dropout=dropout, alpha=0.2)
        self.add_module('out_attention', self.out_attention)

    def forward(self, enc_inputs):
        x = torch.cat([att(enc_inputs).view(-1,1,self.hidden_dim) for att in self.attentions], dim=1) # [batch_size, n_heads, hidden_dim]
        # print(x.shape)
        x = self.out_attention(x) # [batch_size, out_features]

        return x


class GAT(nn.Module):
    def __init__(self, fv='res18-128', dropout=0.0, NUM_HEADS=6, hid_dim=32 ,num_classes=10,NUM_LAYERS=4):
        super(GAT, self).__init__()
        if fv == 'matlab':
            MODEL_DIM = 1097
        else:
            MODEL_DIM = int(fv.split("-")[-1])

        self.encoder = EncoderLayer(MODEL_DIM, hid_dim, hid_dim, NUM_HEADS,dropout)


        self.proj = nn.Linear(hid_dim, num_classes)


    def forward(self, enc_inputs):
        enc_outputs = self.encoder(enc_inputs)
        out = self.proj(enc_outputs)

        # print(enc_inputs.shape,enc_outputs.shape,out.shape)
        return torch.sigmoid(out)


if __name__ == "__main__":
    pass
