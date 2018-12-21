import os
import math
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from dataset import digit2index, index2digit, vocab_size


class Encoder(nn.Module):
    
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        
        self.emb_dim = emb_dim
        
        self.enc_hid_dim = enc_hid_dim
        
        self.dec_hid_dim = dec_hid_dim
        
        self.dropout = dropout
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        embedded = self.dropout(self.embedding(src))
        
        outputs, hidden = self.rnn(embedded)
        
        # outputs = [sent len, batch size, hid dim * num directions]
        # hidden = [n layers * num directions, batch size, hid dim]
        
        # hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        # outputs are always from the last layer
        
        # hidden [-2, :, : ] is the last of the forwards RNN 
        # hidden [-1, :, : ] is the last of the backwards RNN
        
        # initial decoder hidden is final hidden state of the forwards and backwards encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        
        # outputs = [sent len, batch size, enc hid dim * 2]
        # hidden = [batch size, dec hid dim]
        
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        
        self.v = nn.Parameter(torch.rand(dec_hid_dim))
        
    def forward(self, hidden, encoder_outputs):
        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]
        
        #repeat encoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        
        #hidden = [batch size, src sent len, dec hid dim]
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        
        #energy = [batch size, src sent len, dec hid dim]
        
        energy = energy.permute(0, 2, 1)
        
        #energy = [batch size, dec hid dim, src sent len]
        
        #v = [dec hid dim]
        
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        
        #v = [batch size, 1, dec hid dim]
                
        attention = torch.bmm(v, energy).squeeze(1)
        
        #energy = [batch size, src len]
        
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        
        self.out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):
             
        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src sent len, batch size, enc hid dim * 2]
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        #embedded = [1, batch size, emb dim]
        
        a = self.attention(hidden, encoder_outputs)  
        #a = [batch size, src len]
        
        a = a.unsqueeze(1)
        #a = [batch size, 1, src len]
        
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        #encoder_outputs = [batch size, src sent len, enc hid dim * 2]
        
        weighted = torch.bmm(a, encoder_outputs)
        #weighted = [batch size, 1, enc hid dim * 2]
        
        weighted = weighted.permute(1, 0, 2)
        #weighted = [1, batch size, enc hid dim * 2]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)
        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]
            
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        #output = [sent len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        
        #sent len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()
        
        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)
        
        output = self.out(torch.cat((output, weighted, embedded), dim=1))
        #output = [bsz, output dim]
        
        return output, hidden.squeeze(0)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg,teacher_forcing_ratio=0.95):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        
        batch_size = src.shape[1]

        max_len = src.shape[0]
    
        vocab_size = self.decoder.output_dim
        # print('trg_vocab_size:',trg_vocab_size)
        
        # zeros tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, vocab_size).to(self.device)
        # print(outputs)
        
        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        # first input to the decoder is the <GO> tokens
        # output = torch.tensor([digit2index.get('<GO>')] * batch_size)
        # <GO> - 1
        output = torch.tensor([1] * batch_size)
    
        # 为什么 range 从 1 开始？
        for t in range(1, max_len):
            
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            
            # output
            # [-3.6458e-01, -4.8386e-01,  5.8300e-01, -6.9097e-01,  2.2829e-01,
            # 1.2765e-01, -1.0911e-02,  1.4811e-01,  2.9055e-01,  2.4679e-01,
            # 8.2105e-02,  1.9709e-01,  2.9024e-01]
            # torch.Size([32, 13])
            # print(output, output.size())
            
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio

            # print('output.max(1):', output.max(1)) 返回每行的最大值及其 在 vocab——size 位置
            """
            (       
            tensor([0.8058, 0.5901, 0.6381, 0.5621, 0.5681, 0.8082, 0.8403, 0.6904, 0.6290,
            0.6781, 0.8079, 0.8254, 0.8949, 0.5739, 0.4605, 0.7892, 0.9507, 0.7178,
            0.7695, 0.8277, 0.6578, 0.7333, 0.7485, 0.8497, 0.6989, 0.5069, 0.5275,
            0.8578, 0.6744, 0.7343, 0.6102, 0.7272], grad_fn=<MaxBackward0>), 

            tensor([10,  9, 10,  9,  5, 10,  4,  4,  9,  7, 10, 10,  5,  9,  7,  4, 12,  7,
            10,  5, 10, 10, 10,  4, 10,  9,  9,  7,  9, 10,  9, 10])
            )
            [32]
            """
            
            top1 = output.max(1)[1]
            # _, topi = output.data.topk(1)
            # index = topi[0][0]
            # top1 = index

            # print(top1)


            # print(top1.size()) torch.Size([32])

            # if 2 in list(top1.numpy()):
            #     lst = [0 if i==2 else i for i in list(top1.numpy())]
            # #     # eos_idx = lst.index(0)
            # #     # lst[eos_idx] = 2
            #     top1 = torch.tensor(lst).view(32)

            # print('output:',top1)

            # print('trg[t]: ',trg[t])
            """
            trg[t]: tensor([ 8,  7,  8,  4,  8,  9,  8,  5, 10,  5, 12,  9, 11,  7,  8,  9,  9,  7,
         5, 12, 10,  4, 10,  7,  5,  9, 10, 12, 11,  6, 11,  7])
            """
            # print('trg[t]: ',trg[t].size())
            # torch.Size([32])

            output = (trg[t] if teacher_force else top1)

            # if top1 == 2:
            #     break

            # print('output:',output)
        # print('decoder outputs',outputs,outputs.size())
        return outputs