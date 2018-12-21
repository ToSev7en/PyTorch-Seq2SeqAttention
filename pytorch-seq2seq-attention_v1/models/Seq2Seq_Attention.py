import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

MAX_LENGTH = 21

class DigitsSeq:
    def __init__(self):
        self.digit2index = {}
        self.digit2count = {}
        self.index2digit = {0: "<SOS>", 1: "<EOS>", 2:"<PAD>"}
        # Count SOS and EOS
        self.n_digits = 3
        
    def __len__(self):
        return len(self.index2digit)

    def add_digits_seq(self, digits_sequence):
        for digit in digits_sequence:
            self.add_digit(digit)

    def add_digit(self, digit):
        if digit not in self.digit2index:
            self.digit2index[digit] = self.n_digits
            self.digit2count[digit] = 1
            self.index2digit[self.n_digits] = digit
            self.n_digits += 1
        else:
            self.digit2count[digit] += 1

# The encoder of a seq2seq network is a RNN that outputs some value
# for every word from the input sentence. 
# For every input word the encoder outputs a vector and a hidden state, 
# and uses the hidden state for the next input word.
class GRUEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUEncoder, self).__init__()

        self.hidden_size = hidden_size

        # vocab_size, embedding_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # here I use GRU instead
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        # 输出 output 和 状态向量 hidden
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)


# 解码器
class GRUDecoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)

        # use relu activitation function
        output = F.relu(output)


        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=24):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Attention 
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)