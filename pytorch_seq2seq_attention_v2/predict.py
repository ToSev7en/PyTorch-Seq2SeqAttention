import os
import math
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchtext.data import Field, BucketIterator, TabularDataset
from torch.utils.data import TensorDataset

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

from utils import build_vocab, load_datasets
from dataset import Digits, DigitSeqDataset, DigitSeqTestDataset
from dataset import digit2index, index2digit, vocab_size
from models import Encoder, Decoder, Attention, Seq2Seq

# build vocab and vocab2index mapping
"""
0: '<PAD>',
1: '<UNK>',
2: '<GO>',
3: '<EOS>',
4: '6',
5: '2',
6: '7',
7: '8',
8: '9',
9: '5',
10: '1',
11: '4',
12: '3'
"""

def predict(model, iterator, criterion):
    
    model.eval()

    preds = []
    
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(iterator)):
            
            src = batch.get('input_seq_tensor').squeeze().permute(1, 0) 
            # trg = batch.get('output_seq_tensor').squeeze().permute(1, 0) 

            trg = None

            #turn off teacher forcing
            output = model(src, trg, 0) 
            
            
            topv, topi = output.data.topk(1)

            # print(topi)
            """
            [[ 5],
            [ 2],
            [ 2],
            [ 9],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [10],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 2],
            [ 4],
            [ 2],
            [ 2],
            [ 2]]
            """

            # print(topi.size()) -> torch.Size([22, 32, 1])


            decoder_output = topi.squeeze().detach()

            # print(decoder_output)
            """
            tensor([[12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
                    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
                    [ 8,  8, 10, 12,  9,  6,  5,  8,  5, 12,  8,  8,  8,  8,  7,  6, 12,  6,
                    11,  5,  8,  8, 11,  8,  5, 12,  8, 11,  9,  9, 11,  8],
                    [10,  8,  5,  4,  9,  8,  7, 11,  4,  8, 12,  8, 11,  8,  5, 10,  9,  7,
                    12,  4, 12,  6,  6, 10,  9, 12,  8,  7,  7,  8, 12,  7],
                    [11,  5,  4, 12,  5,  6,  5, 10, 12,  4, 12,  4,  7,  4, 12,  4,  4,  4,
                    4, 12,  8,  4,  5, 11, 11,  9,  6, 11,  9,  8,  2, 11],
                    [ 2,  2,  7,  9,  2,  2,  2,  8, 10,  2,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2,  2,  2,  2,  2,  7,  2,  2,  2,  2,  2,  2],
                    [ 2,  2,  2,  9,  8,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2, 12,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2]])
            """

            decoder_output = decoder_output[1:].permute(1, 0)

            # print(decoder_output,decoder_output.size())
            """
            tensor([[11,  7, 12,  4, 11,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2],
                    [ 8,  6,  6,  7,  9,  5,  7,  6,  4,  8,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2],
                    [ 6,  8,  7,  8, 12,  7,  7,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2],
                    [11,  8, 11,  7, 10,  5,  6, 12,  5, 11,  6,  8,  9,  8,  2,  2,  2,  2,
                    2,  2,  2],
                    [ 6,  9,  7,  8,  9,  7,  6,  5,  8,  6, 12,  4,  4,  9,  2,  2,  2,  2,
                    2,  2,  2],
                    [ 6, 12, 12,  7,  7,  5, 10,  4,  5,  9, 11,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  9],
                    [ 4,  9, 12, 10,  5, 12, 10,  6,  4,  6,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2],
                    [ 8,  7,  5,  9,  9,  6,  9,  5, 11,  2,  2,  2,  2,  2,  2,  2,  2,  2,
                    2,  2,  2]]) torch.Size([32, 21])
            """
            
            preds.append([0 if digit_idx.item()==2 else digit_idx.item() for seq in decoder_output for digit_idx in seq])

    return preds#.view()


if __name__=='__main__':
    
    INPUT_DIM = len(digit2index)
    OUTPUT_DIM = len(digit2index)

    ENC_EMB_DIM = 8
    DEC_EMB_DIM = 8

    ENC_HID_DIM = 64
    DEC_HID_DIM = 64

    ENC_DROPOUT = 0
    DEC_DROPOUT = 0


    SAVE_DIR = './checkpoints/'

    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_attention_model.pt')

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)
    
    print('load model from checkpoints...')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    optimizer = optim.Adam(model.parameters())

    # <PAD> index = 0
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.eval()

    print('load test inputs dataset:')
    test_input_csv = './dataset/task8_test_input.csv'
    test_inputs = pd.read_csv(test_input_csv, header=None)
    test_input_dataset = DigitSeqTestDataset(test_inputs)
    test_loader = torch.utils.data.DataLoader(test_input_dataset, batch_size=32, shuffle=False)
    test_iterator = test_loader

    print('predicting...')
    test_result = predict(model, test_iterator,criterion)

    pred_array = np.array([index2digit[item] for batch in test_result for item in batch]).reshape(1000,32,21).reshape(32000,21)
    pd.DataFrame([list("".join(list(i)).replace('<PAD>','0')[:-1]) for i in pred_array]).to_csv('reverse_seq_predictions .csv', header=None,index=False)

    print('Generating reversed sequence csv done!')