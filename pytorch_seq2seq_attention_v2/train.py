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

torch.manual_seed(1)

from utils import build_vocab, load_datasets
from dataset import Digits, DigitSeqDataset, DigitSeqTestDataset
from dataset import digit2index, index2digit, vocab_size
from models import Encoder, Decoder, Attention, Seq2Seq


# datasets
train_input_csv = './dataset/task8_train_input.csv'
train_output_csv = './dataset/task8_train_output.csv'
test_input_csv = './dataset/task8_test_input.csv'

# Load Datasets
print('Load Datasets')
train_inputs, train_outputs, test_inputs = load_datasets(train_input_csv, train_output_csv, test_input_csv)

# Split train and valid datasets. ratio = 0.3
print('Split train and valid datasets')
train_inputs, valid_inputs, train_outputs, valid_outputs = train_test_split(train_inputs, train_outputs, test_size=0.3, random_state=42)


# build vocab and vocab2index mapping
"""
'<PAD>': 0, 
'<GO>': 1, 
'<EOS>': 2, 
'<UNK>': 3, 
'1': 4, 
'2': 5, 
'3': 6, 
'4': 7, 
'5': 8, 
'6': 9, 
'7': 10, 
'8': 11, 
'9': 12
"""
# digit2index, index2digit, vocab_size = build_vocab('123456789')

train_input_output_dataset = DigitSeqDataset(train_inputs, train_outputs)
valid_input_output_dataset = DigitSeqDataset(valid_inputs, valid_outputs)

train_loader = torch.utils.data.DataLoader(train_input_output_dataset, batch_size=32, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_input_output_dataset, batch_size=32, shuffle=False)


def train(model, iterator, optimizer, criterion, clip):
    print('training...')

    """
    First, we'll set the model into "training mode" with model.train(). 
    This will turn on dropout (and batch normalization, which we aren't using) 
    and then iterate through our data iterator.
    """
    model.train()

    epoch_loss = 0
    
    for i, batch in tqdm(enumerate(iterator)):
        
        # get the source and target sentences from the batch,  X and  Y
        src = batch.get('input_seq_tensor').squeeze().permute(1,0)
        trg = batch.get('output_seq_tensor').squeeze().permute(1,0)
        
        # zero the gradients calculated from the last batch
        optimizer.zero_grad()
        
        # feed the source and target into the model to get the output,  Ŷ 
        output = model(src, trg, 1.0)
        # print(output, output.size()) -> torch.Size([22, 32, 13])

        # print('model output:',output)

        # pred = output[1:].contiguous().view(-1, output.shape[2]).argmax(1, keepdim=False).view(21,32).permute(1,0)

        # print(pred,pred.size())
        
        # pred_lst = [0 if digit.item()==2 else digit.item() for seq in pred for digit in seq]

        # pred = torch.tensor(pred_lst).view(32,21)

        # print('pred:',pred, pred.size())
        
        """
        as the loss function only works on 2d inputs with 1d targets we need to flatten each of them with .view
        we also don't want to measure the loss of the <sos> token, 
        hence we slice off the first column of the output and target tensors
        """
        loss = criterion(output[1:].view(-1, output.shape[2]), trg[1:].contiguous().view(-1))
        
        # calculate the gradients with loss.backward()
        loss.backward()
        
        # clip the gradients to prevent them from exploding (a common issue in RNNs)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # update the parameters of our model by doing an optimizer step
        optimizer.step()
        
        # sum the loss value to a running total
        epoch_loss += loss.item()

    # Finally, we return the loss that is averaged over all batches.
    return epoch_loss / len(iterator)



def evaluate(model, iterator, criterion):
    print('evaluating...')
    
    model.eval()

    print(model)

    epoch_loss = 0

    correct = 0
    
    with torch.no_grad():
        for i, batch in tqdm(enumerate(iterator)):
            src = batch.get('input_seq_tensor').squeeze().permute(1, 0)
            trg = batch.get('output_seq_tensor').squeeze().permute(1, 0)

            # print(src,src.size())
            # print(trg)
            # print(trg.size())

            """
            [[ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
            1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
            [ 6, 11,  4,  6,  9,  8,  7, 10,  7, 12,  5,  8,  4,  4,  9,  6,  8,  5,
            10,  6,  7,  5,  4,  4,  9,  4,  4,  8,  6,  4,  8,  5],
            [ 4,  7, 11, 10, 11,  8,  4,  7, 10, 12,  6,  6,  9,  5,  5,  6,  8,  7,
            11,  6,  7,  8, 11,  9,  5, 10,  9, 10, 10,  5, 11,  7],
            [ 4, 10, 11,  6,  9,  4,  6, 12,  2, 12,  7, 10,  9, 12,  7,  2, 11, 12,
            9,  5,  6,  4,  9,  7,  8,  9,  4,  8, 11, 11, 10,  5],
            [ 8, 10,  8,  4,  4,  4,  5,  9,  0, 11, 11, 11, 12,  8, 10,  0,  6,  2,
            11,  5,  9,  6,  4, 11, 11,  9, 10,  4,  6,  6,  6,  4],
            [11, 11,  9, 10,  5,  8,  8,  5,  0,  2, 12,  4,  9,  8,  8,  0, 11,  0,
            7,  5,  5,  4,  9, 11,  8,  2,  4, 12,  2,  5,  5,  4],
            [ 6, 12,  6,  4,  2,  2,  7, 12,  0,  0,  5, 12,  4,  9,  7,  0, 12,  0,
            7,  5, 10,  5,  6,  9,  7,  0,  4,  8,  0, 12,  2, 10],
            [ 8,  7, 12,  9,  0,  0,  7,  2,  0,  0, 11,  7, 11,  5,  4,  0,  5,  0,
            6,  4,  2,  6,  6,  9,  6,  0,  5, 12,  0,  6,  0,  2]]
            torch.Size([22, 32])
            """

            # turn off teacher forcing
            output = model(src, trg, 0) 

            # print(output)
            """
            [
            [[-0.0788,  0.0934,  0.1492,  ..., -0.0456, -0.0774, -0.0299],
            [ 0.2416,  0.3212,  0.2847,  ..., -0.1371,  0.1734,  0.2213],
            [-0.1152,  0.0938,  0.1734,  ..., -0.0049, -0.0787,  0.0410],
            ...,
            [ 0.2418,  0.3506,  0.2690,  ..., -0.1100,  0.1444,  0.3129],
            [ 0.1861,  0.2770,  0.1419,  ..., -0.0458,  0.2547,  0.1294],
            [ 0.2225,  0.2923,  0.3220,  ..., -0.0942,  0.1761,  0.2604]],
            ]

            "torch.Size([22, 32, 13])"

            """
            

            # topv, topi = output.data.topk(1)

            # print(topi)

            # decoder_output = topi.squeeze().detach()

            # print(decoder_output)

            # decoder_output = decoder_output[1:].permute(1, 0)

            # print(decoder_output)
            '[10,  8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]'

            # print(decoder_output.size())
            'torch.Size([32, 21])'

            
            # preds.append([digit_idx.item() for seq in decoder_output for digit_idx in seq])

            """用于训练 C 类别 classes 的分类问题
            输入 input 包含了每一类别的概率或score.

            输入 input Tensor 的大小是 (minibatch, C) 或 (minibatch,C,d1,d2,...,dK). K≥2K≥2 表示 KK-dim 场景.

            输入 target 是类别 class 的索引([0,C−1], C 是类别 classes 总数.)

            672 X 13      torch.Size([672])
            """
            # output[:1] 为
            # trg[:1] 为 <GO> 的 index

            # print(output[1:].view(-1, output.shape[2]).size())
            'torch.Size([672, 13])'

            # print(trg[1:].contiguous().view(-1))
            # print(trg[1:].contiguous().view(-1).size())
            'torch.Size([672]) -> 32 x 21' 

            loss_pred = output[1:].view(-1, output.shape[2])
            loss_trg = trg[1:].contiguous().view(-1)

            loss = criterion(loss_pred, loss_trg)

            # print(output[1:].contiguous().view(-1, output.shape[2]).argmax(1, keepdim=False).view(21,32).permute(1,0))
           
            # output.shape[2] -> 13
            pred = output[1:].contiguous().view(-1, output.shape[2]).argmax(1, keepdim=False).view(21,32).permute(1,0)
            actual = trg[1:].contiguous().view(21, 32).permute(1,0)
            # print('pred',pred)

            # print('trg:', trg[1:].contiguous().view(21, 32).permute(1,0))
            
            # print(pred.size()) -> torch.Size([32, 21])
            
            # pred_lst = [0 if digit.item()==2 else digit.item() for seq in pred for digit in seq]
            # pred_lst = [digit.item() for seq in pred for digit in seq]

            # pred_lst = [0 if digit==2 else digit for batch in list(pred.numpy()) for seq in batch for digit in seq]

            # pred = torch.tensor(pred_lst).view(32,21)

            # print('pred:',pred, pred.size())

            # print('trg:',trg[1:].contiguous().view(21, 32).permute(1,0))

            # torch.Size([32, 21])
            'torch.Size([672, 1])'

            # print(trg[1:].contiguous().view(-1))

            # print(trg[1:].contiguous().view(-1).view_as(pred).size())
            # torch.Size([672, 1]

            # 比较相等 tensor.eq()
            # view_as 返回被视作与给定的 tensor 相同大小的原tensor

            # print(pred.size())
            'torch.Size([672, 1])'
            'trg[1:].contiguous().view(-1) -> torch.Size([672]) -> 32 x 21'

            # torch.eq(input, other, out=None) → Tensor    
            # other可以为Tensor或者float，判断两个是否相等，得到0 1 Tensor

            # trg[1:].contiguous().view(21, 32).permute(1,0)

            # correct += list(pred.ne(trg).sum(1).numpy()).count(0)
            correct += list(pred.ne(actual.view_as(pred)).sum(1).numpy()).count(0)
            # correct += (pred == trg[1:].contiguous().view(21, 32).permute(1,0).view_as(pred)).sum()

            # correct += pred.eq(trg[1:].contiguous().view(-1, 21).view_as(pred)).sum(1).item()

            # print(correct)

            epoch_loss += loss.item()
    
    accuracy = correct / (len(iterator)*32)
    
    print(f'correct nums: {correct} , accuracy: {accuracy}')
        
    return epoch_loss / len(iterator)


if __name__=='__main__':

    # print(digit2index)

    INPUT_DIM = len(digit2index)
    OUTPUT_DIM = len(digit2index)

    ENC_EMB_DIM = 8
    DEC_EMB_DIM = 8

    ENC_HID_DIM = 64
    DEC_HID_DIM = 64

    ENC_DROPOUT = 0
    DEC_DROPOUT = 0

    attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

    model = Seq2Seq(enc, dec, device).to(device)

    print(model)

    optimizer = optim.Adam(model.parameters())

    criterion = nn.CrossEntropyLoss() #ignore_index=digit2index.get('<PAD>')


    EPOCHS = 1
    CLIP = 10
    SAVE_DIR = './checkpoints/'
    MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'seq2seq_attention_model.pt')

    best_valid_loss = float('inf')

    if not os.path.isdir(f'{SAVE_DIR}'):
        os.makedirs(f'{SAVE_DIR}')
        
    train_iterator = train_loader
    valid_iterator = valid_loader

    for epoch in range(EPOCHS):
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        
        valid_loss = evaluate(model, valid_iterator, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            torch.save(model,'model.pkl')
        
        
        print(f'| Epoch: {epoch+1:03} | Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')