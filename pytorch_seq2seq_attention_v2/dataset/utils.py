
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data


class Digits:
    def __init__(self):
        self.digit2index = {"<PAD>": 0, "<GO>": 1, "<EOS>": 2, "<UNK>": 3}
        self.digit2count = {}
        self.index2digit = {0: "<PAD>", 1: "<GO>", 2:"<EOS>", 3:"<UNK>"}
        self.n_digits = 4
        
    def __len__(self):
        return self.n_digits

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

def build_vocab(sequence_corpus):
    digit_mapping = Digits()
    digit_mapping.add_digits_seq('123456789')
    digit2index = digit_mapping.digit2index
    index2digit = digit_mapping.index2digit
    vocab_size = len(digit_mapping)
    return digit2index, index2digit, vocab_size

digit2index, index2digit, vocab_size = build_vocab('123456789')


class DigitSeqDataset(data.Dataset):
    """DigitSeqDataset
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

    def __init__(self, inputs, outputs):
        """
            csv_file (string): Path to the csv file with annotations.
        """
        self.input_digit_seq_dataframe = inputs
        self.output_digit_seq_dataframe = outputs

        
    def pad_sentence(self, digits_sequence, max_digits_len, pad_token_id):
        '''
        对 batch 中的 digits序列进行补全，保证 batch 中的每行都有相同的 sequence_length
        参数：
        - sentence batch
        - pad_token_id: <PAD> 对应索引号
        '''
        # max_digits = max([len(digits) for digits in digits_seq_batch])
        return [digits_sequence + [pad_token_id] * (max_digits_len - len(digits_sequence))]
    
    def __len__(self):
        return len(self.input_digit_seq_dataframe)
    

    def __getitem__(self, idx):
        """
        """
        input_digit_seq = self.input_digit_seq_dataframe.iloc[idx, :].as_matrix()
    
        input_digit2index_seq = [digit2index.get(str(i), digit2index.get('UNK')) for i in input_digit_seq if i != 0]
        
        input_digit2index_seq.insert(0, digit2index.get('<GO>'))
        
        input_digit2index_seq.append(digit2index.get('<EOS>'))
        
        # zero padding
        input_digit2index_seq = self.pad_sentence(input_digit2index_seq, 22, digit2index.get('<PAD>'))
        
        input_digit2index_seq_tensor = torch.tensor(input_digit2index_seq)#.view(1, -1)
        
        
        
        output_digit_seq = self.output_digit_seq_dataframe.iloc[idx, :].as_matrix()
        
        output_digit2index_seq = [digit2index.get(str(i), digit2index.get('<UNK>')) for i in output_digit_seq if i != 0]
        
        output_digit2index_seq.insert(0, digit2index.get('<GO>'))
        
        output_digit2index_seq.append(digit2index.get('<EOS>'))
        
        # zero padding
        output_digit2index_seq = self.pad_sentence(output_digit2index_seq, 22, digit2index.get('<PAD>'))
        
        output_digit2index_seq_tensor = torch.tensor(output_digit2index_seq)#.view(1, -1)
        
        
        # seq_pairs = {'input_seq': input_digit_seq, 'output_seq': output_digit_seq}
        
        seq_pairs = [input_digit2index_seq, output_digit2index_seq]
        
        seq_tensor_pairs = {'input_seq_tensor': input_digit2index_seq_tensor, 'output_seq_tensor': output_digit2index_seq_tensor}
        
        return seq_tensor_pairs


class DigitSeqTestDataset(data.Dataset):
    """DigitSeqDataset"""

    def __init__(self, inputs):
        """
            csv_file (string): Path to the csv file with annotations.
        """
        self.input_digit_seq_dataframe = inputs

        
    def pad_sentence(self, digits_sequence, max_digits_len, pad_token_id):
        '''
        对 batch 中的 digits序列进行补全，保证 batch 中的每行都有相同的 sequence_length
        参数：
        - sentence batch
        - pad_int: <PAD>对应索引号
        '''
        # max_digits = max([len(digits) for digits in digits_seq_batch])
        return [digits_sequence + [pad_token_id] * (max_digits_len - len(digits_sequence))]
    
    def __len__(self):
        return len(self.input_digit_seq_dataframe)
    

    def __getitem__(self, idx):
        """
        在类的__getitem__函数中完成读取工作。
        这样是为了减小内存开销，只要在需要用到的时候才读入。
        """
        input_digit_seq = self.input_digit_seq_dataframe.iloc[idx, :].as_matrix()
    
        input_digit2index_seq = [digit2index.get(str(i), digit2index.get('UNK')) for i in input_digit_seq if i != 0]
        
        input_digit2index_seq.insert(0, digit2index.get('<GO>'))
        
        input_digit2index_seq.append(digit2index.get('<EOS>'))
        
        input_digit2index_seq = self.pad_sentence(input_digit2index_seq, 22, digit2index.get('<PAD>'))
        
        input_digit2index_seq_tensor = torch.tensor(input_digit2index_seq)#.view(1, -1)
        
        seq_tensor_pairs = {'input_seq_tensor': input_digit2index_seq_tensor}
        
        
        return seq_tensor_pairs