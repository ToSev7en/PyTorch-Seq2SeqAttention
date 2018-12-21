class DigitSeqDataset(data.Dataset):
    """DigitSeqDataset"""

    def __init__(self, intput_csv, output_csv):
        """
            csv_file (string): Path to the csv file with annotations.
        """
        self.input_digit_seq_dataframe = pd.read_csv(intput_csv, header=None)[32:]
        self.output_digit_seq_dataframe = pd.read_csv(output_csv, header=None)[32:]

        
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
        
        # print('GO',input_digit2index_seq)
        
        input_digit2index_seq.append(digit2index.get('<EOS>'))
        
        # print(input_digit2index_seq)
        
        input_digit2index_seq = self.pad_sentence(input_digit2index_seq, 22, digit2index.get('<PAD>'))
        
        input_digit2index_seq_tensor = torch.tensor(input_digit2index_seq)#.view(1, -1)
        
        
        output_digit_seq = self.output_digit_seq_dataframe.iloc[idx, :].as_matrix()
        
        output_digit2index_seq = [digit2index.get(str(i), digit2index.get('<UNK>')) for i in output_digit_seq if i != 0]
        
        output_digit2index_seq.insert(0, digit2index.get('<GO>'))
        
        output_digit2index_seq.append(digit2index.get('<EOS>'))
        
        output_digit2index_seq = self.pad_sentence(output_digit2index_seq, 22, digit2index.get('<PAD>'))
        
        output_digit2index_seq_tensor = torch.tensor(output_digit2index_seq)#.view(1, -1)
        
        
        
        # seq_pairs = {'input_seq': input_digit_seq, 'output_seq': output_digit_seq}
        
        seq_pairs = [input_digit2index_seq, output_digit2index_seq]
        
        seq_tensor_pairs = {'input_seq_tensor': input_digit2index_seq_tensor, 'output_seq_tensor': output_digit2index_seq_tensor}
        
        # print(seq_tensor_pairs)
        
        return seq_tensor_pairs



class TxtDataset(Dataset):
    def __init__(self, pickle_file_path, wv_length=300):
        
        with open(pickle_file_path, 'rb') as f:
            data = pickle.load(f)
            
        self.lengths = [d[0].shape[0] for d in data]
        self.labels = [d[1]-1 for d in data]
        self.max_length = max(self.lengths)
        
        # 样本数*最大长度*词向量长度
        self.data = np.zeros((len(data), self.max_length, wv_length), dtype=np.float)
        for i, d in enumerate(data):
            
            self.data[i, :self.lengths[i], :] = np.squeeze(d[0], 1)
 
    def __getitem__(self, index):
        # label = np.zeros(max(self.labels)+1).astype(int)
        # label[self.labels[index]] = 1
        # return self.data[index], torch.from_numpy(label), self.lengths[index]
        return self.data[index], self.labels[index], self.lengths[index]
 
    def __len__(self):
        return len(self.labels)



def tokenize_digits(digits_sequence):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok for tok in digits_sequence]

tokenize_digits('39419468983611214')

class DigitsSeqDataset(data.Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    # return tensor
    def __getitem__(self, index):
        input, output = self.inputs[index], self.outputs[index]
        input = torch.Tensor(input)
        output = torch.Tensor(output)
        return input, output

    def __len__(self):
        return len(self.inputs)

dataset = MyDataset(images, labels)


import numpy as np
input_dataset = np.loadtxt('./dataset/task8_train_input.csv', delimiter=',', dtype=np.int32)
output_dataset = np.loadtxt('./dataset/task8_train_output.csv', delimiter=',', dtype=np.int32)
input_dataset.shape
x = torch.from_numpy(input_dataset[:, :])
y = torch.from_numpy(input_dataset[:, :])


for step, batch in enumerate(train_loader):
    if step == 1:
        print('|batch_inputs:',batch.get('input_seq_tensor'))
        print(" ")
        print('|batch_outputs:',batch.get('output_seq_tensor'))
    #print('Epoch:', 1, '|Step:', step, '|batch_x:',batch_x, '|batch_y', batch_y)

eee = [digit2index.get(str(i), digit2index.get('UNK')) for i in input_digit_seq if i != 0]


with open('./dataset/task8_train_input.csv', 'r') as f:
    source_data = [''.join(line.split(',')).strip('\n').split('0')[0] for line in f.readlines()]
    

with open('./dataset/task8_train_output.csv', 'r') as f:
    target_data = [''.join(line.split(',')).strip('\n').split('0')[0] for line in f.readlines()]


def extract_character_vocab(data):
    """
    :param data:
    :return: 字符映射表
    """
    special_words = ['<PAD>','<UNK>','<GO>','<EOS>']
    set_words = list(set([character for line in data for character in line]))
    int_to_vocab = {idx:word for idx,word in enumerate(special_words + set_words)}
    vocab_to_int = {word:idx for idx,word in int_to_vocab.items()}

    return int_to_vocab,vocab_to_int

# 得到输入和输出的字符
# 
# 映射表
source_int_to_letter,source_letter_to_int = extract_character_vocab(source_data+target_data)

target_int_to_letter,target_letter_to_int = extract_character_vocab(source_data+target_data)

# 将每一行转换成字符id的list
source_int = [[source_letter_to_int.get(letter,source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data]

target_int =[ [target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] for line in target_data]