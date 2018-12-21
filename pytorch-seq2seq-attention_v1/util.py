import time
import math

from dataset import DigitSeqDataset, DigitsSeq

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def read_digits_sequence():
    
    with open('./dataset/task8_train_input.csv', 'r') as f:
        input_data = [''.join(line.split(',')).strip('\n').split('0')[0] for line in f.readlines()]
    

    with open('./dataset/task8_train_output.csv', 'r') as f:
        output_data = [''.join(line.split(',')).strip('\n').split('0')[0] for line in f.readlines()]
        
    inputs = DigitsSeq()
    
    outputs = DigitsSeq()
    
    seq_pairs = [list(value) for value in zip(input_data, output_data)]
    
    return inputs, outputs, seq_pairs


def prepare_data():
    """
    准备数据
    """
    inputs, outputs, seq_pairs = read_digits_sequence()
    
    print("Read %s digits sequence pairs" % len(seq_pairs))
    
    for pair in seq_pairs:
        inputs.add_digits_seq(pair[0])
        outputs.add_digits_seq(pair[1])

    return inputs, outputs, seq_pairs


def indexesFromSentence(lang, sentence):
    return [lang.digit2index[digit] for digit in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(seq_pair):
    input_tensor = tensorFromSentence(inputs, seq_pair[0])
    target_tensor = tensorFromSentence(outputs, seq_pair[1])
    return (input_tensor, target_tensor)