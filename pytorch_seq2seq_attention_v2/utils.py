import pandas as pd
from dataset import Digits

def build_vocab(sequence_corpus):
    digit_mapping = Digits()
    digit_mapping.add_digits_seq('123456789')
    digit2index = digit_mapping.digit2index
    index2digit = digit_mapping.index2digit
    vocab_size = len(digit_mapping)
    return digit2index, index2digit, vocab_size


def load_datasets(train_input_csv, train_output_csv, test_input_csv):
    inputs = pd.read_csv(train_input_csv, header=None)
    outputs = pd.read_csv(train_output_csv, header=None)
    test_inputs = pd.read_csv(test_input_csv, header=None)
    return inputs, outputs, test_inputs