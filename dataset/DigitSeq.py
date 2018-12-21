from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd


# class DigitsSeq:
#     """
    
#     """
#     def __init__(self):
#         self.digit2index = {}
#         self.digit2count = {}
#         self.index2digit = {0: "SOS", 1: "EOS"}
#         # Count SOS and EOS
#         self.n_digits = 2
        
#     def __len__(self):
#         return len(self.index2digit)

#     def add_digits_seq(self, digits_sequence):
#         for digit in digits_sequence:
#             self.add_digit(digit)

#     def add_digit(self, digit):
#         if digit not in self.digit2index:
#             self.digit2index[digit] = self.n_digits
#             self.digit2count[digit] = 1
#             self.index2digit[self.n_digits] = digit
#             self.n_digits += 1
#         else:
#             self.digit2count[digit] += 1


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

class DigitSeqDataset(Dataset):
    """DigitSeqDataset"""

    def __init__(self, intput_csv, output_csv):
        """
            csv_file (string): Path to the csv file with annotations.
        """
        self.input_digit_seq_frame = pd.read_csv(intput_csv, header=None)
        self.output_digit_seq_frame = pd.read_csv(output_csv, header=None)

        
    def __len__(self):
        return len(self.input_digit_seq_frame)
    

    def __getitem__(self, idx):
        """
        在类的__getitem__函数中完成读取工作。
        这样是为了减小内存开销，只要在需要用到的时候才读入。
        """
        input_digit_seq = self.input_digit_seq_frame.iloc[idx, :].as_matrix()
        input_digit_seq = input_digit_seq.astype('int')
        intput_digit_seq_tensor = torch.from_numpy(input_digit_seq)
        
        
        output_digit_seq = self.output_digit_seq_frame.iloc[idx, :].as_matrix()
        output_digit_seq = output_digit_seq.astype('int')
        output_digit_seq_tensor = torch.from_numpy(output_digit_seq)
        
        seq_pairs = {'input_seq': input_digit_seq, 'output_seq': output_digit_seq}
        seq_pairs = [ input_digit_seq, output_digit_seq]
        seq_tensor_pairs = {'input_seq_tensor': intput_digit_seq_tensor, 'output_seq_tensor': output_digit_seq_tensor}
        return seq_tensor_pairs