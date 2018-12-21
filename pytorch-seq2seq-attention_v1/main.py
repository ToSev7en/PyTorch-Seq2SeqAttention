import os
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from util import *
from config import opt
from models import GRUEncoder, AttnDecoder

"""
Dataloader

inputs
outputs

对数据对预处理：

input sequence 添加 eos，或者 length 属性 来控制是否到了结束 位置

outputs sequence 不需要添加 go，在程序中添加，至于 eos 还是加上吧

"""
teacher_forcing_ratio = 0.5

MAX_LENGTH = 20

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """
    Seq2Seq+Attention 训练过程
    """

    # initialize hidden state
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    
    # print(input_length) 21
    
    target_length = target_tensor.size(0)
    
    # 这里 encoder_outputs 记录的就是编码到每一个单词产生的语义向量，比如 10 个英语单词的句子就应该有10个语义向量
    # torch.zeros 进行 zero padding 
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    # 一个个单词 feed encoder
    for ei in range(input_length):
        
        # 获取到每一步 Encoder 的输出
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        
        # 记录改单词处的语义向量
        encoder_outputs[ei] = encoder_output[0, 0]

        
    # 先输入 <GO> 来启动 Decoder，然后将 sequence 输入 Decoder
    # <GO> 表示 sequence 开始，这里这样的话就不用在 sequence 中添加 <GO> token 了
    decoder_input = torch.tensor([[target_letter_to_int.get('<GO>')]], device=device)

    # Decoder的 hidden state 就是 Encoder 最后一步输出的语义向量
    decoder_hidden = encoder_hidden

    # 以一定的概率使用 teacher_forcing
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        """
        Teacher forcing: Feed the target as the next input
        利用已知的上一步真实的单词去预测下一个单词
        """
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # 计算 loss
            loss += criterion(decoder_output, target_tensor[di])
            
            # Teacher forcing
            decoder_input = target_tensor[di]

    else:
        """
        Without teacher forcing: use its own predictions as the next input
        利用自己上一步预测的单词作为输入预测下一个单词

        获取 decoder_output ，利用 topk(1) 得到值最大的索引，作为下一步 decoder 的输入
        """
        for di in range(target_length):

            """
            decoder_output:

            tensor([[-2.5267, -2.4569, -2.2682, -2.9591, -2.5931, -2.8346, -2.4738, -2.4122,
                    -2.6883, -2.5823, -2.4317, -2.8451, -2.5012]], grad_fn=<LogSoftmaxBackward>)

            torch.Size([1, 13])

            """
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            # topk 返回的前 k 最大值及其索引
            # 此处 k 为 1                 
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
            'decoder_output.topk(1) -> (tensor([[-2.1516]], grad_fn=<TopkBackward>), tensor([[10]]))'
            'topv -> tensor([[-2.3059]], grad_fn=<TopkBackward>)'
            'topi -> tensor([[0]])'
            'topi.shape -> torch.Size([1, 1])'

            
            topv, topi = decoder_output.topk(1)

            
            """
            squeeze
            函数功能：去除 size 为 1 的维度，包括行和列。当维度大于等于2时，squeeze()无作用。
            其中squeeze(0)代表若第一维度值为1则去除第一维度，squeeze(1)代表若第二维度值为1则去除第二维度。
            
            tensor.detach()
            返回一个新的Tensor，从当前图中脱离出来，该 tensor 不会要求更新梯度，也就是梯度在这中断了。
            注意：该新的Tensor与原Tensor共享内存。
            """
            # detach from history as input
            decoder_input = topi.squeeze().detach()

            """
            Decoder_Output: tensor([[-2.5014, -2.8472, -2.3635, -2.3492, -2.8655, -2.3466, -2.5064, -3.1898,
                    -2.9688, -2.3176, -2.6453, -2.2796, -2.6324]], grad_fn=<LogSoftmaxBackward>)
            Decoder_Output.shape: torch.Size([1, 13])
            Decoder_Output.topk(1): (tensor([[-2.2796]], grad_fn=<TopkBackward>), tensor([[11]]))
            
            topv: tensor([[-2.2796]], grad_fn=<TopkBackward>)
            
            topi.shape: torch.Size([1, 1])
            
            decoder_input: tensor(11)

            """

            loss += criterion(decoder_output, target_tensor[di])
            
            # 遇到 sequence's <EOS> token，表示翻译句子停止了
            # 如果一个tensor只有一个元素，那么可以使用.item()方法取出这个元素作为普通的 python 数字。
            if decoder_input.item() == target_letter_to_int.get('<EOS>'):
                break

    loss.backward()

    encoder_optimizer.step()
    
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    
    start = time.time()
    
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    
    training_pairs = [random.choice(seq_pairs) for i in range(n_iters)]
    
    
    """
    NLLLoss 的 输入 是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率. 
    适合网络的最后一层是log_softmax. 损失函数 nn.CrossEntropyLoss() 与 NLLLoss() 相同, 
    唯一的不同是它为我们去做 softmax.
    """
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):

        training_pair = training_pairs[iter - 1]

        input_tensor = torch.from_numpy(np.array(training_pair[0]).reshape(-1,1))
        target_tensor = torch.from_numpy(np.array(training_pair[1]).reshape(-1,1))

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)



def evaluate(encoder, decoder, input_sequence, max_length=MAX_LENGTH):
    # To run the model, pass in a  vector
    # Here we don't need to train, so the code is wrapped in torch.no_grad()
    with torch.no_grad():
        
        input_tensor = input_sequence
        
        input_length = input_tensor.size()[0]
        
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            
            encoder_outputs[ei] += encoder_output[0, 0]
            
        # SOS
        decoder_input = torch.tensor([[target_letter_to_int.get('<GO>')]], device=device)  

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            
            if topi.item() == target_letter_to_int.get('<EOS>'):
                decoded_words.append('<EOS>')
                break
            else:
                # 序号 to digit
                decoded_words.append(source_int_to_letter[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(source_target_pair)
        
        print('>', pair[0])
        
        # 之前插入了 EOS, 去除最后一个字符
        print('=', pair[1]) 
        
        # 原始序列映射
        digit_id_sequence = [source_letter_to_int.get(str(letter), source_letter_to_int['<UNK>']) for letter in pair[0]]
        print("~", digit_id_sequence)
        
        # 转换为 tensor
        input_tensor = torch.tensor(digit_id_sequence).view(-1,1)
        # print(input_tensor)
        
        output_words, attentions = evaluate(encoder, decoder, input_tensor)
        
        output_sentence = ', '.join(output_words[:-1])
        
        print('< [', output_sentence,']',sep='')
        print('')


# torch.__version__ >= 0.5
@torch.no_grad() 
def test(**kwargs):
    '''
    测试（inference）
    '''
    pass

@torch.no_grad() 
def predict(**kwargs):
    '''
    测试（inference）
    '''
    pass

def help():
    """
    打印帮助的信息： python file.py help
    """
    
    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)

if __name__=='__main__':
    import fire
    fire.Fire()

    hidden_size = 8

    # 得到输入和输出的字符映射表

    source_int_to_letter,source_letter_to_int = extract_character_vocab(source_data+target_data)

    target_int_to_letter,target_letter_to_int = extract_character_vocab(source_data+target_data)

    # 将每一行转换成字符id的list
    source_int = [[source_letter_to_int.get(letter,source_letter_to_int['<UNK>'])
                for letter in line] for line in source_data]

    # 在 output sequence 后添加 <EOS> tag
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data]

    # inputs, outputs, seq_pairs 
    encoder = GRUEncoder(len(target_letter_to_int), hidden_size)

    attn_decoder = AttnDecoder(hidden_size, len(target_letter_to_int), dropout_p=0.1)

    # all 32000
    trainIters(encoder, attn_decoder, 30000, print_every=500)