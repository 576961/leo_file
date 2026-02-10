
import torch
from torch import nn
from torch.utils import data
import math
import os
import collections
import time
import matplotlib.pyplot as plt
import csv
from nltk.translate.bleu_score import corpus_bleu
#import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler


with open('C:/Users/Lenovo/Desktop/deu.txt', 'r', encoding='utf-8') as f:
    raw_text = f.read()
print(1)
def preprocess_nmt(text):
    """预处理数据集"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集"""
    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) >= 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    return source, target

def pad_nmt(text, max_len):
    for i in range(len(text)):
        if len(text[i]) + 2 >= max_len:
            text[i] = ['<bos>'] + text[i][:max_len-2] + ['<eos>']
        else:
            text[i] = ['<bos>'] + text[i] + ['<pad>']*(max_len-len(text[i])-2) + ['<eos>']
    return text


def create_vocab(text):
    num = 0
    dic_count = {}
    for sen in text:
        for word in sen:
            if word not in dic_count:
                dic_count[word] = 1
            else:
                dic_count[word] += 1

    for sen in text:
        for i in range(len(sen)):
            if dic_count[sen[i]] < 2:
                sen[i] = '<unk>'
    vocab = {}
    for sen in text:
        for word in sen:
            if word not in vocab:
                vocab[word] = num
                num += 1
    return vocab

def build(text, vocab):
    for sen in text:
        for i in range(len(sen)):
            sen[i] = vocab[sen[i]]
    return text

text = preprocess_nmt(raw_text)
source, target = tokenize_nmt(text)
source = pad_nmt(source, max_len=10)
target = pad_nmt(target, max_len=10)
vocab_source = create_vocab(source)
vocab_target = create_vocab(target)
source_num = build(source, vocab_source)
target_num = build(target, vocab_target)

print(source_num[:50])
print(target_num[:50])
print(len(source_num))
print(len(target_num))


# 掩蔽softmax
def masked_softmax(X, valid_lens=None):
    #X: 3D张量，形状为(batch_size, n, d),valid_lens: 1D或2D张量，表示每个序列的有效长度

    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)

    # 确保valid_lens是2D
    if valid_lens.dim() == 1:
        valid_lens = valid_lens.unsqueeze(1)
    # 获取序列长度
    d = X.shape[-1]

    # 创建位置掩码
    mask = torch.arange(d, device=X.device).unsqueeze(0) < valid_lens.unsqueeze(-1) #利用了广播机制

    # 应用掩码并计算softmax
    X_masked = X.masked_fill(~mask, -1e6)  # 使用-1e6而不是最小值，避免梯度问题
    return nn.functional.softmax(X_masked, dim=-1)


# 点积注意力
class scaled_dot_product_attention(nn.Module):
    def __init__(self, **kwargs):
        '''query_size = key_size'''
        super().__init__(**kwargs)

    def forward(self, queries, keys, values,
                valid_lens=None):  # querie, keys, values都是三维张量，大小为(batch_size, num_steps, embed_size)
        weights = masked_softmax(torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[2]), valid_lens)
        return torch.bmm(weights, values)


# 多头注意力
class multi_attention(nn.Module):
    def __init__(self, num_head, input_size, query_size, key_size, value_size, **kwargs):
        super().__init__(**kwargs)
        self.num_head = num_head
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.W_Q = nn.Linear(input_size, query_size * num_head, bias=False)  # 将num_head个投影矩阵拼在一起
        self.W_K = nn.Linear(input_size, key_size * num_head, bias=False)
        self.W_V = nn.Linear(input_size, query_size * num_head, bias=False)
        self.W_O = nn.Linear(num_head * value_size, input_size, bias=False)
        self.attention = scaled_dot_product_attention()

    def forward(self, queries, keys, values, valid_lens=None):
        projected_queries, projected_keys, projected_values = self.W_Q(queries), self.W_K(keys), self.W_V(values)
        for i in range(self.num_head):

            head_now = self.attention(projected_queries[:, :, i * self.query_size:(i + 1) * self.query_size],
                                      projected_keys[:, :, i * self.key_size:(i + 1) * self.key_size],
                                      projected_values[:, :, i * self.value_size:(i + 1) * self.value_size],
                                      valid_lens)
            if (i == 0):
                head_concated = head_now
            else:
                head_concated = torch.concat((head_concated, head_now), dim=2)

        return self.W_O(head_concated)


# 前馈网络（MLP）
class ffn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, input):
        return self.layer_2(self.activation(self.layer_1(input)))


# 残差连接后进行层规范化
class add_norm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

#!!!dropout重复了2次?

# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, dropout, num_hiddens = 1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        m, n = X.shape[0], X.shape[1]
        Y = torch.sin(torch.arange(m)).reshape(m, 1)
        for i in range(1,n):
            if i%2 == 0:
                new_Y = torch.sin((torch.arange(m)).reshape(m,1)/pow(10000,i/n))
            else:
                new_Y = torch.cos((torch.arange(m)).reshape(m,1)/pow(10000,(i+1)/n))
            Y = torch.cat([Y,new_Y],1)
        X = X + Y
        return self.dropout(X)


# 编码器中的基础块
class encoder_block(nn.Module):
    def __init__(self, normalized_shape, dropout,
                 num_head, input_size, query_size, key_size, value_size,
                 hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.add_norm_1 = add_norm(normalized_shape, dropout)
        self.add_norm_2 = add_norm(normalized_shape, dropout)
        self.multi_attention = multi_attention(num_head, input_size, query_size, key_size, value_size)
        self.ffn = ffn(input_size, hidden_size, output_size)

    def forward(self, input, enc_valid_lens=None):
        res_1 = self.add_norm_1(input, self.multi_attention(input, input, input, enc_valid_lens))
        res_2 = self.add_norm_2(res_1, self.ffn(res_1))
        return res_2


# 编码器
class transformer_encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_blocks,
                 normalized_shape, dropout,
                 num_head, input_size, query_size, key_size, value_size,
                 hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.Sequential()
        for i in range(self.num_blocks):
            self.blocks.add_module('i', encoder_block(normalized_shape, dropout,
                                                      num_head, input_size, query_size, key_size, value_size,
                                                      hidden_size, output_size, **kwargs))

    def forward(self, X, enc_valid_lens):
        X = self.embedding(X) * math.sqrt(self.embed_size)  # 把每个分量的范围变成(-1,1)
        X += PositionalEncoding(X)
        for i, block in enumerate(self.blocks):
            X = block(X, enc_valid_lens)

        return X


# 解码器中的基础块
class decoder_block(nn.Module):
    def __init__(self, normalized_shape, dropout,
                 num_head, input_size, query_size, key_size, value_size,
                 hidden_size, output_size, i, **kwargs):  # 参数i表示第i个块(从0开始)
        super().__init__(**kwargs)
        self.add_norm_1 = add_norm(normalized_shape, dropout)
        self.add_norm_2 = add_norm(normalized_shape, dropout)
        self.add_norm_3 = add_norm(normalized_shape, dropout)
        self.multi_attention_1 = multi_attention(num_head, input_size, query_size, key_size, value_size)
        self.multi_attention_2 = multi_attention(num_head, input_size, query_size, key_size, value_size)
        self.ffn = ffn(input_size, hidden_size, output_size)
        self.i = i

    def forward(self, input, state):
        enc_outputs, enc_valid_lens = state[0], state[1]

        if state[2][self.i] is None:
            key_values = input
        else:
            key_values = torch.cat((state[2][self.i], input), axis=1)

        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps, _ = input.shape
            # dec_valid_lens的开头:(batch_size,num_steps),
            # 其中每一行是[1,2,...,num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=input.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        res_1 = self.add_norm_1(input, self.multi_attention_1(input, key_values, key_values, dec_valid_lens))
        res_2 = self.add_norm_2(res_1, self.multi_attention_2(res_1, enc_outputs, enc_outputs, enc_valid_lens))
        res_3 = self.add_norm_3(res_2, self.ffn(res_2))
        return res_3, state


# 解码器
class transformer_decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_blocks,
                 normalized_shape, dropout,
                 num_head, input_size, query_size, key_size, value_size,
                 hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        num_blocks = num_blocks
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.final_linear = nn.Linear(output_size, vocab_size)
        self.blocks = nn.Sequential()
        self.num_blocks = num_blocks

        for i in range(self.num_blocks):
            self.blocks.add_module('i', decoder_block(normalized_shape, dropout,
                                                      num_head, input_size, query_size, key_size, value_size,
                                                      hidden_size, output_size, i, **kwargs))

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blocks]

    def forward(self, X, state):
        X = self.embedding(X)
        X += PositionalEncoding(X)
        for i, block in enumerate(self.blocks):
            X, state = block(X, state)

        return self.final_linear(X), state


# transformer网络
class transformer(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y, enc_valid_lens):
        enc_outputs = self.encoder(X, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(Y, state)

num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = load_data(batch_size, num_steps)

encoder = TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
decoder = TransformerDecoder(
    len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)

enc = transformer_encoder()
dec = transformer_decoder(24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0)
net = transformer(enc, dec)
net.eval()

X = torch.ones((2, 100, 24))
