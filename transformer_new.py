
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
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
from transformers import AutoTokenizer


def preprocess(text, out_path=None):
    """预处理数据集，并可将结果写入文件(out_path)。"""
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    processed = ''.join(out)
    if out_path is not None:
        # 确保以utf-8写入，避免Windows默认编码问题
        try:
            with open(out_path, 'w', encoding='utf-8') as fout:
                fout.write(processed)
        except Exception:
            # 若路径目录不存在，尝试创建目录后重写
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as fout:
                fout.write(processed)
    return processed

def tokenize(text, num_examples=4000000):
    """按空格分词并统计句长（保留为回退）。"""
    L = []
    count1 = 0
    count2 = 0
    count3 = 0
    for i, line in enumerate(text.split('\n')):
        if i > num_examples:
            break
        l = [tok for tok in line.split(' ') if tok != '']
        if len(l) <= 10:
            count1 += 1
        if len(l) <= 20:
            count2 += 1
        if len(l) <= 30:
            count3 += 1
        L.append(l)
    print(f"句子长度统计: <=10: {count1}, <=20: {count2}, <=30: {count3}")
    return L


def learn_bpe_from_text(text, num_merges=2000, show_progress=True):
    """从文本学习 BPE 合并规则，返回合并对列表（tuple pairs）。

    如果 `show_progress=True`，会打印学习进度、已用时间与 ETA，方便估算总时长。
    """
    vocab = {}
    for line in text.split('\n'):
        for w in line.split():
            if w == '':
                continue
            symbols = tuple(list(w) + ['</w>'])
            vocab[symbols] = vocab.get(symbols, 0) + 1

    merges = []
    start_time = time.time()
    report_every = max(1, num_merges // 100)  # 最多显示 ~100 次更新

    for i in range(num_merges):
        pairs = {}
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] = pairs.get((a, b), 0) + freq
        if not pairs:
            if show_progress:
                print(f"BPE: no more pairs at step {i}, finished early.")
            break
        best = max(pairs, key=pairs.get)
        merges.append(best)
        # 应用合并到词表
        new_vocab = {}
        for word, freq in vocab.items():
            w = list(word)
            j = 0
            new_word = []
            while j < len(w):
                if j < len(w) - 1 and (w[j], w[j + 1]) == best:
                    new_word.append(w[j] + w[j + 1])
                    j += 2
                else:
                    new_word.append(w[j])
                    j += 1
            new_vocab[tuple(new_word)] = new_vocab.get(tuple(new_word), 0) + freq
        vocab = new_vocab

        # 打印进度与 ETA
        if show_progress and ((i + 1) % report_every == 0 or i == 0):
            elapsed = time.time() - start_time
            per_step = elapsed / (i + 1)
            remaining_steps = max(0, num_merges - (i + 1))
            eta = per_step * remaining_steps
            # 格式化为人类可读
            def _fmt(sec):
                if sec >= 3600:
                    return f"{sec/3600:.1f}h"
                if sec >= 60:
                    return f"{sec/60:.1f}m"
                return f"{sec:.1f}s"

            print(f"BPE merges: {i+1}/{num_merges} — elapsed {_fmt(elapsed)}, ETA {_fmt(eta)}")

    return merges


def apply_bpe_to_word(word, merges):
    """对单词应用 BPE 合并规则，返回子词列表（不含</w>标记）。"""
    token = list(word) + ['</w>']
    for pair in merges:
        i = 0
        while i < len(token) - 1:
            if (token[i], token[i + 1]) == pair:
                token[i] = token[i] + token[i + 1]
                token.pop(i + 1)
                if i > 0:
                    i -= 1
            else:
                i += 1
    if token and token[-1] == '</w>':
        token = token[:-1]
    return token


def tokenize_bpe(text, merges, num_examples=4000000):
    """使用 BPE 合并规则对文本进行分词，返回嵌套 token 列表。"""
    L = []
    for i, line in enumerate(text.split('\n')):
        if i > num_examples:
            break
        toks = []
        for w in line.split():
            if w == '':
                continue
            toks.extend(apply_bpe_to_word(w, merges))
        L.append(toks)
    return L

def tokenize_auto(text, tokenizer, seq_len, num_examples=4000000):
    """使用 BPE 合并规则对文本进行分词，返回嵌套 token 列表。"""
    L = []
    for i, line in enumerate(text.split('\n')):
        if i >= num_examples:
            break
        L.append(line)
    return tokenizer(L, padding=True, truncation=True, max_length=seq_len, return_tensors="pt")


def pad(text, max_len):
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
            if dic_count[sen[i]] < 10:  # 词频小于10的单词被替换为<unk>
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



'''
# 遍历DataLoader中的批次
for batch_idx, batch_data in enumerate(dataloader):
    print(f"\n--- 批次 {batch_idx} ---")
    print(len(batch_data),len(batch_data[0]),len(batch_data[0][0]))  # [batch_size, seq_len]
    break

vocab_size = len(vocab_source)
embedding_dim = 32

embedding = nn.Embedding(
    num_embeddings=vocab_size,  # 词表大小
    embedding_dim=embedding_dim,  # 嵌入维度
    padding_idx=vocab_source['<pad>']   # 填充标记的索引
)

print(source[:50])
'''


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

    def forward(self, X):  #X: (batch_size, num_steps, embedding_dim)
        m, n = X.shape[1], X.shape[2]
        Y = torch.sin(torch.arange(m)).reshape(m, 1)
        for i in range(1,n):
            if i%2 == 0:
                new_Y = torch.sin((torch.arange(m)).reshape(m,1)/pow(10000,i/n))
            else:
                new_Y = torch.cos((torch.arange(m)).reshape(m,1)/pow(10000,(i+1)/n))
            Y = torch.cat([Y,new_Y],1)
        X = X + torch.unsqueeze(Y, dim=0).to(X.device)
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
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_blocks,
                 normalized_shape, dropout,
                 num_head, input_size, query_size, key_size, value_size,
                 hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.num_blocks = num_blocks
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.Sequential()
        self.dropout = dropout
        for i in range(self.num_blocks):
            self.blocks.add_module('i', encoder_block(normalized_shape, dropout,
                                                      num_head, input_size, query_size, key_size, value_size,
                                                      hidden_size, output_size, **kwargs))

    def forward(self, X, enc_valid_lens):
        X = self.embedding(X) * math.sqrt(self.embed_size)  # 把每个分量的范围变成(-1,1)
        pos_enc = PositionalEncoding(self.dropout)
        X += pos_enc(X)
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
class TransformerDecoder(nn.Module):
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
        self.dropout = dropout
        
        for i in range(self.num_blocks):
            self.blocks.add_module('i', decoder_block(normalized_shape, dropout,
                                                      num_head, input_size, query_size, key_size, value_size,
                                                      hidden_size, output_size, i, **kwargs))

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blocks]

    def forward(self, X, state):
        X = self.embedding(X)
        pos_enc = PositionalEncoding(self.dropout)
        X += pos_enc(X)
        for i, block in enumerate(self.blocks):
            X, state = block(X, state)

        return self.final_linear(X), state


# transformer网络
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y, enc_valid_lens):
        enc_outputs = self.encoder(X, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(Y, state)


'''
# 超参数设置1
seq_len, embedding_dim = 20, 32
num_layers, dropout, batch_size = 2, 0.1, 64
# Ensure the attention/input sizes match the embedding dimension
ffn_num_input, ffn_num_hiddens, ffn_num_output, num_heads = 32, 64, 32, 4
key_size, query_size, value_size = 32, 32, 32
# LayerNorm normalized shape must match embedding dimension
norm_shape = [32]
'''

# 超参数设置2
seq_len, embedding_dim = 30, 512
num_layers, dropout, batch_size = 6, 0.1, 64
# Ensure the attention/input sizes match the embedding dimension
ffn_num_input, ffn_num_hiddens, ffn_num_output, num_heads = 512, 2048, 512, 8
key_size, query_size, value_size = 64, 64, 64
# LayerNorm normalized shape must match embedding dimension
norm_shape = [512]

preprocessed = True
if not preprocessed:
    en_path = r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.en'
    de_path = r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.de'
    with open(en_path, 'r', encoding='utf-8') as f:
        raw_text_1 = f.read()
    with open(de_path, 'r', encoding='utf-8') as f:
        raw_text_2 = f.read()
        
    print(1)

    text_1 = preprocess(raw_text_1, en_path + '.preprocessed.txt')
    print(2)
    text_2 = preprocess(raw_text_2, de_path + '.preprocessed.txt')
    print(2)
else:
    with open(r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.en.preprocessed.txt', 'r', encoding='utf-8') as f:
        text_1 = f.read()
    with open(r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.de.preprocessed.txt', 'r', encoding='utf-8') as f:
        text_2 = f.read()

r'''
# 使用 BPE 分词
num_bpe_merges = 32000
en_bpe_path = r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.en.bpe'
de_bpe_path = r'C:\Users\Lenovo\PycharmProjects\ProjectTransformer\train.de.bpe'
en_merges = learn_bpe_from_text(text_1, num_bpe_merges)
de_merges = learn_bpe_from_text(text_2, num_bpe_merges)
# 保存合并表，便于复现或查看
with open(en_bpe_path, 'w', encoding='utf-8') as f:
    for a, b in en_merges:
        f.write(f"{a} {b}\n")
with open(de_bpe_path, 'w', encoding='utf-8') as f:
    for a, b in de_merges:
        f.write(f"{a} {b}\n")
'''
   
tokenizer = AutoTokenizer.from_pretrained("./my_bert2bert_tokenizer")

'''
# Ensure the tokenizer has an unknown token to avoid WordPiece errors
try:
    vocab_dict = tokenizer.get_vocab()
except Exception:
    vocab_dict = {}

if getattr(tokenizer, 'unk_token', None) is None or ('[UNK]' not in vocab_dict):
    tokenizer.add_special_tokens({'unk_token': '[UNK]'})
'''
#text_1 = text_1.split('\n')[:4000000]
#text_2 = text_2.split('\n')[:4000000]
#print(1)
#print(tokenizer(text_2[760292], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids'])        
#1266800
'''
for i in range(1360000, 14000000):
    #break
    if i%100 == 0:
        print(i)
    tokenizer(text_1[i], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']
    tokenizer(text_2[i], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']


for i in range(1360000, 14000000):
    #break
    #if i%100 == 0:
    #    print(i)
    try:
        tokenizer(text_1[i], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']
        tokenizer(text_2[i], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']
    except Exception as e:
        print(i)
'''

'''
source_tensor = tokenizer(text_1[:100], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']
target_tensor = tokenizer(text_2[:100], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']
print(source_tensor.shape, target_tensor.shape)

error_list = []

with open("error_list.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        error_list.append(int(line.strip()))

text_clc_1 = text_1[:100]
text_clc_2 = text_2[:100]
for i in range(10, 400000):
    if i not in error_list:
        text_clc_1.extend(text_1[i*10:(i+1)*10])
        text_clc_2.extend(text_2[i*10:(i+1)*10])
print(len(text_clc_1), len(text_clc_2))

for i in range(1, 38890):
    #break
    if i%100 == 0:
        print(i)
    try:
        source_tensor = torch.cat((source_tensor, tokenizer(text_clc_1[i*100:(i+1)*100], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']), dim=0)
        target_tensor = torch.cat((target_tensor, tokenizer(text_clc_2[i*100:(i+1)*100], padding=True, truncation=True, max_length=seq_len, return_tensors="pt")['input_ids']), dim=0)
    except Exception as e:
        count += 1
        error_list.append(i)
        print(f"error point {i}")

torch.save(source_tensor, 'source_tensor.pt')
torch.save(target_tensor, 'target_tensor.pt')
'''

#print(f"Total errors: {count}")
source_tensor = torch.load('source_tensor.pt')
target_tensor = torch.load('target_tensor.pt')
print(source_tensor[0])
print(target_tensor[0])
print(source_tensor.shape, target_tensor.shape)
vocab_dict = tokenizer.get_vocab()
vocab_source = vocab_dict
vocab_target = vocab_dict
print(len(vocab_source), len(vocab_target))


'''
source = tokenizer(text_1)
target = tokenizer(text_2)
source = pad(source, max_len=seq_len)
target = pad(target, max_len=seq_len)
print(len(source), len(target))

vocab_source = create_vocab(source)
vocab_target = create_vocab(target)
source_num = build(source, vocab_source)
target_num = build(target, vocab_target)
source_tensor = torch.tensor(source_num, dtype=torch.long)
target_tensor = torch.tensor(target_num, dtype=torch.long)
'''


'''

dataset = TensorDataset(source_tensor, target_tensor)
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False  # 是否丢弃最后一个不完整的批次
)

encoder = TransformerEncoder(len(vocab_source), embedding_dim, num_layers,
                             norm_shape, dropout, num_heads,
                             ffn_num_input, query_size, key_size, value_size,
                             ffn_num_hiddens, ffn_num_output)
decoder = TransformerDecoder(len(vocab_target), embedding_dim, num_layers,
                             norm_shape, dropout, num_heads,
                             ffn_num_input, query_size, key_size, value_size,
                             ffn_num_hiddens, ffn_num_output)
net = Transformer(encoder, decoder)


net.eval()
X = torch.ones((batch_size, seq_len), dtype=torch.long)
valid_lens = torch.ones(batch_size, dtype=torch.long)
Y, _ = net(X, X, valid_lens)
print(Y.shape)
'''


