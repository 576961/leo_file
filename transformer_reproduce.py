
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
from transformers import AutoTokenizer
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

#预处理
def get_data(path, n, batch_size):
    source, target = [], []
    num_batchs = 4000
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i > (n)*num_batchs*batch_size:
                break
            if i > (n-1)*num_batchs*batch_size and len(row) >= 2:
                source.append(row[0])
                target.append(row[1])
            
            i+=1
    return source, target

def preprocess(path_in, path_out):
    punctuation = [',', '.', '!', '?']
    source, target = [], []
    with open(path_in, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        n = 0
        for row in reader:
            if n%10000 == 0:
                print(n)
            n+=1
            if len(row) >= 2:
                x = row[0].replace('\u202f', ' ').replace('\xa0', ' ').lower()
                y = row[1].replace('\u202f', ' ').replace('\xa0', ' ').lower()
                i, j = 1, 1
                len_1, len_2 = len(x), len(y)
                while i < len_1:
                    if (x[i] in punctuation) and (x[i-1] != ' '):
                        x = x[:i] + ' ' + x[i:]
                        len_1 += 1
                        i += 1
                    i += 1
                while j < len_2:
                    if (y[j] in punctuation) and (y[j-1] != ' '):
                        y = y[:j] + ' ' + y[j:]
                        len_2 += 1
                        j += 1
                    j += 1  
                source.append(x)
                target.append(y)
    
    with open(path_out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for x,y in zip(source, target):
            writer.writerow([x, y])

#preprocess('C:/Users/23062/Desktop/wmt14_gr-en/wmt14_translate_de-en_train.csv', 'C:/Users/23062/Desktop/wmt14_gr-en/wmt14_translate_de-en_train_preprocessed.csv')



#print(source[0])
#print(source[:10], target[:10])
'''
#制作词表
class Vocab:  #@save
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)
'''


def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array_nmt(lines, vocab, tokenizer, num_steps):
    """将机器翻译的文本序列转换成小批量"""
    #lines = [[vocab[w] for w in l] for l in lines]
    lines_new = []
    for l in lines:
        tokens = tokenizer.tokenize(l)
        lines_new.append([vocab[token] for token in tokens])

    lines_new = [l + [vocab['[SEP]']] for l in lines_new]
    array = torch.tensor([truncate_pad(l, num_steps, vocab['[PAD]']) for l in lines_new])
    valid_len = (array != vocab['[PAD]']).type(torch.int32).sum(1)
    return array, valid_len

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

'''
def load_data_nmt(source, target, batch_size, num_steps):
    """返回翻译数据集的迭代器和词表"""
    src_vocab = Vocab(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    
    return src_vocab, tgt_vocab
'''

#test_1
'''print(len(src_vocab))
for X, X_valid_len, Y, Y_valid_len in data_iter:
    res = nn.Embedding(len(src_vocab), 8)
    break

print(res.shape)'''


#掩蔽softmax
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        count = 0
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                now_valid_len = valid_lens[count]
                for k in range(now_valid_len, X.shape[2]):
                    X[i][j][k] = -1e6
                count +=1
        
        return nn.functional.softmax(X, dim=-1)


#点积注意力
class scaled_dot_product_attention(nn.Module):
    def __init__(self, **kwargs):
        '''query_size = key_size'''
        super().__init__(**kwargs)

    def forward(self, queries, keys, values, valid_lens = None):  #querie, keys, values都是三维张量，大小为(batch_size, num_steps, embed_size)
        weights = masked_softmax(torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(queries.shape[2]), valid_lens)
        return torch.bmm(weights, values)

#多头注意力
class multi_attention(nn.Module):
    def __init__(self, num_head, input_size, query_size, key_size, value_size, **kwargs):
        super().__init__(**kwargs)
        self.num_head = num_head
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.W_Q = nn.Linear(input_size, query_size*num_head, bias = False) #将num_head个投影矩阵拼在一起
        self.W_K = nn.Linear(input_size, key_size*num_head, bias = False)
        self.W_V = nn.Linear(input_size, query_size*num_head, bias = False)
        self.W_O = nn.Linear(num_head*value_size, input_size, bias = False)
        self.attention = scaled_dot_product_attention()

    def forward(self, queries, keys, values, valid_lens = None):
        projected_queries, projected_keys, projected_values = self.W_Q(queries), self.W_K(keys), self.W_V(values)
        for i in range(self.num_head):
            
            head_now = self.attention(projected_queries[:, :, i*self.query_size:(i+1)*self.query_size],
                                 projected_keys[:, :, i*self.key_size:(i+1)*self.key_size],
                                 projected_values[:, :, i*self.value_size:(i+1)*self.value_size],
                                 valid_lens)
            if (i == 0):
                head_concated = head_now
            else:
                head_concated = torch.concat((head_concated, head_now), dim = 2)

        return self.W_O(head_concated)
            

#前馈网络（MLP）
class ffn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, input):
        return self.layer_2(self.activation(self.layer_1(input)))
    
#残差连接后进行层规范化
class add_norm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


#位置编码
def position_code(X):  #X为三维张量，在后两个维度上做位置编码
    a = X.shape[1]
    b = X.shape[2] #X有偶数列，故b为偶数
    P = torch.zeros((a, b)).to(X.device)
    for i in range(a):
        for j in range(b//2):
            P[i][2*j] = math.sin(i/(pow(10000, 2*j/b)))
            P[i][2*j+1] = math.cos(i/(pow(10000, 2*j/b)))

    return P.unsqueeze(0).repeat(X.shape[0], 1, 1)


#编码器中的基础块
class encoder_block(nn.Module):
    def __init__(self, normalized_shape, dropout,
                num_head, input_size, query_size, key_size, value_size,
                hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.add_norm_1 = add_norm(normalized_shape, dropout)
        self.add_norm_2 = add_norm(normalized_shape, dropout)
        self.multi_attention = multi_attention(num_head, input_size, query_size, key_size, value_size)
        self.ffn = ffn(input_size, hidden_size, output_size)

    def forward(self, input, enc_valid_lens = None):
        res_1 = self.add_norm_1(input, self.multi_attention(input, input, input, enc_valid_lens) )
        res_2 = self.add_norm_2(res_1, self.ffn(res_1))
        return res_2


#编码器
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
        X = self.embedding(X) * math.sqrt(self.embed_size)  #把每个分量的范围变成(-1,1)
        X += position_code(X)
        for i, block in enumerate(self.blocks):
            X = block(X, enc_valid_lens)

        return X


#解码器中的基础块
class decoder_block(nn.Module):
    def __init__(self, normalized_shape, dropout,
                num_head, input_size, query_size, key_size, value_size,
                hidden_size, output_size, i, **kwargs):  #参数i表示第i个块(从0开始)
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



#解码器
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
        X += position_code(X)
        for i, block in enumerate(self.blocks):
            X, state = block(X, state)
        
        return self.final_linear(X), state


#transformer网络
class transformer(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, X, Y, enc_valid_lens):
        enc_outputs = self.encoder(X, enc_valid_lens)
        state = self.decoder.init_state(enc_outputs, enc_valid_lens)
        return self.decoder(Y, state)



#test_2
'''vocab_size = len(src_vocab)
embed_size, num_blocks, normalized_shape, dropout, num_head = 32, 2, 32, 0.1, 8
input_size, query_size, key_size, value_size, hidden_size, output_size = 32, 32, 32, 32, 100, 32
X = torch.zeros((2,12), dtype=torch.long)
net_1 = transformer_encoder(vocab_size, embed_size, num_blocks,
                normalized_shape, dropout,
                num_head, input_size, query_size, key_size, value_size,
                hidden_size, output_size)
net_1.eval()
valid_lens = torch.tensor([2,3])
print(X.shape)
Y = net_1(X, valid_lens)
print(Y.shape)'''

#24, 24, 24, 24, [100, 24], 24, 48, 8, 0.5, 0
#test3
'''valid_lens = torch.tensor([2,3])
encoder_blk = encoder_block(normalized_shape, dropout,
                num_head, input_size, query_size, key_size, value_size,
                hidden_size, output_size)
decoder_blk = decoder_block(32, 0.1, 4, 32, 32, 32, 32, 200, 32, 1)
decoder_blk.eval()
X = torch.zeros((2, 100, 32))
state = [encoder_blk(X, valid_lens), valid_lens, [None]*4]
print(decoder_blk(X, state)[0].shape)'''

#test4


'''X = torch.zeros((2, 100), dtype = torch.long)
Y = torch.zeros((2, 100), dtype = torch.long)
enc_valid_lens = torch.tensor([2,3])
net_1 = transformer_encoder(vocab_size_1, embed_size, num_blocks,
                normalized_shape, dropout,
                num_head, input_size, query_size, key_size, value_size,
                hidden_size, output_size)
net_2 = transformer_decoder(vocab_size_2, embed_size, num_blocks, normalized_shape, dropout, num_head, input_size, 
                            query_size, key_size, value_size, hidden_size, output_size)
net = transformer(net_1, net_2)
net.eval()
Z, _ = net(X, Y, enc_valid_lens)
print(len(tgt_vocab))
print(Z.shape)
'''



class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        a, b = weights.shape
        for i in range(a):
            for j in range(valid_len[i], b):
                weights[i][j] = 0
        
        self.reduction='none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).mean(dim=1)
        return weighted_loss


#使用GPU
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def grad_clipping(net, theta):
    """裁剪梯度"""
    params = [p for p in net.parameters() if p.requires_grad and p.grad is not None]
    if not params:
        return
    # 计算所有梯度的L2范数
    grad_norm = torch.sqrt(sum(torch.sum(p.grad ** 2) for p in params))
    if grad_norm > theta:
        for p in params:
            p.grad.data *= theta / grad_norm  # 按比例缩小梯度

class TransformerLRScheduler(_LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps, k):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.k = k
        super(TransformerLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        step_num = self._step_count + (self.k - 1) * self.warmup_steps
        lr = 0.1*self.d_model ** (-0.5) * min(step_num ** (-0.5), step_num * self.warmup_steps ** (-1.5))
        return [lr for _ in self.optimizer.param_groups]

#训练
def train_seq2seq(net, data_iter, lr, tgt_vocab, device, k):
    """训练序列到序列模型"""
    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])
    if k == 1:
        net.apply(xavier_init_weights)
    
    
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-8, lr=0.001)
    scheduler = TransformerLRScheduler(optimizer, 512, 4000, k)
    loss = MaskedSoftmaxCELoss()  
    net.train()
    #animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[10, num_epochs])

    
    count = 0
    y_axis = []
    l_loss = []
    l_valid_lens = []
    #timer = d2l.Timer()
    #metric = d2l.Accumulator(2)  # 训练损失总和，词元数量
    for batch in data_iter:
        start_time = time.time() 
    
        optimizer.zero_grad()
        X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
        
        bos = torch.tensor([tgt_vocab['[CLS]']] * Y.shape[0],
                        device=device).reshape(-1, 1)
        dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学??
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间1：{execution_time:.6f} 秒")
        
        start_time = time.time()

        Y_hat, _ = net(X, dec_input, X_valid_len)
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间2：{execution_time:.6f} 秒")
        
        start_time = time.time()

        l = loss(Y_hat, Y, Y_valid_len)
        l.sum().backward()      # 损失函数的标量进行“反向传播”
        
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间3：{execution_time:.6f} 秒")
        
        start_time = time.time()

        grad_clipping(net, 1)  #梯度裁剪
        num_tokens = Y_valid_len.sum()
        value_1, value_2 = l.sum().item(), num_tokens.item()
        l_loss.append(value_1)
        l_valid_lens.append(value_2)
        if count > 20:
            break

        if count % 50 == 0:
            print(count)
            mean_loss = value_1/value_2
            print(mean_loss)
            y_axis.append(mean_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(current_lr)
        
        if count % 1000 == 0:
            torch.save(net.state_dict(), "C:/Users/23062/Desktop/transformer/test_transformer_model_"+str(k)+".pth")


        optimizer.step()
        scheduler.step()
        count += 1

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"代码执行时间4：{execution_time:.6f} 秒")

    x_axis = [10*i for i in range(len(y_axis))]
    '''plt.figure(figsize = (8,5))
    plt.plot(x_axis, y_axis, marker = 'o', linestyle = '-', color = 'blue', linewidth = 2, markersize = 1)

    title = "mean loss after every 10 batchs in " + str((k-1)*6000+1) + "~" + str(k*6000) +" th steps"
    plt.title(title, fontsize = 14)
    plt.xlabel("num_batchs", fontsize = 12)
    plt.ylabel("loss_mean", fontsize = 12)
    plt.show()

    print("hi")
'''
    torch.save(net.state_dict(), "C:/Users/23062/Desktop/transformer/test_transformer_model_"+str(k)+".pth")
    
    with open("C:/Users/23062/Desktop/transformer/test_loss_"+str(k)+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(l_loss)

    with open("C:/Users/23062/Desktop/transformer/test_valid_lens_"+str(k)+".csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(l_valid_lens)
    
    print("hello")
        

'''  
#超参数设置(英德)
embed_size, num_blocks, normalized_shape, dropout, num_head = 32, 3, 32, 0.1, 8
input_size, query_size, key_size, value_size, hidden_size, output_size = 32, 32, 32, 32, 128, 32
batch_size, num_steps, lr, num_epochs, device = 64, 15, 0.005, 5, try_gpu()
data_iter, src_vocab, tgt_vocab = load_data_nmt(source, target, batch_size, num_steps)
vocab_size_1, vocab_size_2 = len(src_vocab), len(tgt_vocab)
print(len(data_iter), vocab_size_1, vocab_size_2)


#超参数设置(英德)
embed_size, num_blocks, normalized_shape, dropout, num_head = 64, 3, 64, 0.1, 8
input_size, query_size, key_size, value_size, hidden_size, output_size = 64, 64, 64, 64, 128, 64
batch_size, num_steps, lr, num_epochs, device = 64, 10, 0.005, 5, try_gpu()
data_iter, src_vocab, tgt_vocab = load_data_nmt(source, target, batch_size, num_steps)
vocab_size_1, vocab_size_2 = len(src_vocab), len(tgt_vocab)
print(len(data_iter), vocab_size_1, vocab_size_2)
'''

#超参数设置(原论文)
#n = 6

#for n in range(6, 11):
    #print("n = ", n)
n = 1
embed_size, num_blocks, normalized_shape, dropout, num_head = 512, 6, 512, 0.1, 8
input_size, query_size, key_size, value_size, hidden_size, output_size = 512, 64, 64, 64, 2048, 512
batch_size, num_steps, lr, num_epochs, device = 24, 20, 0.005, 5, try_gpu()

target, source = get_data('C:/Users/23062/Desktop/wmt14_gr-en/wmt14_translate_de-en_train_preprocessed.csv', 11, batch_size)
print(len(source), len(target))

src_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
src_vocab = src_tokenizer.get_vocab()
tgt_tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
tgt_vocab = tgt_tokenizer.get_vocab()
#src_vocab, tgt_vocab = load_data_nmt(source, target, batch_size, num_steps)
src_array, src_valid_len = build_array_nmt(source, src_vocab, src_tokenizer, num_steps)
tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, tgt_tokenizer, num_steps)
data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
data_iter = load_array(data_arrays, batch_size)


vocab_size_1, vocab_size_2 = len(src_vocab), len(tgt_vocab)
print(len(data_iter), vocab_size_1, vocab_size_2)


encoder = transformer_encoder(vocab_size_1, embed_size, num_blocks,
                            normalized_shape, dropout,
                            num_head, input_size, query_size, key_size, value_size,
                            hidden_size, output_size)
decoder = transformer_decoder(vocab_size_2, embed_size, num_blocks, 
                            normalized_shape, dropout, 
                            num_head, input_size, query_size, key_size, value_size, 
                            hidden_size, output_size)


net = transformer(encoder, decoder)


#net.load_state_dict(torch.load("C:/Users/23062/Desktop/transformer/latest_transformer_model_1.pth"))

train_seq2seq(net, data_iter, lr, tgt_vocab, device, n)

#net.load_state_dict(torch.load("C:/Users/23062/Desktop/transformer/latest_transformer_model_5.pth"))

#train_seq2seq(net, data_iter, lr, tgt_vocab, device, 2)

#net.load_state_dict(torch.load("C:/Users/23062/Desktop/transformer/transformer_model_2.pth"))
#train_seq2seq(net, data_iter, lr, tgt_vocab, device, 3)

#net.load_state_dict(torch.load("C:/Users/23062/Desktop/transformer/new_transformer_model_1.pth"))

#print(f"耗时: {time.time()-start:.2f}秒")

net.eval()

'''net.eval()  # 设置为评估模式（关闭 Dropout 等）
X = torch.zeros((2, 100), dtype = torch.long)
Y = torch.zeros((2, 100), dtype = torch.long)
enc_valid_lens = torch.tensor([2,3])
Z, _ = net(X, Y, enc_valid_lens)
print(Z.shape)'''



def predict(input_sentence):
    input = [src_vocab[token] for token in src_tokenizer.tokenize(input_sentence.lower())] + [src_vocab['[SEP]']]
    input_valid_len = torch.tensor([len(input)])
    input = truncate_pad(input, num_steps, src_vocab['[PAD]'])

    input = torch.unsqueeze(torch.tensor(input, dtype = torch.long), dim = 0)  #形状：1*num_steps
    #print(input)
    output_mid = net.encoder(input, input_valid_len)
    state = net.decoder.init_state(output_mid, input_valid_len)
    output = torch.unsqueeze(torch.tensor([tgt_vocab['[CLS]']], dtype = torch.long), dim = 0)
    list_output = []
    flag = True
    for i in range(num_steps):
        Y, state = net.decoder(output, state)
        output = Y.argmax(dim = 2)
        if flag:
            flag = False
            #print(Y[0][0][5], Y[0][0][27])
        pred = output.squeeze(dim = 0).type(torch.int32).item()
        if pred == tgt_vocab['[SEP]']:
            break
        list_output.append(pred)
    return tgt_tokenizer.decode(list_output, skip_special_tokens=True)



def bleu(pred_seq, label_seq, k):  #@save
    """计算BLEU"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)

    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score


def get_data_2(path, n):
    inputs, labels = [], []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i > n:
                break
            if i > 0 and len(row) >= 2:
                inputs.append(row[1])
                labels.append(row[0])
            i+=1 
    return inputs, labels

inputs, labels = get_data_2('C:/Users/23062/Desktop/wmt14_gr-en/wmt14_translate_de-en_train_preprocessed.csv', 100)
print(len(labels))

def get_data_3(path, n):
    ids = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        i = 0
        for row in reader:
            if i > n:
                break
            if i > 0 and len(row) >= 2:
                ids.append(row[0])

            i+=1 
    return ids

#ids = get_data_3('C:/Users/23062/Desktop/wmt14_gr-en/scores_sorted.csv', 20)
#print(len(ids))
'''
predictions = []

for i in range(len(inputs)):
    input = inputs[i]
    label = labels[i]
    output = predict(input)
    if i%5 == 0:
        print(i)
        print(input)
        print(output)
        print(label)
    predictions.append(output.split(' '))
    labels[i] = label.split(' ')
    

    
print(corpus_bleu(predictions, labels, weights=(1.0, 0, 0, 0)))




with open('C:/Users/23062/Desktop/wmt14_gr-en/scores.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for i in range(len(scores)):
        writer.writerow(scores[i])

print(count)
'''


