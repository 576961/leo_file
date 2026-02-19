
import torch
from torch import nn
import numpy as np
import math

'''
dic = {}
dic['a'] = 0
dic['b'] = 3
dic['c'] = 2
dic[3] = 1

print(dic)

reversed_dic = {value: key for key, value in dic.items()}
print(reversed_dic)


dic_inv = {}

for i in range(len(dic)):
    dic_inv[dic.values()[i]] = dic.keys()[i]

print(dic_inv)
'''

'''
a = torch.arange(10)
a = torch.sin(a)
#print(a)
x = torch.arange(6).reshape(2,3)
y = torch.arange(9).reshape(3,3)
z = torch.cat([x,y],0)
#print(x)
#print(z)
print(z.shape)


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

P = PositionalEncoding(dropout = 0.1)
Q = PositionalEncoding(dropout = 0)
X = torch.randn(2,3)
#print(X)
X1 = P(X)
X2 = Q(X)
#print(X1)
#print(X2)

class Ffn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(**kwargs)
        self.layer_1 = nn.Linear(input_size, hidden_size)
        self.layer_2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, input):
        return self.layer_2(self.activation(self.layer_1(input)))

ffn = Ffn(4, 4, 8)
ffn.eval()
X = ffn(torch.ones((2, 3, 4)))
#print(X)

class add_norm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

F = add_norm(3,0)
X = torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.float32)
#print(F(X,X))

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
                count += 1

        return nn.functional.softmax(X, dim=-1)

Y = masked_softmax(torch.tensor([[[14,8],[8,5]]], dtype=torch.float32)/math.sqrt(3), valid_lens=None)
print(Y)

class scaled_dot_product_attention(nn.Module):
    def __init__(self, **kwargs):
        #query_size = key_size
        super().__init__(**kwargs)

    def forward(self, queries, keys, values,
                valid_lens=None):  # querie, keys, values都是三维张量，大小为(batch_size, num_steps, embed_size)
        weights = masked_softmax(torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(queries.shape[2]), valid_lens)
        print(weights)
        return torch.bmm(weights, values)
X = torch.tensor([[[1,2,3],[0,1,2]]], dtype=torch.float32)
F = scaled_dot_product_attention()
print(torch.bmm(Y,X))
Y = F(X,X,X)
print(Y)
'''


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

X = torch.tensor([[[1,2,3],[0,1,2]],[[1,2,3],[0,1,2]]], dtype=torch.float32)
valid_lens = torch.tensor([[[2,3],[1,2]]])
print(masked_softmax(X, valid_lens))



'''
valid_lens = torch.tensor([[1,2,3],[2,1,2]], dtype=torch.float32)
seq_len = 4
print(valid_lens.unsqueeze(-1).shape)
print(torch.arange(seq_len).unsqueeze(0).shape)
mask = torch.arange(seq_len).unsqueeze(0) < valid_lens.unsqueeze(-1)
print(mask)

X = torch.arange(4)
print(X)
mask = X < 3
print(mask)'''


def cal_circle_area(r):
    '''计算圆的面积'''
    return math.pi * r**2

# 写一个快速排序函数
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    less = arr[:len(arr) // 2]
    greater = arr[len(arr) // 2:]
    return quicksort(less) + [pivot] + quicksort(greater)

X = torch.ones((32, 10, 293), dtype=torch.float32)
print(X.reshape(-1, 293).shape)

Y = torch.ones((32, 10), dtype=torch.float32)
print(Y.reshape(-1).shape)