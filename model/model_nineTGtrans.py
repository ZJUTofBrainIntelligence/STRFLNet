import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PLI import Graph
from torchdiffeq import odeint
from STODE6 import STODE
from crosstransformer_only import Cross
from selftransformer_temporal_position2 import Selftrans

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class attention(nn.Module):
    def __init__(self, dropout=0.0):
        super(attention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        # 没加掩码
        att = torch.bmm(q, k.transpose(1, 2))
        att = F.softmax(att, dim=2)
        # 添加dropout
        att = self.dropout(att)
        # 和V做点积
        out = torch.bmm(att, v)
        return out, att


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, h_num, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.h_dim = dim // h_num
        self.h_num = h_num
        self.l_k = nn.Linear(dim, self.h_dim * h_num)
        self.l_v = nn.Linear(dim, self.h_dim * h_num)
        self.l_q = nn.Linear(dim, self.h_dim * h_num)
        self.att = attention(dropout)
        self.linear = nn.Linear(self.h_dim * h_num, dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, k, v, q):
        # 残差连接
        batchsize = k.size(0)
        res = q
        h_dim = self.h_dim
        h_num = self.h_num
        k = self.l_k(k)
        v = self.l_v(v)
        q = self.l_q(q)
        k = k.reshape(batchsize * h_num, -1, h_dim)
        v = v.reshape(batchsize * h_num, -1, h_dim)
        q = q.reshape(batchsize * h_num, -1, h_dim)
        out, att = self.att(q, k, v)
        out = out.reshape(batchsize, -1, h_dim * h_num)
        out = self.linear(out)
        out = self.dropout(out)
        out = self.layernorm(res + out)
        return out, att

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(62, 128)
        self.fc = nn.Linear(128, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.linear(x)
        x = self.fc(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x[:, -1, :]
class EncoderLayer(nn.Module):
    def __init__(self, nodes=62, seq_len=200, hidden_channels=64, h_num=4, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.seq_len =  seq_len
        self.nodes = nodes
        self.att = MultiHeadAttention(self.nodes, h_num, dropout)
        self.stodeg = STODE(hidden_channels)
        self.start_conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=(1, 1))
        self.start_conv = nn.Conv2d(in_channels=1, out_channels=hidden_channels, kernel_size=(1, 1))
        self.conv_layer = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 1))
        self.Crosstrans = Cross(chan_num=62, class_num=4, Feature_num=62)
        self.Selftrans = Selftrans('seed iv')
        self.pooling_layer = nn.AvgPool2d((1, 10))
        self.pooling_layer1 = nn.AvgPool2d((1, 10))
        self.linear = nn.Linear(128, 64)
        self.adj = Graph(freqrange=(13, 30), srate=200, TH=0.1, init_alpha=2.0)
    def forward(self, inputs):
        #step1：特征提取
        out, att = self.att(inputs, inputs, inputs)
        adj = self.adj.get_adjacency(inputs)
        B = inputs.size(0)
        # step2：时间信息提取
        out1 = self.Selftrans(out)
        # step3：时空信息提取（时空信息变为三维）
        out2 = out.reshape(out.shape[0], -1, out.shape[1], out.shape[2])
        out2 = self.start_conv(out2 )
        out2 = self.stodeg(out2, adj)
        out2 = out2.reshape(out2.shape[0], 64, out2.shape[2]*4, out2.shape[3])
        out2 = self.conv_layer(out2)
        out2 = torch.squeeze(out2)
        # step4：中间态
        out3 = out1 + out2
        # step5：多模态融合(简单融合)
        out1 = self.Crosstrans(out3, out1)
        out2 = self.Crosstrans(out3, out2)
        AB = self.Crosstrans(out1, out2)
        BA = self.Crosstrans(out2, out1)
        out3 = AB + BA
        out3 = self.Selftrans(out3)
        return out3

class Conv2d_res(nn.Module):

    def __init__(self, in_channels, out_channels, residual=True, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2, stride=2),
            nn.BatchNorm2d(out_channels),
        )
        if not residual:
            self.residual = lambda x: 0
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels),
            )


    def forward(self, x):
        res = self.residual(x)
        x = self.conv(x)
        return mish(x + res)

def mish(x):
    return x * (torch.tanh(F.softplus(x)))

class GODE(nn.Module):
    def __init__(self, frame=200, method='euler', t=1, step_size=0.5):
        super(GODE, self).__init__()
        self.t = t
        self.flag = round(self.t / step_size)
        self.encoder_layer = EncoderLayer(nodes=62, seq_len=frame, hidden_channels=64, h_num=4, dropout=0.0)
        self.linear = LinearModel()
        self.decode = nn.Sequential(
                    # Conv2d_res(256, 256),  # T, V -> T//2, V//2
                    # Conv2d_res(128, 64),  # T, V -> T//2, V//2
                    Conv2d_res(64, 3),  # T, V -> T//2, V//2
                )
    def forward(self, input):
        input = input.to(torch.float32)
        x = self.encoder_layer(input)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    x1 = torch.rand(3, 200, 62).to(device)#(32,60,63)
    #model =GODE(frame=200, method='euler', t=1, step_size=0.5).to(device)
    model = GODE().to(device)
    ma=model(x1)
    print(ma.shape)