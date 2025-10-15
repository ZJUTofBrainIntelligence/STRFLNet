import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data


# Transformer 部分
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.n_head = 16

    def forward(self, Q, K, V):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            self.d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.

        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        # attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context = ScaledDotProductAttention(self.d_k)(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)  # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]

        return nn.LayerNorm(self.d_model).cuda()(output + residual)


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(self.d_model).cuda()(output + residual)  # [batch_size, seq_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, d_ff):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, d_v)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_input1, enc_input2):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.enc_self_attn(enc_input1, enc_input2, enc_input2)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_k, d_v, d_ff):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_k, d_v, d_ff) for _ in range(n_layers)])

    def forward(self, enc_input1, enc_input2):
        global enc
        enc_outputs1 = enc_input1
        enc_outputs2 = enc_input2
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc = layer(enc_outputs1, enc_outputs2)
        return enc

class Cross(nn.Module):
    def __init__(self, chan_num = 62, class_num = 3, Feature_num=64):
        super(Cross, self).__init__()
        self.chan_num = chan_num
        self.class_num = class_num
        self.band_num = Feature_num
        self.encoder = Encoder(n_layers=2, n_heads=8, d_model=self.band_num * 2, d_k=8, d_v=8, d_ff=10)
        self.encoder1 = Encoder(n_layers=2, n_heads=8, d_model=self.band_num, d_k=8, d_v=8, d_ff=10)
        self.encoder2 = Encoder(n_layers=2, n_heads=8, d_model=self.band_num, d_k=8, d_v=8, d_ff=10)
        self.linear = nn.Linear(self.chan_num * self.band_num * 2, 64)
        self.A = torch.rand((1, self.chan_num * self.chan_num), dtype=torch.float32, requires_grad=False).cuda()
        self.linear2 = nn.Linear(64, self.class_num)

        self.fc = nn.Linear(self.chan_num * self.band_num * 2, 64)

    def forward(self, input1, input2):
        # [n, 32, 8]
        # A_ds = self.GATENet(self.A)
        # A_ds = A_ds.reshape(self.chan_num, self.chan_num)
        # de = x[:, :, :self.band_num]
        # psd = x[:, :, self.band_num:]
        feat1 = self.encoder1(input1, input2)
        #feat2 = self.encoder2(input2, input1)

        # [n, 32, 8]
        #feat0 = torch.cat([feat1, feat2], dim=2)
        #feat = self.encoder(feat0 ,feat0)
        # feat = feat.reshape(-1, self.chan_num * self.band_num * 2)
        # feat = self.linear(feat)
        # out = self.linear2(feat)

        #tsne = feat.reshape(x.shape[0], -1)  # feat.view(x.shape[0],-1)
        return feat1

if __name__ == '__main__':
    x1 = torch.rand(32, 62, 64).cuda()
    x2 = torch.rand(32, 62, 64).cuda()
    model = Cross().cuda()
    ma = model(x1, x2)
    print(ma.shape)