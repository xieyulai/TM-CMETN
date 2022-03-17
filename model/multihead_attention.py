import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(Q, K, V, mask, dropout=None):
    # Q, K, V are (B, *(H), seq_len, d_model//H = d_k)
    # mask is     (B,    1,       1,               Ss)
    d_k = Q.size(-1)
    QKt = Q.matmul(K.transpose(-1, -2))
    sm_input = QKt / np.sqrt(d_k)

    if mask is not None:
        sm_input = sm_input.masked_fill(mask == 0, -float('inf'))

    softmax = F.softmax(sm_input, dim=-1)
    out = softmax.matmul(V)

    if dropout is not None:
        out = dropout(out)

    return out


class MultiHeadedAttention(nn.Module):

    def __init__(self, d_model_Q, d_model_K, d_model_V, H, d_model=None, dout_p=0.0):
        super(MultiHeadedAttention, self).__init__()
        self.d_model_Q = d_model_Q
        self.d_model_K = d_model_K
        self.d_model_V = d_model_V
        self.H = H
        self.d_model = d_model
        self.dout_p = dout_p
        self.d_model_H = self.d_model // H

        if d_model is None:
            print('d_model is None!')
            self.d_model = self.d_model_Q

        self.linear_Q2d = nn.Linear(self.d_model_Q, self.d_model)
        self.linear_K2d = nn.Linear(self.d_model_K, self.d_model)
        self.linear_V2d = nn.Linear(self.d_model_V, self.d_model)
        self.linear_d2Q = nn.Linear(self.d_model, self.d_model_Q)

        self.dropout = nn.Dropout(self.dout_p)

        assert self.d_model % H == 0

    def forward(self, Q, K, V, mask):
        B, Sq, Dq = Q.shape

        Q = self.linear_Q2d(Q)
        K = self.linear_K2d(K)
        V = self.linear_V2d(V)

        Q = Q.view(B, -1, self.H, self.d_model_H).permute(0, 2, 1, 3)
        K = K.view(B, -1, self.H, self.d_model_H).permute(0, 2, 1, 3)
        V = V.view(B, -1, self.H, self.d_model_H).permute(0, 2, 1, 3)

        if mask is not None:
            # the same mask for all heads -> (B, 1, 1, Sm2)
            mask = mask.unsqueeze(1)

        att_out = attention(Q, K, V, mask, self.dropout)

        att_out = att_out.permute(0, 2, 1, 3).contiguous().view((B, Sq, -1))

        att_out = self.linear_d2Q(att_out)

        return att_out
