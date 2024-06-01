import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 d_in, d_out,
                 context_length,
                 dropout,
                 num_heads,
                 qkv_bias=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def do_w(self, w, b, n):
        w = w.view(b, n, self.num_heads, self.head_dim)
        w = w.transpose(1, 2)
        return w

    def weights(self, x):
        b, n, _ = x.shape
        keys = self.do_w(self.W_key(x), b, n)
        queries = self.do_w(self.W_query(x), b, n)
        values = self.do_w(self.W_value(x), b, n)
        return queries, keys, values

    def attention(self, x, q, k):
        _, n, __ = x.shape
        attn_scores = q @ k.transpose(2, 3)
        attn_scores.masked_fill_(
            self.mask.bool()[:n, :n],
            -torch.inf
        )
        attn_weights = torch.softmax(
            attn_scores / k.shape[-1] ** 0.5,
            dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        return attn_weights

    def context(self, x, v, aw):
        b, n, _ = x.shape
        context_vec = (aw @ v).transpose(1, 2)
        context_vec = context_vec.contiguous().view(
            b, n, self.d_out
        )
        context_vec = self.out_proj(context_vec)

        return context_vec

    def forward(self, x):
        queries, keys, values = self.weights(x)

        return self.context(
            x,
            values,
            self.dropout(
                self.attention(x, queries, keys)))
