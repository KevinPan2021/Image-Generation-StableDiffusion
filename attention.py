# implement self attention and cross attention
# self attention: key, value, and query are from same input matrix
# cross attention: key[mat1], value[mat2], and query[mat2] are from 2 input matrics

import torch
from torch import nn
from torch.nn import functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_heads, d_embed, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        
        self.in_proj = nn.Linear(d_embed, 3*d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        
    def forward(self, x, causal_mask=False):
        # x: (batch, seq_len, d_embed)
        input_shape = x.shape
        batch, seq_len, d_embed = input_shape
        
        intermim_shape = (batch, seq_len, self.n_heads, self.d_head)
        
        # (batch, seq_len, d_embed) -> 3*(batch, seq_len, d_embed)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (batch, seq_len, d_embed) -> (batch, n_heads, seq_len, d_head)
        q = q.view(intermim_shape).transpose(1, 2)
        k = k.view(intermim_shape).transpose(1, 2)
        v = v.view(intermim_shape).transpose(1, 2)
        
        # batch, n_heads, seq_len, seq_len
        weight = q @ k.transpose(-1, -2)
        
        # apply mask
        if causal_mask:
            # upper triangular matrix filled with -inf
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)
        
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        # (batch, n_heads, seq_len, seq_len) @ (batch, n_heads, seq_len, d_head) -> (batch, n_heads, seq_len, d_head)
        output = weight @ v
        # (batch, n_heads, seq_len, d_head) -> (batch, seq_len, n_heads, d_head)
        output = output.transpose(1, 2).reshape(input_shape)
        output = self.out_proj(output)
        # (batch, seq_len, d_embed)
        return output




class CrossAttention(nn.Module):
    def __init__(self, n_heads, d_embed, d_cross, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads
        
        
    def forward(self, latent, context):
        # latent: (batch, seq_len_Q, dim_Q)
        # context: (batch, seq_len_KV, dim_KV)
        input_shape = latent.shape
        batch, seq_length, d_embed = input_shape
        interim_shape = (batch, -1, self.n_heads, self.d_head)
        
        # query * Wq
        q = self.q_proj(latent)
        k = self.k_proj(context)
        v = self.v_proj(context)
        
        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)
        
        weight = q @ k.transpose(-1, -2)
        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, dim=-1)
        
        output = weight @ v
        
        output = output.transpose(1, 2).contiguous()
        output = output.view(input_shape)
        output = self.out_proj(output)
        
        return output
    
    
        