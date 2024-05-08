# text encoder

import torch
from torch import nn

from attention import SelfAttention


class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab, n_embd, n_tokens):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embd)
        self.position_embedding = nn.Parameter(torch.zeros(n_tokens, n_embd))
    
    def forward(self, tokens):
        # (batch, seq_len) -> (batch, seq_len, dim)
        x = self.token_embedding(tokens)
        x += self.position_embedding
        
        return x
    
    

class CLIPLayer(nn.Module):
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(n_embd)
        self.attention = SelfAttention(n_head, n_embd)
        self.layernorm_2 = nn.LayerNorm(n_embd)
        self.linear_1 = nn.Linear(n_embd, 4*n_embd)
        self.linear_2 = nn.Linear(4*n_embd, n_embd)
    
    
    def forward(self, x):
        # batch, seq_len, dim
        residue = x
        
        # self attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)
        x += residue
        
        # feedforward
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        # GELU activation function
        x = x * torch.sigmoid(1.702*x)
        x = self.linear_2(x)
        
        x += residue
        
        return x
        
        
        
        
class CLIP(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([CLIPLayer(12, 768) for i in range(12)])
        self.layernorm = nn.LayerNorm(768)
    
    
    def forward(self, tokens):
        tokens = tokens.type(torch.long)
        
        # (batch, seq_len) -> (batch, seq_len, dim)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        # (batch, seq_len, dim)
        output = self.layernorm(state)
        return output