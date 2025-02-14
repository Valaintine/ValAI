import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 
 
class MultiHeadAttention(nn.Module): 
    """Multi-head attention mechanism.""" 
    def __init__(self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.1 
    ): 
        super().__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.dropout = dropout 
        self.head_dim = embed_dim // num_heads 
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads" 
