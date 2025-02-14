import torch 
import torch.nn as nn 
from .attention import MultiHeadAttention 
 
class TransformerBlock(nn.Module): 
    """Transformer block for personality encoding.""" 
    def __init__(self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.1 
    ): 
        super().__init__() 
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout) 
        self.feed_forward = nn.Sequential( 
            nn.Linear(embed_dim, ff_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(ff_dim, embed_dim) 
        ) 
