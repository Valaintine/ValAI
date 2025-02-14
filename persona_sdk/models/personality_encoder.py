import torch 
import torch.nn as nn 
from typing import Dict, List, Optional 
import numpy as np 
from pathlib import Path 
 
class PersonalityEncoder(nn.Module): 
    """Encodes personality traits into embeddings.""" 
    def __init__(self, 
        embedding_dim: int = 1024, 
        hidden_dim: int = 512, 
        num_layers: int = 4, 
        dropout: float = 0.1 
    ): 
        super().__init__() 
        self.embedding_dim = embedding_dim 
        # Trait embedding layers 
        self.trait_embedder = nn.Sequential( 
            nn.Linear(512, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, embedding_dim) 
        ) 
