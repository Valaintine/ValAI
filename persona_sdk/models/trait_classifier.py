import torch 
import torch.nn as nn 
from typing import Dict, List, Optional 
import numpy as np 
 
class TraitClassifier(nn.Module): 
    """Classifies and quantifies personality traits.""" 
    def __init__(self, 
        num_traits: int = 128, 
        hidden_dim: int = 256, 
        dropout: float = 0.1 
    ): 
        super().__init__() 
        self.num_traits = num_traits 
        # Trait classification layers 
        self.classifier = nn.Sequential( 
            nn.Linear(512, hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_dim, num_traits) 
        ) 
