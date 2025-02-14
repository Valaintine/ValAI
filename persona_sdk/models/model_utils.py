import torch 
import numpy as np 
from typing import Optional, Tuple 
 
def get_device(cuda_index: Optional[int] = None) -
    """Get appropriate device for model computation.""" 
    if cuda_index is not None and torch.cuda.is_available(): 
        return torch.device(f'cuda:{cuda_index}') 
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
