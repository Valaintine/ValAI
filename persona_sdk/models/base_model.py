import torch 
import torch.nn as nn 
from typing import Any, Dict 
 
class BaseModel(nn.Module): 
    """Base class for all models in the SDK.""" 
    def __init__(self): 
        super().__init__() 
 
    def save_checkpoint(self, path: str, metadata: Dict[str, Any] = None): 
        """Save model checkpoint with metadata.""" 
        checkpoint = { 
            'model_state_dict': self.state_dict(), 
            'metadata': metadata or {} 
        } 
        torch.save(checkpoint, path) 
 
    def load_checkpoint(self, path: str) -, Any]: 
        """Load model checkpoint and return metadata.""" 
        checkpoint = torch.load(path) 
        self.load_state_dict(checkpoint['model_state_dict']) 
        return checkpoint.get('metadata', {}) 
