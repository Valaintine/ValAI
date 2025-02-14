import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Dict, Optional, List 
import numpy as np 
from pathlib import Path 
 
class OneShotLearner(nn.Module): 
    """One-shot learning model for personality trait extraction.""" 
    def __init__(self, 
        model_path: Optional[str] = None, 
        device: Optional[torch.device] = None, 
        embedding_dim: int = 512 
    ): 
        super().__init__() 
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        # Feature extraction backbone 
        self.feature_extractor = nn.Sequential( 
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
            self._make_layer(64, 64, 3), 
            self._make_layer(64, 128, 4, stride=2), 
            self._make_layer(128, 256, 6, stride=2), 
            self._make_layer(256, 512, 3, stride=2), 
            nn.AdaptiveAvgPool2d((1, 1)) 
        ) 
