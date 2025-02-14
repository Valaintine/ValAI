import torch 
import torchvision.transforms as transforms 
from PIL import Image 
import numpy as np 
from typing import Union, Tuple, Optional 
from pathlib import Path 
import logging 
 
logger = logging.getLogger(__name__) 
 
class ImageProcessor: 
    """Handles image processing and feature extraction.""" 
    def __init__( 
        self, 
        image_size: Tuple[int, int] = (224, 224), 
        normalize_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406), 
        normalize_std: Tuple[float, float, float] = (0.229, 0.224, 0.225) 
    ): 
        self.transform = transforms.Compose([ 
            transforms.Resize(image_size), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=normalize_mean, std=normalize_std) 
        ]) 
 
    def extract_features(self, image_input: Union[str, Path, Image.Image]) -
        """Extract features from an image.""" 
        try: 
            # Load image if path provided 
            if isinstance(image_input, (str, Path)): 
                image = Image.open(image_input).convert('RGB') 
            else: 
                image = image_input 
 
            # Apply transformations 
            tensor = self.transform(image) 
            tensor = tensor.unsqueeze(0) 
 
            return tensor 
 
        except Exception as e: 
            logger.error(f"Error processing image: {str(e)}") 
            raise RuntimeError(f"Failed to process image: {str(e)}") 
