import pytest 
from persona_sdk.utils import ImageProcessor, PromptParser, validate_traits 
from PIL import Image 
import numpy as np 
 
def test_image_processor(): 
    processor = ImageProcessor() 
    # Create dummy image 
    image = Image.new('RGB', (224, 224)) 
    features = processor.extract_features(image) 
    assert features.shape == (1, 3, 224, 224) 
 
def test_prompt_parser(): 
    parser = PromptParser() 
    prompt = "A friendly and professional assistant" 
    traits = parser.parse(prompt) 
    assert "friendly" in traits or "professional" in traits 
