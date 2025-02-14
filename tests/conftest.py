import pytest 
import torch 
import numpy as np 
from pathlib import Path 
from persona_sdk import PersonalityBuilder 
 
@pytest.fixture 
def api_key(): 
    return "test_api_key" 
 
@pytest.fixture 
def personality_builder(api_key): 
    return PersonalityBuilder(api_key=api_key) 
 
@pytest.fixture 
def sample_personality_embedding(): 
    return np.random.randn(1024).astype(np.float32) 
 
@pytest.fixture 
def sample_traits(): 
    return { 
        "formality": 0.8, 
        "expertise": 0.9, 
        "friendliness": 0.7 
    } 
