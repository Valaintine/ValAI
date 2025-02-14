import pytest 
import torch 
from persona_sdk.models import OneShotLearner, PersonalityEncoder, TraitClassifier 
 
def test_one_shot_learner(): 
    model = OneShotLearner() 
    assert model is not None 
    # Test forward pass 
    x = torch.randn(1, 3, 224, 224) 
    output = model(x) 
    assert output.shape[-1] == 256  # embedding dimension 
 
def test_personality_encoder(): 
    model = PersonalityEncoder() 
    assert model is not None 
    # Test encoding 
    traits = {"formality": 0.8, "expertise": 0.9} 
    embedding = model.encode_traits(traits) 
    assert embedding.shape[-1] == 1024  # embedding dimension 
