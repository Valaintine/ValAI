import pytest 
from persona_sdk.core.chat_engine import ChatEngine 
import numpy as np 
 
def test_chat_initialization(api_key, sample_personality_embedding, sample_traits): 
    engine = ChatEngine( 
        personality_embedding=sample_personality_embedding, 
        traits=sample_traits, 
        config={}, 
        api_key=api_key 
    ) 
    assert engine is not None 
 
def test_chat_response(api_key, sample_personality_embedding, sample_traits): 
    engine = ChatEngine( 
        personality_embedding=sample_personality_embedding, 
        traits=sample_traits, 
        config={}, 
        api_key=api_key 
    ) 
    response = engine.chat("Hello!") 
    assert isinstance(response, str) 
    assert len(response) 
