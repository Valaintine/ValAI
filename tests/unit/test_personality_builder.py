import pytest 
from persona_sdk import PersonalityBuilder 
import torch 
from pathlib import Path 
 
def test_initialization(api_key): 
    builder = PersonalityBuilder(api_key=api_key) 
    assert builder is not None 
    assert builder.api_key == api_key 
 
def test_from_template(personality_builder): 
    chatbot = personality_builder.from_template("professional") 
    assert chatbot is not None 
    assert "formality" in chatbot.traits 
 
def test_from_prompt(personality_builder): 
    prompt = "A friendly and knowledgeable assistant" 
    chatbot = personality_builder.from_prompt(prompt) 
    assert chatbot is not None 
    assert chatbot.personality_embedding is not None 
