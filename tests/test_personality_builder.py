import pytest 
from persona_sdk import PersonaBuilder 
 
def test_template_creation(): 
    builder = PersonaBuilder(api_key="test_key") 
    chatbot = builder.from_template("professional") 
    assert chatbot is not None 
