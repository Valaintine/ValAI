import pytest 
from persona_sdk import PersonalityBuilder 
 
def test_template_to_chat(api_key): 
    # Test full pipeline from template creation to chat 
    builder = PersonalityBuilder(api_key=api_key) 
    chatbot = builder.from_template("professional") 
    response = chatbot.chat("Hello!") 
    assert response is not None 
    assert len(response) 
