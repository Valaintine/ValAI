import pytest 
from persona_sdk import PersonalityBuilder 
 
def test_multi_turn_conversation(api_key): 
    builder = PersonalityBuilder(api_key=api_key) 
    chatbot = builder.from_template("professional") 
    # Test multiple conversation turns 
    responses = [] 
    messages = [ 
        "Hello!", 
        "How can you help me?", 
        "Thank you!" 
    ] 
    for message in messages: 
        response = chatbot.chat(message) 
        responses.append(response) 
        assert response is not None 
    assert len(responses) == len(messages) 
