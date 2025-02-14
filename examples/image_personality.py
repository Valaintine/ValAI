import os 
from pathlib import Path 
from persona_sdk import PersonaBuilder 
 
def create_image_based_personality(): 
    api_key = os.getenv("PERSONA_API_KEY", "your_api_key_here") 
    builder = PersonaBuilder(api_key=api_key) 
 
    # Path to character image 
    image_path = Path("assets/character.jpg") 
 
    # Additional traits to consider 
    traits = [ 
        "friendly", 
        "professional", 
        "knowledgeable", 
        "empathetic" 
    ] 
 
    # Create chatbot from image 
    chatbot = builder.from_image(image_path, traits) 
 
    # Example interaction 
    conversations = [ 
        "Hi! How can you help me today?", 
        "What's your approach to problem-solving?", 
        "Can you tell me more about your expertise?" 
    ] 
 
    for conversation in conversations: 
        print(f"\nUser: {conversation}") 
        response = chatbot.chat(conversation) 
        print(f"Assistant: {response}") 
 
    # Print personality analysis 
    print("\nPersonality Analysis:") 
    print(chatbot.get_personality_info()) 
 
if __name__ == "__main__": 
    create_image_based_personality() 
