import os 
from persona_sdk import PersonaBuilder 
 
def main(): 
    # Initialize the SDK with your API key 
    api_key = os.getenv("PERSONA_API_KEY", "your_api_key_here") 
    builder = PersonaBuilder(api_key=api_key) 
 
    # Create a chatbot from template 
    chatbot = builder.from_template("professional") 
 
    # Start a conversation 
    response = chatbot.chat("Hello! Can you help me with a business inquiry?") 
    print(f"Chatbot: {response}") 
 
    # Get personality info 
    personality_info = chatbot.get_personality_info() 
    print("\nPersonality traits:") 
    print(personality_info["traits"]) 
 
if __name__ == "__main__": 
    main() 
