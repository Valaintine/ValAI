import os 
from persona_sdk import PersonaBuilder 
from pathlib import Path 
 
def create_custom_personality(): 
    api_key = os.getenv("PERSONA_API_KEY", "your_api_key_here") 
    builder = PersonaBuilder(api_key=api_key) 
 
    ### Create from prompt 
    prompt =    "A witty professor who specializes in quantum physics. They love making complex concepts simple using creative analogies. Their communication style is informal but informative, and they often use humor to explain difficult topics." 
    
    additional_context = { 
        "expertise_level": "expert", 
        "communication_style": "informal", 
        "humor_level": "high", 
        "teaching_approach": "analogy-based" 
    } 
 
    chatbot = builder.from_prompt(prompt, additional_context) 
 
    # Example conversation 
    questions = [ 
        "Can you explain quantum entanglement?", 
        "How does this relate to quantum computing?", 
        "That's fascinating! Can you give me another analogy?" 
    ] 
 
    for question in questions: 
        print(f"\nUser: {question}") 
        response = chatbot.chat(question) 
        print(f"Professor: {response}") 
 
if __name__ == "__main__": 
    create_custom_personality() 
