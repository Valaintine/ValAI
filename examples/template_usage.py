import os 
from persona_sdk import PersonaBuilder 
 
def demonstrate_templates(): 
    api_key = os.getenv("PERSONA_API_KEY", "your_api_key_here") 
    builder = PersonaBuilder(api_key=api_key) 
 
    # Available templates 
    templates = ["professional", "casual", "academic"] 
 
    for template_name in templates: 
        print(f"\nTesting {template_name} template:") 
        # Create chatbot with template 
        chatbot = builder.from_template(template_name) 
 
        # Test standard questions 
        questions = [ 
            "How would you handle a customer complaint?", 
            "What's your approach to problem-solving?", 
            "Can you explain a complex topic?" 
        ] 
 
        for question in questions: 
            print(f"\nUser: {question}") 
            response = chatbot.chat(question) 
            print(f"Assistant: {response}") 
 
        # Show personality info 
        print("\nTemplate Personality Info:") 
        print(chatbot.get_personality_info()) 
 
if __name__ == "__main__": 
    demonstrate_templates() 
