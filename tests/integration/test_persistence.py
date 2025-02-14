import pytest 
from persona_sdk import PersonalityBuilder 
import tempfile 
from pathlib import Path 
 
def test_save_load_personality(api_key): 
    builder = PersonalityBuilder(api_key=api_key) 
    chatbot = builder.from_template("professional") 
    # Save personality 
    with tempfile.TemporaryDirectory() as tmpdir: 
        save_path = Path(tmpdir) / "personality.json" 
        builder.export_personality(chatbot, save_path) 
        # Load personality 
        loaded_chatbot = builder.import_personality(save_path) 
        assert loaded_chatbot is not None 
