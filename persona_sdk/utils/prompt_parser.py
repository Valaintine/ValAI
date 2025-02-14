import re 
from typing import Dict, List, Optional 
import spacy 
import json 
from pathlib import Path 
import logging 
 
logger = logging.getLogger(__name__) 
 
class PromptParser: 
    """Parses natural language prompts into structured personality traits.""" 
    def __init__(self): 
        self.nlp = spacy.load("en_core_web_sm") 
        self.trait_patterns = self._load_trait_patterns() 
 
    def parse(self, prompt: str, style_guide: Optional[Dict] = None) -
        """Parse a natural language prompt into traits.""" 
        doc = self.nlp(prompt.lower()) 
        traits = {} 
 
        # Extract personality traits 
        for pattern in self.trait_patterns: 
            matches = pattern.search(prompt.lower()) 
            if matches: 
                trait_value = self._calculate_trait_value(matches) 
                traits[matches.group("trait")] = trait_value 
 
        # Apply style guide modifications 
        if style_guide: 
            traits = self._apply_style_guide(traits, style_guide) 
 
        return traits 
