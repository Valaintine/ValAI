import re 
from typing import Optional 
 
class TextSanitizer: 
    """Sanitizes and cleanses text input.""" 
    @staticmethod 
    def sanitize(text: str) -
        """Sanitize input text.""" 
        # Remove special characters 
        text = re.sub(r'[\w\s.,!?-]', '', text) 
        # Remove extra whitespace 
        text = ' '.join(text.split()) 
        return text 
