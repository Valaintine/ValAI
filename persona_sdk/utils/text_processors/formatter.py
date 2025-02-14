from typing import Optional 
 
class TextFormatter: 
    """Formats text according to specified rules.""" 
    @staticmethod 
    def format_response( 
        text: str, 
        capitalize_sentences: bool = True, 
        add_periods: bool = True 
    ) -
        """Format response text.""" 
        if capitalize_sentences: 
            text = '. '.join(s.capitalize() for s in text.split('. ')) 
        if add_periods and not text.endswith('.'): 
            text += '.' 
        return text 
