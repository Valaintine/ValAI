from typing import List, Dict, Optional 
import torch 
from transformers import PreTrainedTokenizer 
 
class TokenProcessor: 
    """Handles token processing and manipulation.""" 
    def __init__(self, tokenizer: PreTrainedTokenizer): 
        self.tokenizer = tokenizer 
 
    def process_tokens( 
        self, 
        text: str, 
        max_length: Optional[int] = None 
    ) -
        """Process text into tokens.""" 
        tokens = self.tokenizer.encode( 
            text, 
            max_length=max_length, 
            truncation=True if max_length else False 
        ) 
        return { 
            'input_ids': tokens, 
            'length': len(tokens) 
        } 
