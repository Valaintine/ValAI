 
import logging
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)

class ChatEngine:
    """
    Core chat engine that handles conversation management and response generation
    while maintaining personality consistency.
    """
    def __init__(
        self,
        personality_embedding: np.ndarray,
        traits: Dict,
        config: Dict,
        api_key: str,
        model_path: Optional[str] = None
    ):
        self.personality_embedding = personality_embedding
        self.traits = traits
        self.config = config
        self.api_key = api_key
        self.conversation_history = []
        self.session_state = {}
        
        # Initialize components
        self.tokenizer = self._initialize_tokenizer(model_path)
        self.model = self._initialize_model(model_path)
        self.personality_manager = PersonalityManager(
            personality_embedding,
            traits,
            config.get("personality_settings", {})
        )
        
        # Load generation settings
        self.generation_config = config.get("generation", {})
        self.max_history_length = config.get("conversation", {}).get("max_history_length", 10)
        
        logger.info("Chat engine initialized successfully")

    def chat(self, message: str, context: Optional[Dict] = None) -> Dict:
        """
        Process a user message and generate a response.
        
        Args:
            message (str): User input message
            context (Dict, optional): Additional context for response generation
            
        Returns:
            Dict: Response containing generated text and metadata
        """
        try:
            # Preprocess message
            processed_message = self._preprocess_message(message)
            
            # Update conversation history
            self._update_history({"role": "user", "content": processed_message})
            
            # Prepare generation context
            generation_context = self._prepare_context(context)
            
            # Generate response
            response = self._generate_response(
                processed_message,
                generation_context
            )
            
            # Postprocess response
            final_response = self._postprocess_response(response)
            
            # Update history with response
            self._update_history({"role": "assistant", "content": final_response["content"]})
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in chat processing: {str(e)}")
            return {
                "content": "I apologize, but I encountered an error processing your message.",
                "error": str(e),
                "status": "error"
            }

    def _initialize_tokenizer(self, model_path: Optional[str] = None) -> AutoTokenizer:
        """Initialize the tokenizer for text processing."""
        try:
            if model_path:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                model_name = self.config.get("model", {}).get("name", "gpt2")
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add special tokens if needed
            special_tokens = {
                "pad_token": "<pad>",
                "sep_token": "<sep>",
                "personality_token": "<personality>"
            }
            tokenizer.add_special_tokens(special_tokens)
            
            return tokenizer
            
        except Exception as e:
            logger.error(f"Error initializing tokenizer: {str(e)}")
            raise RuntimeError(f"Failed to initialize tokenizer: {str(e)}")

    def _initialize_model(self, model_path: Optional[str] = None) -> AutoModelForCausalLM:
        """Initialize the language model for response generation."""
        try:
            if model_path:
                model = AutoModelForCausalLM.from_pretrained(model_path)
            else:
                model_name = self.config.get("model", {}).get("name", "gpt2")
                model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Move model to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            # Resize token embeddings for special tokens
            model.resize_token_embeddings(len(self.tokenizer))
            
            return model
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def _preprocess_message(self, message: str) -> str:
        """
        Preprocess the user message before generation.
        
        Args:
            message (str): Raw user message
            
        Returns:
            str: Processed message
        """
        # Basic preprocessing
        processed = message.strip()
        
        # Apply any necessary filters or transformations
        processed = self._apply_filters(processed)
        
        return processed

    def _apply_filters(self, text: str) -> str:
        """Apply content filters and transformations to text."""
        # Implementation of content filtering logic
        return text

    def _prepare_context(self, additional_context: Optional[Dict] = None) -> Dict:
        """
        Prepare the generation context including personality and history.
        
        Args:
            additional_context (Dict, optional): Additional context to include
            
        Returns:
            Dict: Prepared context for generation
        """
        context = {
            "personality": self.personality_manager.get_context(),
            "history": self._format_history(),
            "system_state": self.session_state
        }
        
        if additional_context:
            context.update(additional_context)
            
        return context

    def _format_history(self) -> str:
        """Format conversation history for context window."""
        formatted_history = []
        
        for entry in self.conversation_history[-self.max_history_length:]:
            role = entry["role"]
            content = entry["content"]
            formatted_history.append(f"{role}: {content}")
            
        return "\n".join(formatted_history)

    def _generate_response(self, message: str, context: Dict) -> Dict:
        """
        Generate a response using the language model.
        
        Args:
            message (str): Processed user message
            context (Dict): Generation context
            
        Returns:
            Dict: Generated response and metadata
        """
        try:
            # Prepare input for model
            input_text = self._prepare_model_input(message, context)
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            input_ids = input_ids.to(self.model.device)
            
            # Get generation parameters
            gen_params = {
                "max_length": self.generation_config.get("max_length", 150),
                "min_length": self.generation_config.get("min_length", 10),
                "temperature": self.generation_config.get("temperature", 0.7),
                "top_p": self.generation_config.get("top_p", 0.9),
                "top_k": self.generation_config.get("top_k", 50),
                "repetition_penalty": self.generation_config.get("repetition_penalty", 1.2),
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            
            # Generate response
            start_time = time.time()
            output_ids = self.model.generate(
                input_ids,
                **gen_params
            )
            generation_time = time.time() - start_time
            
            # Decode response
            response_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract actual response from generated text
            response_text = self._extract_response(response_text, input_text)
            
            return {
                "content": response_text,
                "metadata": {
                    "generation_time": generation_time,
                    "input_tokens": len(input_ids[0]),
                    "output_tokens": len(output_ids[0]),
                    "status": "success"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise RuntimeError(f"Failed to generate response: {str(e)}")

    def _prepare_model_input(self, message: str, context: Dict) -> str:
        """Prepare the formatted input for the model."""
        personality_context = context["personality"]
        history = context["history"]
        
        template = f"""
<personality>
{personality_context}
</personality>

<history>
{history}
</history>

<input>
{message}
</input>

<response>"""
        
        return template.strip()

    def _extract_response(self, generated_text: str, input_text: str) -> str:
        """Extract the actual response from generated text."""
        response = generated_text[len(input_text):].strip()
        return response.split("</response>")[0].strip()

    def _postprocess_response(self, response: Dict) -> Dict:
        """
        Postprocess the generated response.
        
        Args:
            response (Dict): Raw response from generation
            
        Returns:
            Dict: Processed response
        """
        # Apply personality-specific modifications
        content = self.personality_manager.apply_personality(response["content"])
        
        # Update response with processed content
        response["content"] = content
        
        # Add timestamp
        response["metadata"]["timestamp"] = time.time()
        
        return response

    def _update_history(self, entry: Dict):
        """Update conversation history with new entry."""
        self.conversation_history.append(entry)
        
        # Trim history if needed
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def reset_conversation(self):
        """Reset the conversation history and session state."""
        self.conversation_history = []
        self.session_state = {}
        logger.info("Conversation reset")

    def save_conversation(self, file_path: Union[str, Path]):
        """
        Save the current conversation to a file.
        
        Args:
            file_path (Union[str, Path]): Path to save the conversation
        """
        save_data = {
            "conversation_history": self.conversation_history,
            "session_state": self.session_state,
            "personality_traits": self.traits,
            "timestamp": time.time()
        }
        
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)

    def load_conversation(self, file_path: Union[str, Path]):
        """
        Load a conversation from a file.
        
        Args:
            file_path (Union[str, Path]): Path to the conversation file
        """
        with open(file_path, 'r') as f:
            load_data = json.load(f)
            
        self.conversation_history = load_data["conversation_history"]
        self.session_state = load_data["session_state"]
        
        logger.info(f"Conversation loaded from {file_path}")

class PersonalityManager:
    """Manages personality traits and their application to responses."""
    
    def __init__(self, personality_embedding: np.ndarray, traits: Dict, config: Dict):
        self.personality_embedding = personality_embedding
        self.traits = traits
        self.config = config
        
    def get_context(self) -> str:
        """Get formatted personality context."""
        context_parts = []
        
        for trait, value in self.traits.items():
            context_parts.append(f"{trait}: {value}")
            
        return "\n".join(context_parts)
        
    def apply_personality(self, text: str) -> str:
        """Apply personality traits to modify response text."""
        # Implementation of personality-specific modifications
        return text