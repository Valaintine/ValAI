import logging
from typing import Dict, Optional, Union, List
import torch
import numpy as np
from pathlib import Path
import json
from PIL import Image
from datetime import datetime

from ..models.one_shot_learner import OneShotLearner
from ..models.personality_encoder import PersonalityEncoder
from ..models.trait_classifier import TraitClassifier
from ..utils.image_processor import ImageProcessor
from ..utils.prompt_parser import PromptParser
from ..templates.template_manager import TemplateManager
from ..utils.validators import validate_traits, validate_template

logger = logging.getLogger(__name__)

class PersonalityBuilder:
    """
    Main class for building chatbot personalities using various methods including
    templates, images, and natural language prompts.
    """
    
    def __init__(
        self,
        api_key: str,
        config_path: Optional[Path] = None,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the PersonalityBuilder.
        
        Args:
            api_key (str): API key for authentication
            config_path (Path, optional): Path to custom configuration
            cache_dir (Path, optional): Directory for caching models and embeddings
        """
        self.api_key = api_key
        self.config = self._load_config(config_path)
        self.cache_dir = cache_dir or Path.home() / ".persona_sdk" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_components()
        
        logger.info("PersonalityBuilder initialized successfully")

    def _initialize_components(self):
        """Initialize all required components and models."""
        try:
            # Initialize one-shot learner
            self.one_shot_learner = OneShotLearner(
                model_path=self.config.get("models", {}).get("one_shot_path"),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            
            # Initialize personality encoder
            self.personality_encoder = PersonalityEncoder(
                embedding_dim=self.config.get("models", {}).get("embedding_dim", 1024),
                cache_dir=self.cache_dir
            )
            
            # Initialize trait classifier
            self.trait_classifier = TraitClassifier(
                num_traits=self.config.get("models", {}).get("num_traits", 128)
            )
            
            # Initialize utility components
            self.template_manager = TemplateManager()
            self.image_processor = ImageProcessor()
            self.prompt_parser = PromptParser()
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            raise RuntimeError(f"Failed to initialize components: {str(e)}")

    def from_template(
        self,
        template_name: str,
        modifications: Optional[Dict] = None,
        validate: bool = True
    ) -> 'Chatbot':
        """
        Create a chatbot from a predefined template.
        
        Args:
            template_name (str): Name of the template to use
            modifications (Dict, optional): Modifications to apply to the template
            validate (bool): Whether to validate the template configuration
            
        Returns:
            Chatbot: Configured chatbot instance
        """
        logger.info(f"Creating chatbot from template: {template_name}")
        
        try:
            # Load and validate template
            template = self.template_manager.load_template(template_name)
            if validate:
                validate_template(template)
                
            # Apply modifications if provided
            if modifications:
                template = self.template_manager.apply_modifications(template, modifications)
                if validate:
                    validate_template(template)
                    
            # Generate personality embedding
            personality_embedding = self.personality_encoder.encode_template(template)
            
            # Create and return chatbot
            return self._create_chatbot(
                personality_embedding=personality_embedding,
                traits=template,
                template_name=template_name
            )
            
        except Exception as e:
            logger.error(f"Error creating chatbot from template: {str(e)}")
            raise RuntimeError(f"Failed to create chatbot from template: {str(e)}")

    def from_image(
        self,
        image_path: Union[str, Path],
        traits: Optional[List[str]] = None,
        min_confidence: float = 0.75
    ) -> 'Chatbot':
        """
        Create a chatbot personality from an image.
        
        Args:
            image_path (Union[str, Path]): Path to the image
            traits (List[str], optional): Additional traits to consider
            min_confidence (float): Minimum confidence threshold for trait detection
            
        Returns:
            Chatbot: Configured chatbot instance
        """
        logger.info(f"Creating chatbot from image: {image_path}")
        
        try:
            # Load and process image
            image = Image.open(image_path)
            image_features = self.image_processor.extract_features(image)
            
            # Apply one-shot learning to extract personality traits
            personality_traits = self.one_shot_learner.infer_traits(
                image_features,
                min_confidence=min_confidence
            )
            
            # Add additional traits if provided
            if traits:
                additional_traits = self.trait_classifier.classify_traits(traits)
                for trait, value in additional_traits.items():
                    if value["confidence"] >= min_confidence:
                        personality_traits[trait] = value
                        
            # Validate combined traits
            validate_traits(personality_traits)
            
            # Generate personality embedding
            personality_embedding = self.personality_encoder.encode_traits(personality_traits)
            
            # Save metadata
            metadata = {
                "source": "image",
                "image_path": str(image_path),
                "creation_date": datetime.now().isoformat(),
                "additional_traits": traits,
                "min_confidence": min_confidence
            }
            
            # Create and return chatbot
            return self._create_chatbot(
                personality_embedding=personality_embedding,
                traits=personality_traits,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating chatbot from image: {str(e)}")
            raise RuntimeError(f"Failed to create chatbot from image: {str(e)}")

    def from_prompt(
        self,
        prompt: str,
        additional_context: Optional[Dict] = None,
        style_guide: Optional[Dict] = None
    ) -> 'Chatbot':
        """
        Create a chatbot personality from a natural language prompt.
        
        Args:
            prompt (str): Natural language description of the personality
            additional_context (Dict, optional): Additional context or constraints
            style_guide (Dict, optional): Style and tone guidelines
            
        Returns:
            Chatbot: Configured chatbot instance
        """
        logger.info("Creating chatbot from prompt")
        
        try:
            # Parse prompt into traits
            parsed_traits = self.prompt_parser.parse(
                prompt,
                style_guide=style_guide
            )
            
            # Add additional context if provided
            if additional_context:
                context_traits = self.prompt_parser.parse_context(additional_context)
                parsed_traits.update(context_traits)
                
            # Validate traits
            validate_traits(parsed_traits)
            
            # Generate personality embedding
            personality_embedding = self.personality_encoder.encode_prompt(parsed_traits)
            
            # Save metadata
            metadata = {
                "source": "prompt",
                "original_prompt": prompt,
                "creation_date": datetime.now().isoformat(),
                "additional_context": additional_context,
                "style_guide": style_guide
            }
            
            # Create and return chatbot
            return self._create_chatbot(
                personality_embedding=personality_embedding,
                traits=parsed_traits,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error creating chatbot from prompt: {str(e)}")
            raise RuntimeError(f"Failed to create chatbot from prompt: {str(e)}")

    def export_personality(
        self,
        chatbot: 'Chatbot',
        export_path: Union[str, Path]
    ):
        """
        Export a chatbot's personality configuration for later use.
        
        Args:
            chatbot (Chatbot): Chatbot instance to export
            export_path (Union[str, Path]): Path to save the export file
        """
        export_data = {
            "personality_embedding": chatbot.personality_embedding.tolist(),
            "traits": chatbot.traits,
            "metadata": chatbot.metadata,
            "config": chatbot.config,
            "export_date": datetime.now().isoformat()
        }
        
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        logger.info(f"Personality exported to {export_path}")

    def import_personality(
        self,
        import_path: Union[str, Path]
    ) -> 'Chatbot':
        """
        Import a previously exported personality configuration.
        
        Args:
            import_path (Union[str, Path]): Path to the exported personality file
            
        Returns:
            Chatbot: Configured chatbot instance
        """
        with open(import_path, 'r') as f:
            import_data = json.load(f)
            
        personality_embedding = np.array(import_data["personality_embedding"])
        
        return self._create_chatbot(
            personality_embedding=personality_embedding,
            traits=import_data["traits"],
            metadata=import_data["metadata"],
            config=import_data.get("config")
        )

    def _create_chatbot(
        self,
        personality_embedding: np.ndarray,
        traits: Dict,
        metadata: Optional[Dict] = None,
        template_name: Optional[str] = None,
        config: Optional[Dict] = None
    ) -> 'Chatbot':
        """
        Internal method to create and configure a chatbot instance.
        
        Args:
            personality_embedding (np.ndarray): Encoded personality embedding
            traits (Dict): Personality traits and characteristics
            metadata (Dict, optional): Additional metadata about the personality
            template_name (str, optional): Name of the template if used
            config (Dict, optional): Custom configuration to use
            
        Returns:
            Chatbot: Configured chatbot instance
        """
        from .chat_engine import ChatEngine
        
        # Use provided config or default
        config = config or self.config
        
        # Create metadata if not provided
        metadata = metadata or {
            "creation_date": datetime.now().isoformat(),
            "template_name": template_name
        }
        
        return ChatEngine(
            personality_embedding=personality_embedding,
            traits=traits,
            config=config,
            api_key=self.api_key,
            metadata=metadata
        )

    def _load_config(self, config_path: Optional[Path] = None) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path (Path, optional): Path to configuration file
            
        Returns:
            Dict: Configuration dictionary
        """
        from ..config.config_loader import load_config
        return load_config(config_path)

    def get_available_templates(self) -> List[str]:
        """
        Get a list of available personality templates.
        
        Returns:
            List[str]: List of template names
        """
        return self.template_manager.list_templates()

    def get_template_details(self, template_name: str) -> Dict:
        """
        Get detailed information about a specific template.
        
        Args:
            template_name (str): Name of the template
            
        Returns:
            Dict: Template details and configuration
        """
        return self.template_manager.get_template_details(template_name)