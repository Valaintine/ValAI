import os 
import yaml 
import logging 
from pathlib import Path 
from typing import Dict, Optional, Union 
 
logger = logging.getLogger(__name__) 
 
class ConfigurationError(Exception): 
    """Raised when there's an error in configuration.""" 
    pass 
 
def load_config(config_path: Optional[Union[str, Path]] = None) -
    """Load configuration from file or use defaults.""" 
    try: 
        # Load default configuration 
        default_config_path = Path(__file__).parent / 'default_config.yaml' 
        with open(default_config_path, 'r') as f: 
            config = yaml.safe_load(f) 
 
        # If custom config provided, merge with defaults 
        if config_path: 
            config_path = Path(config_path) 
            if not config_path.exists(): 
                raise ConfigurationError(f"Config file not found: {config_path}") 
 
            with open(config_path, 'r') as f: 
                custom_config = yaml.safe_load(f) 
            config = deep_merge(config, custom_config) 
 
        # Apply environment variable overrides 
        config = apply_env_overrides(config) 
 
        return config 
 
    except Exception as e: 
        logger.error(f"Error loading configuration: {str(e)}") 
        raise ConfigurationError(f"Failed to load configuration: {str(e)}") 
 
def deep_merge(dict1: Dict, dict2: Dict) -
    """Recursively merge two dictionaries.""" 
    result = dict1.copy() 
 
    for key, value in dict2.items(): 
        if key in result and isinstance(result[key], dict) and isinstance(value, dict): 
            result[key] = deep_merge(result[key], value) 
        else: 
            result[key] = value 
 
    return result 
 
def apply_env_overrides(config: Dict) -
    """Apply environment variable overrides to configuration.""" 
    env_prefix = 'PERSONA_' 
    result = config.copy() 
 
    for env_var, value in os.environ.items(): 
        if env_var.startswith(env_prefix): 
            # Convert environment variable to config key 
            key_path = env_var[len(env_prefix):].lower().split('_') 
            current = result 
 
            # Navigate to the correct nested dictionary 
            for key in key_path[:-1]: 
                if key not in current: 
                    current[key] = {} 
                current = current[key] 
 
            # Set the value 
            try: 
                # Try to parse as int or float first 
                parsed_value = yaml.safe_load(value) 
                current[key_path[-1]] = parsed_value 
            except yaml.YAMLError: 
                # If parsing fails, use the string value 
                current[key_path[-1]] = value 
 
    return result 
