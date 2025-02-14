from typing import Dict, Any 
import jsonschema 
from pathlib import Path 
import yaml 
import logging 
 
logger = logging.getLogger(__name__) 
 
def validate_traits(traits: Dict[str, Any]) -
    """Validate personality traits.""" 
    try: 
        schema_path = Path(__file__).parent.parent / 'schemas' / 'traits_schema.yaml' 
        with open(schema_path) as f: 
            schema = yaml.safe_load(f) 
 
        jsonschema.validate(traits, schema) 
        return True 
    except Exception as e: 
        logger.error(f"Trait validation error: {str(e)}") 
        return False 
