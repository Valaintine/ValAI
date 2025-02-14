import yaml 
from pathlib import Path 
from typing import Dict, Optional 
import jsonschema 
 
class TemplateValidator: 
    """Validates personality templates against schemas.""" 
    def __init__(self): 
        schema_path = Path(__file__).parent / 'schemas' / 'base_schema.yaml' 
        with open(schema_path) as f: 
            self.base_schema = yaml.safe_load(f) 
 
    def validate(self, template: Dict) -
        """Validate a template against the schema.""" 
        try: 
            jsonschema.validate(template, self.base_schema) 
            return True 
        except jsonschema.exceptions.ValidationError: 
            return False 
