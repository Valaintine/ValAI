import yaml 
import logging 
from typing import Dict, List, Optional, Union 
from pathlib import Path 
import json 
from datetime import datetime 
 
logger = logging.getLogger(__name__) 
 
class TemplateManager: 
    """Manages personality templates and their modifications.""" 
    def __init__(self, custom_template_path: Optional[Path] = None): 
        self.base_path = Path(__file__).parent / 'base_templates' 
        self.custom_path = custom_template_path or Path.home() / '.persona_sdk' / 'templates' 
        self.custom_path.mkdir(parents=True, exist_ok=True) 
        self._cache = {} 
 
    def load_template(self, template_name: str) -
        """Load a template by name.""" 
        if template_name in self._cache: 
            return self._cache[template_name].copy() 
 
        # Try custom templates first 
        custom_path = self.custom_path / f"{template_name}.yaml" 
        if custom_path.exists(): 
            template = self._load_yaml(custom_path) 
        else: 
            # Fall back to base templates 
            base_path = self.base_path / f"{template_name}.yaml" 
            if not base_path.exists(): 
                raise ValueError(f"Template not found: {template_name}") 
            template = self._load_yaml(base_path) 
 
        self._cache[template_name] = template.copy() 
        return template 
 
    def save_template(self, template_name: str, template: Dict): 
        """Save a custom template.""" 
        template['metadata'] = { 
            'created_at': datetime.now().isoformat(), 
            'version': '1.0' 
        } 
        path = self.custom_path / f"{template_name}.yaml" 
        with open(path, 'w') as f: 
            yaml.safe_dump(template, f) 
        self._cache[template_name] = template.copy() 
 
    def list_templates(self) -
        """List all available templates.""" 
        base_templates = [p.stem for p in self.base_path.glob('*.yaml')] 
        custom_templates = [p.stem for p in self.custom_path.glob('*.yaml')] 
        return sorted(set(base_templates + custom_templates)) 
