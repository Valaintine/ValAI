import json 
from pathlib import Path 
from typing import Any, Dict, Optional 
import time 
import logging 
 
logger = logging.getLogger(__name__) 
 
class CacheManager: 
    """Manages caching of model outputs and computations.""" 
    def __init__( 
        self, 
        cache_dir: Optional[Path] = None, 
        max_size: int = 1000, 
        ttl: int = 3600 
    ): 
        self.cache_dir = cache_dir or Path.home() / '.persona_sdk' / 'cache' 
        self.cache_dir.mkdir(parents=True, exist_ok=True) 
        self.max_size = max_size 
        self.ttl = ttl 
        self.cache = {} 
 
    def get(self, key: str) -
        """Get item from cache.""" 
        if key in self.cache: 
            item = self.cache[key] 
                return item['data'] 
            else: 
                del self.cache[key] 
        return None 
