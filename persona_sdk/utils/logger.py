import logging 
from pathlib import Path 
from typing import Optional 
 
def setup_logger( 
    name: str = "persona_sdk", 
    level: int = logging.INFO, 
    log_file: Optional[Path] = None 
) -
    """Set up logging configuration.""" 
    logger = logging.getLogger(name) 
    logger.setLevel(level) 
 
    formatter = logging.Formatter( 
        '(name)s - (message)s' 
    ) 
 
    # Console handler 
    console_handler = logging.StreamHandler() 
    console_handler.setFormatter(formatter) 
    logger.addHandler(console_handler) 
 
    # File handler if log_file provided 
    if log_file: 
        file_handler = logging.FileHandler(log_file) 
        file_handler.setFormatter(formatter) 
        logger.addHandler(file_handler) 
 
    return logger 
