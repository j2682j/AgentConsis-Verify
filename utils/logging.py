"""gnaa.utils.logging 模組。"""

import logging
import sys
from typing import Optional

def setup_logger(
    name: str = "hello_agents",
    level: str = "INFO",
    format_string: Optional[str] = None
) -> logging.Logger:
    """setup_logger 的主要實作。"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            format_string or 
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def get_logger(name: str = "hello_agents") -> logging.Logger:
    """get_logger 的主要實作。"""
    return logging.getLogger(name) 
