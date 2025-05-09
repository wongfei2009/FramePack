"""
Configure logging settings for the diffusers_helper package.
Logs are only written to the console.
"""

import logging
import os

# Get the logger for the package
logger = logging.getLogger('diffusers_helper')

# Set log level from environment variable or default to INFO
DEFAULT_LOG_LEVEL = os.environ.get('FRAMEPACK_LOG_LEVEL', 'INFO').upper()
LOG_LEVEL = getattr(logging, DEFAULT_LOG_LEVEL, logging.INFO)
logger.setLevel(LOG_LEVEL)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(LOG_LEVEL)

# Create formatter and add it to the handler
formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(console_handler)

def set_log_level(level):
    """
    Set the logging level for the diffusers_helper package.
    
    Args:
        level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logger.setLevel(numeric_level)
    for handler in logger.handlers:
        handler.setLevel(numeric_level)
    
    return f"Log level set to {level}"
