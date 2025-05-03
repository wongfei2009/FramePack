"""
Parameter configuration module for FramePack.

Handles saving, loading, and resetting of UI parameters.
"""

import os
import json
import logging
from pathlib import Path

# Set up logging
logger = logging.getLogger(__name__)

# Import mode settings
from framepack.config.modes.video_mode_settings import MODE_TYPE_NORMAL

# Default parameter values optimized for FramePack-F1
DEFAULT_PARAMETERS = {
    # Basic parameters
    "seed": 31337,
    "total_latent_sections": 5,  # Increased for F1 to leverage anti-drifting capability
    "resolution_scale": "Full (1x)",
    
    # Mode settings
    "generation_mode": MODE_TYPE_NORMAL,
    "video_length_preset": "5 seconds",
    "enable_keyframe_copy": True,
    
    # Advanced parameters
    "use_teacache": True,
    "teacache_thresh": 0.15,
    "steps": 25,
    "gs": 10.0,  # Distilled CFG Scale optimized for F1
    "cfg": 1.0,
    "rs": 0.0,
    "gpu_memory_preservation": 6.0,
    "latent_window_size": 9,
    "enable_optimization": False,  # PyTorch optimizations for attention and BFloat16 conversion
    "mp4_crf": 16,
    "end_frame_strength": 0.5,  # EndFrame influence in forward-only sampling
    "movement_scale": 0.1,      # Controls the amount of camera movement in generated videos (0.0-1.0)
    
    # FP8 and LoRA parameters
    "fp8_optimization": False,  # Enable FP8 quantization of transformer weights
    "lora_multiplier": 0.8,     # Multiplier for LoRA weights when applied
    "lora_dropdown": "None",    # Selected LoRA file (None or filename)
    
    # Other parameters as needed
    "prompt": "",
    "n_prompt": ""
}

class ParameterConfig:
    """
    Handles parameter configuration for the FramePack application.
    
    This class provides methods to save, load, and reset UI parameters.
    """
    
    def __init__(self):
        """Initialize the parameter configuration handler."""
        # Get the root directory of the application
        self.root_dir = self._get_root_dir()
        self.config_dir = os.path.join(self.root_dir, "config")
        self.config_file = os.path.join(self.config_dir, "parameters.json")
        
        # Ensure the config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Initialize with default parameters
        self.parameters = DEFAULT_PARAMETERS.copy()
        
        # Load parameters from file (if exists)
        self.load_parameters()
        
        # Log the loaded parameters
        logger.info(f"Parameter config initialized with: {self.parameters}")
    
    def _get_root_dir(self):
        """Get the root directory of the application."""
        # This assumes this file is in framepack/config/parameter_config.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up two levels to get to the root directory
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        logger.info(f"Using root directory: {root_dir}")
        return root_dir
    
    def load_parameters(self):
        """
        Load parameters from the configuration file.
        
        If the file doesn't exist or is invalid, default values are used.
        """
        try:
            if os.path.exists(self.config_file):
                logger.info(f"Loading parameters from {self.config_file}")
                with open(self.config_file, 'r') as f:
                    loaded_params = json.load(f)
                    
                    # Start with defaults and update with loaded values
                    # This ensures any new parameters added in updates will have defaults
                    self.parameters = DEFAULT_PARAMETERS.copy()
                    self.parameters.update(loaded_params)
                    
                    # Log the loaded parameters
                    logger.info(f"Parameters loaded successfully: {self.parameters}")
            else:
                # Use default parameters if config file doesn't exist
                self.parameters = DEFAULT_PARAMETERS.copy()
                logger.info(f"Config file not found at {self.config_file}. Using default parameters.")
                
                # Save defaults to create the file
                self.save_parameters()
                
                # Log the default parameters
                logger.info(f"Default parameters saved to {self.config_file}: {self.parameters}")
        except Exception as e:
            logger.error(f"Error loading parameters: {e}")
            # Use defaults if there's an error
            self.parameters = DEFAULT_PARAMETERS.copy()
            logger.info(f"Using default parameters due to error: {self.parameters}")
    
    def save_parameters(self):
        """Save the current parameters to the configuration file."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                json.dump(self.parameters, f, indent=4)
            logger.info(f"Parameters saved to {self.config_file}")
            logger.debug(f"Saved parameters: {self.parameters}")
            return True
        except Exception as e:
            logger.error(f"Error saving parameters: {e}")
            return False
    
    def reset_parameters(self):
        """Reset all parameters to their default values."""
        self.parameters = DEFAULT_PARAMETERS.copy()
        self.save_parameters()
        logger.info("Parameters reset to defaults")
        return self.parameters
    
    def update_parameter(self, name, value):
        """
        Update a single parameter value and save the configuration.
        
        Args:
            name: Parameter name
            value: New parameter value
        """
        if name in self.parameters:
            self.parameters[name] = value
            self.save_parameters()
            logger.debug(f"Parameter '{name}' updated to '{value}'")
    
    def update_parameters(self, parameters_dict):
        """
        Update multiple parameters at once and save the configuration.
        
        Args:
            parameters_dict: Dictionary of parameter names and values
        """
        for name, value in parameters_dict.items():
            if name in self.parameters:
                self.parameters[name] = value
        
        self.save_parameters()
        logger.debug(f"Updated multiple parameters: {', '.join(parameters_dict.keys())}")
    
    def get_parameter(self, name):
        """
        Get the value of a parameter.
        
        Args:
            name: Parameter name
            
        Returns:
            Parameter value, or None if not found
        """
        return self.parameters.get(name)
    
    def get_all_parameters(self):
        """
        Get all parameters.
        
        Returns:
            Dictionary of all parameters
        """
        return self.parameters.copy()


# Create a singleton instance
param_config = ParameterConfig()
