"""
Utility functions for handling LoRA files.
"""

import os

def get_lora_files_directory():
    """
    Get the path to the LoRA files directory.
    
    Returns:
        str: Path to the LoRA files directory
    """
    # Get the path relative to the project root
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    lora_dir = os.path.join(root_dir, 'local_models', 'lora')
    
    # Create the directory if it doesn't exist
    os.makedirs(lora_dir, exist_ok=True)
    
    return lora_dir

def list_lora_files():
    """
    Get a list of available LoRA files in the local_models/lora directory.
    
    Returns:
        list: List of dictionaries containing LoRA file information:
              - 'name': Display name (filename without extension)
              - 'path': Full path to the file
    """
    lora_dir = get_lora_files_directory()
    lora_files = []
    
    # Check if directory exists
    if not os.path.exists(lora_dir):
        print(f"LoRA directory does not exist: {lora_dir}")
        return lora_files
    
    # Scan directory for .safetensors files
    try:
        for filename in os.listdir(lora_dir):
            if filename.endswith('.safetensors'):
                # Create full path
                file_path = os.path.join(lora_dir, filename)
                
                # Get display name without extension
                display_name = os.path.splitext(filename)[0]
                
                # Add to the list
                lora_files.append({
                    'name': display_name,
                    'path': file_path
                })
    except Exception as e:
        print(f"Error scanning LoRA directory: {e}")
    
    # Sort by name for consistent display
    lora_files.sort(key=lambda x: x['name'].lower())
    
    return lora_files
