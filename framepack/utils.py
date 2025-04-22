"""
Utility functions for FramePack.
"""

import os
import sys
import importlib.util

def debug_import_paths(module_name="sageattention"):
    """
    Print debugging information about Python's import system for a specific module.
    
    Args:
        module_name: Name of the module to debug
        
    Returns:
        Boolean indicating if the module was found
    """
    # Check if the module is already imported
    if module_name in sys.modules:
        return True
    
    # Try to find the module in sys.path
    found = False
    for path in sys.path:
        spec = importlib.util.find_spec(module_name, [path])
        if spec is not None:
            found = True
            break
    
    return found

def setup_sage_attention():
    """
    Attempt to import and setup Sage Attention.
    
    Returns:
        Tuple containing (sageattn, sageattn_varlen, has_sage_attn)
    """
    try:
        from sageattention import sageattn, sageattn_varlen
        print("Successfully imported Sage Attention")
        has_sage_attn = True
    except ImportError:
        # Create dummy functions that do nothing but allow the code to run
        def sageattn(*args, **kwargs):
            return None
        def sageattn_varlen(*args, **kwargs):
            return None
        sageattn.is_available = lambda: False
        sageattn_varlen.is_available = lambda: False
        has_sage_attn = False
        print("Sage Attention not imported!")
    
    return sageattn, sageattn_varlen, has_sage_attn

def prepare_outputs_directory(outputs_folder='./outputs/'):
    """
    Prepare the outputs directory for saving results.
    
    Args:
        outputs_folder: Path to outputs directory
        
    Returns:
        Path to outputs directory
    """
    os.makedirs(outputs_folder, exist_ok=True)
    return outputs_folder

def prepare_generation_subfolder(outputs_folder='./outputs/', job_id=None):
    """
    Prepare a subfolder within the outputs directory for a specific generation job.
    
    Args:
        outputs_folder: Path to main outputs directory
        job_id: Generation job ID (timestamp). If None, a new one will be generated.
        
    Returns:
        Tuple of (subfolder_path, job_id)
    """
    from diffusers_helper.utils import generate_timestamp
    
    # Generate job ID if not provided
    if job_id is None:
        job_id = generate_timestamp()
    
    # Create subfolder path
    subfolder_path = os.path.join(outputs_folder, job_id)
    
    # Create the subfolder
    os.makedirs(subfolder_path, exist_ok=True)
    
    return subfolder_path, job_id
