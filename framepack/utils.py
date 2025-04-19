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
        print("Sage Attention not imported - install with 'pip install sageattention==1.0.6'")
    
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
