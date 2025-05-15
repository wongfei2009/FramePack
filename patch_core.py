#!/usr/bin/env python3
"""
This script patches the SageAttention core.py file to add RTX 5090 (sm_120) support.
It enables proper initialization of the SM120_ENABLED flag.
"""

import os
import re

def patch_core_py():
    """
    Patch the core.py file of SageAttention to add proper SM120 support.
    """
    core_py_path = 'sageattention/core.py'
    
    # Check if core.py exists
    if not os.path.exists(core_py_path):
        print(f"Error: {core_py_path} not found!")
        return False
    
    # Read the original file
    with open(core_py_path, 'r') as f:
        content = f.read()
    
    # Check if it already has SM120 import block
    if 'SM120_ENABLED' in content:
        print("core.py already has SM120_ENABLED flag. Check the implementation.")
        
        # If the flag exists but is not properly initialized with an import block, fix it
        if 'from . import _qattn_sm120' not in content:
            # Find the SM90 import block
            sm90_block_pattern = re.compile(r'try:.*?from \. import _qattn_sm90.*?SM90_ENABLED = True.*?except:.*?SM90_ENABLED = False', re.DOTALL)
            sm90_block_match = sm90_block_pattern.search(content)
            
            if sm90_block_match:
                sm90_block = sm90_block_match.group(0)
                sm120_block = """
try:
    from . import _qattn_sm120
    SM120_ENABLED = True
except:
    SM120_ENABLED = False"""
                
                # Add SM120 import block after SM90 block
                new_content = content.replace(sm90_block, sm90_block + sm120_block)
                
                # Write the patched content
                with open(core_py_path, 'w') as f:
                    f.write(new_content)
                
                print("Successfully added SM120 import block to core.py.")
                return True
            else:
                print("Could not find SM90 import block. Manual inspection needed.")
                return False
        else:
            print("SM120 import block already exists. No changes needed.")
    
    # If SM120_ENABLED doesn't exist at all, add it
    # First, find the SM90 import block
    sm90_block_pattern = re.compile(r'try:.*?from \. import _qattn_sm90.*?SM90_ENABLED = True.*?except:.*?SM90_ENABLED = False', re.DOTALL)
    sm90_block_match = sm90_block_pattern.search(content)
    
    if sm90_block_match:
        sm90_block = sm90_block_match.group(0)
        sm120_block = """

try:
    from . import _qattn_sm120
    SM120_ENABLED = True
except:
    SM120_ENABLED = False"""
        
        # Add SM120 import block after SM90 block
        new_content = content.replace(sm90_block, sm90_block + sm120_block)
        
        # Check and fix sageattn_qk_int8_pv_fp8_cuda function to handle SM120
        fp8_cuda_pattern = r'assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."'
        fp8_cuda_replacement = r'assert SM89_ENABLED or SM120_ENABLED, "SM89 or SM120 kernel is not available. Make sure your GPUs have compute capability 8.9 or 12.0."'
        
        new_content = new_content.replace(fp8_cuda_pattern, fp8_cuda_replacement)
        
        # Write the patched content
        with open(core_py_path, 'w') as f:
            f.write(new_content)
        
        print("Successfully patched core.py to add RTX 5090 (sm_120) support.")
        return True
    else:
        print("Could not find SM90 import block in core.py. Manual inspection needed.")
        return False

def main():
    """Main function."""
    if not patch_core_py():
        exit(1)

if __name__ == "__main__":
    main()
