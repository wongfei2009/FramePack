#!/usr/bin/env python3
"""
This script patches the SageAttention setup.py file to add RTX 5090 (sm_120) support.
"""

import os
import re

def patch_setup_py():
    """
    Patch the setup.py file of SageAttention to add sm_120 support for RTX 5090.
    """
    setup_py_path = 'setup.py'
    
    # Check if setup.py exists
    if not os.path.exists(setup_py_path):
        print(f"Error: {setup_py_path} not found!")
        return False
    
    # Read the original file
    with open(setup_py_path, 'r') as f:
        content = f.read()
    
    # Check if it already has sm_120 support
    if 'sm_120' in content or 'compute_120' in content:
        print("setup.py already has sm_120 support. No changes needed.")
        return True
    
    # Patch content for RTX 5090 support
    
    # 1. Add HAS_SM120 detection
    new_content = re.sub(
        r'HAS_SM80 = .+?\n\s*HAS_SM86 = .+?\n\s*HAS_SM90 = .+?\n',
        r'HAS_SM80 = compute_capabilities is not None and any(cc.startswith("8.0") for cc in compute_capabilities)\nHAS_SM86 = compute_capabilities is not None and any(cc.startswith("8.6") for cc in compute_capabilities)\nHAS_SM90 = compute_capabilities is not None and any(cc.startswith("9.0") for cc in compute_capabilities)\nHAS_SM120 = compute_capabilities is not None and any(cc.startswith("12.0") for cc in compute_capabilities)\n',
        content
    )
    
    # 2. Add sm_120 extensions if needed
    sm90_extension_pattern = r'(if HAS_SM90:.*?ext_modules\.append\(.*?\))'
    sm90_extension_match = re.search(sm90_extension_pattern, new_content, re.DOTALL)
    
    if sm90_extension_match:
        sm90_extension = sm90_extension_match.group(1)
        sm120_extension = sm90_extension.replace('HAS_SM90', 'HAS_SM120')
        sm120_extension = sm120_extension.replace('_qattn_sm90', '_qattn_sm120')
        sm120_extension = sm120_extension.replace('pybind_sm90.cpp', 'pybind_sm120.cpp')
        sm120_extension = sm120_extension.replace('qk_int_sv_f8_cuda_sm90.cu', 'qk_int_sv_f8_cuda_sm120.cu')
        
        # Insert SM120 extension block after SM90 block
        new_content = new_content.replace(sm90_extension, sm90_extension + '\n\n# RTX 5090 (Blackwell) extension\n' + sm120_extension)
    
    # 3. Ensure compute_capabilities is properly set
    compute_caps_pattern = r'compute_capabilities = \[(.*?)\]'
    compute_caps_match = re.search(compute_caps_pattern, new_content)
    
    if compute_caps_match:
        compute_caps = compute_caps_match.group(1)
        if '"12.0"' not in compute_caps and "'12.0'" not in compute_caps:
            # Add sm_120 to the compute capabilities list
            new_compute_caps = compute_caps
            if compute_caps.strip():
                new_compute_caps += ', "12.0"'
            else:
                new_compute_caps = '"8.0", "8.6", "9.0", "12.0"'
            new_content = new_content.replace(compute_caps_match.group(0), f'compute_capabilities = [{new_compute_caps}]')
    
    # Write the patched content
    with open(setup_py_path, 'w') as f:
        f.write(new_content)
    
    print("Successfully patched setup.py to add RTX 5090 (sm_120) support.")
    return True

def main():
    """Main function."""
    if not patch_setup_py():
        exit(1)

if __name__ == "__main__":
    main()
