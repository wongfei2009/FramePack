"""
Main application entry point for FramePack.
"""

import os
import sys
import argparse
from custom_asyncio_policy import apply_asyncio_fixes

# Apply custom asyncio fixes before any other imports that might use asyncio
apply_asyncio_fixes()

# Import from diffusers_helper
from diffusers_helper.hf_login import login
from diffusers_helper.thread_utils import AsyncStream

# Import from framepack package
from framepack.utils import debug_import_paths, setup_sage_attention, prepare_outputs_directory
from framepack.models import FramePackModels
from framepack.ui import create_ui

# Parse command line arguments
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--share', action='store_true')
    parser.add_argument("--server", type=str, default='0.0.0.0')
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()

def main():
    """Main application entry point."""
    # Parse command line arguments
    args = parse_args()
    print(args)
    
    # Initialize SageAttention
    debug_import_paths("sageattention")
    sageattn, sageattn_varlen, has_sage_attn = setup_sage_attention()
    
    # Create outputs directory
    outputs_folder = prepare_outputs_directory()
    
    # Load models
    models = FramePackModels()
    models.load_models(has_sage_attn=has_sage_attn)
    
    # Create a global stream for communication
    stream = AsyncStream()
    
    # Create UI
    block = create_ui(models, stream)
    
    # Launch the application
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        max_threads=20    # Increase number of worker threads for handling requests
    )

if __name__ == "__main__":
    main()
