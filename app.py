"""
Main application entry point for FramePack.
"""

import argparse
import os
from custom_asyncio_policy import apply_asyncio_fixes

# Apply custom asyncio fixes before any other imports that might use asyncio
apply_asyncio_fixes()

# Apply RunPod-specific fixes if running in RunPod environment
if os.environ.get('RUNPOD_POD_ID') or os.environ.get('RUNPOD') or 'runpod' in os.environ.get('HOME', '').lower():
    print("RunPod environment detected, applying Gradio compatibility fixes...")
    from runpod_gradio_fix import apply_fixes
    apply_fixes()

# Import from diffusers_helper
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
    prepare_outputs_directory()
    
    # Load models
    models = FramePackModels()
    models.load_models(has_sage_attn=has_sage_attn)
    
    # Create a global stream for communication
    stream = AsyncStream()
    
    # Create UI with config
    block = create_ui(models, stream)
    
    # Get outputs directory from environment or default
    import os
    outputs_dir = os.environ.get('OUTPUTS_DIR', '/workspace/framepack/outputs')
    
    # Launch the application
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        max_threads=20,    # Increase number of worker threads for handling requests
        allowed_paths=[outputs_dir],  # Allow Gradio to serve files from this path
        root_path="",      # Empty root_path to handle proxy correctly
        ssl_verify=False   # Disable SSL verification for proxy environments
    )
    
    print(f"Allowed paths for Gradio: {outputs_dir}")

if __name__ == "__main__":
    main()
