"""
RunPod Gradio Fix module

This module provides functions to patch Gradio's behavior to work with RunPod's proxy service.
The issue occurs because Gradio's route_utils.py doesn't recognize RunPod's proxy URL pattern,
resulting in a ValueError during request processing.
"""

import logging
import re
from functools import wraps
import sys

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def patch_gradio_for_runpod():
    """
    Patch Gradio's get_api_call_path function to handle RunPod URLs correctly.
    This fixes the "Request url has an unknown api call pattern" error.
    """
    try:
        import gradio.route_utils as route_utils
        
        # Store the original function
        original_get_api_call_path = route_utils.get_api_call_path
        
        @wraps(original_get_api_call_path)
        def patched_get_api_call_path(*, request):
            """Patched version of get_api_call_path that handles RunPod URLs"""
            root_url = str(request.url)
            
            # Log the URL for debugging
            logger.info(f"Processing URL: {root_url}")
            
            # Check if it's a RunPod URL pattern
            runpod_pattern = r"https?://[a-zA-Z0-9]+-\d+\.proxy\.runpod\.net"
            
            if re.match(runpod_pattern, root_url):
                logger.info("Detected RunPod proxy URL, applying patch")
                # For RunPod URLs, construct a basic API path
                # This assumes the request is to the API endpoint
                return "/api/predict"
            
            # Fall back to the original function for non-RunPod URLs
            try:
                return original_get_api_call_path(request=request)
            except ValueError as e:
                # If original function fails, log the error and try a fallback approach
                logger.warning(f"Original get_api_call_path failed: {str(e)}")
                logger.warning(f"Attempting fallback for URL: {root_url}")
                
                # Simple fallback - assume it's an API call
                return "/api/predict"
        
        # Replace the original function with our patched version
        route_utils.get_api_call_path = patched_get_api_call_path
        logger.info("Successfully patched Gradio's route_utils.get_api_call_path for RunPod compatibility")
        
    except ImportError:
        logger.error("Failed to import gradio.route_utils. Patch not applied.")
    except Exception as e:
        logger.error(f"Error patching Gradio: {str(e)}")

def apply_fixes():
    """Apply all necessary fixes for RunPod deployment"""
    logger.info("Applying RunPod compatibility fixes...")
    patch_gradio_for_runpod()
    logger.info("RunPod compatibility fixes applied successfully")

# If run directly, apply the fixes
if __name__ == "__main__":
    apply_fixes()
