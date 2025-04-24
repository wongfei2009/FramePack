"""
Keyframe handler module for FramePack.

Provides functions for managing keyframes and their relationships based on the selected mode.
"""

import logging
import numpy as np
from framepack.config.modes.video_mode_settings import (
    get_copy_targets, get_important_keyframes, get_video_seconds,
    ui_to_code_index, code_to_ui_index
)

# Set up logging
logger = logging.getLogger(__name__)

def print_keyframe_debug_info():
    """Print debug information about keyframes."""
    from framepack.config.modes.video_mode_settings import (
        VIDEO_MODE_SETTINGS, MODE_TYPE_NORMAL, MODE_TYPE_LOOP
    )
    
    logger.debug("Keyframe Debug Information:")
    for length_name, length_data in VIDEO_MODE_SETTINGS.items():
        logger.debug(f"Length: {length_name} ({length_data['seconds']}s)")
        
        keyframes = length_data.get("keyframes", {})
        
        # Normal mode
        normal_config = keyframes.get(MODE_TYPE_NORMAL, [])
        logger.debug(f"  Normal Mode: {len(normal_config)} keyframes")
        for i, config in enumerate(normal_config):
            pos = config["position"]
            copy_to = config.get("copy_to", [])
            logger.debug(f"    Keyframe {i}: position {pos}, copy to {copy_to}")
        
        # Loop mode
        loop_config = keyframes.get(MODE_TYPE_LOOP, [])
        logger.debug(f"  Loop Mode: {len(loop_config)} keyframes")
        for i, config in enumerate(loop_config):
            pos = config["position"]
            copy_to = config.get("copy_to", [])
            logger.debug(f"    Keyframe {i}: position {pos}, copy to {copy_to}")

def unified_keyframe_change_handler(src_ui_index, img, mode, length, enable_copy):
    """
    Handle a keyframe change and update dependent keyframes.
    
    Args:
        src_ui_index: Source keyframe UI index (1-based)
        img: Image data
        mode: Selected mode type
        length: Selected length mode
        enable_copy: Whether automatic keyframe copying is enabled
        
    Returns:
        List of image updates for dependent keyframes
    """
    # Convert UI index to code index (0-based)
    src_code_index = ui_to_code_index(src_ui_index)
    
    if src_code_index is None:
        return []
    
    # Get updates
    updates_dict = process_keyframe_change(src_code_index, img, mode, length, enable_copy)
    
    # Convert updates to list
    max_keyframes = 10  # Maximum number of keyframes
    updates = [None] * max_keyframes
    
    for target_idx, target_img in updates_dict.items():
        if 0 <= target_idx < max_keyframes:
            updates[target_idx] = target_img
    
    return updates

def unified_mode_length_change_handler(mode, length, section_number_inputs):
    """
    Handle changes when the mode or length is changed.
    
    Args:
        mode: Selected mode type
        length: Selected length mode
        section_number_inputs: List of section number input components
        
    Returns:
        Updates for input image, end frame, and section images
    """
    from framepack.config.modes.video_mode_settings import MODE_TYPE_LOOP
    
    # Get the number of seconds for the selected length
    seconds = get_video_seconds(length)
    
    # Get important positions for keyframes
    important_positions = get_important_keyframes(mode, length)
    logger.debug(f"Important positions for {mode}, {length}: {important_positions}")
    
    # For loop mode, set end frame to match the start frame (input image)
    use_same_end_frame = (mode == MODE_TYPE_LOOP)
    
    # Create empty updates for all section images
    section_updates = [None] * 10  # Assuming maximum of 10 sections
    
    # Return updates for input image, end frame, and section images
    return None, None, *section_updates, seconds

def unified_input_image_change_handler(input_image, mode, length, enable_keyframe_copy):
    """
    Handle input image changes and update dependent keyframes.
    
    Args:
        input_image: The input image
        mode: Selected mode type
        length: Selected length mode
        enable_keyframe_copy: Whether automatic keyframe copying is enabled
        
    Returns:
        Updates for end frame and section images
    """
    from framepack.config.modes.video_mode_settings import MODE_TYPE_LOOP
    
    # For loop mode, set end frame to match the start frame (input image)
    end_frame_update = input_image if mode == MODE_TYPE_LOOP else None
    
    # Process keyframe changes
    updates = []
    
    if input_image is not None and enable_keyframe_copy:
        # Apply auto-copy logic for keyframe 0 (input image)
        src_code_index = 0  # Input image is keyframe 0
        updates_dict = {}
        
        copy_targets = get_copy_targets(mode, length, src_code_index)
        for target_idx in copy_targets:
            updates_dict[target_idx] = input_image
        
        # Convert updates to list
        max_keyframes = 10  # Maximum number of keyframes
        updates = [None] * max_keyframes
        
        for target_idx, target_img in updates_dict.items():
            if 0 <= target_idx < max_keyframes:
                updates[target_idx] = target_img
    
    # Return updates for end frame and section images
    return end_frame_update, *updates

def process_keyframe_change(src_idx, img, mode, length, enable_copy):
    """
    Process a keyframe change and determine which other keyframes should be updated.
    
    Args:
        src_idx: Source keyframe index (0-based)
        img: Image data
        mode: Selected mode type
        length: Selected length mode
        enable_copy: Whether automatic keyframe copying is enabled
        
    Returns:
        Dictionary of target indices and image updates
    """
    if img is None or not enable_copy:
        return {}
    
    updates = {}
    if mode and length:
        copy_targets = get_copy_targets(mode, length, src_idx)
        for target_idx in copy_targets:
            updates[target_idx] = img
    
    return updates
