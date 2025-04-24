"""
Video mode settings module for FramePack.

Defines the available video modes, durations, and keyframe configurations.
"""

import logging

# Set up logging
logger = logging.getLogger(__name__)

# Define mode types
MODE_TYPE_NORMAL = "Normal Mode"
MODE_TYPE_LOOP = "Loop Mode"

# Define video mode settings
VIDEO_MODE_SETTINGS = {
    "5 seconds": {
        "seconds": 5.0,
        "keyframes": {
            MODE_TYPE_NORMAL: [
                {"position": 0, "copy_to": []}
            ],
            MODE_TYPE_LOOP: [
                {"position": 0, "copy_to": [1, 2]}
            ]
        }
    },
    "10 seconds": {
        "seconds": 10.0,
        "keyframes": {
            MODE_TYPE_NORMAL: [
                {"position": 0, "copy_to": []},
                {"position": 1, "copy_to": []}
            ],
            MODE_TYPE_LOOP: [
                {"position": 0, "copy_to": [1, 3]},
                {"position": 2, "copy_to": []}
            ]
        }
    },
    "15 seconds": {
        "seconds": 15.0,
        "keyframes": {
            MODE_TYPE_NORMAL: [
                {"position": 0, "copy_to": []},
                {"position": 1, "copy_to": []},
                {"position": 2, "copy_to": []}
            ],
            MODE_TYPE_LOOP: [
                {"position": 0, "copy_to": [1, 4]},
                {"position": 2, "copy_to": []},
                {"position": 3, "copy_to": []}
            ]
        }
    },
    "30 seconds": {
        "seconds": 30.0,
        "keyframes": {
            MODE_TYPE_NORMAL: [
                {"position": 0, "copy_to": []},
                {"position": 1, "copy_to": []},
                {"position": 2, "copy_to": []},
                {"position": 3, "copy_to": []}
            ],
            MODE_TYPE_LOOP: [
                {"position": 0, "copy_to": [1, 5]},
                {"position": 2, "copy_to": []},
                {"position": 3, "copy_to": []},
                {"position": 4, "copy_to": []}
            ]
        }
    },
    "60 seconds": {
        "seconds": 60.0,
        "keyframes": {
            MODE_TYPE_NORMAL: [
                {"position": 0, "copy_to": []},
                {"position": 2, "copy_to": []},
                {"position": 4, "copy_to": []},
                {"position": 6, "copy_to": []}
            ],
            MODE_TYPE_LOOP: [
                {"position": 0, "copy_to": [1, 7]},
                {"position": 2, "copy_to": []},
                {"position": 4, "copy_to": []},
                {"position": 6, "copy_to": []}
            ]
        }
    }
}

def get_video_modes():
    """
    Get the list of available video mode names.
    
    Returns:
        List of video mode names
    """
    return list(VIDEO_MODE_SETTINGS.keys())

def get_video_seconds(mode_name):
    """
    Get the seconds value for a given mode name.
    
    Args:
        mode_name: Name of the video mode
        
    Returns:
        Number of seconds for the mode, or default value if not found
    """
    if mode_name in VIDEO_MODE_SETTINGS:
        return VIDEO_MODE_SETTINGS[mode_name]["seconds"]
    # Default fallback
    logger.warning(f"Video mode '{mode_name}' not found. Using default value.")
    return 5.0

def get_important_keyframes(mode, length):
    """
    Get the list of important keyframe positions for a given mode and length.
    
    Args:
        mode: Mode type (normal or loop)
        length: Video length mode name
        
    Returns:
        List of keyframe positions
    """
    mode_settings = VIDEO_MODE_SETTINGS.get(length, {}).get("keyframes", {})
    mode_config = mode_settings.get(mode, [])
    
    # Extract positions
    positions = [config["position"] for config in mode_config]
    return positions

def get_copy_targets(mode, length, src_idx):
    """
    Get the list of keyframe indices that should copy from the source keyframe.
    
    Args:
        mode: Mode type (normal or loop)
        length: Video length mode name
        src_idx: Source keyframe index
        
    Returns:
        List of target keyframe indices
    """
    mode_settings = VIDEO_MODE_SETTINGS.get(length, {}).get("keyframes", {})
    mode_config = mode_settings.get(mode, [])
    
    for config in mode_config:
        if config["position"] == src_idx:
            return config.get("copy_to", [])
    return []

def get_max_keyframes_count():
    """
    Get the maximum number of keyframes across all modes.
    
    Returns:
        Maximum number of keyframes
    """
    max_count = 0
    for mode_data in VIDEO_MODE_SETTINGS.values():
        keyframes = mode_data.get("keyframes", {})
        for mode_type in keyframes.values():
            count = len(mode_type)
            max_count = max(max_count, count)
    return max_count

def get_total_sections(mode, length):
    """
    Get the total number of sections for a given mode and length.
    
    Args:
        mode: Mode type (normal or loop)
        length: Video length mode name
        
    Returns:
        Total number of sections
    """
    # For normal mode, just return the number of seconds divided by 5
    if mode == MODE_TYPE_NORMAL:
        seconds = get_video_seconds(length)
        return max(1, int(seconds / 5))
    
    # For loop mode, add 1 more section to ensure proper looping
    if mode == MODE_TYPE_LOOP:
        seconds = get_video_seconds(length)
        return max(2, int(seconds / 5) + 1)
    
    # Default fallback
    seconds = get_video_seconds(length)
    return max(1, int(seconds / 5))

def generate_keyframe_guide_html():
    """
    Generate HTML guide for keyframe usage in different modes.
    
    Returns:
        HTML string explaining keyframe usage
    """
    html = "<div style='background-color: #f5f5f5; padding: 10px; border-radius: 5px;'>"
    html += "<h4>Keyframe Guide</h4>"
    html += "<p><strong>Normal Mode:</strong> Use keyframes to define key points in your video.</p>"
    html += "<p><strong>Loop Mode:</strong> First and last keyframes are automatically connected to create a seamless loop.</p>"
    html += "<ul>"
    html += "<li>Keyframe 0 is always the starting frame</li>"
    html += "<li>In Loop Mode, the end connects back to the beginning</li>"
    html += "</ul>"
    html += "</div>"
    return html

def ui_to_code_index(ui_index):
    """
    Convert UI index (1-based) to code index (0-based).
    
    Args:
        ui_index: UI index (1-based)
        
    Returns:
        Code index (0-based)
    """
    return ui_index - 1 if ui_index is not None else None

def code_to_ui_index(code_index):
    """
    Convert code index (0-based) to UI index (1-based).
    
    Args:
        code_index: Code index (0-based)
        
    Returns:
        UI index (1-based)
    """
    return code_index + 1 if code_index is not None else None

def handle_mode_length_change(mode, length, section_number_inputs):
    """
    Handle changes when the mode or length is changed.
    
    Args:
        mode: Selected mode type
        length: Selected length mode
        section_number_inputs: List of section number input components
        
    Returns:
        Updates for the section number inputs
    """
    important_positions = get_important_keyframes(mode, length)
    
    # Update section numbers
    updates = []
    for i, input_component in enumerate(section_number_inputs):
        if i < len(important_positions):
            updates.append(code_to_ui_index(important_positions[i]))
        else:
            updates.append(None)  # Clear other inputs
    
    return updates

def process_keyframe_change(src_idx, img, mode, length, enable_copy):
    """
    Process a keyframe change and determine which other keyframes should be updated.
    
    Args:
        src_idx: Source keyframe index
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
