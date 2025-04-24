"""
Initialization module for framepack.config.modes package.
"""

from framepack.config.modes.video_mode_settings import (
    MODE_TYPE_NORMAL,
    MODE_TYPE_LOOP,
    get_video_modes,
    get_video_seconds,
    get_important_keyframes,
    get_copy_targets,
    get_max_keyframes_count,
    get_total_sections,
    generate_keyframe_guide_html
)

from framepack.config.modes.keyframe_handler import (
    unified_keyframe_change_handler,
    unified_mode_length_change_handler,
    unified_input_image_change_handler,
    print_keyframe_debug_info
)

__all__ = [
    'MODE_TYPE_NORMAL',
    'MODE_TYPE_LOOP',
    'get_video_modes',
    'get_video_seconds',
    'get_important_keyframes',
    'get_copy_targets',
    'get_max_keyframes_count',
    'get_total_sections',
    'generate_keyframe_guide_html',
    'unified_keyframe_change_handler',
    'unified_mode_length_change_handler',
    'unified_input_image_change_handler',
    'print_keyframe_debug_info'
]
