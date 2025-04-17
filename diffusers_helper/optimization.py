"""
Optimization utilities for FramePack
"""
import torch
import numpy as np
from diffusers_helper.memory import get_cuda_free_memory_gb


def configure_teacache(transformer, vram_gb=None, steps=25):
    """
    Configure TeaCache settings based on available VRAM for optimal performance
    
    Args:
        transformer: The transformer model to configure
        vram_gb: Available VRAM in GB (if None, will be auto-detected)
        steps: Number of diffusion steps
        
    Returns:
        transformer: The configured transformer
    """
    if vram_gb is None:
        vram_gb = get_cuda_free_memory_gb()
        
    print(f"Configuring TeaCache with {vram_gb:.2f}GB VRAM available")
    
    # Adaptive TeaCache threshold based on available VRAM
    if vram_gb > 20:
        # High memory mode - maximize quality
        rel_l1_thresh = 0.1  # Lower threshold = more frequent calculation = better quality
        print("High VRAM mode: Using quality-focused TeaCache settings")
    elif vram_gb > 10:
        # Medium memory mode - balanced
        rel_l1_thresh = 0.15  # Default value
        print("Medium VRAM mode: Using balanced TeaCache settings")
    else:
        # Low memory mode - maximize speed
        rel_l1_thresh = 0.2  # Higher threshold = less frequent calculation = faster
        print("Low VRAM mode: Using speed-focused TeaCache settings")
    
    # Configure the TeaCache
    transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=rel_l1_thresh)
    
    return transformer


def optimize_for_inference(transformer, high_vram=False):
    """
    Apply various optimizations to the transformer model for faster inference
    
    Args:
        transformer: The transformer model to optimize
        high_vram: Whether running in high VRAM mode
        
    Returns:
        transformer: The optimized transformer
    """
    # Always use fp32 for output for better quality
    transformer.high_quality_fp32_output_for_inference = True
    
    # Optimize attention mechanisms
    # Already handled via the SageAttention configuration
    
    # Disable gradient checkpointing if not needed
    if high_vram:
        transformer.disable_gradient_checkpointing()
    else:
        transformer.enable_gradient_checkpointing()
    
    # Set the global inference options
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
    
    return transformer
