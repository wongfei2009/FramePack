import torch
import gc
import time

def aggressive_memory_cleanup():
    """More aggressive memory cleanup between processing steps"""
    # First round - standard PyTorch cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Second round - trigger Python garbage collection
    gc.collect()
    
    # Final cleanup
    torch.cuda.empty_cache()
    
    # Return current memory stats
    if torch.cuda.is_available() and hasattr(torch.cuda, 'memory_stats'):
        stats = torch.cuda.memory_stats()
        reserved = stats["reserved_bytes.all.current"] / (1024**3)
        allocated = stats["allocated_bytes.all.current"] / (1024**3)
        return {
            "reserved_gb": reserved,
            "allocated_gb": allocated,
            "free_gb": reserved - allocated
        }
    return None

def configure_teacache(transformer, vram_gb, steps=25, rel_l1_thresh=None):
    """Configure TeaCache settings based on available VRAM, model parameters, and step count
    
    This enhanced version optimizes the rel_l1_thresh parameter based on multiple factors:
    1. Available VRAM - higher VRAM allows more cache entries
    2. Step count - higher step counts benefit from more aggressive caching
    3. Model precision - different data types have different memory requirements
    
    Returns the transformer with optimized TeaCache settings.
    """
    if not hasattr(transformer, 'initialize_teacache'):
        print("Model doesn't support TeaCache")
        return transformer
    
    # Base TeaCache parameters
    teacache_params = {
        'enable_teacache': True,
        'num_steps': steps
    }
    
    # Determine model precision for better parameter tuning
    precision = 'unknown'
    if hasattr(transformer, 'dtype'):
        if transformer.dtype == torch.float32:
            precision = 'float32'
        elif transformer.dtype == torch.bfloat16:
            precision = 'bfloat16'
        elif transformer.dtype == torch.float16:
            precision = 'float16'
    
    # Calculate optimal threshold based on step count
    # Fewer steps benefit from more aggressive caching (lower threshold)
    step_factor = max(0.5, min(1.0, steps / 30))
    
    # Adjust cache_size_multiplier if supported
    if hasattr(transformer, 'initialize_teacache') and 'cache_size_multiplier' in transformer.initialize_teacache.__code__.co_varnames:
        # Calculate optimal cache size based on VRAM and precision
        # Higher precision needs more memory per cache entry
        if precision == 'float32':
            cache_multiplier = min(vram_gb * 0.15, 2.5)  # More conservative for float32
        elif precision == 'bfloat16':
            cache_multiplier = min(vram_gb * 0.20, 3.0)  # Balanced for bfloat16
        else:  # float16 or unknown
            cache_multiplier = min(vram_gb * 0.25, 3.5)  # More aggressive for float16
            
        teacache_params['cache_size_multiplier'] = cache_multiplier
        print(f"Setting TeaCache size multiplier to {cache_multiplier:.2f}")
    
    # Dynamically adjust rel_l1_thresh based on multiple factors
    if hasattr(transformer, 'rel_l1_thresh') or (hasattr(transformer, 'initialize_teacache') and 'rel_l1_thresh' in transformer.initialize_teacache.__code__.co_varnames):
        # Always use the provided threshold from UI
        final_thresh = float(rel_l1_thresh)
        
        # Ensure the threshold is within safe limits
        final_thresh = max(0.08, min(0.25, final_thresh))
        print(f"Using TeaCache threshold: {final_thresh:.4f}")
        
        teacache_params['rel_l1_thresh'] = final_thresh
    
    print(f"TeaCache configuration: {teacache_params}")
    transformer.initialize_teacache(**teacache_params)
    
    return transformer

def smart_batch_size(height, width, available_vram_gb):
    """Calculate optimal batch size based on image dimensions and available VRAM"""
    # Rough estimation based on typical memory usage patterns
    pixels = height * width
    if pixels > 1024*1024:  # 1M+ pixels (higher than HD)
        base_multiplier = 0.5
    elif pixels > 512*512:  # HD-range
        base_multiplier = 0.75
    else:  # SD or lower
        base_multiplier = 1.0
        
    # Calculate based on VRAM
    if available_vram_gb > 20:  # High-end cards
        return min(4, max(1, int(available_vram_gb * 0.15 * base_multiplier)))
    elif available_vram_gb > 10:  # Mid-range cards
        return min(2, max(1, int(available_vram_gb * 0.1 * base_multiplier)))
    else:  # Low VRAM
        return 1

def optimize_for_inference(transformer, high_vram=False, enable_compile=False):
    """Apply various optimizations for inference"""
    # Apply torch.compile optimization if enabled, regardless of VRAM size
    if enable_compile and hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
        try:
            # Only compile the most intensive operations
            if hasattr(transformer, 'model') and hasattr(transformer.model, 'norm'):
                print("Applying PyTorch 2.0+ compile optimizations...")
                # Store original forward function
                if not hasattr(transformer, '_original_norm_forward'):
                    transformer._original_norm_forward = transformer.model.norm.forward
                
                # Apply torch compile to the normalization layers
                transformer.model.norm.forward = torch.compile(
                    transformer.model.norm.forward,
                    mode="reduce-overhead", 
                    fullgraph=False
                )
                print("Applied torch.compile optimization to normalization layers")
        except Exception as e:
            print(f"Failed to apply torch.compile: {e}")
            # Restore original if needed
            if hasattr(transformer, '_original_norm_forward'):
                transformer.model.norm.forward = transformer._original_norm_forward
    
    # Apply VRAM-dependent optimizations only for high VRAM systems
    if high_vram:
        # Set maximum batch sizes for attention operations
        if hasattr(transformer, 'set_attention_optimization'):
            transformer.set_attention_optimization(True)
    
    return transformer
