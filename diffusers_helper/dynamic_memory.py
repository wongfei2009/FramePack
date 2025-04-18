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

def optimize_model_sequence(models, operations, target_device, preserved_memory_gb=2):
    """Intelligently manage models for a sequence of operations
    
    Args:
        models: Dictionary of models {name: model_object}
        operations: List of operations, each containing the models needed
        target_device: Target device to move models to
        preserved_memory_gb: GB of memory to preserve
    """
    results = {}
    current_models_on_device = set()
    
    for op_name, required_models in operations:
        print(f"Executing operation: {op_name}")
        
        # Models to load
        to_load = set(required_models) - current_models_on_device
        
        # Models to unload
        to_unload = current_models_on_device - set(required_models)
        
        # Unload unnecessary models
        for model_name in to_unload:
            print(f"  Unloading {model_name}")
            models[model_name].to('cpu')
            current_models_on_device.remove(model_name)
            aggressive_memory_cleanup()
        
        # Load required models
        for model_name in to_load:
            print(f"  Loading {model_name}")
            models[model_name].to(target_device)
            current_models_on_device.add(model_name)
        
        # Allow time for operation
        time.sleep(0.01)  # Small pause for device synchronization
        
    return results

def configure_optimal_teacache(transformer, vram_gb):
    """Configure TeaCache settings based on available VRAM"""
    if not hasattr(transformer, 'initialize_teacache'):
        print("Model doesn't support TeaCache")
        return transformer
        
    # Base settings
    teacache_params = {
        'enable_teacache': True
    }
    
    # Adjust cache size based on available VRAM
    if vram_gb > 20:  # High-end cards (RTX 3090, 4090, etc)
        teacache_params['cache_size_multiplier'] = 1.0  # Full cache
    elif vram_gb > 12:  # Mid-range cards (RTX 3080, etc)
        teacache_params['cache_size_multiplier'] = 0.75  # 75% cache
    elif vram_gb > 8:  # Lower-end cards (RTX 3060, etc)
        teacache_params['cache_size_multiplier'] = 0.5  # 50% cache
    else:  # Very limited VRAM
        teacache_params['cache_size_multiplier'] = 0.3  # Minimal cache
        
    print(f"Configuring TeaCache with: {teacache_params}")
    transformer.initialize_teacache(**teacache_params)
    
    return transformer
