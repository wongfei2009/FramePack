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

def configure_teacache(transformer, vram_gb, steps=25):
    """Configure TeaCache settings based on available VRAM"""
    if not hasattr(transformer, 'initialize_teacache'):
        print("Model doesn't support TeaCache")
        return transformer
    
    # The model only supports enable_teacache and num_steps parameters
    # Adjusting the threshold based on available VRAM instead of using cache_size_multiplier
    teacache_params = {
        'enable_teacache': True,
        'num_steps': steps
    }
    
    # Adjust rel_l1_thresh based on available VRAM
    # Lower threshold = more cache hits = faster but potentially lower quality
    # Higher threshold = fewer cache hits = slower but higher quality
    if hasattr(transformer, 'rel_l1_thresh'):
        if vram_gb > 20:  # High-end cards (RTX 3090, 4090, etc)
            print(f"Using optimized TeaCache for high VRAM ({vram_gb:.1f} GB)")
            teacache_params['rel_l1_thresh'] = 0.20  # More quality focused
        elif vram_gb > 12:  # Mid-range cards (RTX 3080, etc)
            print(f"Using optimized TeaCache for medium VRAM ({vram_gb:.1f} GB)")
            teacache_params['rel_l1_thresh'] = 0.15  # Balanced
        elif vram_gb > 8:  # Lower-end cards (RTX 3060, etc)
            print(f"Using optimized TeaCache for low VRAM ({vram_gb:.1f} GB)")
            teacache_params['rel_l1_thresh'] = 0.12  # More speed focused
        else:  # Very limited VRAM
            print(f"Using optimized TeaCache for very low VRAM ({vram_gb:.1f} GB)")
            teacache_params['rel_l1_thresh'] = 0.10  # Maximum speed
    
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

def optimize_for_inference(transformer, high_vram=False):
    """Apply various optimizations for inference"""
    # Skip optimizations for low VRAM mode to avoid unexpected behavior
    if not high_vram:
        return transformer
    
    # Try to apply torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, 'compile') and torch.__version__ >= '2.0.0':
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
    
    # Set maximum batch sizes for attention operations
    if hasattr(transformer, 'set_attention_optimization'):
        transformer.set_attention_optimization(True)
    
    return transformer

def warmup_model(transformer, device, dtype=torch.bfloat16):
    """Perform a small forward pass to warm up the model"""
    print("Warming up model with a test forward pass...")
    try:
        # Create small dummy inputs for warmup
        dummy_latents = torch.zeros((1, 16, 4, 32, 32), device=device, dtype=dtype)
        dummy_timestep = torch.ones((1,), device=device, dtype=torch.float32)
        
        # Dummy encoder inputs (needed for the model's forward function)
        dummy_encoder_hidden_states = torch.zeros((1, 16, 4096), device=device, dtype=dtype)
        dummy_encoder_attention_mask = torch.ones((1, 16), device=device, dtype=torch.bool)
        dummy_pooled_projections = torch.zeros((1, 768), device=device, dtype=dtype)
        dummy_guidance = torch.zeros((1,), device=device, dtype=torch.float32)
        
        # Warm up with a single forward pass
        with torch.no_grad():
            _ = transformer(
                hidden_states=dummy_latents, 
                timestep=dummy_timestep,
                encoder_hidden_states=dummy_encoder_hidden_states,
                encoder_attention_mask=dummy_encoder_attention_mask,
                pooled_projections=dummy_pooled_projections,
                guidance=dummy_guidance
            )
            
        print("Model warmup completed successfully")
        return True
    except Exception as e:
        print(f"Model warmup failed: {e}")
        return False
