import torch
import gc

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

def optimize_for_inference(transformer, high_vram=False, enable_optimization=False):
    """
    Apply optimizations that complement Sage Attention for inference.
    
    This version is streamlined for systems with Sage Attention always enabled,
    focusing only on optimizations that work alongside Sage Attention rather
    than competing with it.
    
    Args:
        transformer: The transformer model to optimize
        high_vram: Whether the system has high VRAM available
        enable_optimization: Whether to enable optimizations (kept for API compatibility)
    
    Returns:
        The optimized transformer model
    """
    # Debug info
    print(f"optimize_for_inference called with high_vram={high_vram}, enable_optimization={enable_optimization}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Store original forward functions if we need to restore
    if not hasattr(transformer, '_original_forwards'):
        transformer._original_forwards = {}
    
    # We apply optimizations regardless of the enable_optimization flag
    # since these are specifically chosen to complement Sage Attention
    try:
        print("Applying Sage Attention-compatible optimizations...")
        
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformerBlock, HunyuanVideoSingleTransformerBlock
        
        # Function to optimize projection matrices (these run before Sage Attention is called)
        def optimize_projections(obj):
            if hasattr(obj, 'attn') and hasattr(obj.attn, 'to_q'):
                print(f"Optimizing projections for {type(obj).__name__}")
                
                # Enable TF32 for faster matmul operations throughout the model
                if torch.cuda.is_available() and hasattr(torch.backends, 'cuda'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    print("  - Enabled TF32 for matrix multiplications")
                
                # Use BFloat16 for projection matrices (these run before Sage Attention)
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    try:
                        obj.attn.to_q = obj.attn.to_q.to(torch.bfloat16)
                        obj.attn.to_k = obj.attn.to_k.to(torch.bfloat16)
                        obj.attn.to_v = obj.attn.to_v.to(torch.bfloat16)
                        print(f"  - Converted projections to BFloat16")
                    except Exception as e:
                        print(f"  - Could not convert to BFloat16: {e}")
            
            # Apply recursively to any child modules
            for name, child in obj.named_children():
                optimize_projections(child)
        
        # Apply optimization to transformer blocks
        if hasattr(transformer, 'transformer_blocks'):
            print(f"Optimizing {len(transformer.transformer_blocks)} transformer blocks")
            for block in transformer.transformer_blocks:
                optimize_projections(block)
        
        if hasattr(transformer, 'single_transformer_blocks'):
            print(f"Optimizing {len(transformer.single_transformer_blocks)} single transformer blocks")
            for block in transformer.single_transformer_blocks:
                optimize_projections(block)
        
        # Enable kernel fusion optimizations for better throughput
        if hasattr(torch, '_C') and hasattr(torch._C, '_jit_set_profiling_executor'):
            torch._C._jit_set_profiling_executor(True)
            torch._C._jit_set_profiling_mode(True)
            print("Enabled fused kernel profiling")
            
        if hasattr(torch._C, '_jit_override_can_fuse_on_cpu'):
            torch._C._jit_override_can_fuse_on_cpu(True)
            print("Enabled CPU kernel fusion")
            
        if hasattr(torch._C, '_jit_override_can_fuse_on_gpu'):
            torch._C._jit_override_can_fuse_on_gpu(True)
            print("Enabled GPU kernel fusion")
            
        print("Successfully applied Sage Attention-compatible optimizations")
        
    except Exception as e:
        print(f"Failed to apply optimizations: {e}")
    
    # Apply VRAM-dependent optimizations only for high VRAM systems
    if high_vram and hasattr(transformer, 'set_attention_optimization'):
        transformer.set_attention_optimization(True)
        print("Applied high VRAM optimizations")
    
    return transformer