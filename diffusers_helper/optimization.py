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
    """Apply various optimizations for inference"""
    # Debug info
    print(f"optimize_for_inference called with high_vram={high_vram}, enable_optimization={enable_optimization}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torch.compile available: {hasattr(torch, 'compile')}")
    
    # Store original forward functions if we need to restore
    if not hasattr(transformer, '_original_forwards'):
        transformer._original_forwards = {}
    
    # Apply optimizations if enabled, regardless of VRAM size
    if enable_optimization and torch.__version__ >= '2.0.0':
        try:
            print("Applying transformer inference optimizations...")
            
            # Apply BFloat16 conversion for specific classes if they exist
            from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformerBlock, HunyuanVideoSingleTransformerBlock
            
            # Function to apply efficient attention operations
            def optimize_attention(obj):
                if hasattr(obj, 'attn') and hasattr(obj.attn, 'to_q'):
                    print(f"Optimizing attention for {type(obj).__name__}")
                    # Force efficient attention operations
                    if torch.cuda.is_available() and hasattr(torch.backends, 'cuda'):
                        torch.backends.cuda.matmul.allow_tf32 = True
                        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
                            torch.backends.cuda.enable_flash_sdp(True)
                            print("  - Enabled Flash Attention")
                        if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                            torch.backends.cuda.enable_mem_efficient_sdp(True)
                            print("  - Enabled Memory-Efficient Attention")
                        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
                            torch.backends.cuda.enable_math_sdp(True)
                            print("  - Enabled Math Attention")
                    
                    # Use BFloat16 for attention computation
                    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                        try:
                            obj.attn.to_q = obj.attn.to_q.to(torch.bfloat16)
                            obj.attn.to_k = obj.attn.to_k.to(torch.bfloat16)
                            obj.attn.to_v = obj.attn.to_v.to(torch.bfloat16)
                            print(f"  - Converted attention projections to BFloat16")
                        except Exception as e:
                            print(f"  - Could not convert to BFloat16: {e}")
                
                # Apply recursively to any modules
                for name, child in obj.named_children():
                    optimize_attention(child)
            
            # Apply optimization to transformer blocks
            if hasattr(transformer, 'transformer_blocks'):
                print(f"Optimizing {len(transformer.transformer_blocks)} transformer blocks")
                for block in transformer.transformer_blocks:
                    optimize_attention(block)
            
            if hasattr(transformer, 'single_transformer_blocks'):
                print(f"Optimizing {len(transformer.single_transformer_blocks)} single transformer blocks")
                for block in transformer.single_transformer_blocks:
                    optimize_attention(block)
            
            # Enable fused kernels for attention operations if available
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
                
            # Removed deprecated nvFuser-related code
                
            print("Successfully applied alternative optimization techniques")
            
        except Exception as e:
            print(f"Failed to apply optimizations: {e}")
    
    # Apply VRAM-dependent optimizations only for high VRAM systems
    if high_vram:
        # Set maximum batch sizes for attention operations
        if hasattr(transformer, 'set_attention_optimization'):
            transformer.set_attention_optimization(True)
    
    return transformer