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
    
    # Enable TF32 for faster matmul operations throughout the model - do this once globally
    if torch.cuda.is_available() and hasattr(torch.backends, 'cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True
        print("Enabled TF32 for matrix multiplications")
    
    # Check if BFloat16 is supported - do this once globally
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if not bf16_supported:
        print("BFloat16 is not supported on this device")
    
    # We apply optimizations regardless of the enable_optimization flag
    # since these are specifically chosen to complement Sage Attention
    try:
        print("Applying Sage Attention-compatible optimizations...")
        
        # Function to safely convert a module to BFloat16
        def safe_to_bf16(module, name):
            if not bf16_supported:
                return module
                
            try:
                if hasattr(module, 'to') and callable(module.to):
                    return module.to(torch.bfloat16)
                return module
            except Exception as e:
                print(f"  - Could not convert {name} to BFloat16: {e}")
                return module
        
        # Function to optimize projection matrices and other components
        def optimize_module(obj):
            obj_name = type(obj).__name__
            
            # 1. Optimize attention projections
            if hasattr(obj, 'attn'):
                attn = obj.attn
                # Handle the case where attn might be None in some models
                if attn is not None and hasattr(attn, 'to_q') and attn.to_q is not None:
                    print(f"Optimizing attention projections for {obj_name}")
                    
                    if bf16_supported:
                        # Main attention projections
                        if hasattr(attn, 'to_q') and attn.to_q is not None:
                            attn.to_q = safe_to_bf16(attn.to_q, 'to_q')
                            
                        if hasattr(attn, 'to_k') and attn.to_k is not None:
                            attn.to_k = safe_to_bf16(attn.to_k, 'to_k')
                            
                        if hasattr(attn, 'to_v') and attn.to_v is not None:
                            attn.to_v = safe_to_bf16(attn.to_v, 'to_v')
                        
                        # Output projections
                        if hasattr(attn, 'to_out') and isinstance(attn.to_out, list):
                            for i, layer in enumerate(attn.to_out):
                                attn.to_out[i] = safe_to_bf16(layer, f'to_out[{i}]')
                        
                        # Additional projections for cross-attention
                        if hasattr(attn, 'add_q_proj'):
                            attn.add_q_proj = safe_to_bf16(attn.add_q_proj, 'add_q_proj')
                            
                        if hasattr(attn, 'add_k_proj'):
                            attn.add_k_proj = safe_to_bf16(attn.add_k_proj, 'add_k_proj')
                            
                        if hasattr(attn, 'add_v_proj'):
                            attn.add_v_proj = safe_to_bf16(attn.add_v_proj, 'add_v_proj')
                            
                        if hasattr(attn, 'to_add_out'):
                            attn.to_add_out = safe_to_bf16(attn.to_add_out, 'to_add_out')
                        
                        print(f"  - Converted attention projections to BFloat16")
            
            # 2. Optimize FeedForward networks
            if hasattr(obj, 'ff') and obj.ff is not None:
                print(f"Optimizing feed-forward network for {obj_name}")
                
                if bf16_supported:
                    # For standard FeedForward modules
                    if hasattr(obj.ff, 'net') and isinstance(obj.ff.net, torch.nn.Sequential):
                        for i, layer in enumerate(obj.ff.net):
                            obj.ff.net[i] = safe_to_bf16(layer, f'ff.net[{i}]')
                    
                    # For FeedForward modules with direct linear projections
                    if hasattr(obj.ff, 'proj_in'):
                        obj.ff.proj_in = safe_to_bf16(obj.ff.proj_in, 'ff.proj_in')
                        
                    if hasattr(obj.ff, 'proj_out'):
                        obj.ff.proj_out = safe_to_bf16(obj.ff.proj_out, 'ff.proj_out')
                        
                    print(f"  - Converted feed-forward networks to BFloat16")
            
            # 3. Optimize additional projection layers in SingleTransformerBlock
            if hasattr(obj, 'proj_mlp'):
                print(f"Optimizing proj_mlp for {obj_name}")
                
                if bf16_supported:
                    obj.proj_mlp = safe_to_bf16(obj.proj_mlp, 'proj_mlp')
                    print(f"  - Converted proj_mlp to BFloat16")
            
            if hasattr(obj, 'proj_out'):
                print(f"Optimizing proj_out for {obj_name}")
                
                if bf16_supported:
                    obj.proj_out = safe_to_bf16(obj.proj_out, 'proj_out')
                    print(f"  - Converted proj_out to BFloat16")
            
            # 4. Special handling for ClipVisionProjection
            if "ClipVisionProjection" in obj_name:
                print(f"Optimizing {obj_name}")
                
                if bf16_supported:
                    if hasattr(obj, 'up'):
                        obj.up = safe_to_bf16(obj.up, 'up')
                    
                    if hasattr(obj, 'down'):
                        obj.down = safe_to_bf16(obj.down, 'down')
                        
                    print(f"  - Converted ClipVisionProjection layers to BFloat16")
            
            # 5. Special handling for Embedding processors
            if "CombinedTimestep" in obj_name:
                print(f"Optimizing {obj_name}")
                
                if bf16_supported:
                    # Convert timestep embedders
                    if hasattr(obj, 'timestep_embedder'):
                        obj.timestep_embedder = safe_to_bf16(obj.timestep_embedder, 'timestep_embedder')
                    
                    # Convert guidance embedder if present
                    if hasattr(obj, 'guidance_embedder'):
                        obj.guidance_embedder = safe_to_bf16(obj.guidance_embedder, 'guidance_embedder')
                    
                    # Convert text embedder
                    if hasattr(obj, 'text_embedder'):
                        obj.text_embedder = safe_to_bf16(obj.text_embedder, 'text_embedder')
                        
                    print(f"  - Converted embedding components to BFloat16")
            
            # Apply recursively to any child modules
            for name, child in obj.named_children():
                optimize_module(child)
        
        # Apply optimization to transformer blocks
        if hasattr(transformer, 'transformer_blocks'):
            print(f"Optimizing {len(transformer.transformer_blocks)} transformer blocks")
            for block in transformer.transformer_blocks:
                optimize_module(block)
        
        if hasattr(transformer, 'single_transformer_blocks'):
            print(f"Optimizing {len(transformer.single_transformer_blocks)} single transformer blocks")
            for block in transformer.single_transformer_blocks:
                optimize_module(block)
        
        # Optimize embedding components
        if hasattr(transformer, 'time_text_embed'):
            print(f"Optimizing time_text_embed")
            optimize_module(transformer.time_text_embed)
        
        if hasattr(transformer, 'context_embedder'):
            print(f"Optimizing context_embedder")
            optimize_module(transformer.context_embedder)
        
        if hasattr(transformer, 'x_embedder'):
            print(f"Optimizing x_embedder")
            optimize_module(transformer.x_embedder)
        
        if hasattr(transformer, 'clean_x_embedder') and transformer.clean_x_embedder is not None:
            print(f"Optimizing clean_x_embedder")
            optimize_module(transformer.clean_x_embedder)
        
        if hasattr(transformer, 'image_projection') and transformer.image_projection is not None:
            print(f"Optimizing image_projection")
            optimize_module(transformer.image_projection)
        
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
        print("PyTorch optimization enabled")
        
    except Exception as e:
        print(f"Failed to apply optimizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Apply VRAM-dependent optimizations only for high VRAM systems
    if high_vram and hasattr(transformer, 'set_attention_optimization'):
        transformer.set_attention_optimization(True)
        print("Applied high VRAM optimizations")
    
    return transformer