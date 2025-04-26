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

def optimize_for_inference(transformer, high_vram=False):
    """
    Apply potentially beneficial optimizations for inference, assuming the model
    is already loaded in the desired precision (e.g., BF16).

    Optimizations applied:
    - Enables TF32 for matrix multiplications on compatible hardware.
    - Attempts to enable PyTorch JIT kernel fusion.
    - Applies model-specific high-VRAM optimizations if available.

    Args:
        transformer: The transformer model to optimize.
        high_vram: Boolean indicating if high VRAM is available.    

    Returns:
        The potentially optimized transformer model.
    """
    # Debug info
    print(f"optimize_for_inference called with high_vram={high_vram}")
    print(f"PyTorch version: {torch.__version__}")

    optimizations_applied = []

    # 1. Enable TF32 for faster matmul operations
    # This is a global setting beneficial for Ampere+ GPUs, independent of model dtype.
    if torch.cuda.is_available() and hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'matmul'):
        try:
            # Check current value before setting
            current_tf32 = torch.backends.cuda.matmul.allow_tf32
            if not current_tf32:
                torch.backends.cuda.matmul.allow_tf32 = True
                print("Enabled TF32 for CUDA matrix multiplications.")
                optimizations_applied.append("TF32")
            else:
                 print("TF32 for CUDA matrix multiplications already enabled.")
                 optimizations_applied.append("TF32 (already enabled)")
        except Exception as e:
            print(f"Warning: Could not enable TF32. Error: {e}")

    # 2. Enable kernel fusion optimizations
    # Attempts to fuse small operations into larger kernels to reduce overhead.
    # Uses internal PyTorch JIT flags, which might change across versions.
    try:
        fused_kernels_enabled = False
        # Check if JIT features are available
        if hasattr(torch, '_C'):
            if hasattr(torch._C, '_jit_set_profiling_executor') and hasattr(torch._C, '_jit_set_profiling_mode'):
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
                fused_kernels_enabled = True
                print("Enabled JIT profiling executor and mode for fusion.")
            if hasattr(torch._C, '_jit_override_can_fuse_on_cpu'):
                torch._C._jit_override_can_fuse_on_cpu(True)
                fused_kernels_enabled = True
                print("Enabled CPU kernel fusion override.")
            if hasattr(torch._C, '_jit_override_can_fuse_on_gpu'):
                torch._C._jit_override_can_fuse_on_gpu(True)
                fused_kernels_enabled = True
                print("Enabled GPU kernel fusion override.")

        if fused_kernels_enabled:
            optimizations_applied.append("Kernel Fusion")
        else:
             print("Could not find JIT fusion settings (might be unavailable in this PyTorch version).")

    except Exception as e:
        print(f"Warning: Could not enable kernel fusion settings. Error: {e}")
        # Optionally uncomment traceback for debugging:
        # traceback.print_exc()

    # 3. Apply High VRAM specific optimizations (if the model supports it)
    # This depends on the specific transformer implementation having this method.
    if high_vram and hasattr(transformer, 'set_attention_optimization'):
        try:
            transformer.set_attention_optimization(True)
            print("Applied high VRAM optimization via transformer.set_attention_optimization(True).")
            optimizations_applied.append("High VRAM Attention")
        except Exception as e:
            print(f"Warning: Could not apply high VRAM optimization. Error: {e}")
            # Optionally uncomment traceback for debugging:
            # traceback.print_exc()
    elif high_vram:
         print("High VRAM mode active, but transformer does not have 'set_attention_optimization' method.")

    # Final summary
    if optimizations_applied:
        print(f"Successfully applied optimizations: {', '.join(optimizations_applied)}")
    else:
        print("No specific optimizations were applied or enabled in this call (some might have been pre-enabled).")

    return transformer