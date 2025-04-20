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



