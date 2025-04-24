import torch
import time
import numpy as np
from collections import defaultdict

class PerformanceTracker:
    """Track performance metrics across multiple runs"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all tracked metrics"""
        self.timers = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.step_times = []
        self.total_time = 0
        self.start_time = None
        self.cache_stats = []
        
    def start_timer(self, name=None):
        """Start a named timer"""
        if self.start_time is None:
            self.start_time = time.time()
            
        if name is not None:
            setattr(self, f"{name}_start", time.time())
            
    def end_timer(self, name):
        """End a named timer and record the elapsed time"""
        if hasattr(self, f"{name}_start"):
            elapsed = time.time() - getattr(self, f"{name}_start")
            self.timers[name].append(elapsed)
            return elapsed
        return 0
        
    def get_start_time(self):
        """Get the time when tracking started"""
        return self.start_time if self.start_time is not None else time.time()
        
    def track_memory(self, name=None):
        """Track current GPU memory usage"""
        if torch.cuda.is_available():
            current_mem = torch.cuda.memory_allocated() / (1024**3)
            max_mem = torch.cuda.max_memory_allocated() / (1024**3)
            free_mem = torch.cuda.memory_reserved() / (1024**3) - current_mem
            
            if name:
                self.memory_stats[f"{name}_current"].append(current_mem)
                self.memory_stats[f"{name}_max"].append(max_mem)
                self.memory_stats[f"{name}_free"].append(free_mem)
            
            return {
                "current_gb": current_mem,
                "max_gb": max_mem,
                "free_gb": free_mem
            }
        return None
        
    def track_step_time(self, step_time):
        """Record time for a single generation step"""
        self.step_times.append(step_time)
        
    def track_cache_stats(self, hits, misses, total_queries):
        """Track TeaCache performance statistics"""
        self.cache_stats.append({
            'hits': hits,
            'misses': misses,
            'total': total_queries,
            'hit_rate': hits / total_queries if total_queries > 0 else 0
        })
        
        # Print a debug message to confirm we're tracking stats
        print(f"TeaCache stats tracked: +{hits} hits, +{misses} misses, {hits}/{total_queries} ({hits/total_queries*100:.1f}% hit rate)")
        
    def get_summary(self):
        """Generate a performance summary"""
        if self.start_time is not None:
            self.total_time = time.time() - self.start_time
            
        summary = {
            "total_time": self.total_time,
            "timers": {}
        }
        
        # Process timer data
        for name, times in self.timers.items():
            if times:
                summary["timers"][name] = {
                    "mean": np.mean(times),
                    "min": np.min(times),
                    "max": np.max(times),
                    "total": np.sum(times),
                    "count": len(times)
                }
                
        # Process step time data
        if self.step_times:
            summary["steps"] = {
                "mean": np.mean(self.step_times),
                "min": np.min(self.step_times),
                "max": np.max(self.step_times),
                "total": np.sum(self.step_times),
                "count": len(self.step_times),
                "steps_per_sec": 1.0 / np.mean(self.step_times) if np.mean(self.step_times) > 0 else 0
            }
            
        # Process cache statistics
        if self.cache_stats:
            total_hits = sum(stat['hits'] for stat in self.cache_stats)
            total_queries = sum(stat['total'] for stat in self.cache_stats)
            avg_hit_rate = total_hits / total_queries if total_queries > 0 else 0
            
            summary["cache"] = {
                "hits": total_hits,
                "total": total_queries,
                "hit_rate": avg_hit_rate
            }
            
        # Process memory statistics
        if torch.cuda.is_available():
            summary["memory"] = {
                "current_gb": torch.cuda.memory_allocated() / (1024**3),
                "max_gb": torch.cuda.max_memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3)
            }
            
            # Add tracked memory stats
            for name, values in self.memory_stats.items():
                if values:
                    summary["memory"][name] = {
                        "mean": np.mean(values),
                        "max": np.max(values)
                    }
                    
        return summary
        
    def print_summary(self):
        """Print a formatted performance summary"""
        summary = self.get_summary()
        
        print("\n===== PERFORMANCE SUMMARY =====")
        print(f"Total time: {summary['total_time']:.2f} seconds")
        
        if "steps" in summary:
            steps = summary["steps"]
            print(f"\nStep Performance:")
            print(f"  Average: {steps['mean']:.4f} sec/step ({steps['steps_per_sec']:.2f} steps/sec)")
            print(f"  Min/Max: {steps['min']:.4f}/{steps['max']:.4f} sec")
            print(f"  Total steps: {steps['count']}")
            
        print("\nComponent Timings:")
        for name, timer in summary["timers"].items():
            print(f"  {name}: {timer['mean']:.4f} sec avg, {timer['total']:.2f} sec total ({timer['count']} calls)")
            
        # Always print TeaCache statistics
        if "cache" in summary:
            cache = summary["cache"]
            print(f"\nTeaCache Performance:")
            print(f"  Cache Hits: {cache['hits']} / {cache['total']} queries")
            print(f"  Hit Rate: {cache['hit_rate']:.2%}")
        else:
            # Even if we don't have cache stats, report that
            print(f"\nTeaCache Performance: No statistics collected")
            
        if "memory" in summary:
            mem = summary["memory"]
            print(f"\nMemory Usage:")
            print(f"  Current: {mem['current_gb']:.2f} GB")
            print(f"  Peak: {mem['max_gb']:.2f} GB")
            print(f"  Reserved: {mem['reserved_gb']:.2f} GB")
            
        print("================================\n")


# Global tracker instance for easy import
performance_tracker = PerformanceTracker()
