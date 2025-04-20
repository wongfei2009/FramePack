import asyncio
import sys
import warnings
from functools import wraps

# Only needed on Windows
if sys.platform.startswith("win"):
    import asyncio.proactor_events
    
    # Get the actual class that we need to patch
    _ProactorBasePipeTransport = asyncio.proactor_events._ProactorBasePipeTransport
    
    # Define a wrapper to silence connection reset errors
    def silence_connection_reset(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except ConnectionResetError as e:
                # Silently handle the specific connection reset error
                if str(e) == "[WinError 10054] An existing connection was forcibly closed by the remote host":
                    return None
                raise  # Re-raise other types of ConnectionResetError
        return wrapper
    
    # Patch the _call_connection_lost method
    _ProactorBasePipeTransport._call_connection_lost = silence_connection_reset(_ProactorBasePipeTransport._call_connection_lost)

def apply_asyncio_fixes():
    """Apply fixes for asyncio on Windows to silence connection reset errors."""
    if sys.platform.startswith("win"):
        # Suppress the deprecation warning related to the ProactorEventLoop
        warnings.filterwarnings(
            "ignore", 
            message="There is no current event loop", 
            category=DeprecationWarning
        )
        
        # Silently apply fixes without printing to console
        # print("Applied custom asyncio Windows fixes to suppress connection reset errors")
