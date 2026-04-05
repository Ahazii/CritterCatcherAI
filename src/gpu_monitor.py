"""
GPU Monitoring Module for CritterCatcherAI
Provides real-time GPU utilization tracking and logging.
"""
import logging
import time
import threading
from typing import Dict, Optional


class GPUMonitor:
    """Monitor GPU usage and log periodically."""
    
    def __init__(self, logger: Optional[logging.Logger] = None, log_interval: int = 5, log_on_idle: bool = False):
        """
        Initialize GPU monitor.
        
        Args:
            logger: Logger instance for output (creates default if None)
            log_interval: Seconds between log entries
            log_on_idle: Whether to log when GPU is idle (0% usage)
        """
        # Create logger if not provided
        if logger is None:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
            
        self.log_interval = log_interval
        self.log_on_idle = log_on_idle
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self.gpu_available = False
        self.handle = None
        self.gpu_name = "No GPU detected"
        self.pynvml = None
        
        # Try to initialize NVML (NVIDIA Management Library)
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            
            # Get device count to validate GPU access
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                self.logger.warning("No NVIDIA GPUs detected")
                return
            
            self.gpu_available = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            
            # Decode if bytes (Python 3)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode('utf-8')
            
            # Test that we can actually read GPU stats
            try:
                test_util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.logger.info(f"GPU monitoring available: {self.gpu_name}")
            except Exception as test_error:
                self.logger.warning(f"GPU detected but monitoring unavailable: {test_error}")
                self.gpu_available = False
                
        except ImportError:
            self.logger.warning("pynvml not installed - GPU monitoring unavailable")
        except Exception as e:
            self.logger.warning(f"GPU monitoring initialization failed: {e}")
            # Clean up partial initialization
            if self.pynvml:
                try:
                    self.pynvml.nvmlShutdown()
                except:
                    pass
    
    def start_monitoring(self):
        """Start background GPU monitoring thread."""
        if not self.gpu_available or self._monitoring:
            if not self.gpu_available:
                self.logger.debug("Cannot start GPU monitoring - GPU not available")
            return
        
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"GPU monitoring started for {self.gpu_name}")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._thread:
            self._thread.join(timeout=self.log_interval + 1)
    
    def _monitor_loop(self):
        """Background monitoring loop with error recovery."""
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        while self._monitoring:
            try:
                utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                
                gpu_usage = utilization.gpu
                mem_used_mb = memory.used // (1024 * 1024)
                mem_total_mb = memory.total // (1024 * 1024)
                mem_percent = (memory.used / memory.total) * 100
                
                # Reset error counter on successful read
                consecutive_errors = 0
                
                # Skip logging if idle and log_on_idle is False
                if gpu_usage == 0 and not self.log_on_idle:
                    time.sleep(self.log_interval)
                    continue
                
                # Log GPU usage
                self.logger.info(
                    f"GPU: {gpu_usage}% utilization | "
                    f"Memory: {mem_used_mb}MB / {mem_total_mb}MB ({mem_percent:.1f}%)"
                )
                
            except Exception as e:
                consecutive_errors += 1
                self.logger.warning(
                    f"GPU monitoring error (attempt {consecutive_errors}/{max_consecutive_errors}): {e}"
                )
                
                # Only stop after repeated failures
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(
                        f"GPU monitoring stopped after {max_consecutive_errors} consecutive errors. "
                        "GPU may be unavailable or driver issue detected."
                    )
                    self._monitoring = False
                    break
                
                # Wait longer before retry on error
                time.sleep(self.log_interval * 2)
                continue
            
            time.sleep(self.log_interval)
    
    def get_current_stats(self) -> Dict:
        """
        Get current GPU stats (for API endpoint).
        
        Returns:
            Dictionary with usage_percent, name, memory stats, or error info
        """
        if not self.gpu_available:
            return {
                "usage_percent": 0.0,
                "name": "No GPU detected",
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "memory_percent": 0.0,
                "error": "GPU not available"
            }
        
        try:
            utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            return {
                "usage_percent": float(utilization.gpu),
                "name": self.gpu_name,
                "memory_used_mb": memory.used // (1024 * 1024),
                "memory_total_mb": memory.total // (1024 * 1024),
                "memory_percent": float((memory.used / memory.total) * 100)
            }
        except Exception as e:
            return {
                "usage_percent": 0.0,
                "name": self.gpu_name,
                "memory_used_mb": 0,
                "memory_total_mb": 0,
                "memory_percent": 0.0,
                "error": str(e)
            }
    
    def log_operation(self, operation: str, stage: str = "start"):
        """
        Log GPU usage for a specific operation.
        
        Args:
            operation: Name of the operation (e.g., "YOLO detection", "CLIP analysis")
            stage: Either "start" or "end"
        """
        if not self.gpu_available:
            return
        
        try:
            stats = self.get_current_stats()
            usage = stats.get('usage_percent', 0.0)
            
            if stage == "start":
                self.logger.info(f"GPU: Starting {operation} (current: {usage:.1f}%)")
            else:
                self.logger.info(f"GPU: {operation} complete (usage: {usage:.1f}%)")
        except Exception as e:
            self.logger.debug(f"Could not log GPU operation: {e}")
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        self.stop_monitoring()
        if self.gpu_available and self.pynvml:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass
