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
    
    def __init__(self, logger: logging.Logger, log_interval: int = 5, log_on_idle: bool = False):
        """
        Initialize GPU monitor.
        
        Args:
            logger: Logger instance for output
            log_interval: Seconds between log entries
            log_on_idle: Whether to log when GPU is idle (0% usage)
        """
        self.logger = logger
        self.log_interval = log_interval
        self.log_on_idle = log_on_idle
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self.gpu_available = False
        self.handle = None
        self.gpu_name = "No GPU detected"
        
        # Try to initialize NVML (NVIDIA Management Library)
        try:
            import pynvml
            self.pynvml = pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            
            # Decode if bytes (Python 3)
            if isinstance(self.gpu_name, bytes):
                self.gpu_name = self.gpu_name.decode('utf-8')
                
            self.logger.info(f"GPU monitoring available: {self.gpu_name}")
        except ImportError:
            self.logger.warning("pynvml not installed - GPU monitoring unavailable")
        except Exception as e:
            self.logger.warning(f"GPU monitoring unavailable: {e}")
    
    def start_monitoring(self):
        """Start background GPU monitoring thread."""
        if not self.gpu_available or self._monitoring:
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
        """Background monitoring loop."""
        while self._monitoring:
            try:
                utilization = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                memory = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                
                gpu_usage = utilization.gpu
                mem_used_mb = memory.used // (1024 * 1024)
                mem_total_mb = memory.total // (1024 * 1024)
                mem_percent = (memory.used / memory.total) * 100
                
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
                self.logger.error(f"GPU monitoring error: {e}")
                break
            
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
    
    def __del__(self):
        """Cleanup NVML on destruction."""
        if self.gpu_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass
