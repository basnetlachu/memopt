"""
Health Monitor - Phase 4

Periodic sanity checks for memory leaks, fragmentation, and anomalies.
Logs warnings for operators - does NOT auto-restart or modify system.

PERFORMANCE IMPACT: None on inference (runs in background thread, read-only checks).
"""

import threading
import time
import torch
from typing import Optional, Dict


class HealthMonitor:
    """
    Periodic health checks for long-running deployments.
    
    Design:
    - Runs in background thread (every 5 minutes by default)
    - Compares current state to baseline
    - Logs warnings, does NOT restart or modify system
    - Read-only checks (no side effects)
    
    Checks:
    - Memory growth (detects leaks)
    - KV cache fragmentation (if exposed by paged cache)
    - Anomalous metrics (sudden spikes)
    
    Use Case:
    - Early warning system for operators
    - Detect gradual degradation over days/weeks
    - Production observability
    
    Integration:
    - Started by production server if enabled
    - Completely optional (disabled by default)
    """
    
    def __init__(
        self,
        model,  # OptimizedLLM instance
        check_interval_sec: int = 300,  # 5 minutes
        enable: bool = False
    ):
        """
        Args:
            model: OptimizedLLM instance to monitor
            check_interval_sec: How often to run checks (seconds)
            enable: Enable health monitoring (default False)
        """
        self.model = model
        self.interval = check_interval_sec
        self.enable = enable
        self.baseline_memory_gb: Optional[float] = None
        self.baseline_kv_blocks: Optional[int] = None
        self.running = False
        self.thread = None
        self.checks_run = 0
        self.warnings_issued = 0
    
    def start(self):
        """Start background health monitor."""
        if not self.enable:
            return
        
        # Establish baseline
        self.baseline_memory_gb = self._get_memory_usage_gb()
        self.baseline_kv_blocks = self._get_kv_blocks_allocated()
        
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        
        print(f"[HealthMonitor] Started (interval: {self.interval}s, "
              f"baseline memory: {self.baseline_memory_gb:.2f}GB)")
    
    def stop(self):
        """Stop background health monitor."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print(f"[HealthMonitor] Stopped ({self.checks_run} checks, {self.warnings_issued} warnings)")
    
    def _monitor_loop(self):
        """Periodic health check loop."""
        while self.running:
            time.sleep(self.interval)
            
            try:
                self._run_health_checks()
                self.checks_run += 1
            except Exception as e:
                print(f"[HealthMonitor] Check error: {e}")
    
    def _run_health_checks(self):
        """Run all health checks."""
        self._check_memory_growth()
        self._check_kv_fragmentation()
        self._check_metrics_sanity()
    
    def _check_memory_growth(self):
        """Detect unbounded memory growth."""
        current_gb = self._get_memory_usage_gb()
        
        if self.baseline_memory_gb is None or self.baseline_memory_gb == 0:
            return
        
        growth_factor = current_gb / self.baseline_memory_gb
        
        # Warning thresholds
        if growth_factor > 3.0:
            self._issue_warning(
                "CRITICAL_MEMORY_GROWTH",
                f"Memory usage grew {growth_factor:.1f}x from baseline "
                f"({self.baseline_memory_gb:.2f}GB → {current_gb:.2f}GB). "
                f"Possible memory leak detected."
            )
        elif growth_factor > 2.0:
            self._issue_warning(
                "HIGH_MEMORY_GROWTH",
                f"Memory usage grew {growth_factor:.1f}x from baseline "
                f"({self.baseline_memory_gb:.2f}GB → {current_gb:.2f}GB). "
                f"Monitor for continued growth."
            )
    
    def _check_kv_fragmentation(self):
        """Detect KV cache fragmentation."""
        # Check if paged KV cache exposes fragmentation stats
        if not hasattr(self.model, 'kv_cache'):
            return
        
        kv_cache = self.model.kv_cache
        
        # Check for fragmentation_pct method
        if hasattr(kv_cache, 'get_fragmentation_pct'):
            frag_pct = kv_cache.get_fragmentation_pct()
            
            if frag_pct > 70:
                self._issue_warning(
                    "CRITICAL_KV_FRAGMENTATION",
                    f"KV cache fragmentation at {frag_pct:.1f}%. "
                    f"Consider restarting to defragment."
                )
            elif frag_pct > 50:
                self._issue_warning(
                    "HIGH_KV_FRAGMENTATION",
                    f"KV cache fragmentation at {frag_pct:.1f}%. "
                    f"Monitor for performance degradation."
                )
    
    def _check_metrics_sanity(self):
        """Check for anomalous metrics."""
        # If metrics are enabled, check for anomalies
        if not hasattr(self.model, 'metrics') or not self.model.metrics.enable:
            return
        
        snapshot = self.model.metrics.snapshot()
        
        # Check for anomalous queue depth
        if snapshot.queue_depth > 5000:
            self._issue_warning(
                "HIGH_QUEUE_DEPTH",
                f"Queue depth at {snapshot.queue_depth}. "
                f"System may be overloaded."
            )
        
        # Check for high rejection rate
        if snapshot.requests_completed > 0:
            rejection_rate = snapshot.requests_rejected / snapshot.requests_completed
            if rejection_rate > 0.5:
                self._issue_warning(
                    "HIGH_REJECTION_RATE",
                    f"Rejection rate at {rejection_rate*100:.1f}%. "
                    f"System is rejecting more than half of requests."
                )
    
    def _get_memory_usage_gb(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        return 0.0
    
    def _get_kv_blocks_allocated(self) -> Optional[int]:
        """Get number of KV cache blocks allocated (if available)."""
        if hasattr(self.model, 'kv_cache'):
            kv_cache = self.model.kv_cache
            if hasattr(kv_cache, 'num_blocks_allocated'):
                return kv_cache.num_blocks_allocated
        return None
    
    def _issue_warning(self, warning_type: str, message: str):
        """Issue warning to operator."""
        self.warnings_issued += 1
        print(f"[HealthMonitor] WARNING [{warning_type}]: {message}")
    
    def get_health_status(self) -> Dict:
        """Get current health status (for API/dashboard)."""
        return {
            'enabled': self.enable,
            'checks_run': self.checks_run,
            'warnings_issued': self.warnings_issued,
            'baseline_memory_gb': self.baseline_memory_gb,
            'current_memory_gb': self._get_memory_usage_gb(),
            'memory_growth_factor': (
                self._get_memory_usage_gb() / self.baseline_memory_gb
                if self.baseline_memory_gb and self.baseline_memory_gb > 0
                else 1.0
            )
        }
