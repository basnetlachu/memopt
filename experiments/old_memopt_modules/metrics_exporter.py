"""
Metrics Exporter - Phase 3

Background thread that exports metrics in Prometheus format.
Can be adapted for Datadog, CloudWatch, or other monitoring systems.

PERFORMANCE IMPACT: None on inference (runs in separate background thread).

MULTI-NODE: All metrics include node_id label for horizontal scaling.
"""

import threading
import time
from typing import Optional

from .production_metrics import ProductionMetrics, MetricsSnapshot
from .node_identity import get_node_id


class PrometheusExporter:
    """
    Background metrics exporter for Prometheus scraping.
    
    Design:
    - Runs in dedicated background thread (daemon)
    - Periodically reads snapshot() from metrics collector
    - Exposes HTTP endpoint for Prometheus scraping (optional)
    - Does NOT block inference
    
    Use Case:
    - Prometheus pulls metrics from /metrics endpoint
    - Datadog agent scrapes metrics
    - CloudWatch exporter pushes metrics
    
    Integration:
    - Started by production server if metrics enabled
    - Completely optional (not required for correctness)
    """
    
    def __init__(
        self,
        metrics: ProductionMetrics,
        export_interval_sec: int = 10,
        enable: bool = False
    ):
        """
        Args:
            metrics: ProductionMetrics instance to export
            export_interval_sec: How often to export metrics (seconds)
            enable: Enable background exporter (default False)
        """
        self.metrics = metrics
        self.interval = export_interval_sec
        self.enable = enable
        self.running = False
        self.thread = None
        self.latest_snapshot: Optional[MetricsSnapshot] = None
    
    def start(self):
        """Start background exporter thread."""
        if not self.enable:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._export_loop, daemon=True)
        self.thread.start()
        print(f"[PrometheusExporter] Started (export interval: {self.interval}s)")
    
    def stop(self):
        """Stop background exporter thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print("[PrometheusExporter] Stopped")
    
    def _export_loop(self):
        """Background loop that periodically exports metrics."""
        while self.running:
            try:
                # Read snapshot from metrics collector (non-blocking)
                snapshot = self.metrics.snapshot()
                self.latest_snapshot = snapshot
                
                # Export to monitoring system
                self._export_snapshot(snapshot)
                
            except Exception as e:
                print(f"[PrometheusExporter] Export error: {e}")
            
            # Sleep until next export interval
            time.sleep(self.interval)
    
    def _export_snapshot(self, snapshot: MetricsSnapshot):
        """
        Export snapshot to monitoring system.
        
        Implementation depends on monitoring stack:
        - Prometheus: Store snapshot, serve via HTTP endpoint
        - Datadog: Push via statsd
        - CloudWatch: Push via boto3
        
        This is a basic implementation that just logs.
        Override this method for your monitoring system.
        """
        # Example: Print metrics (replace with actual export logic)
        if self.interval >= 60:  # Only log if interval is 1min+
            print(f"[Metrics] tokens/s={snapshot.tokens_per_sec:.1f}, "
                  f"batch_size={snapshot.avg_batch_size:.1f}, "
                  f"queue_depth={snapshot.queue_depth}, "
                  f"gpu_mem={snapshot.gpu_memory_allocated_gb:.2f}GB")
    
    def get_prometheus_format(self) -> str:
        """
        Get metrics in Prometheus text format.
        
        Returns:
            Prometheus-formatted metrics string
        
        Use Case:
        - Serve from HTTP endpoint for Prometheus scraping
        - Example: GET /metrics returns this string
        """
        if self.latest_snapshot is None:
            return "# No metrics available yet\n"

        s = self.latest_snapshot
        node_id = get_node_id()

        # All metrics include node_id label for multi-node deployments
        return f"""# HELP memopt_tokens_per_sec Token generation throughput
# TYPE memopt_tokens_per_sec gauge
memopt_tokens_per_sec{{node_id="{node_id}"}} {s.tokens_per_sec:.2f}

# HELP memopt_requests_per_sec Request completion rate
# TYPE memopt_requests_per_sec gauge
memopt_requests_per_sec{{node_id="{node_id}"}} {s.requests_per_sec:.2f}

# HELP memopt_avg_batch_size Average batch size
# TYPE memopt_avg_batch_size gauge
memopt_avg_batch_size{{node_id="{node_id}"}} {s.avg_batch_size:.2f}

# HELP memopt_max_batch_size Maximum batch size observed
# TYPE memopt_max_batch_size gauge
memopt_max_batch_size{{node_id="{node_id}"}} {s.max_batch_size}

# HELP memopt_kv_cache_utilization_pct KV cache utilization percentage
# TYPE memopt_kv_cache_utilization_pct gauge
memopt_kv_cache_utilization_pct{{node_id="{node_id}"}} {s.kv_cache_utilization_pct:.2f}

# HELP memopt_gpu_memory_allocated_gb GPU memory allocated (GB)
# TYPE memopt_gpu_memory_allocated_gb gauge
memopt_gpu_memory_allocated_gb{{node_id="{node_id}"}} {s.gpu_memory_allocated_gb:.3f}

# HELP memopt_gpu_memory_reserved_gb GPU memory reserved (GB)
# TYPE memopt_gpu_memory_reserved_gb gauge
memopt_gpu_memory_reserved_gb{{node_id="{node_id}"}} {s.gpu_memory_reserved_gb:.3f}

# HELP memopt_queue_depth Current request queue depth
# TYPE memopt_queue_depth gauge
memopt_queue_depth{{node_id="{node_id}"}} {s.queue_depth}

# HELP memopt_active_requests Currently processing requests
# TYPE memopt_active_requests gauge
memopt_active_requests{{node_id="{node_id}"}} {s.active_requests}

# HELP memopt_requests_completed_total Total completed requests
# TYPE memopt_requests_completed_total counter
memopt_requests_completed_total{{node_id="{node_id}"}} {s.requests_completed}

# HELP memopt_requests_rejected_total Total rejected requests
# TYPE memopt_requests_rejected_total counter
memopt_requests_rejected_total{{node_id="{node_id}"}} {s.requests_rejected}

# HELP memopt_tokens_generated_total Total tokens generated
# TYPE memopt_tokens_generated_total counter
memopt_tokens_generated_total{{node_id="{node_id}"}} {s.total_tokens_generated}
"""


class DatadogExporter(PrometheusExporter):
    """
    Datadog-specific exporter (example).
    
    Pushes metrics via statsd instead of HTTP scraping.
    """
    
    def _export_snapshot(self, snapshot: MetricsSnapshot):
        """Push metrics to Datadog via statsd."""
        # Example implementation (requires datadog library)
        # from datadog import statsd
        # statsd.gauge('memopt.tokens_per_sec', snapshot.tokens_per_sec)
        # statsd.gauge('memopt.avg_batch_size', snapshot.avg_batch_size)
        # etc.
        pass  # Implement based on your Datadog setup
