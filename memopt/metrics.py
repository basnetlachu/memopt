"""
Phase 2: Observability - Metrics Collection System

Prometheus-compatible metrics for hyperscale deployment monitoring.
Tracks performance, resource usage, degradation, and SLO compliance.
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from collections import defaultdict, deque
from threading import Lock
import json


@dataclass
class MetricValue:
    """Single metric observation with timestamp."""
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Phase 2: Prometheus-compatible metrics collector.

    Collects performance, resource, and degradation metrics for
    hyperscale monitoring and alerting.

    Metrics Types:
    - Counter: Monotonically increasing values (requests_total)
    - Gauge: Current value that can go up/down (queue_depth)
    - Histogram: Distribution of values (latency_ms)
    """

    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of samples to keep for histograms
        """
        self.window_size = window_size
        self._lock = Lock()

        # Metrics storage
        self._counters: Dict[str, float] = defaultdict(float)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Label tracking for multi-dimensional metrics
        self._counter_labels: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauge_labels: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Metric metadata
        self._help_text: Dict[str, str] = {}
        self._metric_types: Dict[str, str] = {}

    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None, help_text: str = ""):
        """
        Increment a counter metric.

        Args:
            name: Metric name (e.g., "requests_total")
            value: Amount to increment (default 1.0)
            labels: Optional labels (e.g., {"status": "200"})
            help_text: Metric description
        """
        with self._lock:
            if help_text and name not in self._help_text:
                self._help_text[name] = help_text
                self._metric_types[name] = "counter"

            if labels:
                label_key = self._serialize_labels(labels)
                self._counter_labels[name][label_key] += value
            else:
                self._counters[name] += value

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None, help_text: str = ""):
        """
        Set a gauge metric to a specific value.

        Args:
            name: Metric name (e.g., "queue_depth")
            value: Current value
            labels: Optional labels
            help_text: Metric description
        """
        with self._lock:
            if help_text and name not in self._help_text:
                self._help_text[name] = help_text
                self._metric_types[name] = "gauge"

            if labels:
                label_key = self._serialize_labels(labels)
                self._gauge_labels[name][label_key] = value
            else:
                self._gauges[name] = value

    def histogram(self, name: str, value: float, help_text: str = ""):
        """
        Observe a value in a histogram.

        Args:
            name: Metric name (e.g., "latency_ms")
            value: Observed value
            help_text: Metric description
        """
        with self._lock:
            if help_text and name not in self._help_text:
                self._help_text[name] = help_text
                self._metric_types[name] = "histogram"

            self._histograms[name].append(value)

    def get_counter(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current counter value."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return self._counter_labels[name].get(label_key, 0.0)
            return self._counters.get(name, 0.0)

    def get_gauge(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current gauge value."""
        with self._lock:
            if labels:
                label_key = self._serialize_labels(labels)
                return self._gauge_labels[name].get(label_key, 0.0)
            return self._gauges.get(name, 0.0)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """
        Get histogram statistics.

        Returns:
            Dict with min, max, mean, p50, p95, p99 percentiles
        """
        with self._lock:
            values = list(self._histograms.get(name, []))

        if not values:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0
            }

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "count": count,
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "mean": sum(sorted_values) / count,
            "p50": sorted_values[int(count * 0.50)],
            "p95": sorted_values[int(count * 0.95)],
            "p99": sorted_values[int(count * 0.99)]
        }

    def _serialize_labels(self, labels: Dict[str, str]) -> str:
        """Serialize labels to a consistent string key."""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metric string
        """
        lines = []

        with self._lock:
            # Export counters
            for name, value in self._counters.items():
                if name in self._help_text:
                    lines.append(f"# HELP {name} {self._help_text[name]}")
                    lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")

            # Export labeled counters
            for name, label_values in self._counter_labels.items():
                if name in self._help_text:
                    lines.append(f"# HELP {name} {self._help_text[name]}")
                    lines.append(f"# TYPE {name} counter")
                for label_key, value in label_values.items():
                    lines.append(f"{name}{{{label_key}}} {value}")

            # Export gauges
            for name, value in self._gauges.items():
                if name in self._help_text:
                    lines.append(f"# HELP {name} {self._help_text[name]}")
                    lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            # Export labeled gauges
            for name, label_values in self._gauge_labels.items():
                if name in self._help_text:
                    lines.append(f"# HELP {name} {self._help_text[name]}")
                    lines.append(f"# TYPE {name} gauge")
                for label_key, value in label_values.items():
                    lines.append(f"{name}{{{label_key}}} {value}")

            # Export histograms as summaries
            for name in self._histograms.keys():
                stats = self.get_histogram_stats(name)
                if name in self._help_text:
                    lines.append(f"# HELP {name} {self._help_text[name]}")
                    lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {stats['mean'] * stats['count']}")
                lines.append(f'{name}{{quantile="0.5"}} {stats["p50"]}')
                lines.append(f'{name}{{quantile="0.95"}} {stats["p95"]}')
                lines.append(f'{name}{{quantile="0.99"}} {stats["p99"]}')

        return "\n".join(lines)

    def export_json(self) -> str:
        """Export all metrics as JSON."""
        data = {
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "histograms": {
                name: self.get_histogram_stats(name)
                for name in self._histograms.keys()
            }
        }
        return json.dumps(data, indent=2)

    def reset(self):
        """Reset all metrics (for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._counter_labels.clear()
            self._gauge_labels.clear()


# Global metrics instance
_global_metrics: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get global metrics collector singleton."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = MetricsCollector()
    return _global_metrics


def init_metrics(window_size: int = 1000) -> MetricsCollector:
    """Initialize global metrics collector."""
    global _global_metrics
    _global_metrics = MetricsCollector(window_size=window_size)
    return _global_metrics


# ============================================================================
# Phase 2: Standard Metrics Registry
# ============================================================================

class SystemMetrics:
    """
    Phase 2: Standard system metrics for hyperscale monitoring.

    These metrics enable SLO tracking, capacity planning, and alerting.
    """

    # Request metrics
    REQUESTS_TOTAL = "memopt_requests_total"
    REQUESTS_ACTIVE = "memopt_requests_active"
    REQUESTS_QUEUED = "memopt_requests_queued"
    REQUESTS_REJECTED = "memopt_requests_rejected_total"

    # Performance metrics
    LATENCY_MS = "memopt_latency_ms"
    THROUGHPUT_TOKENS_PER_SEC = "memopt_throughput_tokens_per_sec"
    TOKENS_GENERATED = "memopt_tokens_generated_total"

    # Resource metrics
    QUEUE_DEPTH = "memopt_queue_depth"
    QUEUE_UTILIZATION = "memopt_queue_utilization"
    KV_CACHE_BLOCKS_USED = "memopt_kv_cache_blocks_used"
    KV_CACHE_BLOCKS_FREE = "memopt_kv_cache_blocks_free"
    KV_CACHE_UTILIZATION = "memopt_kv_cache_utilization"
    MEMORY_USED_GB = "memopt_memory_used_gb"

    # Degradation metrics
    EVICTIONS_TOTAL = "memopt_evictions_total"
    EVICTION_LATENCY_MS = "memopt_eviction_latency_ms"
    SPECULATIVE_FALLBACKS = "memopt_speculative_fallbacks_total"
    CIRCUIT_BREAKER_STATE = "memopt_circuit_breaker_state"

    # SLO metrics
    SLO_LATENCY_TARGET_MS = "memopt_slo_latency_target_ms"
    SLO_VIOLATIONS_TOTAL = "memopt_slo_violations_total"
    SLO_COMPLIANCE_RATIO = "memopt_slo_compliance_ratio"


def record_request_start(metrics: MetricsCollector, request_id: str):
    """Record request start."""
    metrics.counter(SystemMetrics.REQUESTS_TOTAL, help_text="Total requests processed")
    metrics.gauge(SystemMetrics.REQUESTS_ACTIVE,
                  metrics.get_gauge(SystemMetrics.REQUESTS_ACTIVE) + 1,
                  help_text="Currently active requests")


def record_request_complete(metrics: MetricsCollector, request_id: str, latency_ms: float, tokens: int):
    """Record request completion."""
    metrics.gauge(SystemMetrics.REQUESTS_ACTIVE,
                  max(0, metrics.get_gauge(SystemMetrics.REQUESTS_ACTIVE) - 1))
    metrics.histogram(SystemMetrics.LATENCY_MS, latency_ms,
                      help_text="Request latency in milliseconds")
    metrics.counter(SystemMetrics.TOKENS_GENERATED, tokens,
                    help_text="Total tokens generated")


def record_request_rejected(metrics: MetricsCollector, reason: str):
    """Record request rejection."""
    metrics.counter(SystemMetrics.REQUESTS_REJECTED,
                    labels={"reason": reason},
                    help_text="Total requests rejected")


def record_queue_state(metrics: MetricsCollector, depth: int, max_depth: int):
    """Record queue depth."""
    metrics.gauge(SystemMetrics.QUEUE_DEPTH, depth,
                  help_text="Current queue depth")
    metrics.gauge(SystemMetrics.QUEUE_UTILIZATION, depth / max(max_depth, 1),
                  help_text="Queue utilization ratio")


def record_cache_state(metrics: MetricsCollector, used_blocks: int, total_blocks: int):
    """Record KV cache state."""
    metrics.gauge(SystemMetrics.KV_CACHE_BLOCKS_USED, used_blocks,
                  help_text="KV cache blocks in use")
    metrics.gauge(SystemMetrics.KV_CACHE_BLOCKS_FREE, total_blocks - used_blocks,
                  help_text="KV cache blocks available")
    metrics.gauge(SystemMetrics.KV_CACHE_UTILIZATION, used_blocks / max(total_blocks, 1),
                  help_text="KV cache utilization ratio")


def record_eviction(metrics: MetricsCollector, blocks_evicted: int, latency_ms: float):
    """Record cache eviction event."""
    metrics.counter(SystemMetrics.EVICTIONS_TOTAL, blocks_evicted,
                    help_text="Total cache evictions")
    metrics.histogram(SystemMetrics.EVICTION_LATENCY_MS, latency_ms,
                      help_text="Cache eviction latency in milliseconds")
