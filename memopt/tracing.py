"""
Phase 2: Request Tracing and Telemetry

Distributed tracing for performance analysis and debugging.
Tracks request flow through scheduler, cache, inference, and speculative decoding.
"""

import time
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json


class SpanKind(Enum):
    """Span types for request tracing."""
    REQUEST = "request"           # Top-level request
    ADMISSION = "admission"       # Queue admission
    SCHEDULING = "scheduling"     # Batch scheduling
    INFERENCE = "inference"       # Model inference
    CACHE_READ = "cache_read"     # KV cache read
    CACHE_WRITE = "cache_write"   # KV cache write
    EVICTION = "eviction"         # Cache eviction
    SPECULATIVE = "speculative"   # Speculative decoding
    VERIFICATION = "verification" # Draft verification


@dataclass
class Span:
    """
    Trace span representing a single operation.

    Spans form a tree structure representing request flow.
    """
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    kind: SpanKind
    name: str
    start_time: float
    end_time: Optional[float] = None
    status: str = "in_progress"  # in_progress, success, error
    metadata: Dict = field(default_factory=dict)
    events: List[Dict] = field(default_factory=list)

    def finish(self, status: str = "success", **metadata):
        """Mark span as complete."""
        self.end_time = time.time()
        self.status = status
        self.metadata.update(metadata)

    def add_event(self, name: str, **attributes):
        """Add event to span."""
        self.events.append({
            "timestamp": time.time(),
            "name": name,
            "attributes": attributes
        })

    @property
    def duration_ms(self) -> Optional[float]:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "kind": self.kind.value,
            "name": self.name,
            "start_time": self.start_time,
            "status": self.status,
            "metadata": self.metadata
        }
        if self.parent_span_id:
            data["parent_span_id"] = self.parent_span_id
        if self.end_time:
            data["end_time"] = self.end_time
            data["duration_ms"] = self.duration_ms
        if self.events:
            data["events"] = self.events
        return data


@dataclass
class Trace:
    """
    Complete trace for a single request.

    Contains all spans for the request in a tree structure.
    """
    trace_id: str
    start_time: float
    end_time: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    root_span_id: Optional[str] = None

    def add_span(self, span: Span):
        """Add span to trace."""
        self.spans.append(span)
        if span.parent_span_id is None:
            self.root_span_id = span.span_id

    def finish(self):
        """Mark trace as complete."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> Optional[float]:
        """Get total trace duration in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = {
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "spans": [s.to_dict() for s in self.spans]
        }
        if self.end_time:
            data["end_time"] = self.end_time
            data["duration_ms"] = self.duration_ms
        if self.root_span_id:
            data["root_span_id"] = self.root_span_id
        return data

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class Tracer:
    """
    Phase 2: Distributed tracer for request flow.

    Tracks request journey through all system components.
    """

    def __init__(self, max_traces: int = 1000):
        """
        Args:
            max_traces: Maximum number of traces to keep in memory
        """
        self.max_traces = max_traces
        self._traces: Dict[str, Trace] = {}
        self._span_counter = 0

    def start_trace(self, trace_id: str) -> Trace:
        """
        Start a new trace.

        Args:
            trace_id: Unique trace identifier (correlation ID)

        Returns:
            Trace instance
        """
        trace = Trace(trace_id=trace_id, start_time=time.time())
        self._traces[trace_id] = trace

        # Evict oldest traces if limit exceeded
        if len(self._traces) > self.max_traces:
            oldest_id = min(self._traces.keys(), key=lambda k: self._traces[k].start_time)
            del self._traces[oldest_id]

        return trace

    def start_span(
        self,
        trace_id: str,
        kind: SpanKind,
        name: str,
        parent_span_id: Optional[str] = None,
        **metadata
    ) -> Span:
        """
        Start a new span.

        Args:
            trace_id: Trace ID to associate with
            kind: Span kind
            name: Span name
            parent_span_id: Parent span ID
            **metadata: Additional metadata

        Returns:
            Span instance
        """
        self._span_counter += 1
        span_id = f"span_{self._span_counter:08d}"

        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            kind=kind,
            name=name,
            start_time=time.time(),
            metadata=metadata
        )

        # Add to trace if it exists
        if trace_id in self._traces:
            self._traces[trace_id].add_span(span)

        return span

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID."""
        return self._traces.get(trace_id)

    def finish_trace(self, trace_id: str):
        """Mark trace as complete."""
        if trace_id in self._traces:
            self._traces[trace_id].finish()

    def export_trace(self, trace_id: str) -> Optional[str]:
        """Export trace as JSON."""
        trace = self.get_trace(trace_id)
        if trace:
            return trace.to_json()
        return None

    def get_all_traces(self) -> List[Trace]:
        """Get all traces."""
        return list(self._traces.values())


# ============================================================================
# Phase 2: Tracing Context Manager
# ============================================================================

class TracedOperation:
    """
    Context manager for traced operations.

    Usage:
        with TracedOperation(tracer, trace_id, SpanKind.INFERENCE, "forward_pass") as span:
            result = model(inputs)
            span.add_event("tokens_generated", count=10)
    """

    def __init__(
        self,
        tracer: Tracer,
        trace_id: str,
        kind: SpanKind,
        name: str,
        parent_span_id: Optional[str] = None,
        **metadata
    ):
        """
        Args:
            tracer: Tracer instance
            trace_id: Trace ID
            kind: Span kind
            name: Span name
            parent_span_id: Parent span ID
            **metadata: Span metadata
        """
        self.tracer = tracer
        self.trace_id = trace_id
        self.kind = kind
        self.name = name
        self.parent_span_id = parent_span_id
        self.metadata = metadata
        self.span: Optional[Span] = None

    def __enter__(self) -> Span:
        """Start span."""
        self.span = self.tracer.start_span(
            trace_id=self.trace_id,
            kind=self.kind,
            name=self.name,
            parent_span_id=self.parent_span_id,
            **self.metadata
        )
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish span."""
        if self.span:
            if exc_type is None:
                self.span.finish(status="success")
            else:
                self.span.finish(
                    status="error",
                    error_type=exc_type.__name__ if exc_type else None,
                    error_message=str(exc_val) if exc_val else None
                )
        return False


# ============================================================================
# Phase 2: Performance Telemetry
# ============================================================================

@dataclass
class PerformanceSnapshot:
    """Performance metrics snapshot."""
    timestamp: float
    throughput_tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    queue_depth: int
    cache_utilization: float
    active_requests: int
    evictions_per_min: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "queue_depth": self.queue_depth,
            "cache_utilization": self.cache_utilization,
            "active_requests": self.active_requests,
            "evictions_per_min": self.evictions_per_min
        }


class TelemetryCollector:
    """
    Phase 2: Telemetry collector for performance monitoring.

    Aggregates metrics from all components for dashboards and alerting.
    """

    def __init__(self, snapshot_interval: int = 60):
        """
        Args:
            snapshot_interval: Seconds between snapshots
        """
        self.snapshot_interval = snapshot_interval
        self._snapshots: List[PerformanceSnapshot] = []
        self._last_snapshot_time = time.time()

    def record_snapshot(self, snapshot: PerformanceSnapshot):
        """Record performance snapshot."""
        self._snapshots.append(snapshot)
        self._last_snapshot_time = time.time()

        # Keep last 1000 snapshots (configurable retention)
        if len(self._snapshots) > 1000:
            self._snapshots.pop(0)

    def get_recent_snapshots(self, count: int = 10) -> List[PerformanceSnapshot]:
        """Get most recent snapshots."""
        return self._snapshots[-count:]

    def get_snapshots_in_range(self, start_time: float, end_time: float) -> List[PerformanceSnapshot]:
        """Get snapshots within time range."""
        return [
            s for s in self._snapshots
            if start_time <= s.timestamp <= end_time
        ]

    def export_snapshots(self) -> str:
        """Export all snapshots as JSON."""
        return json.dumps([s.to_dict() for s in self._snapshots], indent=2)


# ============================================================================
# Phase 2: Global Tracer and Telemetry
# ============================================================================

_global_tracer: Optional[Tracer] = None
_global_telemetry: Optional[TelemetryCollector] = None


def get_tracer() -> Tracer:
    """Get global tracer singleton."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def get_telemetry() -> TelemetryCollector:
    """Get global telemetry collector singleton."""
    global _global_telemetry
    if _global_telemetry is None:
        _global_telemetry = TelemetryCollector()
    return _global_telemetry


def init_tracing(max_traces: int = 1000) -> Tracer:
    """Initialize global tracer."""
    global _global_tracer
    _global_tracer = Tracer(max_traces=max_traces)
    return _global_tracer


def init_telemetry(snapshot_interval: int = 60) -> TelemetryCollector:
    """Initialize global telemetry collector."""
    global _global_telemetry
    _global_telemetry = TelemetryCollector(snapshot_interval=snapshot_interval)
    return _global_telemetry
