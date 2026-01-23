"""
MemOpt - GPU Memory Bandwidth Profiling & Optimization Platform

Professional bandwidth profiler for GPU-bound AI workloads.
"""

__version__ = "0.2.0"

# Core bandwidth profiling
from .bandwidth_profiler import (
    BandwidthProfiler,
    BandwidthStats,
    compare_bandwidth_profiles,
)

from .bandwidth_analyzer import (
    BandwidthAnalyzer,
    ModelBandwidthProfile,
    profile_model,
)

from .bottleneck_detector import (
    BottleneckDetector,
    Bottleneck,
    BottleneckType,
    BottleneckSeverity,
)

# KV Cache (for lazy allocation optimization - Phase 3)
from .kv_cache import PagedKVCache, CacheStats

# Exceptions
from .exceptions import (
    CacheEvictionError,
    QueueFullError,
    ResourceExhaustedError,
    RequestRejectedError,
)

__all__ = [
    # Bandwidth profiling
    "BandwidthProfiler",
    "BandwidthStats",
    "BandwidthAnalyzer",
    "ModelBandwidthProfile",
    "profile_model",
    "compare_bandwidth_profiles",
    # Bottleneck detection
    "BottleneckDetector",
    "Bottleneck",
    "BottleneckType",
    "BottleneckSeverity",
    # KV Cache (Phase 3)
    "PagedKVCache",
    "CacheStats",
    # Exceptions
    "CacheEvictionError",
    "QueueFullError",
    "ResourceExhaustedError",
    "RequestRejectedError",
]
