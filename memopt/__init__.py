"""
MemOpt - GPU Memory Bandwidth Profiling & Optimization Platform

Professional bandwidth profiler for GPU-bound AI workloads.
"""

__version__ = "0.4.0"  # Phase 3 complete

# Core bandwidth profiling (Phase 1)
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

# Visualization & Reporting (Phase 2)
from .visualizer import (
    BandwidthVisualizer,
    print_bandwidth_chart,
    print_comparison,
    print_dashboard,
)

from .report_generator import (
    ReportGenerator,
    generate_html_report,
)

# Hardware Validation (Phase 4)
# Note: Import directly from module if needed
# from .hardware_validator import HardwareValidator, HardwareValidationResult

# Memory Access Coalescing (Phase 4)
from .memory_coalescing import (
    MemoryAccessCoalescer,
    CoalescedKVCache,
    CoalescingStats,
)

# Optimization Engine (Phase 3)
from .optimization_engine import (
    OptimizationEngine,
    OptimizationResult,
    LazyKVCache,
    LazyTensorAllocator,
    run_optimization_demo,
)

# KV Cache
from .kv_cache import PagedKVCache, CacheStats

# Exceptions
from .exceptions import (
    CacheEvictionError,
    QueueFullError,
    ResourceExhaustedError,
    RequestRejectedError,
)

__all__ = [
    # Phase 1: Bandwidth profiling
    "BandwidthProfiler",
    "BandwidthStats",
    "BandwidthAnalyzer",
    "ModelBandwidthProfile",
    "profile_model",
    "compare_bandwidth_profiles",
    "BottleneckDetector",
    "Bottleneck",
    "BottleneckType",
    "BottleneckSeverity",
    # Phase 2: Visualization
    "BandwidthVisualizer",
    "print_bandwidth_chart",
    "print_comparison",
    "print_dashboard",
    "ReportGenerator",
    "generate_html_report",
    # Phase 4: Hardware Validation
    # "HardwareValidator",  # Import directly if needed
    # Phase 4: Memory Coalescing
    "MemoryAccessCoalescer",
    "CoalescedKVCache",
    "CoalescingStats",
    # Phase 3: Optimization
    "OptimizationEngine",
    "OptimizationResult",
    "LazyKVCache",
    "LazyTensorAllocator",
    "run_optimization_demo",
    "PagedKVCache",
    "CacheStats",
    # Exceptions
    "CacheEvictionError",
    "QueueFullError",
    "ResourceExhaustedError",
    "RequestRejectedError",
]
