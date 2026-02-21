"""
Single source of truth for GPU L2 cache sizes.

All other modules that need L2 cache MB values must import from here.
Values are derived from GPU_SPECS in hardware_counters.py — do not add
a separate dict elsewhere.
"""
from __future__ import annotations

from typing import Dict

# Lazy import to avoid circular dependency (hardware_counters imports profiler modules)
def _build_l2_table() -> Dict[str, float]:
    from .hardware_counters import GPU_SPECS
    return {name: spec.l2_cache_mb for name, spec in GPU_SPECS.items()}


# Module-level dict built once at import time.
GPU_L2_CACHE_MB: Dict[str, float] = _build_l2_table()
