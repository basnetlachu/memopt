"""
MemOpt - Production-Ready GPU Optimization
"""

__version__ = "1.0.0"

# Import from core.model (since your model.py is in core/)
try:
    from .core.model import OptimizedLLM
except ImportError:
    try:
        from memopt.core.model import OptimizedLLM
    except ImportError:
        # Fallback to regular location
        try:
            from memopt.core.model import OptimizedLLM
        except ImportError:
            from memopt.core.model import OptimizedLLM

# Import utilities
from .utils.errors import (
    MemOptError,
    ModelLoadError,
    GenerationError,
    OutOfMemoryError,
    InvalidInputError,
)

from .monitoring.logger import get_logger
from .core.memory_manager import MemoryManager
from .monitoring.metrics import MetricsCollector, ProfileStats

__all__ = [
    'OptimizedLLM',
    'MemOptError',
    'ModelLoadError',
    'GenerationError',
    'OutOfMemoryError',
    'InvalidInputError',
    'get_logger',
    'MemoryManager',
    'MetricsCollector',
    'ProfileStats',
]