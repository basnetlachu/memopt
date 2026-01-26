"""
Hardware Validation Module

Provides Nsight Compute integration for hardware-level memory validation.
"""

from .hardware_validator import HardwareValidator, NsightMetrics
from .run_validation import validate_optimization, ValidationReport

__all__ = [
    'HardwareValidator',
    'NsightMetrics',
    'validate_optimization',
    'ValidationReport'
]
