"""
Memory Optimization Module

Provides memory access coalescing for PyTorch models.
"""

from .memory_coalescer import MemoryCoalescer, CoalescingConfig, CoalescingStats

__all__ = ['MemoryCoalescer', 'CoalescingConfig', 'CoalescingStats']
