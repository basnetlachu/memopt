"""
Memory Pressure Monitoring for Trillion-Token Scale

Tracks GPU memory usage and calculates memory pressure (0.0 to 1.0).
Used by adaptive systems to make dynamic decisions:
- Sliding window size adjustment
- Speculative decoding enable/disable
- KV cache eviction policies
"""

import torch
from typing import Optional


class MemoryPressureMonitor:
    """
    Monitors GPU memory pressure in real-time.

    Memory pressure = used_memory / total_memory

    Production thresholds:
    - < 0.50: Low pressure (can increase windows, enable speculation)
    - 0.50-0.70: Normal pressure (maintain current settings)
    - 0.70-0.85: High pressure (reduce windows, conservative speculation)
    - > 0.85: Critical pressure (minimize windows, disable speculation)
    """

    def __init__(self, device: str = "cuda"):
        """
        Args:
            device: PyTorch device to monitor
        """
        self.device = device if isinstance(device, str) else str(device)

        # Track if CUDA is available
        self.cuda_available = torch.cuda.is_available()

        # Get device index
        if ":" in self.device:
            self.device_idx = int(self.device.split(":")[1])
        else:
            self.device_idx = 0

        # Cache total memory (constant)
        self._total_memory_gb = None
        if self.cuda_available:
            try:
                props = torch.cuda.get_device_properties(self.device_idx)
                self._total_memory_gb = props.total_memory / (1024**3)
            except:
                self._total_memory_gb = 0.0

    def get_memory_pressure(self) -> float:
        """
        Get current memory pressure (0.0 to 1.0).

        Returns:
            Memory pressure ratio. 0.0 = no usage, 1.0 = fully used
        """
        if not self.cuda_available:
            return 0.0

        try:
            # Get current memory usage
            allocated = torch.cuda.memory_allocated(self.device_idx)
            reserved = torch.cuda.memory_reserved(self.device_idx)

            # Use reserved (not allocated) as it's what's actually taken from GPU
            used_memory = reserved

            # Get total memory
            if self._total_memory_gb is None or self._total_memory_gb == 0:
                return 0.0

            total_memory = self._total_memory_gb * (1024**3)

            # Calculate pressure
            pressure = used_memory / total_memory

            return min(1.0, max(0.0, pressure))
        except Exception:
            # If we can't get memory stats, assume moderate pressure
            return 0.5

    def get_memory_stats(self) -> dict:
        """
        Get detailed memory statistics.

        Returns:
            Dictionary with memory stats in GB
        """
        if not self.cuda_available:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'free_gb': 0.0,
                'pressure': 0.0,
            }

        try:
            allocated = torch.cuda.memory_allocated(self.device_idx) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device_idx) / (1024**3)
            total = self._total_memory_gb or 0.0

            pressure = self.get_memory_pressure()

            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': max(0.0, total - reserved),
                'pressure': pressure,
            }
        except Exception:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'free_gb': 0.0,
                'pressure': 0.0,
            }

    def should_reduce_memory_usage(self) -> bool:
        """
        Check if memory usage should be reduced.

        Returns:
            True if memory pressure is high (> 0.70)
        """
        return self.get_memory_pressure() > 0.70

    def is_critical_pressure(self) -> bool:
        """
        Check if memory pressure is critical.

        Returns:
            True if memory pressure is critical (> 0.85)
        """
        return self.get_memory_pressure() > 0.85

    def can_increase_memory_usage(self) -> bool:
        """
        Check if memory usage can be increased safely.

        Returns:
            True if memory pressure is low (< 0.50)
        """
        return self.get_memory_pressure() < 0.50
