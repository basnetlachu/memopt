"""
GPU memory management

PURPOSE: Manage GPU memory to prevent out-of-memory errors
WHY: GPU memory is limited and must be carefully managed
WHEN TO USE: Initialize with model, check memory before/after operations

Features:
- Real-time memory monitoring
- Automatic cache clearing when threshold exceeded
- Memory statistics and reporting
- OOM (Out Of Memory) prevention

Usage:
    memory_manager = MemoryManager(device="cuda", memory_threshold=0.9)
    
    # Before operation
    memory_manager.check_memory()
    
    # Ensure enough memory
    memory_manager.ensure_memory(required_gb=5.0)
    
    # After operation
    memory_manager.clear_cache()
"""

import torch
from typing import Optional, Dict
from memopt.monitoring.logger import get_logger
from memopt.utils.errors import OutOfMemoryError

logger = get_logger(__name__)


class MemoryManager:
    """
    Manages GPU memory allocation and cleanup
    
    WHY THIS IS CRITICAL:
    - GPUs have limited memory (8GB, 12GB, 24GB typical)
    - Running out of memory crashes the program
    - Fragmented memory reduces efficiency
    - Proper management = stable production system
    """
    
    def __init__(
        self,
        device: str = "cuda",
        memory_threshold: float = 0.9,
        auto_clear_cache: bool = True
    ):
        """
        Initialize memory manager
        
        Args:
            device: Device to manage (e.g., 'cuda', 'cuda:0')
            memory_threshold: Alert when memory exceeds this (0.0-1.0)
                             0.9 = alert at 90% usage
            auto_clear_cache: Automatically clear cache when threshold exceeded
        """
        self.device = device
        self.memory_threshold = memory_threshold
        self.auto_clear_cache = auto_clear_cache
        
        # Check if CUDA is available
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, memory management disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Get device index from device string
        if device.startswith('cuda:'):
            self.device_id = int(device.split(':')[1])
        else:
            self.device_id = 0
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(self.device_id)
        self.total_memory = props.total_memory  # Total GPU memory in bytes
        self.device_name = props.name
        
        logger.info(f"Memory manager initialized for {self.device_name}")
        logger.info(f"Total memory: {self.total_memory / 1e9:.2f} GB")
        logger.info(f"Memory threshold: {memory_threshold * 100}%")
    
    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get current memory statistics
        
        Returns:
            Dictionary with:
            - allocated_gb: Currently allocated memory
            - reserved_gb: Reserved by PyTorch (cached)
            - total_gb: Total GPU memory
            - free_gb: Available memory
            - utilization: Fraction of memory in use (0.0-1.0)
        """
        if not self.enabled:
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'free_gb': 0.0,
                'utilization': 0.0,
            }
        
        # Get PyTorch memory stats
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        free = self.total_memory - allocated
        
        return {
            'allocated_gb': allocated / 1e9,
            'reserved_gb': reserved / 1e9,
            'total_gb': self.total_memory / 1e9,
            'free_gb': free / 1e9,
            'utilization': allocated / self.total_memory,
        }
    
    def check_memory(self, log_stats: bool = False) -> bool:
        """
        Check memory usage and alert if high
        
        Args:
            log_stats: Whether to log current stats
            
        Returns:
            True if memory usage is OK (below threshold)
            False if memory usage is high (above threshold)
        """
        if not self.enabled:
            return True
        
        stats = self.get_memory_stats()
        
        if log_stats:
            logger.debug(
                f"Memory: {stats['allocated_gb']:.2f} GB / "
                f"{stats['total_gb']:.2f} GB ({stats['utilization']*100:.1f}%)"
            )
        
        # Check if above threshold
        if stats['utilization'] > self.memory_threshold:
            logger.warning(
                f"High memory usage: {stats['utilization']*100:.1f}% "
                f"({stats['allocated_gb']:.2f} / {stats['total_gb']:.2f} GB)"
            )
            
            # Auto-clear if enabled
            if self.auto_clear_cache:
                logger.info("Auto-clearing cache...")
                self.clear_cache()
            
            return False
        
        return True
    
    def clear_cache(self) -> float:
        """
        Clear GPU cache
        
        WHY: PyTorch caches memory for efficiency, but this can 
        prevent other operations from allocating memory.
        Clearing cache frees this reserved memory.
        
        Returns:
            Amount of memory freed in GB
        """
        if not self.enabled:
            return 0.0
        
        before = torch.cuda.memory_allocated(self.device_id)
        torch.cuda.empty_cache()  # Free cached memory
        after = torch.cuda.memory_allocated(self.device_id)
        
        freed = (before - after) / 1e9
        
        if freed > 0.1:  # Log if freed > 100 MB
            logger.info(f"Cleared cache, freed {freed:.2f} GB")
        
        return freed
    
    def reset_peak_stats(self):
        """
        Reset peak memory statistics
        
        WHY: Useful to measure memory usage for a specific operation
        """
        if self.enabled:
            torch.cuda.reset_peak_memory_stats(self.device_id)
            logger.debug("Reset peak memory statistics")
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage since last reset (in GB)"""
        if not self.enabled:
            return 0.0
        
        peak = torch.cuda.max_memory_allocated(self.device_id)
        return peak / 1e9
    
    def has_available_memory(self, required_gb: float) -> bool:
        """
        Check if required memory is available
        
        Args:
            required_gb: Required memory in GB
            
        Returns:
            True if enough memory available
        """
        if not self.enabled:
            return True
        
        stats = self.get_memory_stats()
        return stats['free_gb'] >= required_gb
    
    def ensure_memory(self, required_gb: float) -> None:
        """
        Ensure required memory is available
        
        Tries to free memory if needed. Raises error if still insufficient.
        
        Args:
            required_gb: Required memory in GB
            
        Raises:
            OutOfMemoryError: If insufficient memory even after cleanup
        """
        if not self.enabled:
            return
        
        # Check if already available
        if self.has_available_memory(required_gb):
            return
        
        logger.info(f"Insufficient memory, attempting to free {required_gb:.2f} GB")
        
        # Try clearing cache
        freed = self.clear_cache()
        
        # Check again
        if self.has_available_memory(required_gb):
            logger.info(f"Successfully freed {freed:.2f} GB")
            return
        
        # Still not enough - raise error
        stats = self.get_memory_stats()
        raise OutOfMemoryError(
            f"Insufficient GPU memory",
            {
                'required_gb': required_gb,
                'available_gb': stats['free_gb'],
                'total_gb': stats['total_gb'],
                'utilized': f"{stats['utilization']*100:.1f}%"
            }
        )
    
    def log_memory_summary(self):
        """Log detailed memory summary (useful for debugging)"""
        if not self.enabled:
            logger.info("Memory management disabled (no CUDA)")
            return
        
        stats = self.get_memory_stats()
        peak = self.get_peak_memory()
        
        logger.info("="*60)
        logger.info("MEMORY SUMMARY")
        logger.info("="*60)
        logger.info(f"Device: {self.device_name}")
        logger.info(f"Total:     {stats['total_gb']:.2f} GB")
        logger.info(f"Allocated: {stats['allocated_gb']:.2f} GB ({stats['utilization']*100:.1f}%)")
        logger.info(f"Reserved:  {stats['reserved_gb']:.2f} GB")
        logger.info(f"Free:      {stats['free_gb']:.2f} GB")
        logger.info(f"Peak:      {peak:.2f} GB")
        logger.info("="*60)