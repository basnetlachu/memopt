"""
Configuration management for MemOpt

PURPOSE: Central configuration for all MemOpt settings
WHY: Makes it easy to adjust behavior without changing code
WHEN TO USE: Import and use get_default_config()
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MemOptConfig:
    """
    Main configuration for MemOpt
    
    All configuration parameters with defaults.
    Adjust these to tune performance vs accuracy.
    """
    
    # Model configuration
    optimization_level: str = "high"  # conservative, balanced, high, aggressive
    quantize_kv_cache: bool = True
    use_flash_attention: bool = True
    use_paged_cache: bool = True
    
    # Memory configuration
    max_memory_gb: Optional[float] = None  # Auto-detect if None
    memory_threshold: float = 0.9  # Alert at 90% usage
    kv_cache_block_size: int = 16
    max_kv_cache_blocks: int = 4096
    
    # Performance configuration
    num_gpus: int = 1
    enable_multi_gpu: bool = False
    
    # Monitoring configuration
    enable_metrics: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    enable_profiling: bool = False
    
    # Safety configuration
    max_requests_per_minute: int = 1000
    request_timeout_seconds: int = 300
    enable_input_validation: bool = True


def get_default_config() -> MemOptConfig:
    """Get default configuration"""
    return MemOptConfig()