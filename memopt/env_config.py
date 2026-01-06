"""
Environment Configuration - Multi-Node Infrastructure

Centralized environment variable configuration for horizontal scaling.
Allows operators to override defaults without code changes.

DESIGN:
- All config reads from environment
- Fallback to sensible defaults
- No global mutable state

PERFORMANCE IMPACT: None (read once at startup)
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """
    Environment-driven configuration for multi-node deployment.
    
    All settings can be overridden via environment variables:
    - MEMOPT_NODE_ID: Unique node identifier
    - MEMOPT_MAX_QUEUE: Maximum queue depth
    - MEMOPT_WORKERS: Number of GPU workers
    - MEMOPT_MAX_BATCH: Maximum batch size
    - MEMOPT_HEALTH_INTERVAL: Health check interval (seconds)
    - PORT: HTTP health endpoint port
    - MEMOPT_ENABLE_HEALTH: Enable health endpoints (default: true)
    - MEMOPT_ENABLE_METRICS: Enable Prometheus metrics (default: false)
    """
    
    # Node identity
    node_id: str
    
    # Queue configuration
    max_queue_depth: int
    
    # Worker configuration
    num_workers: int
    
    # Batch configuration
    max_batch_size: int
    
    # Health monitoring
    health_check_interval_sec: int
    
    # HTTP endpoints
    http_port: int
    enable_health_endpoints: bool
    
    # Metrics
    enable_metrics: bool
    metrics_export_interval_sec: int
    
    @staticmethod
    def from_environment() -> 'EnvironmentConfig':
        """
        Load configuration from environment variables.
        
        Returns:
            EnvironmentConfig with values from env or defaults
        """
        return EnvironmentConfig(
            # Node identity (from node_identity module)
            node_id=os.getenv('MEMOPT_NODE_ID', os.uname().nodename),
            
            # Queue configuration
            max_queue_depth=int(os.getenv('MEMOPT_MAX_QUEUE', '5000')),
            
            # Worker configuration
            num_workers=int(os.getenv('MEMOPT_WORKERS', '1')),
            
            # Batch configuration
            max_batch_size=int(os.getenv('MEMOPT_MAX_BATCH', '32')),
            
            # Health monitoring
            health_check_interval_sec=int(os.getenv('MEMOPT_HEALTH_INTERVAL', '300')),
            
            # HTTP endpoints
            http_port=int(os.getenv('PORT', '8080')),
            enable_health_endpoints=os.getenv('MEMOPT_ENABLE_HEALTH', 'true').lower() == 'true',
            
            # Metrics
            enable_metrics=os.getenv('MEMOPT_ENABLE_METRICS', 'false').lower() == 'true',
            metrics_export_interval_sec=int(os.getenv('MEMOPT_METRICS_INTERVAL', '10')),
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for logging/debugging."""
        return {
            'node_id': self.node_id,
            'max_queue_depth': self.max_queue_depth,
            'num_workers': self.num_workers,
            'max_batch_size': self.max_batch_size,
            'health_check_interval_sec': self.health_check_interval_sec,
            'http_port': self.http_port,
            'enable_health_endpoints': self.enable_health_endpoints,
            'enable_metrics': self.enable_metrics,
        }


def get_env_config() -> EnvironmentConfig:
    """Get environment configuration (convenience function)."""
    return EnvironmentConfig.from_environment()
