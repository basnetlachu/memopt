"""
Configuration management for production serving.
"""
import os
from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class WorkerConfig:
    """Configuration for a single worker."""

    # Model configuration
    model_name: str = "gpt2-xl"
    optimization_level: str = "batch"

    # Worker configuration
    worker_id: int = 0
    gpu_id: int = 0
    host: str = "0.0.0.0"
    port: int = 9000

    # Performance tuning
    max_batch_size: int = 32
    max_queue_size: int = 128
    batch_timeout_ms: int = 10  # Wait time for batching
    max_tokens_per_request: int = 2048

    # Registry configuration
    registry_host: str = "localhost"
    registry_port: int = 6379
    registry_db: int = 0
    heartbeat_interval_seconds: int = 5

    # Metrics
    enable_prometheus: bool = True
    prometheus_port: int = 9100

    @classmethod
    def from_yaml(cls, path: str) -> "WorkerConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    @classmethod
    def from_env(cls) -> "WorkerConfig":
        """Load configuration from environment variables."""
        return cls(
            model_name=os.getenv("MODEL_NAME", "gpt2-xl"),
            optimization_level=os.getenv("OPTIMIZATION_LEVEL", "batch"),
            worker_id=int(os.getenv("WORKER_ID", "0")),
            gpu_id=int(os.getenv("GPU_ID", "0")),
            host=os.getenv("WORKER_HOST", "0.0.0.0"),
            port=int(os.getenv("WORKER_PORT", "9000")),
            max_batch_size=int(os.getenv("MAX_BATCH_SIZE", "32")),
            max_queue_size=int(os.getenv("MAX_QUEUE_SIZE", "128")),
            batch_timeout_ms=int(os.getenv("BATCH_TIMEOUT_MS", "10")),
            registry_host=os.getenv("REGISTRY_HOST", "localhost"),
            registry_port=int(os.getenv("REGISTRY_PORT", "6379")),
            heartbeat_interval_seconds=int(os.getenv("HEARTBEAT_INTERVAL", "5")),
        )


@dataclass
class RouterConfig:
    """Configuration for router service."""

    # Router configuration
    host: str = "0.0.0.0"
    port: int = 8000

    # Registry configuration
    registry_host: str = "localhost"
    registry_port: int = 6379
    registry_db: int = 0

    # Routing strategy
    routing_strategy: str = "least_loaded"  # round_robin, least_loaded, cost_aware
    enable_session_affinity: bool = True
    session_ttl_seconds: int = 3600

    # Health checks
    health_check_interval_seconds: int = 10
    unhealthy_threshold: int = 3  # Failed checks before marking unhealthy

    # Timeouts and limits
    request_timeout_seconds: int = 60
    max_retries: int = 2

    # Metrics
    enable_prometheus: bool = True
    prometheus_port: int = 8100

    @classmethod
    def from_yaml(cls, path: str) -> "RouterConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)

    @classmethod
    def from_env(cls) -> "RouterConfig":
        """Load configuration from environment variables."""
        return cls(
            host=os.getenv("ROUTER_HOST", "0.0.0.0"),
            port=int(os.getenv("ROUTER_PORT", "8000")),
            registry_host=os.getenv("REGISTRY_HOST", "localhost"),
            registry_port=int(os.getenv("REGISTRY_PORT", "6379")),
            routing_strategy=os.getenv("ROUTING_STRATEGY", "least_loaded"),
            enable_session_affinity=os.getenv("ENABLE_SESSION_AFFINITY", "true").lower() == "true",
        )


@dataclass
class ClusterConfig:
    """Configuration for multi-node cluster."""

    # Cluster topology
    nodes: List[dict] = field(default_factory=list)  # [{"host": "10.0.1.1", "gpus": 8}, ...]

    # Shared configuration
    model_name: str = "gpt2-xl"
    optimization_level: str = "batch"

    # Registry
    registry_host: str = "redis.cluster.local"
    registry_port: int = 6379

    # Worker defaults
    worker_port_start: int = 9000
    max_batch_size: int = 32

    @classmethod
    def from_yaml(cls, path: str) -> "ClusterConfig":
        """Load cluster configuration from YAML file."""
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(**config)
