"""
Redis-based service registry for worker discovery and health tracking.
"""
import json
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import redis
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkerInfo:
    """Information about a registered worker."""
    worker_id: str
    host: str
    port: int
    gpu_id: int
    model_name: str

    # Real-time metrics
    tokens_per_second: float = 0.0
    queue_depth: int = 0
    current_batch_size: int = 0
    memory_used_gb: float = 0.0
    p95_latency_ms: float = 0.0

    # Health
    last_heartbeat: float = 0.0
    is_healthy: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "WorkerInfo":
        return cls(**data)

    @property
    def endpoint(self) -> str:
        """Get worker endpoint URL."""
        return f"http://{self.host}:{self.port}"

    @property
    def load_score(self) -> float:
        """
        Calculate load score for routing decisions.
        Lower is better.
        """
        # Normalize queue depth (0-1 range assuming max queue 128)
        queue_load = min(self.queue_depth / 128.0, 1.0)

        # Normalize batch size (0-1 range assuming max batch 32)
        batch_load = min(self.current_batch_size / 32.0, 1.0)

        # Combined load (weighted)
        return (queue_load * 0.7) + (batch_load * 0.3)


class WorkerRegistry:
    """
    Redis-based worker registry for service discovery.

    Workers register themselves and send periodic heartbeats.
    Routers query the registry to find available workers.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.redis = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.worker_key_prefix = "memopt:worker:"
        self.worker_list_key = "memopt:workers"
        self.session_key_prefix = "memopt:session:"

    def register_worker(self, worker_info: WorkerInfo) -> bool:
        """
        Register a worker in the registry.

        Returns:
            True if registration successful
        """
        try:
            worker_key = f"{self.worker_key_prefix}{worker_info.worker_id}"

            # Set worker info with TTL (expires if no heartbeat)
            self.redis.setex(
                worker_key,
                time=30,  # 30 second TTL
                value=json.dumps(worker_info.to_dict())
            )

            # Add to worker list (set for uniqueness)
            self.redis.sadd(self.worker_list_key, worker_info.worker_id)

            logger.info(f"Registered worker {worker_info.worker_id} at {worker_info.endpoint}")
            return True

        except Exception as e:
            logger.error(f"Failed to register worker: {e}")
            return False

    def update_worker_metrics(
        self,
        worker_id: str,
        tokens_per_second: float = 0.0,
        queue_depth: int = 0,
        current_batch_size: int = 0,
        memory_used_gb: float = 0.0,
        p95_latency_ms: float = 0.0
    ) -> bool:
        """Update worker metrics (called during heartbeat)."""
        try:
            worker_key = f"{self.worker_key_prefix}{worker_id}"
            worker_data = self.redis.get(worker_key)

            if not worker_data:
                logger.warning(f"Worker {worker_id} not found in registry")
                return False

            worker_info = WorkerInfo.from_dict(json.loads(worker_data))

            # Update metrics
            worker_info.tokens_per_second = tokens_per_second
            worker_info.queue_depth = queue_depth
            worker_info.current_batch_size = current_batch_size
            worker_info.memory_used_gb = memory_used_gb
            worker_info.p95_latency_ms = p95_latency_ms
            worker_info.last_heartbeat = time.time()
            worker_info.is_healthy = True

            # Save with refreshed TTL
            self.redis.setex(
                worker_key,
                time=30,
                value=json.dumps(worker_info.to_dict())
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update worker metrics: {e}")
            return False

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get information about a specific worker."""
        try:
            worker_key = f"{self.worker_key_prefix}{worker_id}"
            worker_data = self.redis.get(worker_key)

            if not worker_data:
                return None

            return WorkerInfo.from_dict(json.loads(worker_data))

        except Exception as e:
            logger.error(f"Failed to get worker: {e}")
            return None

    def get_all_workers(self, only_healthy: bool = True) -> List[WorkerInfo]:
        """Get list of all registered workers."""
        try:
            worker_ids = self.redis.smembers(self.worker_list_key)
            workers = []

            current_time = time.time()

            for worker_id in worker_ids:
                worker = self.get_worker(worker_id)
                if worker:
                    # Check if heartbeat is recent (within 15 seconds)
                    if current_time - worker.last_heartbeat > 15:
                        worker.is_healthy = False

                    if only_healthy and not worker.is_healthy:
                        continue

                    workers.append(worker)
                else:
                    # Worker expired, remove from list
                    self.redis.srem(self.worker_list_key, worker_id)

            return workers

        except Exception as e:
            logger.error(f"Failed to get workers: {e}")
            return []

    def deregister_worker(self, worker_id: str) -> bool:
        """Remove a worker from the registry."""
        try:
            worker_key = f"{self.worker_key_prefix}{worker_id}"
            self.redis.delete(worker_key)
            self.redis.srem(self.worker_list_key, worker_id)

            logger.info(f"Deregistered worker {worker_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to deregister worker: {e}")
            return False

    def set_session_affinity(self, session_id: str, worker_id: str, ttl_seconds: int = 3600):
        """Map a session to a specific worker for prefix caching."""
        try:
            session_key = f"{self.session_key_prefix}{session_id}"
            self.redis.setex(session_key, time=ttl_seconds, value=worker_id)
            return True
        except Exception as e:
            logger.error(f"Failed to set session affinity: {e}")
            return False

    def get_session_worker(self, session_id: str) -> Optional[str]:
        """Get the worker associated with a session."""
        try:
            session_key = f"{self.session_key_prefix}{session_id}"
            return self.redis.get(session_key)
        except Exception as e:
            logger.error(f"Failed to get session worker: {e}")
            return None

    def get_cluster_stats(self) -> Dict:
        """Get aggregate cluster statistics."""
        workers = self.get_all_workers(only_healthy=True)

        total_workers = len(workers)
        total_throughput = sum(w.tokens_per_second for w in workers)
        total_queue_depth = sum(w.queue_depth for w in workers)
        avg_p95_latency = sum(w.p95_latency_ms for w in workers) / total_workers if total_workers > 0 else 0

        return {
            "total_workers": total_workers,
            "total_throughput_tokens_per_second": total_throughput,
            "total_queue_depth": total_queue_depth,
            "average_p95_latency_ms": avg_p95_latency,
            "workers": [w.to_dict() for w in workers]
        }
