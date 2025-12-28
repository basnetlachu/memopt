"""
Phase 3: Cluster-Wide Resource Tracking

Tracks and coordinates resources across all nodes in the cluster.
Enables global scheduling, load balancing, and capacity planning.
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from threading import Lock
from Memopt.distributed_state import DistributedStateBackend, NodeInfo
import json


@dataclass
class NodeResources:
    """Resource state for a single node."""
    node_id: str
    timestamp: float

    # GPU resources
    gpu_count: int
    gpu_memory_total_gb: float
    gpu_memory_used_gb: float
    gpu_utilization_percent: float

    # CPU resources
    cpu_count: int
    cpu_utilization_percent: float

    # Request queue
    queue_depth: int
    queue_capacity: int

    # KV cache
    cache_blocks_used: int
    cache_blocks_total: int

    # Performance
    active_requests: int
    throughput_tokens_per_sec: float
    p95_latency_ms: float

    @property
    def gpu_memory_free_gb(self) -> float:
        """Get free GPU memory."""
        return self.gpu_memory_total_gb - self.gpu_memory_used_gb

    @property
    def gpu_memory_utilization(self) -> float:
        """Get GPU memory utilization ratio."""
        return self.gpu_memory_used_gb / max(self.gpu_memory_total_gb, 1)

    @property
    def queue_utilization(self) -> float:
        """Get queue utilization ratio."""
        return self.queue_depth / max(self.queue_capacity, 1)

    @property
    def cache_utilization(self) -> float:
        """Get cache utilization ratio."""
        return self.cache_blocks_used / max(self.cache_blocks_total, 1)

    @property
    def load_score(self) -> float:
        """
        Calculate node load score (0.0 = idle, 1.0 = fully loaded).

        Combines queue, cache, and GPU utilization.
        """
        return (
            self.queue_utilization * 0.4 +
            self.cache_utilization * 0.3 +
            self.gpu_memory_utilization * 0.3
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "NodeResources":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ClusterResources:
    """Aggregated cluster-wide resources."""
    timestamp: float
    total_nodes: int
    active_nodes: int

    # Aggregated GPU
    total_gpus: int
    total_gpu_memory_gb: float
    used_gpu_memory_gb: float

    # Aggregated capacity
    total_queue_capacity: int
    total_queue_depth: int
    total_cache_blocks: int
    total_cache_used_blocks: int

    # Aggregated performance
    total_active_requests: int
    cluster_throughput_tokens_per_sec: float
    avg_p95_latency_ms: float

    # Node details
    nodes: Dict[str, NodeResources]

    @property
    def cluster_gpu_utilization(self) -> float:
        """Get cluster-wide GPU utilization."""
        return self.used_gpu_memory_gb / max(self.total_gpu_memory_gb, 1)

    @property
    def cluster_queue_utilization(self) -> float:
        """Get cluster-wide queue utilization."""
        return self.total_queue_depth / max(self.total_queue_capacity, 1)

    @property
    def cluster_cache_utilization(self) -> float:
        """Get cluster-wide cache utilization."""
        return self.total_cache_used_blocks / max(self.total_cache_blocks, 1)

    def get_least_loaded_node(self) -> Optional[str]:
        """Get ID of least loaded node."""
        if not self.nodes:
            return None

        return min(self.nodes.items(), key=lambda x: x[1].load_score)[0]

    def get_nodes_by_load(self) -> List[str]:
        """Get node IDs sorted by load (least to most)."""
        return sorted(self.nodes.keys(), key=lambda nid: self.nodes[nid].load_score)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["nodes"] = {k: v.to_dict() for k, v in self.nodes.items()}
        return data


class ClusterResourceTracker:
    """
    Phase 3: Tracks resources across all cluster nodes.

    The leader aggregates resource information from all nodes
    for global scheduling and load balancing decisions.
    """

    def __init__(
        self,
        node_id: str,
        backend: DistributedStateBackend,
        update_interval: float = 1.0
    ):
        """
        Args:
            node_id: This node's identifier
            backend: Distributed state backend
            update_interval: Seconds between updates
        """
        self.node_id = node_id
        self.backend = backend
        self.update_interval = update_interval

        self._lock = Lock()
        self._last_update = 0.0

    def update_node_resources(self, resources: NodeResources):
        """
        Update this node's resource state.

        Args:
            resources: Current resource state
        """
        key = f"/Memopt/resources/nodes/{self.node_id}"
        value = json.dumps(resources.to_dict())

        # Store with TTL (2x update interval for fault tolerance)
        ttl = int(self.update_interval * 2)
        self.backend.set(key, value, ttl=ttl)

        self._last_update = time.time()

    def get_cluster_resources(self) -> ClusterResources:
        """
        Get aggregated cluster resource state.

        Returns:
            ClusterResources with all node information
        """
        # List all node resource keys
        node_keys = self.backend.list_keys("/Memopt/resources/nodes/")

        nodes: Dict[str, NodeResources] = {}
        total_gpus = 0
        total_gpu_memory = 0.0
        used_gpu_memory = 0.0
        total_queue_capacity = 0
        total_queue_depth = 0
        total_cache_blocks = 0
        total_cache_used = 0
        total_active_requests = 0
        total_throughput = 0.0
        latency_sum = 0.0
        latency_count = 0

        for key in node_keys:
            value = self.backend.get(key)
            if value:
                try:
                    resource_data = json.loads(value)
                    resource = NodeResources.from_dict(resource_data)
                    nodes[resource.node_id] = resource

                    # Aggregate
                    total_gpus += resource.gpu_count
                    total_gpu_memory += resource.gpu_memory_total_gb
                    used_gpu_memory += resource.gpu_memory_used_gb
                    total_queue_capacity += resource.queue_capacity
                    total_queue_depth += resource.queue_depth
                    total_cache_blocks += resource.cache_blocks_total
                    total_cache_used += resource.cache_blocks_used
                    total_active_requests += resource.active_requests
                    total_throughput += resource.throughput_tokens_per_sec

                    if resource.p95_latency_ms > 0:
                        latency_sum += resource.p95_latency_ms
                        latency_count += 1

                except Exception:
                    continue

        avg_latency = (latency_sum / latency_count) if latency_count > 0 else 0.0

        return ClusterResources(
            timestamp=time.time(),
            total_nodes=len(nodes),
            active_nodes=len(nodes),
            total_gpus=total_gpus,
            total_gpu_memory_gb=total_gpu_memory,
            used_gpu_memory_gb=used_gpu_memory,
            total_queue_capacity=total_queue_capacity,
            total_queue_depth=total_queue_depth,
            total_cache_blocks=total_cache_blocks,
            total_cache_used_blocks=total_cache_used,
            total_active_requests=total_active_requests,
            cluster_throughput_tokens_per_sec=total_throughput,
            avg_p95_latency_ms=avg_latency,
            nodes=nodes
        )

    def get_node_resources(self, node_id: str) -> Optional[NodeResources]:
        """
        Get resources for a specific node.

        Args:
            node_id: Node identifier

        Returns:
            NodeResources or None if not found
        """
        key = f"/Memopt/resources/nodes/{node_id}"
        value = self.backend.get(key)

        if value:
            try:
                data = json.loads(value)
                return NodeResources.from_dict(data)
            except Exception:
                return None

        return None


# ============================================================================
# Phase 3: Global Scheduler
# ============================================================================

class GlobalScheduler:
    """
    Phase 3: Global request scheduler for multi-node clusters.

    The leader node runs the global scheduler to distribute
    requests across nodes based on current load.
    """

    def __init__(
        self,
        resource_tracker: ClusterResourceTracker,
        scheduling_policy: str = "least_loaded"
    ):
        """
        Args:
            resource_tracker: Cluster resource tracker
            scheduling_policy: Scheduling algorithm
                - "least_loaded": Route to least loaded node
                - "round_robin": Rotate through nodes
                - "random": Random selection
        """
        self.resource_tracker = resource_tracker
        self.scheduling_policy = scheduling_policy

        self._round_robin_index = 0
        self._lock = Lock()

    def select_node(self) -> Optional[str]:
        """
        Select best node for next request.

        Returns:
            Node ID or None if no capacity
        """
        cluster = self.resource_tracker.get_cluster_resources()

        if cluster.active_nodes == 0:
            return None

        if self.scheduling_policy == "least_loaded":
            return cluster.get_least_loaded_node()

        elif self.scheduling_policy == "round_robin":
            with self._lock:
                nodes = list(cluster.nodes.keys())
                if not nodes:
                    return None

                node = nodes[self._round_robin_index % len(nodes)]
                self._round_robin_index += 1
                return node

        elif self.scheduling_policy == "random":
            import random
            nodes = list(cluster.nodes.keys())
            return random.choice(nodes) if nodes else None

        return None

    def can_accept_request(self, tokens_estimate: int = 256) -> bool:
        """
        Check if cluster has capacity for request.

        Args:
            tokens_estimate: Estimated tokens for request

        Returns:
            True if cluster can accept request
        """
        cluster = self.resource_tracker.get_cluster_resources()

        # Check queue capacity
        if cluster.cluster_queue_utilization >= 0.95:
            return False

        # Check cache capacity
        if cluster.cluster_cache_utilization >= 0.95:
            return False

        return True

    def get_cluster_stats(self) -> dict:
        """Get cluster statistics for monitoring."""
        cluster = self.resource_tracker.get_cluster_resources()

        return {
            "total_nodes": cluster.total_nodes,
            "active_nodes": cluster.active_nodes,
            "total_gpus": cluster.total_gpus,
            "gpu_utilization": cluster.cluster_gpu_utilization,
            "queue_utilization": cluster.cluster_queue_utilization,
            "cache_utilization": cluster.cluster_cache_utilization,
            "total_active_requests": cluster.total_active_requests,
            "cluster_throughput": cluster.cluster_throughput_tokens_per_sec,
            "avg_p95_latency_ms": cluster.avg_p95_latency_ms
        }
