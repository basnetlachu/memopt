"""
Phase 3: Distributed State Management

Provides distributed coordination for multi-node deployments.
Supports etcd-based state synchronization, leader election, and distributed locks.
"""

import time
import json
from typing import Dict, Optional, Any, Callable, List
from dataclasses import dataclass, asdict
from threading import Lock, Event
from enum import Enum
import socket


class NodeRole(Enum):
    """Node roles in distributed system."""
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"


@dataclass
class NodeInfo:
    """Information about a cluster node."""
    node_id: str
    hostname: str
    role: NodeRole
    gpu_count: int
    memory_gb: float
    last_heartbeat: float
    state: str = "active"  # active, draining, offline

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        data["role"] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "NodeInfo":
        """Create from dictionary."""
        data["role"] = NodeRole(data["role"])
        return cls(**data)


@dataclass
class ClusterState:
    """Global cluster state."""
    total_nodes: int
    active_nodes: int
    total_gpus: int
    total_memory_gb: float
    leader_node_id: Optional[str]
    nodes: Dict[str, NodeInfo]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_nodes": self.total_nodes,
            "active_nodes": self.active_nodes,
            "total_gpus": self.total_gpus,
            "total_memory_gb": self.total_memory_gb,
            "leader_node_id": self.leader_node_id,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()}
        }


class DistributedStateBackend:
    """
    Abstract backend for distributed state storage.

    Implementations: InMemory (testing), Etcd (production), Redis (alternative)
    """

    def get(self, key: str) -> Optional[str]:
        """Get value for key."""
        raise NotImplementedError

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        """Delete key."""
        raise NotImplementedError

    def watch(self, key: str, callback: Callable[[str, str], None]):
        """Watch key for changes."""
        raise NotImplementedError

    def compare_and_swap(self, key: str, old_value: str, new_value: str) -> bool:
        """Atomic compare-and-swap operation."""
        raise NotImplementedError

    def list_keys(self, prefix: str) -> List[str]:
        """List all keys with prefix."""
        raise NotImplementedError


class InMemoryBackend(DistributedStateBackend):
    """
    In-memory backend for testing.

    NOT FOR PRODUCTION - single node only.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._data: Dict[str, str] = {}
        self._ttls: Dict[str, float] = {}
        self._lock = Lock()
        self._watchers: Dict[str, List[Callable]] = {}

    def get(self, key: str) -> Optional[str]:
        """Get value for key."""
        with self._lock:
            # Check TTL
            if key in self._ttls and time.time() > self._ttls[key]:
                del self._data[key]
                del self._ttls[key]
                return None
            return self._data.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set key-value pair."""
        with self._lock:
            old_value = self._data.get(key)
            self._data[key] = value

            if ttl is not None:
                self._ttls[key] = time.time() + ttl

            # Trigger watchers
            if key in self._watchers:
                for callback in self._watchers[key]:
                    try:
                        callback(key, value)
                    except Exception:
                        pass

            return True

    def delete(self, key: str) -> bool:
        """Delete key."""
        with self._lock:
            if key in self._data:
                del self._data[key]
                if key in self._ttls:
                    del self._ttls[key]
                return True
            return False

    def watch(self, key: str, callback: Callable[[str, str], None]):
        """Watch key for changes."""
        with self._lock:
            if key not in self._watchers:
                self._watchers[key] = []
            self._watchers[key].append(callback)

    def compare_and_swap(self, key: str, old_value: str, new_value: str) -> bool:
        """Atomic compare-and-swap."""
        with self._lock:
            current = self._data.get(key)
            if current == old_value:
                self._data[key] = new_value
                return True
            return False

    def list_keys(self, prefix: str) -> List[str]:
        """List keys with prefix."""
        with self._lock:
            return [k for k in self._data.keys() if k.startswith(prefix)]


class DistributedStateManager:
    """
    Phase 3: Distributed state manager for multi-node coordination.

    Provides:
    - Node registration and discovery
    - Cluster state tracking
    - Key-value storage with TTL
    - Watch/notification system
    """

    def __init__(
        self,
        node_id: str,
        backend: DistributedStateBackend,
        heartbeat_interval: int = 5,
        heartbeat_timeout: int = 15
    ):
        """
        Args:
            node_id: Unique node identifier
            backend: State storage backend
            heartbeat_interval: Seconds between heartbeats
            heartbeat_timeout: Seconds before node considered dead
        """
        self.node_id = node_id
        self.backend = backend
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout

        # Node info
        self.hostname = socket.gethostname()
        self.node_info: Optional[NodeInfo] = None

        # Heartbeat thread control
        self._heartbeat_running = False
        self._heartbeat_event = Event()

    def register_node(
        self,
        gpu_count: int,
        memory_gb: float,
        role: NodeRole = NodeRole.FOLLOWER
    ) -> bool:
        """
        Register this node in the cluster.

        Args:
            gpu_count: Number of GPUs on this node
            memory_gb: Total memory in GB
            role: Initial role

        Returns:
            True if registration successful
        """
        self.node_info = NodeInfo(
            node_id=self.node_id,
            hostname=self.hostname,
            role=role,
            gpu_count=gpu_count,
            memory_gb=memory_gb,
            last_heartbeat=time.time(),
            state="active"
        )

        # Store in distributed state
        key = f"/memopt/nodes/{self.node_id}"
        value = json.dumps(self.node_info.to_dict())

        return self.backend.set(key, value, ttl=self.heartbeat_timeout)

    def unregister_node(self) -> bool:
        """
        Unregister this node from the cluster.

        Returns:
            True if unregistration successful
        """
        self._heartbeat_running = False
        self._heartbeat_event.set()

        key = f"/memopt/nodes/{self.node_id}"
        return self.backend.delete(key)

    def heartbeat(self) -> bool:
        """
        Send heartbeat to update node liveness.

        Returns:
            True if heartbeat successful
        """
        if self.node_info is None:
            return False

        self.node_info.last_heartbeat = time.time()

        key = f"/memopt/nodes/{self.node_id}"
        value = json.dumps(self.node_info.to_dict())

        return self.backend.set(key, value, ttl=self.heartbeat_timeout)

    def start_heartbeat(self):
        """Start background heartbeat thread."""
        import threading

        self._heartbeat_running = True

        def heartbeat_loop():
            while self._heartbeat_running:
                self.heartbeat()
                self._heartbeat_event.wait(self.heartbeat_interval)

        thread = threading.Thread(target=heartbeat_loop, daemon=True)
        thread.start()

    def stop_heartbeat(self):
        """Stop background heartbeat thread."""
        self._heartbeat_running = False
        self._heartbeat_event.set()

    def get_cluster_state(self) -> ClusterState:
        """
        Get current cluster state.

        Returns:
            ClusterState with all node information
        """
        # List all nodes
        node_keys = self.backend.list_keys("/memopt/nodes/")

        nodes: Dict[str, NodeInfo] = {}
        total_gpus = 0
        total_memory = 0.0
        active_nodes = 0
        leader_id = None

        for key in node_keys:
            value = self.backend.get(key)
            if value:
                try:
                    node_data = json.loads(value)
                    node = NodeInfo.from_dict(node_data)
                    nodes[node.node_id] = node

                    if node.state == "active":
                        active_nodes += 1
                        total_gpus += node.gpu_count
                        total_memory += node.memory_gb

                    if node.role == NodeRole.LEADER:
                        leader_id = node.node_id

                except Exception:
                    continue

        return ClusterState(
            total_nodes=len(nodes),
            active_nodes=active_nodes,
            total_gpus=total_gpus,
            total_memory_gb=total_memory,
            leader_node_id=leader_id,
            nodes=nodes
        )

    def set_node_state(self, state: str) -> bool:
        """
        Update node state.

        Args:
            state: New state (active, draining, offline)

        Returns:
            True if update successful
        """
        if self.node_info is None:
            return False

        self.node_info.state = state

        key = f"/memopt/nodes/{self.node_id}"
        value = json.dumps(self.node_info.to_dict())

        return self.backend.set(key, value, ttl=self.heartbeat_timeout)

    def set_node_role(self, role: NodeRole) -> bool:
        """
        Update node role.

        Args:
            role: New role

        Returns:
            True if update successful
        """
        if self.node_info is None:
            return False

        self.node_info.role = role

        key = f"/memopt/nodes/{self.node_id}"
        value = json.dumps(self.node_info.to_dict())

        return self.backend.set(key, value, ttl=self.heartbeat_timeout)

    # Key-value operations

    def get(self, key: str) -> Optional[str]:
        """Get value from distributed state."""
        return self.backend.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in distributed state."""
        return self.backend.set(key, value, ttl=ttl)

    def delete(self, key: str) -> bool:
        """Delete key from distributed state."""
        return self.backend.delete(key)

    def watch(self, key: str, callback: Callable[[str, str], None]):
        """Watch key for changes."""
        self.backend.watch(key, callback)


# ============================================================================
# Phase 3: Global State Manager
# ============================================================================

_global_state_manager: Optional[DistributedStateManager] = None


def get_state_manager() -> Optional[DistributedStateManager]:
    """Get global state manager."""
    return _global_state_manager


def init_state_manager(
    node_id: str,
    backend: Optional[DistributedStateBackend] = None,
    gpu_count: int = 1,
    memory_gb: float = 16.0
) -> DistributedStateManager:
    """
    Initialize global state manager.

    Args:
        node_id: Unique node identifier
        backend: State backend (defaults to runtime-selected backend)
        gpu_count: Number of GPUs on this node
        memory_gb: Total memory in GB

    Returns:
        DistributedStateManager instance
    """
    global _global_state_manager

    if backend is None:
        # Production readiness: Use runtime factory to select backend
        # Dev mode → InMemoryBackend, Prod mode → RedisBackend
        try:
            from memopt.runtime import create_distributed_state_backend
            backend = create_distributed_state_backend()
        except ImportError:
            # Fallback for standalone usage without runtime module
            backend = InMemoryBackend()

    _global_state_manager = DistributedStateManager(
        node_id=node_id,
        backend=backend
    )

    # Register node
    _global_state_manager.register_node(
        gpu_count=gpu_count,
        memory_gb=memory_gb
    )

    # Start heartbeat
    _global_state_manager.start_heartbeat()

    return _global_state_manager
