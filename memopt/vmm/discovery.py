"""
Node discovery — cluster membership via Redis gossip or static hosts.

Each node registers its capabilities in Redis with a TTL. Peers are
discovered via non-blocking SCAN (not KEYS). Failed nodes expire
automatically via TTL.

Falls back to MEMOPT_NODE_HOSTS when Redis is unavailable.

Environment variables:
  MEMOPT_NODE_ID              node identifier (default: hostname)
  MEMOPT_HOST                 advertised host (default: auto-detect)
  MEMOPT_RACK                 rack identifier (default: "unknown")
  MEMOPT_POD                  pod identifier (default: "unknown")
  MEMOPT_REGION               region (default: "unknown")
  MEMOPT_DISCOVERY_TTL_S      node TTL in seconds (default: 30)
  MEMOPT_DISCOVERY_SCAN_S     scan interval in seconds (default: 10)
"""
from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeCapabilities:
    """What this node offers to the cluster."""

    def __init__(self):
        self.node_id: str = ""
        self.host: str = ""
        self.gossip_port: int = 18600
        self.rbp_port: int = 18516
        self.rack: str = "unknown"
        self.pod: str = "unknown"
        self.region: str = "unknown"
        self.gpu_count: int = 0
        self.hbm_total_gb: float = 0.0
        self.hbm_free_gb: float = 0.0
        self.memopt_version: str = "unknown"
        self.image_version: str = "unknown"
        self.git_commit: str = "unknown"
        self.image_source: str = "development"
        self.hardware_backend: str = "unknown"
        self.registered_at: float = 0.0
        self.last_heartbeat: float = 0.0

    def detect(self) -> None:
        """Detect capabilities from local hardware. Never raises."""
        try:
            self.node_id = os.environ.get(
                "MEMOPT_NODE_ID", socket.gethostname())
        except Exception:
            self.node_id = "unknown"

        try:
            self.host = os.environ.get(
                "MEMOPT_HOST", "")
            if not self.host:
                self.host = socket.gethostbyname(socket.gethostname())
        except Exception:
            self.host = "127.0.0.1"

        self.rack = os.environ.get("MEMOPT_RACK", "unknown")
        self.pod = os.environ.get("MEMOPT_POD", "unknown")
        self.region = os.environ.get("MEMOPT_REGION", "unknown")

        # GPU detection routes through the HAL so AMD/CPU-only
        # nodes are treated as first-class (not errors).
        try:
            from memopt.vmm.hal import get_hal
            hal = get_hal()
            self.gpu_count = hal.gpu_count
            self.hbm_total_gb = round(hal.total_hbm_bytes / 1e9, 2)
            self.hbm_free_gb = round(hal.free_hbm_bytes / 1e9, 2)
            self.hardware_backend = hal.backend.value
        except Exception:
            # HAL itself never raises from public methods, but belt
            # and suspenders — fall back to the pre-HAL defaults.
            self.gpu_count = 0
            self.hbm_total_gb = 0.0
            self.hbm_free_gb = 0.0
            self.hardware_backend = "unknown"

        try:
            from memopt import __version__
            self.memopt_version = __version__
        except Exception:
            self.memopt_version = "unknown"

        try:
            from memopt.image_version import get_node_image_info
            img_info = get_node_image_info()
            self.image_version = img_info["image_version"]
            self.git_commit = img_info["git_commit"]
            self.image_source = img_info["image_source"]
        except Exception:
            pass

        self.registered_at = time.time()
        self.last_heartbeat = time.time()

    def update_hbm_free(self) -> None:
        """Update free HBM before registration. Never raises."""
        try:
            import torch
            if torch.cuda.is_available():
                free = 0.0
                for i in range(torch.cuda.device_count()):
                    free_mem, _ = torch.cuda.mem_get_info(i)
                    free += free_mem / 1e9
                self.hbm_free_gb = round(free, 2)
        except Exception:
            pass

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "host": self.host,
            "gossip_port": self.gossip_port,
            "rbp_port": self.rbp_port,
            "rack": self.rack,
            "pod": self.pod,
            "region": self.region,
            "gpu_count": self.gpu_count,
            "hbm_total_gb": self.hbm_total_gb,
            "hbm_free_gb": self.hbm_free_gb,
            "memopt_version": self.memopt_version,
            "image_version": self.image_version,
            "git_commit": self.git_commit,
            "image_source": self.image_source,
            "hardware_backend": self.hardware_backend,
            "registered_at": self.registered_at,
            "last_heartbeat": self.last_heartbeat,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeCapabilities":
        c = cls()
        c.node_id = d.get("node_id", "")
        c.host = d.get("host", "")
        c.gossip_port = d.get("gossip_port", 18600)
        c.rbp_port = d.get("rbp_port", 18516)
        c.rack = d.get("rack", "unknown")
        c.pod = d.get("pod", "unknown")
        c.region = d.get("region", "unknown")
        c.gpu_count = d.get("gpu_count", 0)
        c.hbm_total_gb = d.get("hbm_total_gb", 0.0)
        c.hbm_free_gb = d.get("hbm_free_gb", 0.0)
        c.memopt_version = d.get("memopt_version", "unknown")
        c.image_version = d.get("image_version", "unknown")
        c.git_commit = d.get("git_commit", "unknown")
        c.image_source = d.get("image_source", "development")
        c.hardware_backend = d.get("hardware_backend", "unknown")
        c.registered_at = d.get("registered_at", 0.0)
        c.last_heartbeat = d.get("last_heartbeat", 0.0)
        return c


class NodeDiscovery:
    """
    Cluster node discovery via Redis gossip or static hosts.

    Registers this node in Redis every TTL/2 seconds.
    Scans for peers via SCAN (not KEYS) every scan_interval.
    Calls on_peer_join/on_peer_leave on membership changes.
    Falls back to MEMOPT_NODE_HOSTS when Redis unavailable.
    """

    REGISTRY_KEY_PREFIX = "memopt:discovery:nodes"
    DEFAULT_TTL_S = 30.0
    DEFAULT_SCAN_INTERVAL_S = 10.0

    def __init__(
        self,
        capabilities: NodeCapabilities,
        redis_url: str = "",
        scan_interval_s: float = 0,
        node_ttl_s: float = 0,
        on_peer_join: Optional[Callable] = None,
        on_peer_leave: Optional[Callable] = None,
    ):
        self._caps = capabilities
        self._redis_url = redis_url or os.environ.get("REDIS_URL", "")
        self._ttl_s = node_ttl_s or float(os.environ.get(
            "MEMOPT_DISCOVERY_TTL_S", str(self.DEFAULT_TTL_S)))
        self._scan_interval_s = scan_interval_s or float(os.environ.get(
            "MEMOPT_DISCOVERY_SCAN_S", str(self.DEFAULT_SCAN_INTERVAL_S)))
        self._on_peer_join = on_peer_join
        self._on_peer_leave = on_peer_leave

        self._redis = None
        self._peers: Dict[str, NodeCapabilities] = {}
        self._peers_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._registered = False
        self._backend = "none"

        self._reg_failures = 0
        self._scan_failures = 0
        self._last_scan_at = 0.0

        self._reg_thread: Optional[threading.Thread] = None
        self._scan_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start registration and scan threads."""
        self._stop_event.clear()
        self._init_redis()

        # Registration thread (every TTL/2)
        self._reg_thread = threading.Thread(
            target=self._register_loop,
            daemon=True, name="memopt-discovery-reg")
        self._reg_thread.start()

        # Scan thread (every scan_interval)
        self._scan_thread = threading.Thread(
            target=self._scan_loop,
            daemon=True, name="memopt-discovery-scan")
        self._scan_thread.start()

        logger.info(
            "NodeDiscovery started: node=%s backend=%s ttl=%.0fs",
            self._caps.node_id, self._backend, self._ttl_s)

    def stop(self) -> None:
        """Stop threads and deregister from Redis."""
        self._stop_event.set()
        # Deregister from Redis
        if self._redis is not None:
            try:
                key = f"{self.REGISTRY_KEY_PREFIX}:{self._caps.node_id}"
                self._redis.delete(key)
            except Exception:
                pass
        if self._reg_thread:
            self._reg_thread.join(timeout=3.0)
        if self._scan_thread:
            self._scan_thread.join(timeout=3.0)

    def peers(self) -> List[NodeCapabilities]:
        """Current live peers (excluding self). Never raises."""
        try:
            with self._peers_lock:
                return list(self._peers.values())
        except Exception:
            return []

    def peer_count(self) -> int:
        try:
            with self._peers_lock:
                return len(self._peers)
        except Exception:
            return 0

    def stats(self) -> dict:
        return {
            "node_id": self._caps.node_id,
            "registered": self._registered,
            "peer_count": self.peer_count(),
            "discovery_backend": self._backend,
            "last_scan_at": self._last_scan_at,
            "registration_failures": self._reg_failures,
            "scan_failures": self._scan_failures,
        }

    # ── For testing ──────────────────────────────────────────────────

    def _inject_peer_for_test(self, peer: NodeCapabilities) -> None:
        """Inject a fake peer — triggers on_peer_join callback."""
        with self._peers_lock:
            if peer.node_id not in self._peers:
                self._peers[peer.node_id] = peer
                if self._on_peer_join:
                    try:
                        self._on_peer_join(peer)
                    except Exception:
                        pass

    # ── Private ──────────────────────────────────────────────────────

    def _init_redis(self) -> None:
        if not self._redis_url:
            static = self._static_peers()
            if static:
                self._backend = "static_hosts"
                with self._peers_lock:
                    for p in static:
                        self._peers[p.node_id] = p
            else:
                self._backend = "none"
            return

        try:
            import redis
            self._redis = redis.Redis.from_url(
                self._redis_url,
                socket_timeout=2.0,
                socket_connect_timeout=2.0,
                decode_responses=True)
            self._redis.ping()
            self._backend = "redis_discovery"
            logger.info("NodeDiscovery: Redis connected")
        except Exception as e:
            logger.warning(
                "NodeDiscovery: Redis unavailable (%s), "
                "using static fallback", e)
            self._redis = None
            static = self._static_peers()
            if static:
                self._backend = "static_hosts"
                with self._peers_lock:
                    for p in static:
                        self._peers[p.node_id] = p
            else:
                self._backend = "none"

    def _register_loop(self) -> None:
        """Register every TTL/2 seconds."""
        interval = self._ttl_s / 2
        while not self._stop_event.wait(timeout=interval):
            if self._redis is None:
                continue
            try:
                self._caps.update_hbm_free()
                self._caps.last_heartbeat = time.time()
                key = f"{self.REGISTRY_KEY_PREFIX}:{self._caps.node_id}"
                value = json.dumps(self._caps.to_dict())
                self._redis.setex(
                    key, int(self._ttl_s), value)
                self._registered = True
            except Exception as e:
                self._reg_failures += 1
                if self._reg_failures == 1:
                    logger.warning(
                        "NodeDiscovery: registration failed: %s", e)

    def _scan_loop(self) -> None:
        """Scan for peers every scan_interval."""
        while not self._stop_event.wait(
                timeout=self._scan_interval_s):
            self._do_scan()

    def _do_scan(self) -> None:
        """One scan round."""
        if self._redis is None:
            return

        try:
            # Use SCAN not KEYS — non-blocking
            pattern = f"{self.REGISTRY_KEY_PREFIX}:*"
            keys = []
            cursor = 0
            while True:
                cursor, batch = self._redis.scan(
                    cursor, match=pattern, count=100)
                keys.extend(batch)
                if cursor == 0:
                    break

            if not keys:
                self._last_scan_at = time.time()
                return

            # Pipeline GET — one round-trip
            pipe = self._redis.pipeline(transaction=False)
            for k in keys:
                pipe.get(k)
            values = pipe.execute()

            new_peers: Dict[str, NodeCapabilities] = {}
            for key, value in zip(keys, values):
                if value is None:
                    continue
                try:
                    key_str = key if isinstance(key, str) \
                        else key.decode()
                    node_id = key_str.split(":")[-1]
                    if node_id == self._caps.node_id:
                        continue  # skip self
                    d = json.loads(value if isinstance(value, str)
                                   else value.decode())
                    peer = NodeCapabilities.from_dict(d)
                    new_peers[node_id] = peer
                except Exception:
                    continue

            # Detect joins and leaves
            with self._peers_lock:
                old_ids = set(self._peers.keys())
                new_ids = set(new_peers.keys())

                for nid in new_ids - old_ids:
                    if self._on_peer_join:
                        try:
                            self._on_peer_join(new_peers[nid])
                        except Exception:
                            pass

                for nid in old_ids - new_ids:
                    # Only fire leave if backend is redis
                    # (static peers don't leave)
                    if self._backend == "redis_discovery":
                        if self._on_peer_leave:
                            try:
                                self._on_peer_leave(
                                    self._peers[nid])
                            except Exception:
                                pass

                self._peers = new_peers

            self._last_scan_at = time.time()

        except Exception as e:
            self._scan_failures += 1
            if self._scan_failures == 1:
                logger.warning(
                    "NodeDiscovery: scan failed: %s", e)

    def _static_peers(self) -> List[NodeCapabilities]:
        """Parse MEMOPT_NODE_HOSTS env var as fallback."""
        raw = os.environ.get("MEMOPT_NODE_HOSTS", "")
        if not raw:
            return []
        peers = []
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            p = NodeCapabilities()
            p.host = parts[0]
            p.gossip_port = int(parts[1]) if len(parts) > 1 else 18600
            p.node_id = p.host  # use host as ID for static peers
            peers.append(p)
        return peers
