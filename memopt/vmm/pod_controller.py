"""
Pod Controller — middle tier of three-tier cell architecture.

One Pod Controller manages up to 10,000 GPU nodes. It:
1. Aggregates oracle transitions from nodes into a pod-level oracle
2. Reports pod health to the global control plane

Environment variables:
  MEMOPT_POD_ID               pod identifier
  MEMOPT_POD_SIZE             nodes per pod (default 10000)
  MEMOPT_POD_ORACLE_PULL_S    oracle pull interval (default 1.0)
  REDIS_URL                   Redis for pod GKD
  MEMOPT_CONTROL_PLANE_URL    control plane URL
  MEMOPT_POD_REPORT_S         report interval (default 30.0)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional

from typing import Dict, List

from memopt.vmm.oracle import MemoryOracle

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Pod GKD Cache — LRU in-memory cache for pod-level KV deduplication
# ═══════════════════════════════════════════════════════════════════════════

class PodGKDCache:
    """
    In-memory LRU GKD cache for pod-level deduplication.
    Shared across all nodes in the pod via HTTP endpoints.
    Not a replacement for ScyllaDB — a fast layer in front of it.

    Size: MEMOPT_POD_GKD_CACHE_SIZE (default 100,000 entries, ~20MB)
    """

    DEFAULT_SIZE = 100_000

    def __init__(self, max_size: int = DEFAULT_SIZE):
        self._max_size = max_size
        self._cache: Dict[str, dict] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, content_hash: str,
            seq_len: int) -> Optional[dict]:
        """Look up. Returns entry dict or None. Thread-safe."""
        key = f"{content_hash}:{seq_len}"
        with self._lock:
            if key in self._cache:
                self._hits += 1
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
                self._access_order.append(key)
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, content_hash: str,
            seq_len: int, entry: dict) -> None:
        """Store with LRU eviction. Thread-safe."""
        key = f"{content_hash}:{seq_len}"
        with self._lock:
            if key in self._cache:
                try:
                    self._access_order.remove(key)
                except ValueError:
                    pass
            elif len(self._cache) >= self._max_size:
                if self._access_order:
                    lru_key = self._access_order.pop(0)
                    self._cache.pop(lru_key, None)
            self._cache[key] = entry
            self._access_order.append(key)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100
                        if total > 0 else 0.0)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate_pct": round(hit_rate, 2),
            }


class PodConfig:
    """Configuration for a pod controller."""

    def __init__(self):
        self.pod_id: str = ""
        self.node_id: str = ""
        self.pod_size: int = 10_000
        self.oracle_pull_interval_s: float = 1.0
        self.oracle_top_k_per_node: int = 100
        self.redis_url: str = ""
        self.control_plane_url: str = ""
        self.report_interval_s: float = 30.0
        self.discovery = None  # Optional NodeDiscovery instance

    @classmethod
    def from_env(cls) -> "PodConfig":
        """Read from environment. Never raises."""
        c = cls()
        try:
            import socket
            c.pod_id = os.environ.get("MEMOPT_POD_ID", "")
            c.node_id = os.environ.get(
                "MEMOPT_NODE_ID", socket.gethostname())
            c.pod_size = int(os.environ.get(
                "MEMOPT_POD_SIZE", "10000"))
            c.oracle_pull_interval_s = float(os.environ.get(
                "MEMOPT_POD_ORACLE_PULL_S", "1.0"))
            c.redis_url = os.environ.get("REDIS_URL", "")
            c.control_plane_url = os.environ.get(
                "MEMOPT_CONTROL_PLANE_URL", "")
            c.report_interval_s = float(os.environ.get(
                "MEMOPT_POD_REPORT_S", "30.0"))
        except Exception:
            pass
        return c


class PodOracleAggregator:
    """
    Aggregates oracle transitions from all nodes in the pod
    into a pod-level oracle.

    Pull model: pod controller pulls from each node's /oracle/stats
    endpoint every oracle_pull_interval_s seconds.

    Filters: only merges transitions with confidence >= MIN_CONFIDENCE.

    Push: after aggregation, pushes high-confidence pod transitions
    back to all nodes via /oracle/merge.
    """

    MIN_CONFIDENCE = float(os.getenv(
        "MEMOPT_POD_MIN_CONFIDENCE", "0.5"))

    # Only push back transitions seen on at least this fraction
    # of contributing nodes (reduces noise from single-node outliers)
    MIN_NODE_COVERAGE = float(os.getenv(
        "MEMOPT_POD_MIN_NODE_COVERAGE", "0.1"))

    def __init__(self, pod_oracle: MemoryOracle, config: PodConfig):
        self._oracle = pod_oracle
        self._config = config

        # Track which nodes contributed each transition
        # {(from_block, to_block): set(node_ids)}
        self._transition_nodes: dict[tuple, set] = {}
        self._lock = threading.Lock()

        # Aggregation stats
        self._nodes_pulled = 0
        self._transitions_merged = 0
        self._transitions_skipped = 0
        self._push_backs = 0
        self._last_pull_at = 0.0

    def pull_from_node(self, node_id: str,
                       node_oracle_stats: dict) -> int:
        """
        Merge transitions from one node into pod oracle.

        Only merges transitions with confidence >= MIN_CONFIDENCE.
        Transitions without a confidence field (legacy nodes) are
        accepted unconditionally (confidence defaults to 1.0).

        Tracks which nodes contributed each transition for
        coverage calculation.

        Uses max-count merge: only updates if new count > existing.
        Returns: number of transitions merged. Never raises.
        """
        if node_oracle_stats is None:
            return 0
        try:
            transitions = node_oracle_stats.get("transitions", [])
            if not isinstance(transitions, list):
                return 0

            merged = 0
            for t in transitions:
                if not isinstance(t, dict):
                    continue

                # Support both new (from_block/to_block) and
                # legacy (from/to) key names
                from_b = t.get("from_block", t.get("from"))
                to_b = t.get("to_block", t.get("to"))
                count = t.get("count", 0)
                # Legacy nodes without confidence: trust implicitly
                confidence = t.get("confidence", 1.0)

                if from_b is None or to_b is None:
                    continue

                if confidence < self.MIN_CONFIDENCE:
                    self._transitions_skipped += 1
                    continue

                # Max-count merge into pod oracle
                try:
                    with self._oracle._lock:
                        existing = self._oracle._transitions[
                            from_b][to_b]
                        if count > existing:
                            self._oracle._transitions[
                                from_b][to_b] = count
                            merged += 1
                except Exception:
                    continue

                # Track per-transition node coverage
                key = (from_b, to_b)
                with self._lock:
                    if key not in self._transition_nodes:
                        self._transition_nodes[key] = set()
                    self._transition_nodes[key].add(node_id)

            with self._lock:
                self._transitions_merged += merged
                self._nodes_pulled += 1
                self._last_pull_at = time.time()

            return merged
        except Exception:
            return 0

    def get_pod_transitions(
        self,
        min_node_coverage: Optional[float] = None,
        top_k: int = 200,
    ) -> list[dict]:
        """
        Get pod-level transitions filtered by node coverage.

        Only returns transitions seen on at least min_node_coverage
        fraction of contributing nodes. This filters out single-node
        noise and returns only cluster-wide patterns.

        Returns list sorted by count descending. Thread-safe. Never raises.
        """
        if min_node_coverage is None:
            min_node_coverage = self.MIN_NODE_COVERAGE

        total_possible = max(1, self._nodes_pulled)
        if self._config and self._config.pod_size > 0:
            total_possible = min(
                self._nodes_pulled, self._config.pod_size)
            total_possible = max(1, total_possible)

        try:
            results = []
            with self._oracle._lock:
                for from_block, targets in \
                        self._oracle._transitions.items():
                    total_from = sum(targets.values())
                    for to_block, count in targets.items():
                        key = (from_block, to_block)
                        with self._lock:
                            node_set = self._transition_nodes.get(
                                key, set())
                            n_nodes = len(node_set)

                        coverage = n_nodes / total_possible

                        if coverage < min_node_coverage:
                            continue

                        confidence = (
                            count / total_from
                            if total_from > 0 else 0.0)

                        results.append({
                            "from_block":     from_block,
                            "to_block":       to_block,
                            "count":          count,
                            "confidence":     round(confidence, 4),
                            "node_coverage":  round(coverage, 4),
                            "node_count":     n_nodes,
                        })

            results.sort(key=lambda x: x["count"], reverse=True)
            return results[:top_k]
        except Exception:
            return []

    def stats(self) -> dict:
        with self._lock:
            transition_coverage = len(self._transition_nodes)
        return {
            "nodes_pulled":          self._nodes_pulled,
            "transitions_merged":    self._transitions_merged,
            "transitions_skipped":   self._transitions_skipped,
            "push_backs":            self._push_backs,
            "transition_coverage":   transition_coverage,
            "last_pull_at":          self._last_pull_at,
            "pod_oracle_size":       sum(
                len(v) for v in
                self._oracle._transitions.values())
                if hasattr(self._oracle, "_transitions")
                else 0,
        }


class PodController:
    """
    Middle tier of three-tier cell architecture.
    Manages one pod of up to pod_size GPU nodes.

    Runs as a background service on one designated
    node per pod. Start via:
      memopt pod-controller start --pod-id pod-001
    """

    def __init__(self, config: PodConfig):
        self._config = config
        self._discovery = config.discovery
        self._pod_oracle = MemoryOracle(
            horizon=200,
            max_transitions=500_000)
        self._aggregator = PodOracleAggregator(
            self._pod_oracle, config)
        self._gkd_cache = PodGKDCache(
            max_size=int(os.environ.get(
                "MEMOPT_POD_GKD_CACHE_SIZE",
                str(PodGKDCache.DEFAULT_SIZE))))
        self._running = False
        self._stop_event = threading.Event()
        self._threads: list = []
        self._start_time = 0.0
        self._node_count = 0

    def start(self) -> None:
        """Start background threads."""
        self._running = True
        self._stop_event.clear()
        self._start_time = time.time()

        t1 = threading.Thread(
            target=self._oracle_aggregation_loop,
            daemon=True,
            name="pod-oracle-agg")
        t1.start()
        self._threads.append(t1)

        t2 = threading.Thread(
            target=self._reporting_loop,
            daemon=True,
            name="pod-reporting")
        t2.start()
        self._threads.append(t2)

        logger.info(
            "PodController started: pod=%s capacity=%d",
            self._config.pod_id, self._config.pod_size)

    def stop(self) -> None:
        """Stop gracefully."""
        self._running = False
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=5.0)
        logger.info(
            "PodController stopped: pod=%s",
            self._config.pod_id)

    def get_pod_oracle(self) -> MemoryOracle:
        """Return the pod-level aggregated oracle."""
        return self._pod_oracle

    def pod_stats(self) -> dict:
        uptime = time.time() - self._start_time \
            if self._start_time > 0 else 0
        return {
            "pod_id": self._config.pod_id,
            "node_count": self._node_count,
            "pod_oracle_size": sum(
                len(v) for v in
                self._pod_oracle._transitions.values()),
            "aggregator_stats": self._aggregator.stats(),
            "uptime_seconds": round(uptime, 1),
        }

    # ── Background threads ────────────────────────────────────────────

    def _oracle_aggregation_loop(self) -> None:
        """Pull oracle transitions from nodes, push back periodically."""
        push_every_n = int(os.getenv(
            "MEMOPT_POD_PUSH_EVERY_N", "10"))
        cycle = 0
        while self._running and not self._stop_event.is_set():
            try:
                self._pull_cycle()
            except Exception as e:
                logger.error(
                    "Pod oracle aggregation error: %s", e)

            cycle += 1

            # Push high-confidence pod transitions back to nodes
            if cycle % push_every_n == 0:
                try:
                    self._push_to_nodes()
                except Exception as e:
                    logger.debug("Pod push error: %s", e)

            self._stop_event.wait(
                timeout=self._config.oracle_pull_interval_s)

    def _push_to_nodes(self) -> None:
        """
        Push high-confidence pod transitions back to all nodes.

        Called after each N aggregation cycles. Only pushes transitions
        that pass the node coverage filter. Uses fire-and-forget
        threading — never blocks the aggregation loop.
        """
        try:
            pod_transitions = \
                self._aggregator.get_pod_transitions(top_k=50)

            if not pod_transitions:
                return

            nodes = self._get_nodes_to_pull()
            if not nodes:
                return

            payload = json.dumps({
                "source": "pod",
                "pod_id": self._config.pod_id,
                "transitions": pod_transitions,
            }).encode()

            def push_to_node(node_url: str) -> None:
                try:
                    req = urllib.request.Request(
                        f"{node_url}/oracle/merge",
                        data=payload,
                        headers={
                            "Content-Type": "application/json"},
                        method="POST")
                    urllib.request.urlopen(req, timeout=0.5)
                    self._aggregator._push_backs += 1
                except Exception:
                    pass  # Fire and forget

            for node_url in nodes:
                threading.Thread(
                    target=push_to_node,
                    args=(node_url,),
                    daemon=True).start()

        except Exception as e:
            logger.debug(f"Pod push error: {e}")

    def _get_nodes_to_pull(self) -> list:
        """
        Get node URLs to pull oracle data from.
        Priority: NodeDiscovery peers → MEMOPT_NODE_HOSTS → empty.
        Returns list of "http://host:port" strings.
        """
        # Try NodeDiscovery first
        if self._discovery is not None:
            try:
                peers = self._discovery.peers()
                if peers:
                    return [
                        f"http://{p.host}:8080"
                        for p in peers
                        if p.node_id != self._config.node_id
                    ]
            except Exception:
                pass

        # Fall back to static hosts
        raw = os.environ.get("MEMOPT_NODE_HOSTS", "")
        if raw:
            urls = []
            for entry in raw.split(","):
                entry = entry.strip()
                if entry:
                    host = entry.split(":")[0]
                    urls.append(f"http://{host}:8080")
            return urls

        return []

    def _pull_cycle(self) -> None:
        """One aggregation cycle across discovered peers."""
        node_urls = self._get_nodes_to_pull()
        self._node_count = len(node_urls)

        for node_url in node_urls:
            if not self._running:
                break
            try:
                stats_url = f"{node_url}/oracle/stats"
                req = urllib.request.Request(
                    stats_url, method="GET")
                with urllib.request.urlopen(
                        req, timeout=2.0) as resp:
                    data = json.loads(resp.read())
                    node_id = data.get("node_id", node_url)
                    self._aggregator.pull_from_node(
                        node_id, data)
            except Exception:
                pass  # node unreachable — skip

    def _reporting_loop(self) -> None:
        """Report pod health to control plane."""
        import socket as _socket
        while self._running and not self._stop_event.is_set():
            if self._stop_event.wait(
                    timeout=self._config.report_interval_s):
                break  # stop requested
            if not self._config.control_plane_url:
                continue
            try:
                stats = self.pod_stats()

                # Determine this pod's URL for global oracle pull-back
                my_host = os.getenv(
                    "MEMOPT_HOST", _socket.gethostname())
                my_port = os.getenv(
                    "MEMOPT_SERVING_PORT", "8080")
                pod_url = f"http://{my_host}:{my_port}"

                payload = json.dumps({
                    "pod_id": self._config.pod_id,
                    "node_count": stats["node_count"],
                    "healthy_nodes": stats["node_count"],
                    "pod_oracle_size":
                        stats["pod_oracle_size"],
                    "avg_hbm_free_gb": 0.0,
                    "gkd_hit_rate_pct": 0.0,
                    "pod_url": pod_url,
                    "reported_at": time.time(),
                }).encode()

                url = (f"{self._config.control_plane_url}"
                       f"/api/v1/pods/{self._config.pod_id}")
                req = urllib.request.Request(
                    url, data=payload, method="POST",
                    headers={
                        "Content-Type": "application/json"})
                urllib.request.urlopen(req, timeout=5.0)
            except Exception as e:
                logger.debug(
                    "Pod report to control plane failed: %s", e)
