# Three-tier oracle hierarchy:
#
# Global Oracle (this file)
#   - Pulls from pod oracles every 60s
#   - Identifies cross-pod patterns
#   - Pushes top 20 transitions to pods
#   - Runs in control plane process
#
# Pod Oracle (pod_controller.py)
#   - Pulls from node oracles every 1s
#   - Identifies cross-node patterns
#   - Pushes top 50 transitions to nodes
#   - Runs in pod controller process
#
# Node Oracle (oracle.py)
#   - Updated on every block access
#   - Makes per-request predictions
#   - Receives merges from pod oracle
#   - Runs in serving process
"""
Global Oracle — top tier of three-tier oracle hierarchy.

Aggregates oracle transitions from all pods and identifies
cluster-wide access patterns. Operates on a slow cycle
(default 60s) because global patterns change slowly.

Does NOT make per-request predictions — too slow at global scale.
Instead, pushes high-confidence global transitions down to pod
oracles which push them further down to nodes.

Environment variables:
  MEMOPT_GLOBAL_ORACLE_PULL_S       pull interval (default 60.0)
  MEMOPT_GLOBAL_ORACLE_TOP_K        transitions per pod (default 50)
  MEMOPT_GLOBAL_MIN_POD_COVERAGE    min pod coverage fraction (default 0.05)
  MEMOPT_GLOBAL_MAX_TRANSITIONS     max transitions to track (default 1000000)
  MEMOPT_GLOBAL_PUSH_BACK           enable push-back (default true)
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from typing import Optional

from memopt.vmm.oracle import MemoryOracle

logger = logging.getLogger(__name__)


class GlobalOracleConfig:
    """Configuration for global oracle."""

    def __init__(self):
        self.pull_interval_s: float = float(os.getenv(
            "MEMOPT_GLOBAL_ORACLE_PULL_S", "60.0"))

        self.top_k_per_pod: int = int(os.getenv(
            "MEMOPT_GLOBAL_ORACLE_TOP_K", "50"))

        self.min_pod_coverage: float = float(os.getenv(
            "MEMOPT_GLOBAL_MIN_POD_COVERAGE", "0.05"))

        self.max_transitions: int = int(os.getenv(
            "MEMOPT_GLOBAL_MAX_TRANSITIONS", "1000000"))

        self.control_plane_url: str = os.getenv(
            "MEMOPT_CONTROL_PLANE_URL", "")

        self.push_back_enabled: bool = os.getenv(
            "MEMOPT_GLOBAL_PUSH_BACK", "true").lower() == "true"


class GlobalOracle:
    """
    Top tier of three-tier oracle hierarchy.

    Aggregates oracle transitions from all pods and identifies
    cluster-wide access patterns.

    Operates on a slow cycle (default 60s) because global patterns
    change slowly.

    Does NOT make per-request predictions. Instead it pushes
    high-confidence global transitions down to pod oracles which
    push them further down to nodes.

    This is a background service — it improves prediction quality
    over time but is never on the critical path.
    """

    def __init__(
        self,
        config: Optional[GlobalOracleConfig] = None,
    ):
        self._config = config or GlobalOracleConfig()

        # Global oracle: large horizon, many transitions
        self._oracle = MemoryOracle(
            horizon=500,
            max_transitions=self._config.max_transitions)

        # Track which pods contributed each transition
        # {(from_block, to_block): set(pod_ids)}
        self._transition_pods: dict[tuple, set] = {}
        self._pods_lock = threading.Lock()

        # Known pod controller URLs
        # {pod_id: url}
        self._pod_urls: dict[str, str] = {}
        self._pod_urls_lock = threading.Lock()

        self._running = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self._pods_pulled = 0
        self._transitions_merged = 0
        self._push_backs = 0
        self._last_pull_at = 0.0
        self._errors = 0

    def start(self) -> None:
        """Start global oracle daemon thread."""
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="global-oracle",
            daemon=True)
        self._thread.start()
        logger.info(
            "GlobalOracle started: pull_interval=%ss",
            self._config.pull_interval_s)

    def stop(self) -> None:
        """Stop gracefully."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("GlobalOracle stopped")

    def register_pod(self, pod_id: str, pod_url: str) -> None:
        """
        Register a pod controller URL.
        Called when a pod controller reports to the control plane.
        """
        with self._pod_urls_lock:
            self._pod_urls[pod_id] = pod_url
        logger.debug(
            "GlobalOracle: registered pod %s at %s",
            pod_id, pod_url)

    def get_global_transitions(
        self,
        top_k: int = 100,
    ) -> list[dict]:
        """
        Get global transitions filtered by pod coverage.

        Only returns transitions seen in at least min_pod_coverage
        fraction of known pods. Thread-safe. Never raises.
        """
        try:
            with self._pod_urls_lock:
                total_pods = len(self._pod_urls)

            if total_pods == 0:
                return []

            min_coverage = self._config.min_pod_coverage

            results = []
            with self._oracle._lock:
                for from_block, targets in \
                        self._oracle._transitions.items():
                    total_from = sum(targets.values())
                    for to_block, count in targets.items():
                        key = (from_block, to_block)
                        with self._pods_lock:
                            pod_set = self._transition_pods.get(
                                key, set())
                            n_pods = len(pod_set)

                        coverage = n_pods / total_pods

                        if coverage < min_coverage:
                            continue

                        confidence = (
                            count / total_from
                            if total_from > 0 else 0.0)

                        results.append({
                            "from_block":   from_block,
                            "to_block":     to_block,
                            "count":        count,
                            "confidence":   round(confidence, 4),
                            "pod_coverage": round(coverage, 4),
                            "pod_count":    n_pods,
                        })

            results.sort(key=lambda x: x["count"], reverse=True)
            return results[:top_k]
        except Exception:
            return []

    def stats(self) -> dict:
        """Current global oracle statistics."""
        with self._pod_urls_lock:
            pod_count = len(self._pod_urls)
        with self._pods_lock:
            coverage_count = len(self._transition_pods)

        return {
            "pods_known":           pod_count,
            "pods_pulled":          self._pods_pulled,
            "transitions_merged":   self._transitions_merged,
            "push_backs":           self._push_backs,
            "transition_coverage":  coverage_count,
            "last_pull_at":         self._last_pull_at,
            "errors":               self._errors,
            "pull_interval_s":      self._config.pull_interval_s,
        }

    # ── Private ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main loop — pull, aggregate, push."""
        while not self._stop_event.is_set():
            try:
                self._pull_from_pods()
                if self._config.push_back_enabled:
                    self._push_to_pods()
            except Exception as e:
                self._errors += 1
                logger.error("GlobalOracle loop error: %s", e)

            self._stop_event.wait(
                timeout=self._config.pull_interval_s)

    def _pull_from_pods(self) -> None:
        """Pull oracle transitions from all known pods."""
        with self._pod_urls_lock:
            pod_items = list(self._pod_urls.items())

        if not pod_items:
            return

        logger.debug(
            "GlobalOracle: pulling from %d pods",
            len(pod_items))

        for pod_id, pod_url in pod_items:
            if self._stop_event.is_set():
                break
            try:
                self._pull_one_pod(pod_id, pod_url)
            except Exception as e:
                self._errors += 1
                logger.debug(
                    "GlobalOracle: pull failed from %s: %s",
                    pod_id, e)

    def _pull_one_pod(self, pod_id: str, pod_url: str) -> None:
        """
        Pull transitions from one pod oracle.
        HTTP GET {pod_url}/pod/oracle/stats
        Timeout: 5 seconds per pod.
        """
        url = f"{pod_url}/pod/oracle/stats"
        req = urllib.request.Request(url)

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            data = json.loads(resp.read())

        transitions = data.get("transitions", [])
        if not transitions:
            return

        merged = 0
        for t in transitions:
            try:
                confidence = t.get("confidence", 0.0)

                # Global oracle uses a higher confidence bar
                # than pod oracle — only propagate strong signals
                if confidence < 0.6:
                    continue

                from_block = int(t["from_block"])
                to_block = int(t["to_block"])
                count = int(t.get("count", 1))

                # Max-count merge into global oracle
                # (same semantics as pod oracle)
                merge_count = min(count, 3)
                with self._oracle._lock:
                    existing = self._oracle._transitions[
                        from_block][to_block]
                    if merge_count > existing:
                        self._oracle._transitions[
                            from_block][to_block] = merge_count

                # Track pod coverage
                key = (from_block, to_block)
                with self._pods_lock:
                    if key not in self._transition_pods:
                        self._transition_pods[key] = set()
                    self._transition_pods[key].add(pod_id)

                merged += 1
                self._transitions_merged += 1
            except Exception:
                pass

        self._pods_pulled += 1
        self._last_pull_at = time.time()
        logger.debug(
            "GlobalOracle: merged %d transitions from pod %s",
            merged, pod_id)

    def _push_to_pods(self) -> None:
        """
        Push high-confidence global transitions to all pods.

        Pods will further propagate to their nodes.
        Fire-and-forget threading — never blocks the main loop.
        """
        global_transitions = self.get_global_transitions(top_k=20)

        if not global_transitions:
            return

        with self._pod_urls_lock:
            pod_items = list(self._pod_urls.items())

        payload = json.dumps({
            "source": "global",
            "transitions": global_transitions,
        }).encode()

        def push_one(pod_url: str) -> None:
            try:
                req = urllib.request.Request(
                    f"{pod_url}/oracle/merge",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST")
                urllib.request.urlopen(req, timeout=1.0)
                self._push_backs += 1
            except Exception:
                pass  # Fire and forget

        for _, pod_url in pod_items:
            threading.Thread(
                target=push_one,
                args=(pod_url,),
                daemon=True).start()
