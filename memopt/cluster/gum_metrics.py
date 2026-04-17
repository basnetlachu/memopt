"""
GUM bandwidth and latency tracking.

Measures real transfer performance between nodes.

No fake numbers. No estimated latencies.
Every measurement comes from actual timed block transfers.

On TCP: measures TCP transfer latency.
On RDMA: measures RDMA transfer latency.
The transport layer determines which.
"""
import logging
import os
import statistics
import threading
import time
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)


class TransferMeasurement:
    """One block transfer measurement."""

    __slots__ = [
        "peer_node_id", "bytes_transferred", "latency_ms",
        "success", "timestamp", "transport",
        "rack", "pod", "region",
    ]

    def __init__(
        self,
        peer_node_id: str,
        bytes_transferred: int,
        latency_ms: float,
        success: bool,
        transport: str = "tcp",
        rack: str = "unknown",
        pod: str = "unknown",
        region: str = "unknown",
    ):
        self.peer_node_id = peer_node_id
        self.bytes_transferred = bytes_transferred
        self.latency_ms = latency_ms
        self.success = success
        self.timestamp = time.time()
        self.transport = transport
        self.rack = rack
        self.pod = pod
        self.region = region


class GUMMetrics:
    """
    Tracks GUM transfer performance.

    Rolling window per peer. Aggregated stats for dashboard.
    Thread-safe.

    HONEST: All numbers come from real timed transfers. No simulation.
    """

    DEFAULT_WINDOW = int(os.getenv("MEMOPT_GUM_METRICS_WINDOW", "1000"))

    def __init__(self, window_size: int = 0):
        self._window = window_size or self.DEFAULT_WINDOW
        self._lock = threading.Lock()

        self._peer_measurements: dict[str, deque] = {}

        self._total_transfers = 0
        self._total_bytes = 0
        self._total_failures = 0
        self._total_fallbacks = 0

    def record_transfer(self, measurement: TransferMeasurement) -> None:
        """Record one block transfer. Thread-safe. Never raises."""
        try:
            with self._lock:
                peer = measurement.peer_node_id

                if peer not in self._peer_measurements:
                    self._peer_measurements[peer] = deque(
                        maxlen=self._window)

                self._peer_measurements[peer].append(measurement)

                self._total_transfers += 1

                if measurement.success:
                    self._total_bytes += measurement.bytes_transferred
                else:
                    self._total_failures += 1
        except Exception:
            pass

    def record_fallback(self) -> None:
        """Record that GUM fetch failed and we fell back to recompute."""
        with self._lock:
            self._total_fallbacks += 1

    def peer_stats(self, peer_node_id: str) -> dict:
        """Stats for one peer. Returns empty dict if unknown peer."""
        try:
            with self._lock:
                if peer_node_id not in self._peer_measurements:
                    return {}
                measurements = list(self._peer_measurements[peer_node_id])

            if not measurements:
                return {}

            successes = [m for m in measurements if m.success]
            latencies = [m.latency_ms for m in successes]
            bytes_list = [m.bytes_transferred for m in successes]

            result = {
                "peer_node_id": peer_node_id,
                "total_attempts": len(measurements),
                "successes": len(successes),
                "failures": len(measurements) - len(successes),
                "success_rate_pct": round(
                    len(successes) / len(measurements) * 100, 2
                ) if measurements else 0.0,
                "lat_p50_ms": None,
                "lat_p99_ms": None,
                "lat_min_ms": None,
                "lat_max_ms": None,
                "total_bytes": sum(bytes_list),
                "transport": measurements[-1].transport if measurements else "unknown",
                "rack": measurements[-1].rack if measurements else "unknown",
            }

            if latencies:
                result["lat_p50_ms"] = round(statistics.median(latencies), 3)
                result["lat_min_ms"] = round(min(latencies), 3)
                result["lat_max_ms"] = round(max(latencies), 3)
                if len(latencies) >= 2:
                    idx = int(len(latencies) * 0.99)
                    result["lat_p99_ms"] = round(sorted(latencies)[idx], 3)

            return result

        except Exception:
            return {}

    def global_stats(self) -> dict:
        """Aggregated stats across all peers. Thread-safe."""
        try:
            with self._lock:
                peer_ids = list(self._peer_measurements.keys())
                total = self._total_transfers
                bytes_ = self._total_bytes
                failures = self._total_failures
                fallbacks = self._total_fallbacks

            all_latencies = []
            for peer_id in peer_ids:
                s = self.peer_stats(peer_id)
                if s.get("lat_p50_ms") is not None:
                    all_latencies.append(s["lat_p50_ms"])

            return {
                "total_transfers": total,
                "total_bytes": bytes_,
                "total_failures": failures,
                "total_fallbacks": fallbacks,
                "active_peers": len(peer_ids),
                "overall_p50_ms": round(
                    statistics.median(all_latencies), 3
                ) if all_latencies else None,
                "success_rate_pct": round(
                    (total - failures) / total * 100, 2
                ) if total > 0 else 0.0,
                "note": (
                    "All latencies from real timed transfers. "
                    "TCP on dev. RDMA when IB present."),
            }
        except Exception:
            return {}

    def dashboard_data(self) -> dict:
        """Data for Grafana/dashboard. Returns per-peer and global stats."""
        with self._lock:
            peer_ids = list(self._peer_measurements.keys())

        peers = {p: self.peer_stats(p) for p in peer_ids}

        return {
            "global": self.global_stats(),
            "peers": peers,
            "generated_at": time.time(),
        }


# Module-level singleton
_gum_metrics: Optional[GUMMetrics] = None


def get_gum_metrics() -> GUMMetrics:
    global _gum_metrics
    if _gum_metrics is None:
        _gum_metrics = GUMMetrics()
    return _gum_metrics
