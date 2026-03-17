"""
Metrics collector — aggregates stats() from all three pillars into a
unified Prometheus-compatible registry.

Polls every COLLECT_INTERVAL_S seconds in a daemon thread.
Exposes metrics via collect() for the API server and Grafana.

All metric names follow the convention:
  memopt_{pillar}_{measurement}_{unit}

No external dependencies. Reads only from public stats() methods
that already exist on the pillar objects.

Environment variables:
  MEMOPT_COLLECT_INTERVAL_S  float  default 15.0
"""
from __future__ import annotations
import os
import time
import threading
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

COLLECT_INTERVAL_S = float(os.environ.get("MEMOPT_COLLECT_INTERVAL_S", "15.0"))


@dataclass
class Metric:
    """One Prometheus-style metric sample."""
    name:   str
    value:  float
    labels: Dict[str, str] = field(default_factory=dict)
    help:   str = ""
    type:   str = "gauge"   # gauge | counter | histogram


class MetricRegistry:
    """
    Thread-safe in-memory metric store.
    Stores the latest value for each (name, labels) pair.
    Exposes prometheus_text() for the /metrics endpoint.
    """

    def __init__(self):
        self._metrics: Dict[str, Metric] = {}
        self._lock    = threading.RLock()

    def set(self, name: str, value: float,
            labels: Dict[str, str] = None,
            help: str = "",
            type: str = "gauge") -> None:
        if value is None or (isinstance(value, float) and
                             (value != value)):   # NaN check
            return
        key = f"{name}|{sorted((labels or {}).items())}"
        with self._lock:
            self._metrics[key] = Metric(
                name=name, value=value,
                labels=labels or {}, help=help, type=type
            )

    def get_all(self) -> List[Metric]:
        with self._lock:
            return list(self._metrics.values())

    def prometheus_text(self) -> str:
        """
        Render all metrics in Prometheus text exposition format.
        https://prometheus.io/docs/instrumenting/exposition_formats/
        """
        lines = []
        seen_help: set = set()
        with self._lock:
            metrics = list(self._metrics.values())

        for m in metrics:
            if m.name not in seen_help:
                if m.help:
                    lines.append(f"# HELP {m.name} {m.help}")
                lines.append(f"# TYPE {m.name} {m.type}")
                seen_help.add(m.name)
            label_str = ""
            if m.labels:
                pairs = ",".join(
                    f'{k}="{v}"' for k, v in sorted(m.labels.items())
                )
                label_str = f"{{{pairs}}}"
            lines.append(f"{m.name}{label_str} {m.value}")

        return "\n".join(lines) + "\n"

    def as_dict(self) -> Dict[str, Any]:
        """Return all metrics as a plain dict for JSON serialisation."""
        with self._lock:
            return {
                m.name: {
                    "value":  m.value,
                    "labels": m.labels,
                    "type":   m.type,
                }
                for m in self._metrics.values()
            }


class MetricsCollector:
    """
    Polls stats() from all registered pillar objects and writes
    the results into a MetricRegistry.

    Pillar objects are registered at server startup via register_*().
    The collector runs a background thread that polls every
    COLLECT_INTERVAL_S seconds.

    Usage:
        collector = MetricsCollector()
        collector.register_vmm(vmm_instance)
        collector.register_gkd(gkd_store_instance)
        collector.register_hypervisor(hypervisor_instance)
        collector.register_kernel_hooks(kernel_hooks_module)
        collector.start()

        # In API server:
        text = collector.registry.prometheus_text()
    """

    def __init__(self, interval_s: float = COLLECT_INTERVAL_S):
        self.registry  = MetricRegistry()
        self._interval = interval_s
        self._lock     = threading.RLock()
        self._running  = False

        # Registered pillar objects
        self._vmm:           Optional[Any] = None
        self._gkd:           Optional[Any] = None
        self._hypervisor:    Optional[Any] = None
        self._kernel_hooks:  Optional[Any] = None
        self._power_sampler: Optional[Any] = None

        # Cumulative counters (survive across collection cycles)
        self._tokens_total:  float = 0.0
        self._requests_total: int  = 0

    # ── Registration ───────────────────────────────────────────────────

    def register_vmm(self, vmm) -> None:
        with self._lock:
            self._vmm = vmm

    def register_gkd(self, gkd) -> None:
        with self._lock:
            self._gkd = gkd

    def register_hypervisor(self, hypervisor) -> None:
        with self._lock:
            self._hypervisor = hypervisor

    def register_kernel_hooks(self, kernel_hooks_module) -> None:
        with self._lock:
            self._kernel_hooks = kernel_hooks_module

    def register_power_sampler(self, power_sampler) -> None:
        with self._lock:
            self._power_sampler = power_sampler

    def record_request(self, tokens_generated: int) -> None:
        """Call after each inference request completes."""
        with self._lock:
            self._tokens_total   += tokens_generated
            self._requests_total += 1

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        t = threading.Thread(
            target=self._poll_loop, daemon=True, name="memopt-collector"
        )
        t.start()
        logger.info(
            f"MetricsCollector started (interval={self._interval}s)"
        )

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def collect_once(self) -> None:
        """
        Collect metrics synchronously. Called by the background thread
        and also directly in tests.
        """
        self._collect_vmm()
        self._collect_gkd()
        self._collect_hypervisor()
        self._collect_kernel_hooks()
        self._collect_power()
        self._collect_request_counters()

    # ── Collection methods ─────────────────────────────────────────────

    def _collect_vmm(self) -> None:
        vmm = self._vmm
        if vmm is None:
            return
        try:
            s = vmm.stats()
            r = self.registry
            r.set("memopt_vmm_total_blocks",
                  float(s.get("total_blocks", 0)),
                  help="Total VMM blocks allocated across all tiers")
            r.set("memopt_vmm_hbm_bytes",
                  float(s.get("bytes_per_tier", {}).get("hbm", 0)),
                  help="Bytes currently in HBM tier")
            r.set("memopt_vmm_dram_bytes",
                  float(s.get("bytes_per_tier", {}).get("dram", 0)),
                  help="Bytes currently in DRAM tier")
            r.set("memopt_vmm_nvme_bytes",
                  float(s.get("bytes_per_tier", {}).get("nvme", 0)),
                  help="Bytes currently in NVMe tier")

            # Virtual/physical ratio — the Pillar 1 headline metric
            hbm    = float(s.get("bytes_per_tier", {}).get("hbm", 0))
            total  = sum(
                float(v) for v in s.get("bytes_per_tier", {}).values()
            )
            if hbm > 0:
                r.set("memopt_vmm_virtual_physical_ratio",
                      round(total / hbm, 3),
                      help="Ratio of virtual memory served to physical HBM")

            prefetch = s.get("prefetch", {})
            r.set("memopt_vmm_prefetch_hit_rate_pct",
                  float(prefetch.get("hit_rate_pct", 0)),
                  help="VMM prefetch hit rate percent")
        except Exception as e:
            logger.debug(f"VMM metrics collection failed: {e}")

    def _collect_gkd(self) -> None:
        gkd = self._gkd
        if gkd is None:
            return
        try:
            s = gkd.stats()
            r = self.registry
            r.set("memopt_gkd_total_lookups",
                  float(s.get("total_lookups", 0)),
                  help="Total GKD lookups", type="counter")
            r.set("memopt_gkd_cache_hits",
                  float(s.get("cache_hits", 0)),
                  help="GKD cache hits", type="counter")
            r.set("memopt_gkd_hit_rate_pct",
                  float(s.get("hit_rate_pct", 0)),
                  help="GKD hit rate percent")
            r.set("memopt_gkd_hbm_saved_bytes",
                  float(s.get("estimated_hbm_saved_gb", 0)) * 1e9,
                  help="Estimated HBM bytes saved by GKD deduplication",
                  type="counter")
            r.set("memopt_gkd_collision_detections",
                  float(s.get("collision_detections_total", 0)),
                  help="SHA-256 collision detections (must be 0)",
                  type="counter")
        except Exception as e:
            logger.debug(f"GKD metrics collection failed: {e}")

    def _collect_hypervisor(self) -> None:
        hyp = self._hypervisor
        if hyp is None:
            return
        try:
            s = hyp.stats()
            r = self.registry
            r.set("memopt_cluster_nodes_total",
                  float(s.get("cluster_nodes_total", 0)),
                  help="Total nodes in cluster")
            r.set("memopt_cluster_nodes_reachable",
                  float(s.get("cluster_nodes_reachable", 0)),
                  help="Reachable nodes in cluster")
            r.set("memopt_cluster_borrows_total",
                  float(s.get("borrows_total", 0)),
                  help="Total cross-node memory borrow requests",
                  type="counter")
            r.set("memopt_cluster_bytes_borrowed",
                  float(s.get("bytes_borrowed_gb", 0)) * 1e9,
                  help="Total bytes borrowed from remote nodes",
                  type="counter")
            r.set("memopt_cluster_free_dram_bytes",
                  float(s.get("cluster_free_dram_gb", 0)) * 1e9,
                  help="Total free DRAM across cluster")

            gkd_s = s.get("gkd_stats", {})
            if gkd_s:
                r.set("memopt_gkd_hit_rate_pct",
                      float(gkd_s.get("hit_rate_pct", 0)))
        except Exception as e:
            logger.debug(f"Hypervisor metrics collection failed: {e}")

    def _collect_kernel_hooks(self) -> None:
        hooks = self._kernel_hooks
        if hooks is None:
            return
        try:
            s = hooks.stats()
            r = self.registry

            # Cache stats
            cache_s = s.get("cache", {})
            r.set("memopt_kernel_cache_hits",
                  float(cache_s.get("cache_hits", 0)),
                  help="Kernel cache hits", type="counter")
            r.set("memopt_kernel_cache_hit_rate_pct",
                  float(cache_s.get("hit_rate_pct", 0)),
                  help="Kernel cache hit rate percent")

            # Fallback counts per op
            for op_name, count in s.get("fallback_counts", {}).items():
                r.set("memopt_kernel_fallbacks_total",
                      float(count),
                      labels={"op": op_name},
                      help="Kernel fallback count by op",
                      type="counter")

            # Optimizer stats
            opt_s = s.get("optimizer", {})
            gen_s = opt_s.get("generator_stats", {})
            r.set("memopt_kernel_synthesis_succeeded",
                  float(gen_s.get("total_succeeded", 0)),
                  help="Kernel synthesis successes", type="counter")
            r.set("memopt_kernel_synthesis_discarded",
                  float(gen_s.get("total_discarded", 0)),
                  help="Kernels discarded (not faster than baseline)",
                  type="counter")
        except Exception as e:
            logger.debug(f"Kernel hooks metrics collection failed: {e}")

    def _collect_power(self) -> None:
        sampler = self._power_sampler
        if sampler is None:
            return
        try:
            report = sampler.report()
            if report is None:
                return
            r = self.registry
            r.set("memopt_power_avg_watts",
                  float(getattr(report, "avg_watts", 0)),
                  help="Average GPU power draw in watts")
            r.set("memopt_power_peak_watts",
                  float(getattr(report, "peak_watts", 0)),
                  help="Peak GPU power draw in watts")
        except Exception as e:
            logger.debug(f"Power metrics collection failed: {e}")

    def _collect_request_counters(self) -> None:
        with self._lock:
            tokens   = self._tokens_total
            requests = self._requests_total
        r = self.registry
        r.set("memopt_tokens_generated_total",
              float(tokens),
              help="Total tokens generated since startup",
              type="counter")
        r.set("memopt_requests_total",
              float(requests),
              help="Total inference requests since startup",
              type="counter")

    def _poll_loop(self) -> None:
        while self._running:
            try:
                self.collect_once()
            except Exception as e:
                logger.debug(f"Collection cycle error: {e}")
            time.sleep(self._interval)
