"""
Fleet Intelligence Layer — monitors 100+ GPUs, detects performance drift,
auto-remediates, and reports dollar savings for enterprise leadership.

Architecture:
  - SQLite database stores all metrics, drift events, optimization events
  - Background monitor thread polls nodes every `check_interval` seconds
  - Drift detection compares current tok/s vs established baseline
  - Auto-remediation fires AutoMigrationEngine on critical drift
  - Savings calculation converts speedup to GPU-hours and dollars
"""

import json
import logging
import sqlite3
import statistics
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("memopt.fleet")


@dataclass
class NodeMetrics:
    node_name:            str
    timestamp:            float
    gpu_index:            int
    gpu_name:             str
    vram_used_mb:         int
    vram_total_mb:        int
    gpu_util_pct:         float
    power_watts:          float
    temperature_c:        float
    active_pid:           Optional[int]
    tokens_per_second:    Optional[float]
    optimization_applied: bool
    backend:              str


@dataclass
class DriftEvent:
    node_name:           str
    gpu_index:           int
    pid:                 int
    detected_at:         float
    baseline_tps:        float
    current_tps:         float
    drop_pct:            float
    severity:            str
    auto_remediated:     bool
    remediation_result:  Optional[str]


@dataclass
class FleetSavingsReport:
    period_start:           datetime
    period_end:             datetime
    total_nodes:            int
    total_gpus:             int
    optimized_gpus:         int
    total_tokens_served:    int
    baseline_tokens_served: int
    throughput_multiplier:  float
    gpu_hours_saved:        float
    dollar_savings:         float
    dollar_savings_annual:  float
    gpu_cost_per_hour:      float
    top_savings_nodes:      List[dict]

    def to_text(self) -> str:
        lines = [
            "=" * 68,
            "  FLEET SAVINGS REPORT  —  memopt",
            "=" * 68,
            f"  Period        : {self.period_start.strftime('%Y-%m-%d %H:%M')} → "
            f"{self.period_end.strftime('%Y-%m-%d %H:%M')}",
            f"  Fleet size    : {self.total_nodes} nodes / {self.total_gpus} GPUs",
            f"  Optimized GPUs: {self.optimized_gpus}",
            f"  Avg speedup   : {self.throughput_multiplier:.2f}x",
            f"  GPU-hours saved (period): {self.gpu_hours_saved:.1f} h",
            f"  Savings (period): ${self.dollar_savings:,.2f}",
            f"  Savings (annual est.): ${self.dollar_savings_annual:,.0f}",
            "",
            "  Top savings nodes:",
        ]
        for node in self.top_savings_nodes[:5]:
            lines.append(
                f"    {node['node']:<30}  {node['avg_speedup']:.2f}x  "
                f"${node['annual_saving']:>8,.0f}/yr"
            )
        lines.append("=" * 68)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "period_start":           self.period_start.isoformat(),
            "period_end":             self.period_end.isoformat(),
            "total_nodes":            self.total_nodes,
            "total_gpus":             self.total_gpus,
            "optimized_gpus":         self.optimized_gpus,
            "throughput_multiplier":  round(self.throughput_multiplier, 3),
            "gpu_hours_saved":        round(self.gpu_hours_saved, 2),
            "dollar_savings":         round(self.dollar_savings, 2),
            "dollar_savings_annual":  round(self.dollar_savings_annual, 0),
            "gpu_cost_per_hour":      self.gpu_cost_per_hour,
            "top_savings_nodes":      self.top_savings_nodes,
        }


class FleetIntelligence:

    def __init__(
        self,
        db_path: str = str(Path.home() / ".memopt" / "fleet.db"),
        gpu_cost_per_hour: float = 3.50,
        drift_threshold_warning_pct: float = 10.0,
        drift_threshold_critical_pct: float = 25.0,
        auto_remediate: bool = True,
        check_interval_seconds: int = 30,
    ):
        self.db_path                  = db_path
        self.gpu_cost_per_hour        = gpu_cost_per_hour
        self.drift_threshold_warning  = drift_threshold_warning_pct
        self.drift_threshold_critical = drift_threshold_critical_pct
        self.auto_remediate           = auto_remediate
        self.check_interval           = check_interval_seconds

        self.node_baselines: Dict[str, float]       = {}
        self.active_drift:   Dict[str, DriftEvent]  = {}
        self.fleet_metrics:  Dict[str, NodeMetrics] = {}

        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._db_lock = threading.Lock()

        self._init_db()

    # ── DATABASE ─────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS node_metrics (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_name            TEXT    NOT NULL,
                    timestamp            REAL    NOT NULL,
                    gpu_index            INTEGER,
                    gpu_name             TEXT,
                    vram_used_mb         INTEGER,
                    vram_total_mb        INTEGER,
                    gpu_util_pct         REAL,
                    power_watts          REAL,
                    temperature_c        REAL,
                    active_pid           INTEGER,
                    tokens_per_second    REAL,
                    optimization_applied INTEGER,
                    backend              TEXT
                );

                CREATE TABLE IF NOT EXISTS drift_events (
                    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_name            TEXT    NOT NULL,
                    gpu_index            INTEGER,
                    pid                  INTEGER,
                    detected_at          REAL,
                    baseline_tps         REAL,
                    current_tps          REAL,
                    drop_pct             REAL,
                    severity             TEXT,
                    auto_remediated      INTEGER,
                    remediation_result   TEXT
                );

                CREATE TABLE IF NOT EXISTS optimization_events (
                    id             INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_name      TEXT    NOT NULL,
                    timestamp      REAL    NOT NULL,
                    pid            INTEGER,
                    model_name     TEXT,
                    backend_before TEXT,
                    backend_after  TEXT,
                    tps_before     REAL,
                    tps_after      REAL,
                    speedup        REAL,
                    optimizations  TEXT,
                    status         TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_metrics_node_ts
                    ON node_metrics(node_name, timestamp);
                CREATE INDEX IF NOT EXISTS idx_opt_events_ts
                    ON optimization_events(timestamp);
            """)
        logger.info(json.dumps({"event": "fleet_db_init", "path": self.db_path}))

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ── METRICS INGESTION ─────────────────────────────────────────────────

    def ingest_metrics(self, metrics: NodeMetrics) -> None:
        """Called every scan cycle. Thread-safe."""
        key = f"{metrics.node_name}:gpu{metrics.gpu_index}"
        self.fleet_metrics[key] = metrics

        with self._db_lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO node_metrics
                        (node_name, timestamp, gpu_index, gpu_name,
                         vram_used_mb, vram_total_mb, gpu_util_pct,
                         power_watts, temperature_c, active_pid,
                         tokens_per_second, optimization_applied, backend)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        metrics.node_name, metrics.timestamp, metrics.gpu_index,
                        metrics.gpu_name, metrics.vram_used_mb, metrics.vram_total_mb,
                        metrics.gpu_util_pct, metrics.power_watts, metrics.temperature_c,
                        metrics.active_pid, metrics.tokens_per_second,
                        int(metrics.optimization_applied), metrics.backend,
                    ),
                )

        if metrics.tokens_per_second is not None:
            self._check_drift(key, metrics)

    def record_optimization(
        self,
        node_name: str,
        pid: int,
        model_name: str,
        backend_before: str,
        backend_after: str,
        tps_before: Optional[float],
        tps_after: Optional[float],
        optimizations: List[str],
        status: str = "applied",
    ) -> None:
        speedup = None
        if tps_before and tps_after and tps_before > 0:
            speedup = tps_after / tps_before

        with self._db_lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO optimization_events
                        (node_name, timestamp, pid, model_name,
                         backend_before, backend_after, tps_before, tps_after,
                         speedup, optimizations, status)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        node_name, time.time(), pid, model_name,
                        backend_before, backend_after,
                        tps_before, tps_after, speedup,
                        json.dumps(optimizations), status,
                    ),
                )

        if speedup:
            logger.info(
                json.dumps({
                    "event": "optimization_recorded",
                    "node": node_name, "pid": pid,
                    "speedup": round(speedup, 3),
                    "backend": f"{backend_before}→{backend_after}",
                })
            )

    # ── DRIFT DETECTION ───────────────────────────────────────────────────

    def set_baseline(self, node_key: str, tps: float) -> None:
        self.node_baselines[node_key] = tps
        logger.info(json.dumps({"event": "baseline_set", "node": node_key, "tps": round(tps, 2)}))

    def _check_drift(self, node_key: str, metrics: NodeMetrics) -> None:
        current = metrics.tokens_per_second
        if not current or current <= 0:
            return

        if node_key not in self.node_baselines:
            # First observation — establish baseline
            self.node_baselines[node_key] = current
            return

        baseline = self.node_baselines[node_key]
        drop_pct = ((baseline - current) / baseline) * 100.0

        if drop_pct < self.drift_threshold_warning:
            if node_key in self.active_drift:
                del self.active_drift[node_key]
                logger.info(json.dumps({"event": "drift_resolved", "node": node_key}))
            return

        severity = (
            "critical" if drop_pct >= self.drift_threshold_critical
            else "warning"
        )

        drift = DriftEvent(
            node_name=metrics.node_name,
            gpu_index=metrics.gpu_index,
            pid=metrics.active_pid or -1,
            detected_at=time.time(),
            baseline_tps=baseline,
            current_tps=current,
            drop_pct=drop_pct,
            severity=severity,
            auto_remediated=False,
            remediation_result=None,
        )
        self.active_drift[node_key] = drift

        logger.warning(
            json.dumps({
                "event":    "drift_detected",
                "severity": severity,
                "node":     node_key,
                "drop_pct": round(drop_pct, 1),
                "baseline": round(baseline, 1),
                "current":  round(current, 1),
            })
        )

        # Persist
        with self._db_lock:
            with self._get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO drift_events
                        (node_name, gpu_index, pid, detected_at,
                         baseline_tps, current_tps, drop_pct,
                         severity, auto_remediated, remediation_result)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        drift.node_name, drift.gpu_index, drift.pid,
                        drift.detected_at, drift.baseline_tps, drift.current_tps,
                        drift.drop_pct, drift.severity, 0, None,
                    ),
                )

        if severity == "critical" and self.auto_remediate:
            threading.Thread(
                target=self._auto_remediate,
                args=(node_key, drift),
                daemon=True,
            ).start()

    def _auto_remediate(self, node_key: str, drift: DriftEvent) -> None:
        logger.info(json.dumps({"event": "auto_remediate_start", "node": node_key}))

        try:
            from memopt.migration.engine import AutoMigrationEngine
            from memopt.profiler.roofline import RooflineProfiler

            profiler = RooflineProfiler()
            hw = profiler.profile_gpu(drift.gpu_index)
            hw_dict = {
                "gpu_indices":   [drift.gpu_index],
                "vram_total_mb": hw.vram_total_mb,
                "vram_free_mb":  hw.vram_free_mb,
                "gpu_name":      hw.gpu_name,
            }

            engine = AutoMigrationEngine()
            plan   = engine.build_plan(drift.pid, hw_dict)
            result = engine.execute(plan)

            drift.auto_remediated    = True
            drift.remediation_result = "success" if result.success else f"failed: {result.error}"

            # Update DB
            with self._db_lock:
                with self._get_conn() as conn:
                    conn.execute(
                        """
                        UPDATE drift_events SET auto_remediated=1, remediation_result=?
                        WHERE node_name=? AND detected_at=?
                        """,
                        (drift.remediation_result, drift.node_name, drift.detected_at),
                    )

            logger.info(
                json.dumps({
                    "event":  "auto_remediate_complete",
                    "node":   node_key,
                    "result": drift.remediation_result,
                })
            )

        except Exception as e:
            drift.remediation_result = f"error: {e}"
            logger.error(
                json.dumps({"event": "auto_remediate_error", "node": node_key, "error": str(e)}),
                exc_info=True,
            )

    # ── SAVINGS CALCULATION ───────────────────────────────────────────────

    def calculate_savings(self, hours: float = 24.0) -> FleetSavingsReport:
        """
        Convert measured speedups into GPU-hours and dollar savings.

        Math:
          speedup = tps_after / tps_before
          Without memopt you'd need `speedup` more GPUs to serve the same load.
          GPU-hours saved = optimized_gpus × hours × (1 - 1/speedup)
          Dollar savings  = gpu_hours_saved × gpu_cost_per_hour
        """
        since = time.time() - (hours * 3600)

        with self._get_conn() as conn:
            events = conn.execute(
                """
                SELECT node_name, tps_before, tps_after, speedup, timestamp
                FROM optimization_events
                WHERE timestamp > ? AND status = 'applied'
                  AND tps_before IS NOT NULL AND tps_after IS NOT NULL
                """,
                (since,),
            ).fetchall()

            nodes = conn.execute(
                "SELECT DISTINCT node_name FROM node_metrics WHERE timestamp > ?",
                (since,),
            ).fetchall()

            gpus = conn.execute(
                "SELECT DISTINCT node_name, gpu_index FROM node_metrics WHERE timestamp > ?",
                (since,),
            ).fetchall()

        total_nodes = len(nodes)
        total_gpus  = len(gpus)

        if not events:
            return FleetSavingsReport(
                period_start=datetime.fromtimestamp(since),
                period_end=datetime.now(),
                total_nodes=total_nodes,
                total_gpus=total_gpus,
                optimized_gpus=0,
                total_tokens_served=0,
                baseline_tokens_served=0,
                throughput_multiplier=1.0,
                gpu_hours_saved=0.0,
                dollar_savings=0.0,
                dollar_savings_annual=0.0,
                gpu_cost_per_hour=self.gpu_cost_per_hour,
                top_savings_nodes=[],
            )

        speedups       = [float(e["speedup"]) for e in events if e["speedup"]]
        avg_speedup    = statistics.mean(speedups) if speedups else 1.0
        optimized_gpus = len(set(e["node_name"] for e in events))

        gpu_hours_saved       = optimized_gpus * hours * (1.0 - 1.0 / max(avg_speedup, 1.0))
        dollar_savings        = gpu_hours_saved * self.gpu_cost_per_hour
        dollar_savings_annual = dollar_savings * (8760.0 / hours)

        # Per-node savings
        node_speedups: Dict[str, List[float]] = {}
        for e in events:
            if e["speedup"]:
                node_speedups.setdefault(e["node_name"], []).append(float(e["speedup"]))

        top_nodes = sorted(
            [
                {
                    "node":          node,
                    "avg_speedup":   round(statistics.mean(sps), 3),
                    "annual_saving": round(
                        self.gpu_cost_per_hour * 8760.0 * (1.0 - 1.0 / statistics.mean(sps)), 2
                    ),
                }
                for node, sps in node_speedups.items()
            ],
            key=lambda x: x["annual_saving"],
            reverse=True,
        )[:10]

        return FleetSavingsReport(
            period_start=datetime.fromtimestamp(since),
            period_end=datetime.now(),
            total_nodes=total_nodes,
            total_gpus=total_gpus,
            optimized_gpus=optimized_gpus,
            total_tokens_served=0,
            baseline_tokens_served=0,
            throughput_multiplier=round(avg_speedup, 3),
            gpu_hours_saved=round(gpu_hours_saved, 2),
            dollar_savings=round(dollar_savings, 2),
            dollar_savings_annual=round(dollar_savings_annual, 2),
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            top_savings_nodes=top_nodes,
        )

    # ── FLEET STATUS ──────────────────────────────────────────────────────

    def get_fleet_status(self) -> dict:
        """Current in-memory fleet snapshot — safe to call at any time."""
        return {
            "nodes":          len({m.node_name for m in self.fleet_metrics.values()}),
            "gpus":           len(self.fleet_metrics),
            "active_drift":   len(self.active_drift),
            "critical_drift": sum(
                1 for d in self.active_drift.values() if d.severity == "critical"
            ),
            "metrics_snapshot": {
                k: {
                    "gpu_name":   v.gpu_name,
                    "backend":    v.backend,
                    "tps":        round(v.tokens_per_second, 1) if v.tokens_per_second else None,
                    "util_pct":   round(v.gpu_util_pct, 1),
                    "vram_used":  v.vram_used_mb,
                    "vram_total": v.vram_total_mb,
                    "optimized":  v.optimization_applied,
                }
                for k, v in self.fleet_metrics.items()
            },
            "drift_alerts": {
                k: {
                    "severity":   d.severity,
                    "drop_pct":   round(d.drop_pct, 1),
                    "baseline":   round(d.baseline_tps, 1),
                    "current":    round(d.current_tps, 1),
                    "remediated": d.auto_remediated,
                }
                for k, d in self.active_drift.items()
            },
        }

    def get_recent_drift_events(self, limit: int = 100) -> List[dict]:
        """Query drift events from SQLite. Used by control plane to show alerts."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """
                SELECT node_name, gpu_index, pid, detected_at,
                       baseline_tps, current_tps, drop_pct,
                       severity, auto_remediated, remediation_result
                FROM drift_events
                ORDER BY detected_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_avg_power_baseline(self, hours: float = 24.0) -> float:
        """Average power for un-optimized GPUs over last N hours (watts)."""
        since = time.time() - hours * 3600
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT AVG(power_watts) FROM node_metrics "
                "WHERE timestamp > ? AND optimization_applied = 0 AND power_watts > 0",
                (since,),
            ).fetchone()
        return round(float(row[0]), 1) if row and row[0] is not None else 0.0

    def get_avg_power_optimized(self, hours: float = 24.0) -> float:
        """Average power for optimized GPUs over last N hours (watts)."""
        since = time.time() - hours * 3600
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT AVG(power_watts) FROM node_metrics "
                "WHERE timestamp > ? AND optimization_applied = 1 AND power_watts > 0",
                (since,),
            ).fetchone()
        return round(float(row[0]), 1) if row and row[0] is not None else 0.0

    def get_power_reduction_pct(self, hours: float = 24.0) -> float:
        """Percentage power reduction from optimizations. 0.0 if no data."""
        baseline  = self.get_avg_power_baseline(hours)
        optimized = self.get_avg_power_optimized(hours)
        if baseline <= 0:
            return 0.0
        return round((baseline - optimized) / baseline * 100.0, 1)

    def get_electricity_savings(self, hours: float = 24.0, kwh_cost: float = 0.10) -> float:
        """Estimated electricity savings in dollars over N hours."""
        baseline  = self.get_avg_power_baseline(hours)
        optimized = self.get_avg_power_optimized(hours)
        if baseline <= 0 or optimized >= baseline:
            return 0.0
        kwh_saved = (baseline - optimized) / 1000.0 * hours
        return round(kwh_saved * kwh_cost, 4)

    def get_thermal_profile(self) -> dict:
        """
        Returns temperature and power data before/after optimization.
        Uses existing node_metrics rows — no new columns required.
        Only returns non-zero reductions when both pre and post data exist.
        """
        since = time.time() - 86400
        with self._get_conn() as conn:
            temp_before_row = conn.execute(
                "SELECT AVG(temperature_c) FROM node_metrics "
                "WHERE timestamp > ? AND optimization_applied = 0 AND temperature_c > 0",
                (since,),
            ).fetchone()
            temp_after_row = conn.execute(
                "SELECT AVG(temperature_c) FROM node_metrics "
                "WHERE timestamp > ? AND optimization_applied = 1 AND temperature_c > 0",
                (since,),
            ).fetchone()
            power_before_row = conn.execute(
                "SELECT AVG(power_watts) FROM node_metrics "
                "WHERE timestamp > ? AND optimization_applied = 0 AND power_watts > 0",
                (since,),
            ).fetchone()
            power_after_row = conn.execute(
                "SELECT AVG(power_watts) FROM node_metrics "
                "WHERE timestamp > ? AND optimization_applied = 1 AND power_watts > 0",
                (since,),
            ).fetchone()

        temp_before  = temp_before_row[0]  if temp_before_row  else None
        temp_after   = temp_after_row[0]   if temp_after_row   else None
        power_before = power_before_row[0] if power_before_row else None
        power_after  = power_after_row[0]  if power_after_row  else None

        temp_reduction_c   = 0.0
        temp_reduction_pct = 0.0
        power_reduction_pct = 0.0

        if temp_before and temp_after and temp_before > 0:
            temp_reduction_c   = temp_before - temp_after
            temp_reduction_pct = (temp_reduction_c / temp_before) * 100

        if power_before and power_after and power_before > 0:
            power_reduction_pct = ((power_before - power_after) / power_before) * 100

        health_score = min(100, round(
            (temp_reduction_pct * 0.6) + (power_reduction_pct * 0.4)
        ))

        return {
            "temp_before_c":       round(temp_before  or 0.0, 1),
            "temp_after_c":        round(temp_after   or 0.0, 1),
            "temp_reduction_c":    round(temp_reduction_c,    1),
            "temp_reduction_pct":  round(temp_reduction_pct,  1),
            "power_before_w":      round(power_before or 0.0, 1),
            "power_after_w":       round(power_after  or 0.0, 1),
            "power_reduction_pct": round(power_reduction_pct, 1),
            "health_score":        health_score,
            "data_available":      (temp_before is not None and temp_after is not None),
        }

    # ── BACKGROUND MONITOR ─────────────────────────────────────────────────

    def start_monitoring(
        self,
        node_sampler=None,
    ) -> None:
        """
        Start the background monitoring thread.

        Args:
            node_sampler: Optional callable() → List[NodeMetrics].
                          Provides fresh metrics each cycle.
                          If None, monitoring runs but only processes
                          metrics pushed via ingest_metrics().
        """
        if self._running:
            logger.warning("Monitor already running")
            return

        self._running = True
        self._node_sampler = node_sampler

        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="fleet-monitor",
        )
        self._monitor_thread.start()
        logger.info(json.dumps({"event": "fleet_monitor_started", "interval_s": self.check_interval}))

    def stop_monitoring(self, timeout: float = 5.0) -> None:
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=timeout)
            self._monitor_thread = None
        logger.info(json.dumps({"event": "fleet_monitor_stopped"}))

    def _monitor_loop(self) -> None:
        while self._running:
            try:
                if self._node_sampler:
                    fresh_metrics = self._node_sampler()
                    for m in fresh_metrics:
                        self.ingest_metrics(m)
            except Exception as e:
                logger.error(
                    json.dumps({"event": "monitor_loop_error", "error": str(e)}),
                    exc_info=True,
                )
            time.sleep(self.check_interval)
