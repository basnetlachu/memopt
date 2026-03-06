"""
Tests for FleetIntelligence — no GPU required.
Uses a real SQLite database (temp file), real threading, real time.
"""

import json
import os
import tempfile
import threading
import time
from datetime import datetime

import pytest

from memopt.fleet.intelligence import (
    DriftEvent,
    FleetIntelligence,
    FleetSavingsReport,
    NodeMetrics,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test_fleet.db")


@pytest.fixture
def fleet(db_path):
    fi = FleetIntelligence(
        db_path=db_path,
        gpu_cost_per_hour=3.50,
        drift_threshold_warning_pct=10.0,
        drift_threshold_critical_pct=25.0,
        auto_remediate=False,   # Disable auto-remediation in tests
        check_interval_seconds=1,
    )
    yield fi


def _make_metrics(
    node="node-01", gpu=0, tps=100.0, optimized=True, backend="vllm", **kwargs
) -> NodeMetrics:
    return NodeMetrics(
        node_name=node,
        timestamp=time.time(),
        gpu_index=gpu,
        gpu_name="NVIDIA A100-SXM4-80GB",
        vram_used_mb=26 * 1024,
        vram_total_mb=80 * 1024,
        gpu_util_pct=75.0,
        power_watts=280.0,
        temperature_c=65.0,
        active_pid=12345,
        tokens_per_second=tps,
        optimization_applied=optimized,
        backend=backend,
        **kwargs,
    )


# ── Database init ─────────────────────────────────────────────────────────────

class TestDBInit:
    def test_creates_db_file(self, db_path):
        FleetIntelligence(db_path=db_path)
        assert os.path.exists(db_path)

    def test_creates_all_tables(self, db_path):
        import sqlite3
        FleetIntelligence(db_path=db_path)
        conn = sqlite3.connect(db_path)
        tables = {
            r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        conn.close()
        assert "node_metrics" in tables
        assert "drift_events" in tables
        assert "optimization_events" in tables

    def test_idempotent_init(self, db_path):
        """Calling __init__ twice must not fail (tables already exist)."""
        FleetIntelligence(db_path=db_path)
        FleetIntelligence(db_path=db_path)


# ── Metrics ingestion ─────────────────────────────────────────────────────────

class TestMetricsIngestion:
    def test_ingest_persists_to_db(self, fleet, db_path):
        import sqlite3
        m = _make_metrics(node="node-01", tps=120.0)
        fleet.ingest_metrics(m)

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM node_metrics").fetchall()
        conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "node-01"   # node_name

    def test_ingest_updates_in_memory_state(self, fleet):
        m = _make_metrics(node="node-01", gpu=0, tps=200.0)
        fleet.ingest_metrics(m)
        assert "node-01:gpu0" in fleet.fleet_metrics
        assert fleet.fleet_metrics["node-01:gpu0"].tokens_per_second == 200.0

    def test_ingest_multiple_nodes(self, fleet):
        for node in ["node-01", "node-02", "node-03"]:
            fleet.ingest_metrics(_make_metrics(node=node))
        assert len(fleet.fleet_metrics) == 3

    def test_ingest_thread_safety(self, fleet):
        """Concurrent ingestion from multiple threads must not crash."""
        errors = []

        def worker(i):
            try:
                fleet.ingest_metrics(_make_metrics(node=f"node-{i:02d}", tps=float(i * 10)))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(fleet.fleet_metrics) == 20


# ── Optimization recording ─────────────────────────────────────────────────────

class TestOptimizationRecording:
    def test_record_optimization(self, fleet, db_path):
        import sqlite3
        fleet.record_optimization(
            node_name="node-01",
            pid=12345,
            model_name="meta-llama/Llama-2-13b",
            backend_before="huggingface",
            backend_after="vllm",
            tps_before=19.6,
            tps_after=1200.0,
            optimizations=["turbo_continuous_batching"],
            status="applied",
        )

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM optimization_events").fetchall()
        conn.close()
        assert len(rows) == 1
        # speedup ≈ 1200 / 19.6 ≈ 61.2
        row = rows[0]
        speedup = row[9]
        assert speedup is not None
        assert abs(speedup - 1200.0 / 19.6) < 1.0

    def test_speedup_computed_correctly(self, fleet, db_path):
        import sqlite3
        fleet.record_optimization(
            "n", 1, "m", "hf", "vllm",
            tps_before=100.0, tps_after=400.0,
            optimizations=[], status="applied",
        )
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT speedup FROM optimization_events").fetchone()
        conn.close()
        assert abs(row[0] - 4.0) < 0.01

    def test_record_with_none_tps(self, fleet):
        """Recording with no tps should not crash."""
        fleet.record_optimization(
            "n", 1, "m", "hf", "vllm",
            tps_before=None, tps_after=None,
            optimizations=[], status="applied",
        )


# ── Drift detection ───────────────────────────────────────────────────────────

class TestDriftDetection:
    def test_no_drift_at_baseline(self, fleet):
        key = "node-01:gpu0"
        fleet.set_baseline(key, 100.0)

        m = _make_metrics(node="node-01", gpu=0, tps=100.0)
        fleet.ingest_metrics(m)

        assert key not in fleet.active_drift

    def test_warning_drift_at_15_percent(self, fleet):
        key = "node-01:gpu0"
        fleet.set_baseline(key, 100.0)

        m = _make_metrics(node="node-01", gpu=0, tps=83.0)  # 17% drop
        fleet.ingest_metrics(m)

        assert key in fleet.active_drift
        assert fleet.active_drift[key].severity == "warning"

    def test_critical_drift_at_30_percent(self, fleet):
        key = "node-01:gpu0"
        fleet.set_baseline(key, 100.0)

        m = _make_metrics(node="node-01", gpu=0, tps=65.0)  # 35% drop
        fleet.ingest_metrics(m)

        assert key in fleet.active_drift
        assert fleet.active_drift[key].severity == "critical"

    def test_drift_clears_on_recovery(self, fleet):
        key = "node-01:gpu0"
        fleet.set_baseline(key, 100.0)

        # Trigger drift
        fleet.ingest_metrics(_make_metrics(node="node-01", gpu=0, tps=60.0))
        assert key in fleet.active_drift

        # Recovery
        fleet.ingest_metrics(_make_metrics(node="node-01", gpu=0, tps=98.0))
        assert key not in fleet.active_drift

    def test_drift_event_persisted_to_db(self, fleet, db_path):
        import sqlite3
        key = "node-01:gpu0"
        fleet.set_baseline(key, 100.0)
        fleet.ingest_metrics(_make_metrics(node="node-01", gpu=0, tps=60.0))

        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM drift_events").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_first_observation_becomes_baseline(self, fleet):
        """If no baseline set, first tps becomes the baseline."""
        m = _make_metrics(node="new-node", gpu=0, tps=500.0)
        fleet.ingest_metrics(m)
        key = "new-node:gpu0"
        assert key in fleet.node_baselines
        assert fleet.node_baselines[key] == 500.0

    def test_drop_pct_calculated_correctly(self, fleet):
        key = "node-01:gpu0"
        fleet.set_baseline(key, 200.0)
        fleet.ingest_metrics(_make_metrics(node="node-01", gpu=0, tps=140.0))  # 30% drop

        drift = fleet.active_drift[key]
        assert abs(drift.drop_pct - 30.0) < 0.1

    def test_none_tps_does_not_trigger_drift(self, fleet):
        key = "node-01:gpu0"
        fleet.set_baseline(key, 100.0)
        m = _make_metrics(node="node-01", gpu=0, tps=None)
        m.tokens_per_second = None
        fleet.ingest_metrics(m)
        assert key not in fleet.active_drift


# ── Savings calculation ────────────────────────────────────────────────────────

class TestSavingsCalculation:
    def test_no_events_returns_zero_savings(self, fleet):
        report = fleet.calculate_savings(hours=24)
        assert report.dollar_savings == 0.0
        assert report.gpu_hours_saved == 0.0
        assert report.throughput_multiplier == 1.0

    def test_savings_with_real_speedup(self, fleet):
        fleet.record_optimization(
            node_name="node-01", pid=1, model_name="llama-13b",
            backend_before="huggingface", backend_after="vllm",
            tps_before=20.0, tps_after=1200.0,
            optimizations=["turbo"], status="applied",
        )
        report = fleet.calculate_savings(hours=24)
        assert report.optimized_gpus == 1
        assert report.throughput_multiplier > 10.0
        assert report.gpu_hours_saved > 0.0
        assert report.dollar_savings > 0.0
        assert report.dollar_savings_annual > 365 * report.dollar_savings * 0.5

    def test_multiple_nodes_aggregate(self, fleet):
        for i in range(5):
            fleet.record_optimization(
                node_name=f"node-{i:02d}", pid=i, model_name="m",
                backend_before="hf", backend_after="vllm",
                tps_before=20.0, tps_after=200.0,
                optimizations=[], status="applied",
            )
        report = fleet.calculate_savings(hours=24)
        assert report.optimized_gpus == 5

    def test_top_savings_nodes_sorted(self, fleet):
        speedups = [("node-A", 2.0), ("node-B", 10.0), ("node-C", 5.0)]
        for node, ratio in speedups:
            fleet.record_optimization(
                node, 1, "m", "hf", "vllm",
                tps_before=100.0, tps_after=100.0 * ratio,
                optimizations=[], status="applied",
            )
        report = fleet.calculate_savings(hours=24)
        # Should be sorted descending by annual savings
        savings = [n["annual_saving"] for n in report.top_savings_nodes]
        assert savings == sorted(savings, reverse=True)

    def test_savings_period_filtering(self, fleet, db_path):
        """Events older than the period window must not count."""
        import sqlite3
        old_ts = time.time() - (48 * 3600)  # 48 hours ago

        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO optimization_events "
            "(node_name, timestamp, pid, model_name, backend_before, backend_after, "
            "tps_before, tps_after, speedup, optimizations, status) "
            "VALUES ('old-node', ?, 1, 'm', 'hf', 'vllm', 20.0, 200.0, 10.0, '[]', 'applied')",
            (old_ts,),
        )
        conn.commit()
        conn.close()

        report = fleet.calculate_savings(hours=24)
        assert report.optimized_gpus == 0

    def test_report_to_dict(self, fleet):
        fleet.record_optimization(
            "n", 1, "m", "hf", "vllm",
            100.0, 500.0, [], "applied",
        )
        report = fleet.calculate_savings(hours=24)
        d = report.to_dict()
        assert "dollar_savings" in d
        assert "throughput_multiplier" in d
        assert "top_savings_nodes" in d
        # Should be JSON-serializable
        json.dumps(d)

    def test_report_to_text(self, fleet):
        fleet.record_optimization(
            "node-01", 1, "llama-13b", "hf", "vllm",
            20.0, 1200.0, ["turbo"], "applied",
        )
        report = fleet.calculate_savings(hours=24)
        text = report.to_text()
        assert "FLEET SAVINGS REPORT" in text
        assert "$" in text
        assert "node-01" in text


# ── Fleet status snapshot ─────────────────────────────────────────────────────

class TestFleetStatus:
    def test_status_empty_fleet(self, fleet):
        status = fleet.get_fleet_status()
        assert status["nodes"] == 0
        assert status["gpus"] == 0
        assert status["active_drift"] == 0

    def test_status_counts_nodes_correctly(self, fleet):
        for i in range(3):
            fleet.ingest_metrics(_make_metrics(node=f"node-{i}", gpu=0, tps=100.0))
        status = fleet.get_fleet_status()
        assert status["nodes"] == 3

    def test_status_counts_drift(self, fleet):
        for i in range(2):
            key = f"node-{i}:gpu0"
            fleet.set_baseline(key, 100.0)
            fleet.ingest_metrics(_make_metrics(node=f"node-{i}", gpu=0, tps=50.0))

        status = fleet.get_fleet_status()
        assert status["critical_drift"] == 2

    def test_status_is_json_serializable(self, fleet):
        fleet.ingest_metrics(_make_metrics())
        status = fleet.get_fleet_status()
        json.dumps(status)  # Must not raise


# ── Background monitoring ─────────────────────────────────────────────────────

class TestBackgroundMonitor:
    def test_start_stop_monitoring(self, fleet):
        call_count = [0]

        def sampler():
            call_count[0] += 1
            return [_make_metrics(tps=float(call_count[0] * 10))]

        fleet.start_monitoring(node_sampler=sampler)
        time.sleep(2.5)   # Allow ~2 ticks at 1-second interval
        fleet.stop_monitoring()

        assert call_count[0] >= 2

    def test_double_start_does_not_crash(self, fleet):
        fleet.start_monitoring()
        fleet.start_monitoring()  # Should warn but not raise
        fleet.stop_monitoring()

    def test_monitor_without_sampler_runs_idle(self, fleet):
        fleet.start_monitoring(node_sampler=None)
        time.sleep(1.5)
        fleet.stop_monitoring()
        # No crash is the assertion

    def test_sampler_exception_does_not_stop_monitor(self, fleet):
        call_count = [0]

        def bad_sampler():
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("Simulated sampler failure")
            return [_make_metrics(tps=100.0)]

        fleet.start_monitoring(node_sampler=bad_sampler)
        time.sleep(2.5)
        fleet.stop_monitoring()

        # Monitor must have survived the first failure and continued
        assert call_count[0] >= 2


# ── Baseline management ───────────────────────────────────────────────────────

class TestBaseline:
    def test_set_baseline(self, fleet):
        fleet.set_baseline("node-01:gpu0", 500.0)
        assert fleet.node_baselines["node-01:gpu0"] == 500.0

    def test_update_baseline(self, fleet):
        fleet.set_baseline("node-01:gpu0", 200.0)
        fleet.set_baseline("node-01:gpu0", 500.0)
        assert fleet.node_baselines["node-01:gpu0"] == 500.0
