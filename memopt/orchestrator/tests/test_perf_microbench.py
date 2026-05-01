"""Performance microbench (orchestrator v1 Commit 11; design §2.6,
§3.4). All tests marked @pytest.mark.perf — SKIPPED by default in
the regression run; invoked with `pytest -m perf`. Asserts ordering
only (per §3.4.4 — absolute thresholds are noise across hardware)."""
from __future__ import annotations

import json
import os
import pathlib
import platform
import statistics
import subprocess
import sys
import time
from typing import Dict

import pytest

import memopt
import memopt.orchestrator as orch
from memopt.orchestrator.access import AccessTracker
from memopt.orchestrator.config import OrchestratorConfig
from memopt.orchestrator.coordinator import OrchestratorCoordinator
from memopt.orchestrator.policy import (
    Decision,
    LRUWatermarkPolicy,
    PolicyEngine,
    PolicySnapshot,
)
from memopt.orchestrator.predict import Predictor
from memopt.orchestrator.telemetry import TelemetryCollector
from memopt.substrate.events import Event
from memopt.substrate.manager import AllocationManager


pytestmark = pytest.mark.perf


_TRIALS = 10_000
_WARMUP = 1_000
_CYCLE_PERIOD_TARGET_NS = 50 * 1_000_000  # 50 ms (default cycle)


def _percentile(samples, p):
    s = sorted(samples)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return s[k]


def _summary(samples):
    samples = samples[_WARMUP:]
    return {
        "min_ns": min(samples),
        "median_ns": statistics.median(samples),
        "p99_ns": _percentile(samples, 99),
        "max_ns": max(samples),
        "n": len(samples),
    }


# ── benches ───────────────────────────────────────────────────────


def _bench_event_ingest():
    tracker = AccessTracker()
    samples = []
    for i in range(_TRIALS):
        ev = Event(
            kind="alloc", timestamp_ns=time.monotonic_ns(),
            handle_id=i, tenant="alice", tag="t", size_bytes=1,
            to_placement="hbm",
        )
        t0 = time.perf_counter_ns()
        tracker.record(ev)
        samples.append(time.perf_counter_ns() - t0)
    return _summary(samples)


def _bench_predictor_query():
    p = Predictor(min_confidence=0.0)
    for _ in range(50):
        p.observe("alice", "a")
        p.observe("alice", "b")
    samples = []
    for _ in range(_TRIALS):
        t0 = time.perf_counter_ns()
        p.predict("alice", "a", top_k=10)
        samples.append(time.perf_counter_ns() - t0)
    return _summary(samples)


def _bench_policy_evaluation():
    pol = LRUWatermarkPolicy()
    snap = PolicySnapshot(
        per_tenant_pressure={f"t{i}": 0.95 for i in range(16)},
        per_placement_used_bytes={},
        lru_candidates={
            (f"t{i}", "hbm"): list(range(64))
            for i in range(16)
        },
        predictor=None,
        ts_ns=0,
    )
    samples = []
    for _ in range(_TRIALS):
        t0 = time.perf_counter_ns()
        pol.evaluate(snap)
        samples.append(time.perf_counter_ns() - t0)
    return _summary(samples)


def _bench_cycle_period():
    # Measure the wake-to-wake interval of the coordinator thread.
    cfg = OrchestratorConfig(cycle_period_ms=10)
    mgr = AllocationManager.get()
    coord = OrchestratorCoordinator(
        manager=mgr, config=cfg,
        tracker=AccessTracker(),
        predictor=Predictor(),
        policy_engine=PolicyEngine(),
        telemetry=TelemetryCollector(),
    )
    coord.start()
    try:
        time.sleep(0.5)
        cycles = coord._telemetry.snapshot()["coordinator"]["cycles"]
    finally:
        coord.stop()
    avg_ns = int(0.5 * 1e9 / max(1, cycles))
    return {
        "min_ns": avg_ns, "median_ns": avg_ns,
        "p99_ns": avg_ns, "max_ns": avg_ns, "n": cycles,
    }


def _bench_advisory_turnaround():
    # Cost of a no-op policy callback inside the policy engine —
    # represents the upper bound of the placement="auto" advisory path.
    eng = PolicyEngine()

    class _Noop:
        def evaluate(self, snapshot):
            return []
    eng.register(_Noop())
    snap = PolicySnapshot(
        per_tenant_pressure={}, per_placement_used_bytes={},
        lru_candidates={}, predictor=None, ts_ns=0,
    )
    samples = []
    for _ in range(_TRIALS):
        t0 = time.perf_counter_ns()
        eng.evaluate(snap)
        samples.append(time.perf_counter_ns() - t0)
    return _summary(samples)


def _bench_decision_to_emit():
    AllocationManager.reset()
    h = memopt.alloc(4096, placement="hbm", tenant="alice", tag="t")
    mgr = AllocationManager.get()
    samples = []
    try:
        for _ in range(_TRIALS):
            ev = Event(
                kind="evict", timestamp_ns=time.monotonic_ns(),
                handle_id=h.handle_id, tenant="alice", tag="t",
                size_bytes=4096, from_placement="hbm", to_placement="dram",
            )
            t0 = time.perf_counter_ns()
            mgr._emit_orchestrator_event(ev)
            samples.append(time.perf_counter_ns() - t0)
    finally:
        memopt.free(h)
        AllocationManager.reset()
    return _summary(samples)


# ── tests (10 total) ──────────────────────────────────────────────


# 6 print-only timing tests — always pass; report numbers.
def test_event_ingest_p50_p99(capsys):
    s = _bench_event_ingest()
    print(f"[bench] event_ingest median={s['median_ns']}ns p99={s['p99_ns']}ns")
    assert s["median_ns"] >= 0


def test_predictor_query_p50_p99(capsys):
    s = _bench_predictor_query()
    print(f"[bench] predictor_query median={s['median_ns']}ns p99={s['p99_ns']}ns")
    assert s["median_ns"] >= 0


def test_policy_evaluation_p50_p99(capsys):
    s = _bench_policy_evaluation()
    print(f"[bench] policy_eval median={s['median_ns']}ns p99={s['p99_ns']}ns")
    assert s["median_ns"] >= 0


def test_decision_pump_cycle_steady_state(capsys):
    s = _bench_cycle_period()
    print(f"[bench] cycle_period avg={s['median_ns']}ns over {s['n']} cycles")
    assert s["median_ns"] > 0


def test_advisory_turnaround_p99(capsys):
    s = _bench_advisory_turnaround()
    print(f"[bench] advisory_turnaround p99={s['p99_ns']}ns")
    assert s["p99_ns"] >= 0


def test_decision_to_emit_p50_p99(capsys):
    s = _bench_decision_to_emit()
    print(f"[bench] decision_to_emit median={s['median_ns']}ns p99={s['p99_ns']}ns")
    assert s["median_ns"] >= 0


# 3 ordering-only assertions per §3.4.3.
def test_ordering_event_ingest_lt_predictor_query():
    e = _bench_event_ingest()
    p = _bench_predictor_query()
    assert e["median_ns"] < p["median_ns"], (e, p)


def test_ordering_predictor_query_lt_policy_evaluation():
    p = _bench_predictor_query()
    pol = _bench_policy_evaluation()
    assert p["median_ns"] < pol["median_ns"], (p, pol)


def test_ordering_policy_evaluation_lt_cycle_period():
    pol = _bench_policy_evaluation()
    assert pol["median_ns"] < _CYCLE_PERIOD_TARGET_NS, pol


# 1 JSON-write test.
def test_results_persisted_to_json(tmp_path):
    bench_dir = pathlib.Path("/tmp/memopt-orchestrator-bench")
    bench_dir.mkdir(parents=True, exist_ok=True)
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode().strip()
    except Exception:
        sha = "unknown"
    out_path = bench_dir / f"{sha}.json"
    record = []
    for name, fn in (
        ("event_ingest", _bench_event_ingest),
        ("predictor_query", _bench_predictor_query),
        ("policy_evaluation", _bench_policy_evaluation),
        ("decision_to_emit", _bench_decision_to_emit),
    ):
        s = fn()
        record.append({
            "name": name,
            "median_ns": s["median_ns"],
            "p99_ns": s["p99_ns"],
            "hardware": platform.platform(),
            "python_version": platform.python_version(),
            "ts_ns": time.time_ns(),
        })
    out_path.write_text(json.dumps(record, indent=2))
    assert out_path.exists()
    assert json.loads(out_path.read_text())[0]["name"] == "event_ingest"
