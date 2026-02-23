#!/usr/bin/env python3
"""
MemoptAgent Comprehensive Test Suite
=====================================

4 real GPU tests — all assert on measured values, no mocks.

Run:
    python tests/test_agent_comprehensive.py
    pytest tests/test_agent_comprehensive.py -v -s
"""

from __future__ import annotations

import copy
import sys
import time
import traceback
from typing import Tuple

import torch
import torch.nn as nn

# ── Guard: must have CUDA ────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("FATAL: CUDA not available. These tests require a GPU.")
    sys.exit(1)

from memopt.agent import MemoptAgent, AgentReport


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bench_ms(fn, warmup: int = 5, iters: int = 50) -> float:
    """Median CUDA-event latency in ms."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# ── Results tracker ───────────────────────────────────────────────────────────

class _Result:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.metric = ""
        self.notes = ""

    def ok(self, metric: str, notes: str = "") -> "_Result":
        self.passed = True
        self.metric = metric
        self.notes = notes
        return self

    def fail(self, metric: str, notes: str = "") -> "_Result":
        self.passed = False
        self.metric = metric
        self.notes = notes
        return self


# =============================================================================
# TEST 1: ResNet50 — agent finds and commits a real optimization
# =============================================================================

def test_agent_resnet_finds_optimization() -> _Result:
    r = _Result("resnet_finds_optimization")
    print("\n" + "=" * 64)
    print("TEST 1: ResNet50 — agent must find a real optimization")
    print("=" * 64)

    try:
        import torchvision
    except ImportError:
        print("  SKIP: torchvision not installed")
        r.notes = "torchvision not installed"
        r.passed = True          # skip, not fail
        r.metric = "SKIPPED"
        return r

    model = torchvision.models.resnet50().cuda().eval()
    inp = torch.randn(8, 3, 224, 224, device="cuda")
    sample = {"x": inp}

    agent = MemoptAgent(target_speedup=2.0, max_rounds=5)
    report = agent.run(model, sample)

    # ── Assertions ────────────────────────────────────────────────────────────
    failures = []

    if report.final_speedup < 1.5:
        failures.append(
            f"speedup {report.final_speedup:.2f}x < 1.5x"
        )

    if len(report.optimizations_applied) == 0:
        failures.append("no optimization committed")

    if report.optimized_model is None:
        failures.append("optimized_model is None")

    if not report.honest_ceiling or len(report.honest_ceiling) < 10:
        failures.append("honest_ceiling missing or too short")

    # Verify optimized model actually produces correct-shape output
    if report.optimized_model is not None:
        try:
            with torch.no_grad():
                out = report.optimized_model(inp)
            if out.shape[0] != 8:
                failures.append(f"output batch dim wrong: {out.shape}")
        except Exception as exc:
            failures.append(f"optimized model forward failed: {exc}")

    stop = report.rounds[-1].stop_reason or "(no stop)"
    metric = (
        f"speedup={report.final_speedup:.2f}x | "
        f"applied={report.optimizations_applied}"
    )
    print(f"  {metric}")
    print(f"  stop={stop}")
    print(f"  ceiling={report.honest_ceiling[:80]}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric, f"stop={stop}")


# =============================================================================
# TEST 2: Compute-bound model — agent stops round 1 with OPTIMAL
# =============================================================================

def test_agent_stops_on_compute_bound() -> _Result:
    r = _Result("stops_on_compute_bound")
    print("\n" + "=" * 64)
    print("TEST 2: Large Linear — must stop OPTIMAL, commit nothing")
    print("=" * 64)

    model = nn.Linear(4096, 4096).cuda().eval()
    inp = {"input": torch.randn(64, 4096, device="cuda")}

    agent = MemoptAgent(target_speedup=2.0, max_rounds=10)
    report = agent.run(model, inp)

    failures = []
    stop = report.rounds[-1].stop_reason or ""

    if "OPTIMAL" not in stop:
        failures.append(f"expected OPTIMAL in stop_reason, got: {stop!r}")

    if len(report.optimizations_applied) != 0:
        failures.append(
            f"committed something on compute-bound: {report.optimizations_applied}"
        )

    if report.final_speedup > 1.10:
        failures.append(
            f"speedup {report.final_speedup:.2f}x > 1.10 on compute-bound model"
        )

    if len(report.rounds) > 1:
        failures.append(
            f"took {len(report.rounds)} rounds — should stop after round 1"
        )

    metric = f"rounds={len(report.rounds)} | speedup={report.final_speedup:.3f}x | stop={stop[:50]}"
    print(f"  {metric}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric)


# =============================================================================
# TEST 3: Rollback preserves correctness — output must not diverge
# =============================================================================

def test_rollback_preserves_correctness() -> _Result:
    r = _Result("rollback_preserves_correctness")
    print("\n" + "=" * 64)
    print("TEST 3: BERT — rolled-back opts must not corrupt model output")
    print("=" * 64)

    try:
        from transformers import BertModel, AutoTokenizer
    except ImportError:
        print("  SKIP: transformers not installed")
        r.notes = "transformers not installed"
        r.passed = True
        r.metric = "SKIPPED"
        return r

    import warnings
    warnings.filterwarnings("ignore")

    model = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    enc = tokenizer(
        "test input for correctness verification",
        return_tensors="pt",
    )
    inp = {k: v.cuda() for k, v in enc.items()}

    # Original output (before any agent touch)
    with torch.no_grad():
        orig_out = model(**inp).last_hidden_state.float()

    agent = MemoptAgent(target_speedup=2.0, max_rounds=5)
    report = agent.run(model, inp)

    # Use optimized model if anything was committed, else original
    opt_model = report.optimized_model if report.optimized_model is not None else model

    with torch.no_grad():
        opt_out = opt_model(**inp).last_hidden_state.float()

    max_diff = (orig_out - opt_out).abs().max().item()
    mean_diff = (orig_out - opt_out).abs().mean().item()

    failures = []
    if max_diff >= 0.25:
        failures.append(f"max_diff={max_diff:.4f} >= 0.25 threshold")

    metric = (
        f"max_diff={max_diff:.4f} | mean_diff={mean_diff:.5f} | "
        f"applied={report.optimizations_applied} | "
        f"rolled={report.optimizations_rolled_back}"
    )
    print(f"  {metric}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric)


# =============================================================================
# TEST 4: Cumulative speedup matches independent measurement (±15%)
# =============================================================================

def test_cumulative_speedup_vs_baseline() -> _Result:
    r = _Result("cumulative_vs_baseline")
    print("\n" + "=" * 64)
    print("TEST 4: ResNet50 — agent speedup must match real measurement ±15%")
    print("=" * 64)

    try:
        import torchvision
    except ImportError:
        print("  SKIP: torchvision not installed")
        r.notes = "torchvision not installed"
        r.passed = True
        r.metric = "SKIPPED"
        return r

    original = torchvision.models.resnet50().cuda().eval()
    inp = torch.randn(8, 3, 224, 224, device="cuda")

    # Independent baseline — measured BEFORE agent runs
    baseline_ms = _bench_ms(lambda: original(inp))
    print(f"  independent baseline: {baseline_ms:.2f} ms")

    agent = MemoptAgent(target_speedup=2.0, max_rounds=5)
    report = agent.run(copy.deepcopy(original), {"x": inp})

    metric = f"agent={report.final_speedup:.3f}x | baseline_ms={baseline_ms:.2f}"

    if report.optimized_model is None:
        print(f"  No optimization committed — nothing to verify")
        print(f"  {metric}")
        return r.ok(metric, "no commit — skip ratio check")

    # Independently measure optimized model
    optimized_ms = _bench_ms(lambda: report.optimized_model(inp))
    real_speedup = baseline_ms / optimized_ms

    ratio = min(real_speedup, report.final_speedup) / max(real_speedup, report.final_speedup)
    accuracy_pct = ratio * 100

    metric = (
        f"agent={report.final_speedup:.3f}x | real={real_speedup:.3f}x | "
        f"ratio={ratio:.3f} ({accuracy_pct:.1f}%) | "
        f"baseline={baseline_ms:.2f}ms optimized={optimized_ms:.2f}ms"
    )
    print(f"  {metric}")

    failures = []
    if ratio < 0.85:
        failures.append(
            f"speedup mismatch: agent={report.final_speedup:.3f}x "
            f"real={real_speedup:.3f}x ratio={ratio:.3f} < 0.85"
        )

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric, f"accuracy={accuracy_pct:.1f}%")


# =============================================================================
# Runner
# =============================================================================

def main() -> None:
    print("\n" + "=" * 64)
    print("MemoptAgent Comprehensive Test Suite")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 64)

    tests = [
        test_agent_resnet_finds_optimization,
        test_agent_stops_on_compute_bound,
        test_rollback_preserves_correctness,
        test_cumulative_speedup_vs_baseline,
    ]

    results = []
    for fn in tests:
        t0 = time.time()
        try:
            res = fn()
        except Exception as exc:
            res = _Result(fn.__name__)
            res.fail(f"EXCEPTION: {exc}", traceback.format_exc())
            print(f"  EXCEPTION: {exc}")
            traceback.print_exc()
        elapsed = time.time() - t0
        res.notes = f"{res.notes} [{elapsed:.1f}s]".strip()
        results.append(res)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("RESULTS")
    print("-" * 100)
    hdr = f"{'Test':<34}| {'Result':<8}| {'Key Metric':<42}| Notes"
    print(hdr)
    print("-" * 100)

    passed = 0
    failed = 0
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        if res.metric == "SKIPPED":
            status = "SKIP"
        if res.passed:
            passed += 1
        else:
            failed += 1
        metric = res.metric[:42] if res.metric else ""
        notes  = res.notes[:30] if res.notes else ""
        print(f"{res.name:<34}| {status:<8}| {metric:<42}| {notes}")

    print("-" * 100)
    print(f"Total: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 100)

    if failed > 0:
        sys.exit(1)


# ── pytest compatibility ───────────────────────────────────────────────────────

def test_resnet_finds_optimization_pytest():
    """pytest entry point for test 1."""
    r = test_agent_resnet_finds_optimization()
    assert r.passed, r.notes or r.metric


def test_stops_on_compute_bound_pytest():
    """pytest entry point for test 2."""
    r = test_agent_stops_on_compute_bound()
    assert r.passed, r.notes or r.metric


def test_rollback_correctness_pytest():
    """pytest entry point for test 3."""
    r = test_rollback_preserves_correctness()
    assert r.passed, r.notes or r.metric


def test_speedup_vs_baseline_pytest():
    """pytest entry point for test 4."""
    r = test_cumulative_speedup_vs_baseline()
    assert r.passed, r.notes or r.metric


if __name__ == "__main__":
    main()
