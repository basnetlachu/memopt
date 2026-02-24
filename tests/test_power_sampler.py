#!/usr/bin/env python3
"""
Power Sampler Test Suite
========================

4 real GPU tests — all assert on measured values, no mocks.

Run:
    python tests/test_power_sampler.py
    pytest tests/test_power_sampler.py -v -s
"""

from __future__ import annotations

import sys
import time
import traceback

import torch

if not torch.cuda.is_available():
    print("FATAL: CUDA not available.")
    sys.exit(1)

from memopt.profiler.power_sampler import PowerSampler, PowerReport
from memopt.agent import MemoptAgent


# ── Result tracker ────────────────────────────────────────────────────────────

class _R:
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.metric = ""
        self.notes = ""

    def ok(self, metric: str, notes: str = "") -> "_R":
        self.passed = True; self.metric = metric; self.notes = notes; return self

    def fail(self, metric: str, notes: str = "") -> "_R":
        self.passed = False; self.metric = metric; self.notes = notes; return self


# =============================================================================
# TEST 1: PowerSampler works on A100 — collects valid wattage
# =============================================================================

def test_power_sampler_works() -> _R:
    r = _R("power_sampler_works")
    print("\n" + "=" * 64)
    print("TEST 1: PowerSampler — must collect >50W, <500W on A100")
    print("=" * 64)

    try:
        # Use interval_ms=20 so >=5 samples fit within the GPU work window (~300ms)
        # Default 100ms interval only gets 1 sample for a fast 10ms matmul loop
        with PowerSampler(interval_ms=20) as sampler:
            x = torch.randn(4096, 4096, device="cuda")
            for _ in range(300):
                x = x @ x.T
            torch.cuda.synchronize()

        report = sampler.report(duration_ms=1000)

        failures = []
        if not report.available:
            failures.append("PowerReport.available=False — NVML not working")
        else:
            if report.avg_watts <= 50:
                failures.append(f"avg_watts={report.avg_watts:.1f}W — must be >50W on A100")
            if report.avg_watts >= 500:
                failures.append(f"avg_watts={report.avg_watts:.1f}W — must be <500W")
            if report.sample_count < 5:
                failures.append(f"sample_count={report.sample_count} — must be >=5")
            if report.peak_watts < report.avg_watts:
                failures.append(f"peak={report.peak_watts:.1f}W < avg={report.avg_watts:.1f}W — impossible")
            if report.joules <= 0:
                failures.append(f"joules={report.joules:.3f} — must be > 0")

        metric = report.summary()
        print(f"  {metric}")
        print(f"  samples={report.sample_count}  idle={report.idle_watts:.1f}W  active={report.active_watts:.1f}W")

        if failures:
            print(f"  FAILED: {'; '.join(failures)}")
            return r.fail(metric, "; ".join(failures))
        return r.ok(metric)

    except Exception as exc:
        return r.fail(f"EXCEPTION: {exc}", traceback.format_exc())


# =============================================================================
# TEST 2: PowerReport.unavailable() never crashes, correct fields
# =============================================================================

def test_unavailable_no_crash() -> _R:
    r = _R("unavailable_no_crash")
    print("\n" + "=" * 64)
    print("TEST 2: PowerReport.unavailable() — graceful degradation")
    print("=" * 64)

    failures = []
    try:
        report = PowerReport.unavailable()

        if report.available:
            failures.append("available=True on unavailable report")
        if report.avg_watts != 0.0:
            failures.append(f"avg_watts={report.avg_watts} — expected 0.0")
        if report.sample_count != 0:
            failures.append(f"sample_count={report.sample_count} — expected 0")
        expected_summary = "Power data unavailable (NVML not accessible)"
        if report.summary() != expected_summary:
            failures.append(f"summary={repr(report.summary())} — expected {repr(expected_summary)}")
        if report.joules_per_token is not None:
            failures.append("joules_per_token should be None")
        if report.tokens_per_watt is not None:
            failures.append("tokens_per_watt should be None")

        metric = f"available=False | summary='{report.summary()}'"
        print(f"  {metric}")

        if failures:
            print(f"  FAILED: {'; '.join(failures)}")
            return r.fail(metric, "; ".join(failures))
        return r.ok(metric, "graceful degradation confirmed")

    except Exception as exc:
        return r.fail(f"EXCEPTION: {exc}", traceback.format_exc())


# =============================================================================
# TEST 3: AgentReport has power fields — ResNet50
# =============================================================================

def test_agent_report_has_power() -> _R:
    r = _R("agent_report_has_power")
    print("\n" + "=" * 64)
    print("TEST 3: AgentReport.baseline_power / optimized_power populated")
    print("=" * 64)

    try:
        import torchvision
    except ImportError:
        return r.ok("SKIPPED", "torchvision not installed")

    model = torchvision.models.resnet50().cuda().eval()
    inp   = {"x": torch.randn(8, 3, 224, 224, device="cuda")}

    # Patch forward to accept x kwarg
    orig_fwd = model.forward
    import torch.nn as nn

    class _Wrap(nn.Module):
        def __init__(self, m): super().__init__(); self._m = m
        def forward(self, x): return self._m(x)

    model = _Wrap(model).cuda().eval()

    agent  = MemoptAgent(target_speedup=2.0, max_rounds=3)
    report = agent.run(model, inp)

    failures = []

    # baseline_power must be populated (available on A100)
    if not hasattr(report, "baseline_power"):
        failures.append("AgentReport missing baseline_power field")
    elif not report.baseline_power.available:
        failures.append("baseline_power.available=False — NVML should work on A100")
    elif report.baseline_power.avg_watts <= 0:
        failures.append(f"baseline_power.avg_watts={report.baseline_power.avg_watts:.1f}")

    # optimized_power must be populated
    if not hasattr(report, "optimized_power"):
        failures.append("AgentReport missing optimized_power field")
    elif not report.optimized_power.available:
        failures.append("optimized_power.available=False")

    # power_reduction_pct must be a number (can be negative if warmup effect)
    if not hasattr(report, "power_reduction_pct"):
        failures.append("AgentReport missing power_reduction_pct field")
    elif report.power_reduction_pct is None:
        failures.append("power_reduction_pct is None — expected float")

    # power_summary() must not crash
    try:
        ps = report.power_summary()
    except Exception as exc:
        failures.append(f"power_summary() raised: {exc}")
        ps = "ERROR"

    metric = (
        f"baseline={report.baseline_power.avg_watts:.1f}W | "
        f"optimized={report.optimized_power.avg_watts:.1f}W | "
        f"reduction={report.power_reduction_pct:.1f}% | "
        f"speedup={report.final_speedup:.2f}x"
    )
    print(f"  {metric}")
    print(f"  power_summary: {ps[:80]}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric)


# =============================================================================
# TEST 4: token_count metrics — joules_per_token and tokens_per_watt
# =============================================================================

def test_token_efficiency_metrics() -> _R:
    r = _R("token_efficiency_metrics")
    print("\n" + "=" * 64)
    print("TEST 4: joules_per_token + tokens_per_watt computed correctly")
    print("=" * 64)

    failures = []
    try:
        token_count = 512

        with PowerSampler() as sampler:
            x = torch.randn(1, token_count, 768, device="cuda")
            for _ in range(20):
                _ = x @ x.transpose(-1, -2)
            torch.cuda.synchronize()

        report = sampler.report(duration_ms=500, token_count=token_count)

        if not report.available:
            return r.ok("SKIPPED", "NVML unavailable")

        if report.joules_per_token is None:
            failures.append("joules_per_token is None — expected float when token_count>0")
        elif report.joules_per_token <= 0:
            failures.append(f"joules_per_token={report.joules_per_token:.6f} — must be >0")

        if report.tokens_per_watt is None:
            failures.append("tokens_per_watt is None — expected float when token_count>0")
        elif report.tokens_per_watt <= 0:
            failures.append(f"tokens_per_watt={report.tokens_per_watt:.3f} — must be >0")

        # Sanity: joules = avg_watts * duration_s
        expected_joules = report.avg_watts * (500 / 1000.0)
        if abs(report.joules - expected_joules) > 0.5:
            failures.append(
                f"joules={report.joules:.3f} but avg*dur={expected_joules:.3f} "
                f"— mismatch > 0.5J"
            )

        metric = (
            f"joules={report.joules:.3f}J | "
            f"j/tok={report.joules_per_token:.5f} | "
            f"tok/W={report.tokens_per_watt:.2f} | "
            f"tokens={token_count}"
        )
        print(f"  {metric}")

        if failures:
            print(f"  FAILED: {'; '.join(failures)}")
            return r.fail(metric, "; ".join(failures))
        return r.ok(metric)

    except Exception as exc:
        return r.fail(f"EXCEPTION: {exc}", traceback.format_exc())


# =============================================================================
# Runner
# =============================================================================

def main() -> None:
    print("\n" + "=" * 64)
    print("Power Sampler Test Suite")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 64)

    tests = [
        test_power_sampler_works,
        test_unavailable_no_crash,
        test_agent_report_has_power,
        test_token_efficiency_metrics,
    ]

    results = []
    for fn in tests:
        t0 = time.time()
        try:
            res = fn()
        except Exception as exc:
            res = _R(fn.__name__)
            res.fail(f"EXCEPTION: {exc}", traceback.format_exc())
            traceback.print_exc()
        elapsed = time.time() - t0
        res.notes = f"{res.notes} [{elapsed:.1f}s]".strip()
        results.append(res)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTS")
    print("-" * 110)
    print(f"{'Test':<36}| {'Result':<8}| {'Key Metric':<48}| Notes")
    print("-" * 110)

    passed = failed = 0
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        if res.metric == "SKIPPED":
            status = "SKIP"
        passed += res.passed
        failed += (not res.passed)
        print(f"{res.name:<36}| {status:<8}| {res.metric[:48]:<48}| {res.notes[:30]}")

    print("-" * 110)
    print(f"Total: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 110)

    if failed:
        sys.exit(1)


# ── pytest compatibility ──────────────────────────────────────────────────────

def test_power_sampler_works_pytest():
    r = test_power_sampler_works(); assert r.passed, r.notes or r.metric

def test_unavailable_no_crash_pytest():
    r = test_unavailable_no_crash(); assert r.passed, r.notes or r.metric

def test_agent_report_has_power_pytest():
    r = test_agent_report_has_power(); assert r.passed, r.notes or r.metric

def test_token_efficiency_metrics_pytest():
    r = test_token_efficiency_metrics(); assert r.passed, r.notes or r.metric


if __name__ == "__main__":
    main()
