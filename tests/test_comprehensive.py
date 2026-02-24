#!/usr/bin/env python3
"""
memopt Comprehensive End-to-End Test Suite
==========================================
Tests A–F: inference agent, training agent, power metrics.
All assertions on real GPU. No mocks.
"""
from __future__ import annotations

import copy
import sys
import time
import traceback
import warnings

import torch
import torch.nn as nn

if not torch.cuda.is_available():
    print("FATAL: CUDA not available")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GPU  = torch.cuda.get_device_name(0)
PT   = torch.__version__

results: list[tuple[str, str, str]] = []   # (test, result, metric)

def record(name: str, passed: bool, metric: str) -> None:
    results.append((name, "PASS" if passed else "FAIL", metric))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}: {metric}")


def benchmark(m, inp, iters: int = 50) -> float:
    """Forward-only wall-clock benchmark (ms/iter)."""
    with torch.no_grad():
        for _ in range(5):
            m(**inp)
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    with torch.no_grad():
        for _ in range(iters):
            m(**inp)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def benchmark_training_step(m, optimizer, loss_fn, batch, iters: int = 20) -> float:
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        optimizer.zero_grad()
        s.record()
        out  = m(batch["input"])
        loss = loss_fn(out, batch["target"])
        loss.backward()
        optimizer.step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


def benchmark_forward_only(m, inp_tensor, iters: int = 20) -> float:
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            m(inp_tensor)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    return sorted(times)[len(times) // 2]


# ---------------------------------------------------------------------------
# STEP 0: GradScaler fix verification
# ---------------------------------------------------------------------------

def step0_gradscaler_fix() -> None:
    print("\n" + "=" * 64)
    print("STEP 0: GradScaler deprecation fix")
    print("=" * 64)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        scaler = torch.amp.GradScaler('cuda')
        future_warns = [x for x in w if issubclass(x.category, FutureWarning)]
    ok = (len(future_warns) == 0 and type(scaler).__name__ == "GradScaler")
    metric = f"type={type(scaler).__name__} warnings={len(future_warns)}"
    print(f"  {metric}")
    record("GradScaler fix", ok, metric)


# ---------------------------------------------------------------------------
# STEP 1: Regression — existing pytest suites
# ---------------------------------------------------------------------------

def step1_regression() -> None:
    print("\n" + "=" * 64)
    print("STEP 1: Regression test suites (pytest)")
    print("=" * 64)
    import subprocess, os
    repo = "/repo"
    for suite in ["tests/test_optimization_agent.py", "tests/test_training_agent.py"]:
        path = os.path.join(repo, suite)
        if not os.path.exists(path):
            record(f"Regression: {suite}", True, "SKIP (file not found)")
            continue
        r = subprocess.run(
            ["python3", "-m", "pytest", path, "-v", "--tb=short", "-q"],
            capture_output=True, text=True, cwd=repo
        )
        passed = r.returncode == 0
        # Extract summary line
        summary = ""
        for line in (r.stdout + r.stderr).splitlines():
            if "passed" in line or "failed" in line or "error" in line:
                summary = line.strip()
        record(f"Regression: {suite.split('/')[-1]}", passed, summary or f"rc={r.returncode}")
        if not passed:
            print(f"  STDOUT:\n{r.stdout[-2000:]}")
            print(f"  STDERR:\n{r.stderr[-1000:]}")


# ---------------------------------------------------------------------------
# TEST A: Inference — BERT seq=512
# ---------------------------------------------------------------------------

def test_a_bert() -> None:
    print("\n" + "=" * 64)
    print("TEST A: Inference Agent — BERT seq=512")
    print("=" * 64)
    try:
        from transformers import BertModel, AutoTokenizer
        from memopt.agent.optimization_agent import MemoptAgent

        model     = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        inp_512   = {k: v.cuda() for k, v in tokenizer(
            "x " * 200, return_tensors="pt", truncation=True, max_length=512
        ).items()}

        baseline_ms = benchmark(model, inp_512)
        print(f"  Baseline: {baseline_ms:.2f}ms")

        agent  = MemoptAgent(target_speedup=1.5, max_rounds=5)
        report = agent.run(copy.deepcopy(model), inp_512)

        if report.optimized_model:
            real_ms      = benchmark(report.optimized_model, inp_512)
            real_speedup = baseline_ms / real_ms
        else:
            real_ms      = baseline_ms
            real_speedup = 1.0

        accuracy_pct = 100 * min(real_speedup, report.final_speedup) / max(real_speedup, report.final_speedup)

        print(f"  Agent reported: {report.final_speedup:.3f}x")
        print(f"  Real speedup:   {real_speedup:.3f}x  ({real_ms:.2f}ms)")
        print(f"  Accuracy:       {accuracy_pct:.1f}%")
        print(f"  Applied:        {report.optimizations_applied}")
        print(f"  Rolled back:    {report.optimizations_rolled_back}")
        print(f"  Stop reason:    {report.rounds[-1].stop_reason}")
        print(f"  Ceiling:        {report.honest_ceiling[:80] if report.honest_ceiling else 'None'}")

        assert report.rounds[-1].stop_reason is not None, "Must have stop reason"
        assert report.honest_ceiling, "Must have ceiling"
        assert real_speedup >= 0.90, f"Must not regress: {real_speedup:.2f}x"
        if report.optimized_model:
            ratio = min(real_speedup, report.final_speedup) / max(real_speedup, report.final_speedup)
            assert ratio >= 0.85, f"Reported {report.final_speedup:.2f}x real {real_speedup:.2f}x"

        metric = (f"baseline={baseline_ms:.2f}ms reported={report.final_speedup:.3f}x "
                  f"real={real_speedup:.3f}x accuracy={accuracy_pct:.1f}%")
        record("A: BERT inference seq=512", True, metric)

    except Exception as exc:
        record("A: BERT inference seq=512", False, f"EXCEPTION: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TEST B: Inference — ResNet50 batch=8 (must hit 2.0x)
# ---------------------------------------------------------------------------

def test_b_resnet() -> None:
    print("\n" + "=" * 64)
    print("TEST B: Inference Agent — ResNet50 batch=8")
    print("=" * 64)
    try:
        import torchvision
        from memopt.agent.optimization_agent import MemoptAgent

        model = torchvision.models.resnet50().cuda().eval()
        inp   = torch.randn(8, 3, 224, 224).cuda()
        inp_d = {"x": inp}

        baseline_ms = benchmark(model, inp_d)
        print(f"  Baseline: {baseline_ms:.2f}ms")

        agent  = MemoptAgent(target_speedup=2.0, max_rounds=5)
        report = agent.run(copy.deepcopy(model), inp_d)

        if report.optimized_model:
            real_ms      = benchmark(report.optimized_model, inp_d)
            real_speedup = baseline_ms / real_ms
        else:
            real_ms      = baseline_ms
            real_speedup = 1.0

        print(f"  Agent reported: {report.final_speedup:.3f}x")
        print(f"  Real speedup:   {real_speedup:.3f}x  ({real_ms:.2f}ms)")
        print(f"  Applied:        {report.optimizations_applied}")
        print(f"  Stop reason:    {report.rounds[-1].stop_reason}")

        assert real_speedup >= 2.0, f"Must hit 2.0x, got {real_speedup:.2f}x"
        assert len(report.optimizations_applied) > 0, "Must commit something"
        assert "TARGET_MET" in report.rounds[-1].stop_reason

        metric = f"real={real_speedup:.3f}x applied={report.optimizations_applied}"
        record("B: ResNet50 inference", True, metric)

    except AssertionError as exc:
        record("B: ResNet50 inference", False, str(exc))
    except Exception as exc:
        record("B: ResNet50 inference", False, f"EXCEPTION: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TEST C: Inference — compute-bound (must stop immediately)
# ---------------------------------------------------------------------------

def test_c_compute_bound() -> None:
    print("\n" + "=" * 64)
    print("TEST C: Inference Agent — compute-bound Linear")
    print("=" * 64)
    try:
        from memopt.agent.optimization_agent import MemoptAgent

        model = nn.Linear(4096, 4096).cuda().eval()
        inp   = {"input": torch.randn(64, 4096).cuda()}

        agent  = MemoptAgent(target_speedup=2.0, max_rounds=10)
        report = agent.run(model, inp)

        print(f"  Rounds:      {len(report.rounds)}")
        print(f"  Applied:     {report.optimizations_applied}")
        print(f"  Speedup:     {report.final_speedup:.3f}x")
        print(f"  Stop reason: {report.rounds[-1].stop_reason}")

        assert "OPTIMAL" in report.rounds[-1].stop_reason, \
            f"Expected OPTIMAL, got: {report.rounds[-1].stop_reason}"
        assert len(report.optimizations_applied) == 0
        assert report.final_speedup <= 1.10
        assert len(report.rounds) == 1, f"Must be 1 round, got {len(report.rounds)}"

        metric = (f"rounds={len(report.rounds)} applied=[] "
                  f"speedup={report.final_speedup:.3f}x stop={report.rounds[-1].stop_reason[:30]}")
        record("C: Compute-bound stops", True, metric)

    except AssertionError as exc:
        record("C: Compute-bound stops", False, str(exc))
    except Exception as exc:
        record("C: Compute-bound stops", False, f"EXCEPTION: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TEST D: Training Agent — step profiling + no int8
# ---------------------------------------------------------------------------

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 10),
        )
    def forward(self, x):
        return self.layers(x)


def test_d_training_agent() -> None:
    print("\n" + "=" * 64)
    print("TEST D: Training Agent — step profiling + rollback")
    print("=" * 64)
    try:
        from memopt.agent.training_agent import TrainingAgent

        model     = SimpleNet().cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn   = nn.CrossEntropyLoss()
        batch     = {
            "input":  torch.randn(32, 512).cuda(),
            "target": torch.randint(0, 10, (32,)).cuda(),
        }

        baseline_step_ms    = benchmark_training_step(model, optimizer, loss_fn, batch)
        baseline_forward_ms = benchmark_forward_only(model, batch["input"])

        print(f"  Forward-only:      {baseline_forward_ms:.2f}ms")
        print(f"  Full step:         {baseline_step_ms:.2f}ms")
        print(f"  Step/forward:      {baseline_step_ms/baseline_forward_ms:.2f}x")

        agent  = TrainingAgent(target_speedup=1.2, max_rounds=5)
        report = agent.run(model, optimizer, loss_fn, batch)

        print(f"  Agent speedup:     {report.final_speedup:.3f}x")
        print(f"  Applied:           {report.optimizations_applied}")
        print(f"  Rolled back:       {report.optimizations_rolled_back}")
        print(f"  Loss stable:       {report.loss_stable}")
        print(f"  BF16 enabled:      {report.bf16_enabled}")
        print(f"  Grad ckpt:         {report.gradient_checkpointing}")
        print(f"  Stop reason:       {report.rounds[-1].stop_reason}")

        assert "int8" not in str(report.optimizations_applied), "int8 in applied!"
        assert "int8" not in str(report.optimizations_rolled_back), "int8 in rolled back!"
        assert baseline_step_ms > baseline_forward_ms * 1.5, \
            f"Step {baseline_step_ms:.1f}ms not >1.5x forward {baseline_forward_ms:.1f}ms"
        assert report.loss_stable, "Training unstable"
        assert report.rounds[-1].stop_reason is not None

        ratio  = baseline_step_ms / baseline_forward_ms
        metric = (f"ratio={ratio:.2f}x int8_never=True "
                  f"loss_stable={report.loss_stable} speedup={report.final_speedup:.3f}x")
        record("D: Training step profiling", True, metric)

    except AssertionError as exc:
        record("D: Training step profiling", False, str(exc))
    except Exception as exc:
        record("D: Training step profiling", False, f"EXCEPTION: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TEST E: Training rollback preserves weights + optimizer
# ---------------------------------------------------------------------------

def test_e_training_rollback() -> None:
    print("\n" + "=" * 64)
    print("TEST E: Training rollback state preservation")
    print("=" * 64)
    try:
        from memopt.agent.training_agent import TrainingAgent

        model2    = SimpleNet().cuda()
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        loss_fn   = nn.CrossEntropyLoss()
        batch     = {
            "input":  torch.randn(32, 512).cuda(),
            "target": torch.randint(0, 10, (32,)).cuda(),
        }

        # Populate optimizer momentum buffers with one real step
        out = model2(batch["input"])
        loss_fn(out, batch["target"]).backward()
        optimizer2.step()
        optimizer2.zero_grad()

        agent2 = TrainingAgent(target_speedup=1.2)

        # Save checkpoint
        checkpoint = agent2._save_checkpoint(model2, optimizer2)

        # Corrupt model weights
        original_weight = next(model2.parameters()).data.clone()
        with torch.no_grad():
            for p in model2.parameters():
                p.fill_(999.0)
        assert (next(model2.parameters()) == 999.0).all(), "Corruption failed"

        # Restore
        agent2._restore_checkpoint(model2, optimizer2, checkpoint)

        restored_weight = next(model2.parameters()).data
        weight_err = (restored_weight - original_weight).abs().max().item()

        opt_state  = optimizer2.state.get(next(model2.parameters()), {})
        exp_avg_mean = opt_state.get('exp_avg', torch.zeros(1)).abs().mean().item()

        print(f"  Weight error after restore: {weight_err:.2e}")
        print(f"  Optimizer exp_avg mean:     {exp_avg_mean:.6f}")

        assert weight_err < 1e-6, f"Weight restore failed: {weight_err}"
        assert 'exp_avg' in opt_state, "Optimizer momentum not restored"

        metric = f"weight_err={weight_err:.2e} opt_exp_avg={exp_avg_mean:.6f}"
        record("E: Training rollback", True, metric)

    except AssertionError as exc:
        record("E: Training rollback", False, str(exc))
    except Exception as exc:
        record("E: Training rollback", False, f"EXCEPTION: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# TEST F: Power Metrics
# ---------------------------------------------------------------------------

def test_f_power() -> None:
    from memopt.profiler.power_sampler import PowerSampler, PowerReport

    # F1: Basic sampling
    print("\n" + "=" * 64)
    print("TEST F1: PowerSampler basic sampling")
    print("=" * 64)
    try:
        with PowerSampler(interval_ms=20) as sampler:
            x = torch.randn(2000, 2000).cuda()
            for _ in range(200):
                x = x @ x.T
            torch.cuda.synchronize()

        report = sampler.report(duration_ms=2000)
        print(f"  Available:    {report.available}")
        print(f"  Avg watts:    {report.avg_watts:.1f}W")
        print(f"  Peak watts:   {report.peak_watts:.1f}W")
        print(f"  Samples:      {report.sample_count}")
        print(f"  Summary:      {report.summary()}")

        if report.available:
            assert report.avg_watts > 50,  f"Must draw >50W: {report.avg_watts:.1f}W"
            assert report.avg_watts < 500, f"Must draw <500W: {report.avg_watts:.1f}W"
            assert report.peak_watts >= report.avg_watts
            assert report.sample_count >= 5, f"Need >=5 samples: {report.sample_count}"
            metric = f"avg={report.avg_watts:.1f}W peak={report.peak_watts:.1f}W samples={report.sample_count}"
        else:
            metric = "NVML unavailable (SKIP)"

        record("F1: PowerSampler basic", True, metric)

    except AssertionError as exc:
        record("F1: PowerSampler basic", False, str(exc))
    except Exception as exc:
        record("F1: PowerSampler basic", False, f"EXCEPTION: {exc}")
        traceback.print_exc()

    # F2: Unavailable graceful
    print("\n" + "=" * 64)
    print("TEST F2: PowerReport.unavailable()")
    print("=" * 64)
    try:
        unavail = PowerReport.unavailable()
        assert not unavail.available
        assert "unavailable" in unavail.summary().lower()
        print(f"  Summary: {unavail.summary()}")
        record("F2: Unavailable graceful", True, f"available=False summary=OK")
    except Exception as exc:
        record("F2: Unavailable graceful", False, f"EXCEPTION: {exc}")

    # F3: Power tracked during inference agent run
    print("\n" + "=" * 64)
    print("TEST F3: Power during inference agent run")
    print("=" * 64)
    try:
        import torchvision
        from memopt.agent.optimization_agent import MemoptAgent

        model = torchvision.models.resnet50().cuda().eval()
        inp   = torch.randn(8, 3, 224, 224).cuda()

        agent  = MemoptAgent(target_speedup=2.0, max_rounds=5)
        report = agent.run(copy.deepcopy(model), {"x": inp})

        print(f"  Baseline power:  {report.baseline_power.summary()}")
        print(f"  Optimized power: {report.optimized_power.summary()}")
        if report.power_reduction_pct is not None:
            print(f"  Power reduction: {report.power_reduction_pct:.1f}%")
        print(f"  {report.power_summary()}")

        assert report.baseline_power  is not None, "baseline_power missing"
        assert report.optimized_power is not None, "optimized_power missing"
        if report.baseline_power.available and report.optimized_power.available:
            assert report.power_reduction_pct is not None

        metric = (f"baseline={report.baseline_power.avg_watts:.1f}W "
                  f"optimized={report.optimized_power.avg_watts:.1f}W "
                  f"reduction={report.power_reduction_pct:.1f}%"
                  if report.power_reduction_pct is not None else
                  f"baseline={report.baseline_power.avg_watts:.1f}W optimized={report.optimized_power.avg_watts:.1f}W")
        record("F3: Power during agent run", True, metric)

    except AssertionError as exc:
        record("F3: Power during agent run", False, str(exc))
    except Exception as exc:
        record("F3: Power during agent run", False, f"EXCEPTION: {exc}")
        traceback.print_exc()

    # F4: joules_per_token
    print("\n" + "=" * 64)
    print("TEST F4: joules_per_token")
    print("=" * 64)
    try:
        with PowerSampler(interval_ms=20) as sampler:
            for _ in range(100):
                x = torch.randn(1, 512, 768).cuda()
                _ = torch.nn.functional.layer_norm(x, [768])
            torch.cuda.synchronize()

        token_report = sampler.report(duration_ms=1000, token_count=100)
        print(f"  Available:        {token_report.available}")
        print(f"  Joules per token: {token_report.joules_per_token}")
        print(f"  Tokens per watt:  {token_report.tokens_per_watt}")

        if token_report.available:
            assert token_report.joules_per_token is not None
            assert token_report.joules_per_token > 0
            assert token_report.tokens_per_watt  > 0
            metric = (f"j/tok={token_report.joules_per_token:.5f} "
                      f"tok/W={token_report.tokens_per_watt:.2f}")
        else:
            metric = "NVML unavailable (SKIP)"

        record("F4: joules_per_token", True, metric)

    except AssertionError as exc:
        record("F4: joules_per_token", False, str(exc))
    except Exception as exc:
        record("F4: joules_per_token", False, f"EXCEPTION: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Final results table
# ---------------------------------------------------------------------------

def print_results() -> None:
    total  = len(results)
    passed = sum(1 for _, r, _ in results if r == "PASS")
    failed = total - passed

    print("\n" + "=" * 90)
    print("MEMOPT COMPREHENSIVE TEST RESULTS")
    print(f"GPU: {GPU}  |  PyTorch: {PT}")
    print("=" * 90)
    print(f"{'Test':<38}| {'Result':<8}| Key Metric")
    print("-" * 90)
    for name, result, metric in results:
        print(f"{name:<38}| {result:<8}| {metric[:42]}")
    print("-" * 90)
    print(f"Total: {passed} passed, {failed} failed, {total} total")
    print("=" * 90)

    sys.exit(0 if failed == 0 else 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nMemopt Comprehensive E2E Suite")
    print(f"GPU:     {GPU}")
    print(f"PyTorch: {PT}")

    step0_gradscaler_fix()
    step1_regression()
    test_a_bert()
    test_b_resnet()
    test_c_compute_bound()
    test_d_training_agent()
    test_e_training_rollback()
    test_f_power()

    print_results()
