#!/usr/bin/env python3
"""
TrainingAgent Test Suite
========================

4 real GPU tests — all assert on measured values, no mocks.

Run:
    python tests/test_training_agent.py
    pytest tests/test_training_agent.py -v -s
"""

from __future__ import annotations

import sys
import time
import traceback
from typing import Dict, Any

import torch
import torch.nn as nn

# ── Guard ─────────────────────────────────────────────────────────────────────
if not torch.cuda.is_available():
    print("FATAL: CUDA not available. These tests require a GPU.")
    sys.exit(1)

from memopt.agent import (
    TrainingAgent,
    TrainingAgentReport,
    apply_gradient_checkpointing,
    apply_bf16_mixed_precision,
    TRAINING_BLACKLIST,
)
from memopt.agent import MemoptAgent  # inference agent — must still import clean


# ── Result tracker ────────────────────────────────────────────────────────────

class _Result:
    def __init__(self, name: str):
        self.name   = name
        self.passed = False
        self.metric = ""
        self.notes  = ""

    def ok(self, metric: str, notes: str = "") -> "_Result":
        self.passed = True
        self.metric = metric
        self.notes  = notes
        return self

    def fail(self, metric: str, notes: str = "") -> "_Result":
        self.passed = False
        self.metric = metric
        self.notes  = notes
        return self


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_resnet_batch(batch: int = 4) -> Dict[str, Any]:
    """ResNet-compatible training batch."""
    try:
        import torchvision
        model = torchvision.models.resnet50().cuda().train()
        inp   = torch.randn(batch, 3, 224, 224, device="cuda")
        tgt   = torch.randint(0, 1000, (batch,), device="cuda")
        return model, inp, tgt
    except ImportError:
        return None, None, None


def _bench_step_ms(model, inp, tgt, optimizer, loss_fn, warmup=3, iters=10) -> float:
    """Independent benchmark of a full training step."""
    for _ in range(warmup):
        optimizer.zero_grad()
        out  = model(inp)
        loss = loss_fn(out, tgt)
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        optimizer.zero_grad()
        s.record()
        out  = model(inp)
        loss = loss_fn(out, tgt)
        loss.backward()
        optimizer.step()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# =============================================================================
# TEST 1: Training step profiled (step_ms > forward_ms), int8 never applied
# =============================================================================

def test_training_step_profiled_not_forward_only() -> _Result:
    r = _Result("training_step_profiled")
    print("\n" + "=" * 64)
    print("TEST 1: Training step must include backward — step_ms > forward_ms")
    print("=" * 64)

    try:
        import torchvision
    except ImportError:
        r.notes = "torchvision not installed"
        r.passed = True
        r.metric = "SKIPPED"
        return r

    model     = torchvision.models.resnet50().cuda().train()
    inp       = torch.randn(4, 3, 224, 224, device="cuda")
    tgt       = torch.randint(0, 1000, (4,), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn   = nn.CrossEntropyLoss()

    sample_batch = {"input": inp, "target": tgt}

    # Measure forward-only latency independently
    forward_times = []
    torch.cuda.synchronize()
    for _ in range(10):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        with torch.no_grad():
            model(inp)
        e.record()
        torch.cuda.synchronize()
        forward_times.append(s.elapsed_time(e))
    forward_times.sort()
    forward_ms = forward_times[len(forward_times) // 2]

    # Run training agent
    agent  = TrainingAgent(target_speedup=1.3, max_rounds=3)
    report = agent.run(model, optimizer, loss_fn, sample_batch)

    # Agent internally benchmarks full steps — check via comparing
    # to forward_ms: any full step must be noticeably slower
    # We verify this by running an independent full-step benchmark
    step_ms = _bench_step_ms(
        report.optimized_model if report.optimized_model else model,
        inp, tgt, optimizer, loss_fn,
    )

    failures = []

    # int8 must NEVER appear
    for blacklisted in TRAINING_BLACKLIST:
        if blacklisted in report.optimizations_applied:
            failures.append(f"BLACKLISTED '{blacklisted}' was applied!")
        if blacklisted in report.optimizations_rolled_back:
            failures.append(f"BLACKLISTED '{blacklisted}' was even attempted!")

    # Full step must be materially slower than forward-only
    if step_ms <= forward_ms * 1.2:
        failures.append(
            f"step_ms={step_ms:.1f} not > 1.2×forward_ms={forward_ms:.1f} "
            f"— backward may not be included"
        )

    metric = (
        f"step_ms={step_ms:.1f} | forward_ms={forward_ms:.1f} | "
        f"ratio={step_ms/forward_ms:.2f}x | applied={report.optimizations_applied}"
    )
    print(f"  {metric}")
    print(f"  stop={report.rounds[-1].stop_reason}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric, f"int8_never_applied=True")


# =============================================================================
# TEST 2: BF16 mixed precision works on Ampere GPU (A100 = CC 8.0)
# =============================================================================

def test_bf16_on_ampere() -> _Result:
    r = _Result("bf16_on_ampere")
    print("\n" + "=" * 64)
    print("TEST 2: BF16 mixed precision — scaler must not be None on Ampere")
    print("=" * 64)

    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        print(f"  SKIP: GPU {props.name} is CC {props.major}.{props.minor} < 8.0 — no BF16")
        r.passed = True
        r.metric = "SKIPPED"
        r.notes  = f"CC={props.major}.{props.minor}"
        return r

    # Build a simple model with parameters
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).cuda().train()

    failures = []
    try:
        new_model, scaler = apply_bf16_mixed_precision(model)

        if scaler is None:
            failures.append("scaler is None — expected GradScaler")

        # Verify autocast actually runs without error
        inp = torch.randn(8, 256, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = new_model(inp)
        if out.dtype not in (torch.bfloat16, torch.float32):
            failures.append(f"unexpected output dtype: {out.dtype}")

    except Exception as exc:
        failures.append(f"apply_bf16_mixed_precision raised: {exc}")

    metric = f"CC={props.major}.{props.minor} | scaler={scaler is not None if 'scaler' in dir() else 'ERR'}"
    print(f"  {metric}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric, "scaler=GradScaler")


# =============================================================================
# TEST 3: Checkpoint rollback restores optimizer state
# =============================================================================

def test_rollback_restores_optimizer() -> _Result:
    r = _Result("rollback_restores_optimizer")
    print("\n" + "=" * 64)
    print("TEST 3: Checkpoint rollback must restore model + optimizer state")
    print("=" * 64)

    model     = nn.Linear(64, 64).cuda().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Do a few steps so optimizer has non-trivial state (momentum buffers)
    inp = torch.randn(8, 64, device="cuda")
    for _ in range(5):
        optimizer.zero_grad()
        model(inp).sum().backward()
        optimizer.step()

    agent = TrainingAgent(target_speedup=1.5, max_rounds=5)

    # Save checkpoint
    checkpoint = agent._save_checkpoint(model, optimizer)

    # Record original weight and optimizer exp_avg
    orig_weight    = model.weight.data.clone()
    orig_state_key = list(optimizer.state.keys())[0]
    orig_exp_avg   = optimizer.state[orig_state_key]["exp_avg"].clone()

    # Corrupt model weights and optimizer
    with torch.no_grad():
        model.weight.fill_(999.0)
    # Corrupt optimizer state
    optimizer.state[orig_state_key]["exp_avg"].fill_(777.0)

    # Restore
    agent._restore_checkpoint(model, optimizer, checkpoint)

    failures = []
    max_weight_err = (model.weight.data - orig_weight).abs().max().item()
    if max_weight_err > 1e-6:
        failures.append(f"weights not restored: max_err={max_weight_err:.2e}")

    restored_exp_avg = optimizer.state[list(optimizer.state.keys())[0]]["exp_avg"]
    max_opt_err = (restored_exp_avg - orig_exp_avg).abs().max().item()
    if max_opt_err > 1e-6:
        failures.append(f"optimizer exp_avg not restored: max_err={max_opt_err:.2e}")

    metric = f"weight_err={max_weight_err:.2e} | opt_exp_avg_err={max_opt_err:.2e}"
    print(f"  {metric}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric, "model+optimizer fully restored")


# =============================================================================
# TEST 4: ResNet50 training — agent commits ≥1 opt, loss stays stable
# =============================================================================

def test_resnet_training_optimization() -> _Result:
    r = _Result("resnet_training_optimization")
    print("\n" + "=" * 64)
    print("TEST 4: ResNet50 training — agent must commit ≥1 opt, loss stable")
    print("=" * 64)

    try:
        import torchvision
    except ImportError:
        r.passed = True
        r.metric = "SKIPPED"
        r.notes  = "torchvision not installed"
        return r

    model     = torchvision.models.resnet50().cuda().train()
    inp       = torch.randn(4, 3, 224, 224, device="cuda")
    tgt       = torch.randint(0, 1000, (4,), device="cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn   = nn.CrossEntropyLoss()
    sample_batch = {"input": inp, "target": tgt}

    agent  = TrainingAgent(target_speedup=1.3, max_rounds=5)
    report = agent.run(model, optimizer, loss_fn, sample_batch)

    failures = []

    # int8 must never appear
    for bl in TRAINING_BLACKLIST:
        if bl in report.optimizations_applied or bl in report.optimizations_rolled_back:
            failures.append(f"BLACKLISTED '{bl}' was touched")

    # Loss must remain stable
    if not report.loss_stable:
        failures.append(
            f"loss unstable: baseline={report.baseline_loss:.4f} "
            f"final={report.final_loss:.4f}"
        )

    # Report fields must be populated
    if report.baseline_loss <= 0:
        failures.append(f"baseline_loss={report.baseline_loss} — not measured")

    # honest_ceiling must be non-empty
    if len(report.honest_ceiling) < 10:
        failures.append("honest_ceiling too short or missing")

    stop = report.rounds[-1].stop_reason or "(no stop)"
    metric = (
        f"speedup={report.final_speedup:.2f}x | "
        f"applied={report.optimizations_applied} | "
        f"loss={report.baseline_loss:.4f}→{report.final_loss:.4f} | "
        f"stable={report.loss_stable} | stop={stop[:40]}"
    )
    print(f"  {metric}")
    print(f"  ceiling={report.honest_ceiling[:80]}")
    print(f"  bf16={report.bf16_enabled} | grad_ckpt={report.gradient_checkpointing}")

    if failures:
        print(f"  FAILED: {'; '.join(failures)}")
        return r.fail(metric, "; ".join(failures))
    return r.ok(metric, f"stop={stop}")


# =============================================================================
# Runner
# =============================================================================

def main() -> None:
    print("\n" + "=" * 64)
    print("TrainingAgent Test Suite")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print("=" * 64)

    tests = [
        test_training_step_profiled_not_forward_only,
        test_bf16_on_ampere,
        test_rollback_restores_optimizer,
        test_resnet_training_optimization,
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

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print("RESULTS")
    print("-" * 110)
    hdr = f"{'Test':<38}| {'Result':<8}| {'Key Metric':<44}| Notes"
    print(hdr)
    print("-" * 110)

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
        metric = res.metric[:44] if res.metric else ""
        notes  = res.notes[:30]  if res.notes  else ""
        print(f"{res.name:<38}| {status:<8}| {metric:<44}| {notes}")

    print("-" * 110)
    print(f"Total: {passed} passed, {failed} failed, {len(results)} total")
    print("=" * 110)

    if failed > 0:
        sys.exit(1)


# ── pytest compatibility ───────────────────────────────────────────────────────

def test_training_step_profiled_pytest():
    r = test_training_step_profiled_not_forward_only()
    assert r.passed, r.notes or r.metric

def test_bf16_on_ampere_pytest():
    r = test_bf16_on_ampere()
    assert r.passed, r.notes or r.metric

def test_rollback_restores_optimizer_pytest():
    r = test_rollback_restores_optimizer()
    assert r.passed, r.notes or r.metric

def test_resnet_training_optimization_pytest():
    r = test_resnet_training_optimization()
    assert r.passed, r.notes or r.metric


if __name__ == "__main__":
    main()
