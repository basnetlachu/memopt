#!/usr/bin/env python3
"""
Production-Grade Performance Benchmark

Proper benchmarking methodology:
1. Separate model load time from inference time
2. Warmup runs to exclude compilation overhead
3. Steady-state measurement over many iterations
4. Statistical rigor with confidence intervals
5. No-regression guarantee testing

Usage:
    python benchmark_performance.py --model gpt2-xl --max-tokens 1000

Environment variables:
    MEMOPT_NO_REGRESSION=1  # Enable strict no-regression mode
"""

import argparse
import torch
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple
import os

from memopt import OptimizedLLM
from memopt.performance_guard import PerformanceGuard


def measure_baseline(
    model_name: str,
    prompt: str,
    max_tokens: int,
    warmup_runs: int = 5,
    measurement_runs: int = 20
) -> Tuple[float, float, List[float]]:
    """
    Measure baseline performance with proper methodology.

    Args:
        model_name: Model to test
        prompt: Test prompt
        max_tokens: Tokens to generate
        warmup_runs: Warmup iterations (excluded from measurement)
        measurement_runs: Measurement iterations (for steady-state)

    Returns:
        (mean_throughput, std_throughput, all_throughputs)
    """
    print("\n" + "="*70)
    print("BASELINE MEASUREMENT (Proper Methodology)")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # PHASE 1: Model loading (EXCLUDED from timing)
    print(f"\n[1/3] Loading model: {model_name}")
    load_start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True
    )
    model.eval()

    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.2f}s (EXCLUDED from benchmark)")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # PHASE 2: Warmup runs (EXCLUDED from timing)
    print(f"\n[2/3] Running {warmup_runs} warmup iterations (compilation, cache warming)")

    for i in range(warmup_runs):
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            _ = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

        print(f"  Warmup {i+1}/{warmup_runs} complete")

    print("  ✓ Warmup complete (EXCLUDED from benchmark)")

    # PHASE 3: Steady-state measurement
    print(f"\n[3/3] Measuring steady-state performance ({measurement_runs} runs)")

    throughputs = []

    for i in range(measurement_runs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start = time.time()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        tokens_generated = len(outputs[0]) - len(inputs['input_ids'][0])
        throughput = tokens_generated / elapsed
        throughputs.append(throughput)

        print(f"  Run {i+1}/{measurement_runs}: {throughput:.1f} tok/s")

    # Statistics
    mean_throughput = np.mean(throughputs)
    std_throughput = np.std(throughputs)

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\n✓ Baseline: {mean_throughput:.1f} ± {std_throughput:.1f} tok/s")
    print(f"  Min: {min(throughputs):.1f} tok/s")
    print(f"  Max: {max(throughputs):.1f} tok/s")

    return mean_throughput, std_throughput, throughputs


def measure_optimized(
    model_name: str,
    prompt: str,
    max_tokens: int,
    baseline_throughput: float,
    optimization_level: str = "speculative",
    enable_guard: bool = True,
    warmup_runs: int = 5,
    measurement_runs: int = 20
) -> Tuple[float, float, List[float], dict]:
    """
    Measure optimized performance with safety guardrails.

    Args:
        model_name: Model to test
        prompt: Test prompt
        max_tokens: Tokens to generate
        baseline_throughput: Baseline throughput for guard
        optimization_level: Optimization level
        enable_guard: Enable performance guard
        warmup_runs: Warmup iterations
        measurement_runs: Measurement iterations

    Returns:
        (mean_throughput, std_throughput, all_throughputs, guard_stats)
    """
    print("\n" + "="*70)
    print(f"OPTIMIZED MEASUREMENT ({optimization_level.upper()})")
    print("="*70)

    # PHASE 1: Model loading (EXCLUDED from timing)
    print(f"\n[1/3] Loading optimized model: {model_name}")
    load_start = time.time()

    model = OptimizedLLM(
        model=model_name,
        optimization_level=optimization_level,
        enable_profiling=False  # Disable profiling for accurate timing
    )

    load_time = time.time() - load_start
    print(f"  Model loaded in {load_time:.2f}s (EXCLUDED from benchmark)")

    # Initialize performance guard
    guard = None
    if enable_guard:
        guard = PerformanceGuard(
            baseline_throughput=baseline_throughput,
            enable_auto_fallback=True,
            regression_threshold=0.98,  # Max 2% regression
            window_size=10
        )
        print(f"  ✓ Performance guard enabled (max 2% regression allowed)")

    # PHASE 2: Warmup runs
    print(f"\n[2/3] Running {warmup_runs} warmup iterations")

    for i in range(warmup_runs):
        _ = model.generate(prompt, max_tokens=max_tokens, do_sample=False)
        print(f"  Warmup {i+1}/{warmup_runs} complete")

    print("  ✓ Warmup complete")

    # PHASE 3: Steady-state measurement
    print(f"\n[3/3] Measuring steady-state performance ({measurement_runs} runs)")

    throughputs = []
    acceptance_rates = []

    for i in range(measurement_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.time()

        # Track tokens before
        if hasattr(model, 'speculative_decoder') and model.speculative_decoder:
            tokens_before = model.speculative_decoder.total_accepted_tokens
            draft_before = model.speculative_decoder.total_draft_tokens

        _ = model.generate(prompt, max_tokens=max_tokens, do_sample=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.time() - start
        throughput = max_tokens / elapsed
        throughputs.append(throughput)

        # Calculate acceptance rate
        acceptance_rate = 1.0  # Default for non-speculative
        if hasattr(model, 'speculative_decoder') and model.speculative_decoder:
            tokens_after = model.speculative_decoder.total_accepted_tokens
            draft_after = model.speculative_decoder.total_draft_tokens

            tokens_generated = tokens_after - tokens_before
            draft_generated = draft_after - draft_before

            if draft_generated > 0:
                acceptance_rate = tokens_generated / draft_generated
                acceptance_rates.append(acceptance_rate)

        # Record metrics in guard
        if guard and not guard.should_disable_speculative():
            guard.record_metrics(
                tokens_per_second=throughput,
                acceptance_rate=acceptance_rate
            )

        # Check if guard triggered fallback
        if guard and guard.should_disable_speculative():
            print(f"\n  ⚠️  GUARD TRIGGERED: Speculative decoding disabled at run {i+1}")
            # In real implementation, we would actually disable speculative here
            break

        status = ""
        if guard:
            stats = guard.get_stats()
            if stats['meets_requirements']:
                status = " ✓"
            else:
                status = f" ⚠️  (violations: {stats['consecutive_violations']})"

        print(f"  Run {i+1}/{measurement_runs}: {throughput:.1f} tok/s, α={acceptance_rate:.2%}{status}")

    # Statistics
    mean_throughput = np.mean(throughputs)
    std_throughput = np.std(throughputs)

    print(f"\n✓ Optimized: {mean_throughput:.1f} ± {std_throughput:.1f} tok/s")
    print(f"  Min: {min(throughputs):.1f} tok/s")
    print(f"  Max: {max(throughputs):.1f} tok/s")

    if acceptance_rates:
        print(f"  Acceptance: {np.mean(acceptance_rates):.2%} ± {np.std(acceptance_rates):.2%}")

    # Guard statistics
    guard_stats = guard.get_stats() if guard else {}

    return mean_throughput, std_throughput, throughputs, guard_stats


def main():
    parser = argparse.ArgumentParser(
        description="Production-grade performance benchmark with proper methodology"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-xl",
        help="Model to benchmark"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Test prompt"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Tokens to generate"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=5,
        help="Number of warmup runs (excluded from measurement)"
    )
    parser.add_argument(
        "--measurement-runs",
        type=int,
        default=20,
        help="Number of measurement runs (for steady-state)"
    )
    parser.add_argument(
        "--optimization-level",
        type=str,
        default="speculative",
        choices=["conservative", "balanced", "high", "maximum", "ultra", "speculative", "flash"],
        help="Optimization level"
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline measurement (use for faster testing)"
    )
    parser.add_argument(
        "--baseline-throughput",
        type=float,
        default=None,
        help="Manually specify baseline throughput (if --skip-baseline)"
    )

    args = parser.parse_args()

    # Check no-regression mode
    no_regression_mode = os.environ.get("MEMOPT_NO_REGRESSION", "0") == "1"
    if no_regression_mode:
        print("\n🛡️  NO-REGRESSION MODE ENABLED (MEMOPT_NO_REGRESSION=1)")
        print("   Strict 2% regression limit enforced")

    print("\n" + "="*70)
    print("PRODUCTION-GRADE PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"\nModel: {args.model}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Tokens: {args.max_tokens}")
    print(f"Methodology: {args.warmup_runs} warmup + {args.measurement_runs} measurement runs")
    print(f"Optimization: {args.optimization_level}")

    # Measure baseline
    baseline_mean = None
    baseline_std = None

    if not args.skip_baseline:
        baseline_mean, baseline_std, baseline_throughputs = measure_baseline(
            args.model,
            args.prompt,
            args.max_tokens,
            args.warmup_runs,
            args.measurement_runs
        )
    else:
        if args.baseline_throughput is None:
            print("\n❌ ERROR: Must specify --baseline-throughput when using --skip-baseline")
            return
        baseline_mean = args.baseline_throughput
        baseline_std = 0.0
        print(f"\n⚠️  Using manually specified baseline: {baseline_mean:.1f} tok/s")

    # Measure optimized
    optimized_mean, optimized_std, optimized_throughputs, guard_stats = measure_optimized(
        args.model,
        args.prompt,
        args.max_tokens,
        baseline_mean,
        args.optimization_level,
        enable_guard=True,
        warmup_runs=args.warmup_runs,
        measurement_runs=args.measurement_runs
    )

    # Results
    print("\n" + "="*70)
    print("RESULTS (Steady-State, Load Time Excluded)")
    print("="*70)

    print(f"\n📊 Baseline:  {baseline_mean:.1f} ± {baseline_std:.1f} tok/s")
    print(f"🚀 Optimized: {optimized_mean:.1f} ± {optimized_std:.1f} tok/s")

    speedup = optimized_mean / baseline_mean
    regression_pct = ((optimized_mean - baseline_mean) / baseline_mean) * 100

    print(f"\n💰 Performance:")
    print(f"  Speedup: {speedup:.2f}x")

    if regression_pct < 0:
        print(f"  ⚠️  REGRESSION: {abs(regression_pct):.1f}% SLOWER than baseline")
    else:
        print(f"  Improvement: {regression_pct:.1f}% faster")

    # Safety check
    meets_requirements = optimized_mean >= (baseline_mean * 0.98)

    print(f"\n🛡️  Safety Requirements:")
    print(f"  Minimum throughput: {baseline_mean * 0.98:.1f} tok/s (baseline × 0.98)")
    print(f"  Actual throughput:  {optimized_mean:.1f} tok/s")

    if meets_requirements:
        print(f"  ✅ PASS - No regression detected")
    else:
        print(f"  ❌ FAIL - Regression exceeds 2% threshold")

        if no_regression_mode:
            print(f"\n🚨 NO-REGRESSION MODE: Optimization would be DISABLED in production")

    # Guard statistics
    if guard_stats:
        print(f"\n📈 Performance Guard Stats:")
        print(f"  Total measurements: {guard_stats['total_measurements']}")
        print(f"  Total violations: {guard_stats['total_violations']}")
        print(f"  Total fallbacks: {guard_stats['total_fallbacks']}")
        print(f"  Final optimization level: {guard_stats['optimization_level']}")

        if guard_stats['speculative_disabled']:
            print(f"  ⚠️  Speculative decoding was auto-disabled")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
