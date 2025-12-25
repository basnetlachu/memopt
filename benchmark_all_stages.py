#!/usr/bin/env python3
"""
Comprehensive Multi-Stage Benchmark

Compares ALL optimization stages in a single run:
- Stage 0: Conservative (baseline)
- Stage 1: Balanced (memory allocation optimizations)
- Stage 2: High (continuous batching)

Usage:
    python benchmark_all_stages.py --model gpt2-xl --num-prompts 8
"""

import argparse
import torch
import json
import time
from memopt import OptimizedLLM


def run_stage(
    stage_name: str,
    optimization_level: str,
    model_name: str,
    prompts: list,
    max_tokens: int
):
    """Run benchmark for a specific optimization stage"""
    print(f"\n{'='*70}")
    print(f"RUNNING {stage_name}")
    print(f"Optimization level: {optimization_level}")
    print(f"{'='*70}")

    # Reset CUDA memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load model
    model = OptimizedLLM(
        model=model_name,
        optimization_level=optimization_level,
        enable_profiling=True,
        expected_batch_size=len(prompts),
        expected_seq_len=max_tokens
    )

    # Warmup run
    print("  Warming up...")
    _ = model.generate(prompts[0], max_tokens=50, do_sample=False)
    model.reset_kv_cache()

    # Reset stats after warmup
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    # Actual benchmark
    print(f"  Running {len(prompts)} prompts...")
    start_time = time.time()
    outputs = []

    for i, prompt in enumerate(prompts):
        print(f"    Prompt {i+1}/{len(prompts)}...")
        torch.manual_seed(42 + i)  # For reproducibility

        output = model.generate(
            prompt,
            max_tokens=max_tokens,
            do_sample=False  # Greedy for determinism
        )
        outputs.append(output)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()
    total_time = end_time - start_time

    # Get stats
    stats = model.get_profiling_stats()

    # Get memory stats
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        peak_memory_gb = 0.0

    # Get scheduler info
    scheduler_type = type(model.scheduler).__name__

    # Get KV cache stats
    kv_stats = None
    if model.kv_cache:
        kv_stats = model.kv_cache.get_stats()

    print(f"\n✓ {stage_name} complete:")
    print(f"    Scheduler: {scheduler_type}")
    print(f"    Throughput: {stats.tokens_per_second:.1f} tok/s")
    print(f"    Latency: {stats.latency_per_token_ms:.2f} ms/tok")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Peak memory: {peak_memory_gb:.2f} GB")
    if kv_stats:
        print(f"    KV cache: {kv_stats.used_pages}/{kv_stats.total_pages} blocks ({kv_stats.utilization*100:.1f}%)")

    return {
        'stage_name': stage_name,
        'optimization_level': optimization_level,
        'outputs': outputs,
        'stats': stats,
        'total_time': total_time,
        'peak_memory_gb': peak_memory_gb,
        'scheduler_type': scheduler_type,
        'kv_stats': kv_stats
    }


def validate_correctness(baseline_results, stage_results_list):
    """Validate that all stages produce identical outputs"""
    print(f"\n{'='*70}")
    print("CORRECTNESS VALIDATION")
    print(f"{'='*70}")

    baseline_outputs = baseline_results['outputs']
    all_correct = True

    for stage_result in stage_results_list:
        stage_name = stage_result['stage_name']
        stage_outputs = stage_result['outputs']

        print(f"\n{stage_name} vs Baseline:")
        stage_correct = True

        for i, (baseline_out, stage_out) in enumerate(zip(baseline_outputs, stage_outputs)):
            if baseline_out != stage_out:
                print(f"  ❌ Prompt {i} DIFFERS!")
                print(f"     Baseline: {baseline_out[:80]}...")
                print(f"     {stage_name}: {stage_out[:80]}...")
                stage_correct = False
                all_correct = False
            else:
                print(f"  ✓ Prompt {i}: Identical")

        if stage_correct:
            print(f"  ✅ {stage_name}: ALL OUTPUTS IDENTICAL")
        else:
            print(f"  ❌ {stage_name}: CORRECTNESS FAILURE")

    if all_correct:
        print(f"\n{'='*70}")
        print("✅ ALL STAGES PRODUCE IDENTICAL OUTPUTS")
        print("{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("❌ CORRECTNESS FAILURES DETECTED")
        print("{'='*70}")

    return all_correct


def print_comparison_table(all_results):
    """Print comparison table for all stages"""
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON TABLE")
    print(f"{'='*70}\n")

    # Header
    print(f"{'Stage':<25} {'Throughput':<15} {'Latency':<15} {'Memory':<12} {'Speedup':<10}")
    print(f"{'-'*25} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")

    baseline_throughput = all_results[0]['stats'].tokens_per_second

    for result in all_results:
        stage_name = result['stage_name']
        stats = result['stats']
        throughput = stats.tokens_per_second
        latency = stats.latency_per_token_ms
        memory = result['peak_memory_gb']
        speedup = throughput / baseline_throughput

        speedup_emoji = "✅" if speedup >= 1.0 else "❌"

        print(f"{stage_name:<25} {throughput:>7.1f} tok/s   {latency:>6.2f} ms/tok   "
              f"{memory:>5.2f} GB    {speedup:>4.2f}× {speedup_emoji}")

    print()


def print_detailed_comparison(all_results):
    """Print detailed metrics comparison"""
    print(f"\n{'='*70}")
    print("DETAILED METRICS")
    print(f"{'='*70}")

    baseline = all_results[0]
    baseline_stats = baseline['stats']

    for i, result in enumerate(all_results):
        if i == 0:
            continue  # Skip baseline

        stage_name = result['stage_name']
        stats = result['stats']

        print(f"\n{stage_name} vs Baseline:")
        print(f"{'─'*70}")

        # Throughput
        speedup = stats.tokens_per_second / baseline_stats.tokens_per_second
        print(f"📊 Throughput:")
        print(f"   Baseline: {baseline_stats.tokens_per_second:.1f} tok/s")
        print(f"   {stage_name}: {stats.tokens_per_second:.1f} tok/s")
        print(f"   Speedup: {speedup:.3f}× {'✅' if speedup >= 1.0 else '❌'}")

        # Latency
        latency_improvement = baseline_stats.latency_per_token_ms / stats.latency_per_token_ms
        print(f"\n⏱️  Latency per token:")
        print(f"   Baseline: {baseline_stats.latency_per_token_ms:.2f} ms/tok")
        print(f"   {stage_name}: {stats.latency_per_token_ms:.2f} ms/tok")
        print(f"   Improvement: {latency_improvement:.3f}× {'✅' if latency_improvement >= 1.0 else '❌'}")

        # Memory
        mem_baseline = baseline['peak_memory_gb']
        mem_stage = result['peak_memory_gb']
        mem_change = ((mem_stage - mem_baseline) / mem_baseline) * 100 if mem_baseline > 0 else 0.0

        print(f"\n💾 Peak Memory:")
        print(f"   Baseline: {mem_baseline:.2f} GB")
        print(f"   {stage_name}: {mem_stage:.2f} GB")
        print(f"   Change: {mem_change:+.1f}%")

        # Cost
        cost_reduction = ((baseline_stats.cost_per_1m_tokens_usd - stats.cost_per_1m_tokens_usd) /
                         baseline_stats.cost_per_1m_tokens_usd * 100 if baseline_stats.cost_per_1m_tokens_usd > 0 else 0)
        print(f"\n💰 Cost per 1M tokens:")
        print(f"   Baseline: ${baseline_stats.cost_per_1m_tokens_usd:.2f}")
        print(f"   {stage_name}: ${stats.cost_per_1m_tokens_usd:.2f}")
        print(f"   Reduction: {cost_reduction:.1f}%")

        # Scheduler
        print(f"\n🔧 Scheduler:")
        print(f"   Baseline: {baseline['scheduler_type']}")
        print(f"   {stage_name}: {result['scheduler_type']}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Performance Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or path"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens to generate per prompt"
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=5,
        help="Number of prompts to test"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_all_stages_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="0,1,2",
        help="Stages to test (comma-separated: 0,1,2)"
    )

    args = parser.parse_args()

    # Parse stages
    stages_to_test = [int(s.strip()) for s in args.stages.split(",")]

    # Test prompts
    all_prompts = [
        "Explain how neural networks work in simple terms.",
        "Write a Python function to compute the Fibonacci sequence.",
        "What are the key differences between RAM and storage?",
        "Describe the process of photosynthesis step by step.",
        "How does TCP/IP networking function at a high level?",
        "Explain the concept of recursion with an example.",
        "What makes quantum computing different from classical computing?",
        "Write a short story about a robot learning to paint.",
    ]
    prompts = all_prompts[:args.num_prompts]

    print(f"\n{'='*70}")
    print("MULTI-STAGE OPTIMIZATION BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"Testing stages: {', '.join(f'Stage {s}' for s in stages_to_test)}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")

    # Run all stages
    all_results = []

    stage_configs = {
        0: ("STAGE 0 (Conservative - Baseline)", "conservative"),
        1: ("STAGE 1 (Balanced - Memory Optimized)", "balanced"),
        2: ("STAGE 2 (High - Continuous Batching)", "high"),
    }

    for stage_num in stages_to_test:
        if stage_num not in stage_configs:
            print(f"\n⚠️  Warning: Unknown stage {stage_num}, skipping")
            continue

        stage_name, opt_level = stage_configs[stage_num]

        result = run_stage(
            stage_name=stage_name,
            optimization_level=opt_level,
            model_name=args.model,
            prompts=prompts,
            max_tokens=args.max_tokens
        )

        all_results.append(result)

        # Clean up between stages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(2)  # Let GPU settle

    # Validate correctness
    if len(all_results) > 1:
        baseline_result = all_results[0]
        other_results = all_results[1:]
        all_correct = validate_correctness(baseline_result, other_results)
    else:
        all_correct = True
        print("\n⚠️  Only one stage tested, skipping correctness validation")

    # Print comparison table
    if len(all_results) > 1:
        print_comparison_table(all_results)
        print_detailed_comparison(all_results)

    # Save results
    results_json = {
        'model': args.model,
        'num_prompts': len(prompts),
        'max_tokens': args.max_tokens,
        'stages': {}
    }

    for result in all_results:
        stage_name = result['stage_name']
        results_json['stages'][stage_name] = {
            'optimization_level': result['optimization_level'],
            'scheduler': result['scheduler_type'],
            'throughput_tok_per_sec': result['stats'].tokens_per_second,
            'latency_ms_per_tok': result['stats'].latency_per_token_ms,
            'peak_memory_gb': result['peak_memory_gb'],
            'total_time_sec': result['total_time'],
            'cost_per_1m_tokens_usd': result['stats'].cost_per_1m_tokens_usd,
        }

    # Add comparison metrics
    if len(all_results) > 1:
        baseline_throughput = all_results[0]['stats'].tokens_per_second
        results_json['comparison'] = {}

        for result in all_results[1:]:
            stage_name = result['stage_name']
            speedup = result['stats'].tokens_per_second / baseline_throughput
            results_json['comparison'][stage_name] = {
                'speedup_vs_baseline': speedup,
                'target_met': speedup >= 1.2,  # Arbitrary target
            }

    results_json['correctness_validated'] = all_correct

    with open(args.output, 'w') as f:
        json.dump(results_json, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")

    # Final summary
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    if all_correct:
        print("✅ Correctness: ALL OUTPUTS IDENTICAL")
    else:
        print("❌ Correctness: FAILURES DETECTED")

    if len(all_results) > 1:
        best_stage = max(all_results, key=lambda r: r['stats'].tokens_per_second)
        baseline = all_results[0]
        best_speedup = best_stage['stats'].tokens_per_second / baseline['stats'].tokens_per_second

        print(f"\n🏆 Best Performance: {best_stage['stage_name']}")
        print(f"   Throughput: {best_stage['stats'].tokens_per_second:.1f} tok/s")
        print(f"   Speedup vs baseline: {best_speedup:.3f}×")

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
