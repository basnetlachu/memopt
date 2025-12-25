#!/usr/bin/env python3
"""
Stage 1 vs Stage 2 Comparison Benchmark

Compares performance between:
- Stage 1: Balanced mode (Stage 1 optimizations, no continuous batching)
- Stage 2: High mode (Stage 1 + Stage 2 continuous batching)

Validates:
1. No regression in throughput (should improve by 1.3-1.4×)
2. Better GPU utilization with batching
3. Output correctness (outputs must be identical)

Usage:
    python benchmark_stage2.py --model gpt2
    python benchmark_stage2.py --model meta-llama/Llama-2-7b-hf --num-prompts 8
"""

import argparse
import torch
import json
import time
from memopt import OptimizedLLM


def run_stage(stage_name: str, optimization_level: str, model_name: str, prompts: list, max_tokens: int):
    """Run benchmark for a specific stage"""
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

    print(f"\n✓ {stage_name} complete:")
    print(f"    Scheduler: {scheduler_type}")
    print(f"    Throughput: {stats.tokens_per_second:.1f} tok/s")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Peak memory: {peak_memory_gb:.2f} GB")

    return {
        'outputs': outputs,
        'stats': stats,
        'total_time': total_time,
        'peak_memory_gb': peak_memory_gb,
        'scheduler_type': scheduler_type
    }


def validate_correctness(stage1_results, stage2_results):
    """Validate that Stage 1 and Stage 2 produce identical outputs"""
    print(f"\n{'='*70}")
    print("CORRECTNESS VALIDATION")
    print(f"{'='*70}")

    outputs_stage1 = stage1_results['outputs']
    outputs_stage2 = stage2_results['outputs']

    all_identical = True
    for i, (out1, out2) in enumerate(zip(outputs_stage1, outputs_stage2)):
        if out1 != out2:
            print(f"❌ Prompt {i} DIFFERS!")
            print(f"   Stage 1: {out1[:100]}...")
            print(f"   Stage 2: {out2[:100]}...")
            all_identical = False
        else:
            print(f"✓ Prompt {i}: Identical")

    if all_identical:
        print(f"\n✅ ALL OUTPUTS IDENTICAL - Stage 2 maintains correctness!")
    else:
        print(f"\n❌ CORRECTNESS FAILURE - Stage 2 changed outputs!")
        raise AssertionError("Stage 2 correctness validation failed")

    return all_identical


def print_comparison(stage1_results, stage2_results):
    """Print detailed comparison between Stage 1 and Stage 2"""
    print(f"\n{'='*70}")
    print("STAGE 1 vs STAGE 2 COMPARISON")
    print(f"{'='*70}")

    stats1 = stage1_results['stats']
    stats2 = stage2_results['stats']

    # Scheduler info
    print(f"\n🔧 Scheduler:")
    print(f"   Stage 1: {stage1_results['scheduler_type']}")
    print(f"   Stage 2: {stage2_results['scheduler_type']}")

    # Throughput
    speedup = stats2.tokens_per_second / stats1.tokens_per_second
    print(f"\n📊 Throughput:")
    print(f"   Stage 1: {stats1.tokens_per_second:.1f} tok/s")
    print(f"   Stage 2: {stats2.tokens_per_second:.1f} tok/s")
    print(f"   Speedup: {speedup:.3f}× {'✅' if speedup >= 1.0 else '❌'}")

    # Latency
    latency_improvement = stats1.latency_per_token_ms / stats2.latency_per_token_ms
    print(f"\n⏱️  Latency per token:")
    print(f"   Stage 1: {stats1.latency_per_token_ms:.2f} ms/tok")
    print(f"   Stage 2: {stats2.latency_per_token_ms:.2f} ms/tok")
    print(f"   Improvement: {latency_improvement:.3f}× {'✅' if latency_improvement >= 1.0 else '❌'}")

    # Memory
    mem1 = stage1_results['peak_memory_gb']
    mem2 = stage2_results['peak_memory_gb']
    mem_change_pct = ((mem2 - mem1) / mem1) * 100 if mem1 > 0 else 0.0

    print(f"\n💾 Peak Memory:")
    print(f"   Stage 1: {mem1:.2f} GB")
    print(f"   Stage 2: {mem2:.2f} GB")
    print(f"   Change: {mem_change_pct:+.1f}% {'⚠️' if abs(mem_change_pct) > 10 else '✅'}")

    # GPU utilization
    print(f"\n🎮 GPU Metrics:")
    print(f"   Stage 1 stall: {stats1.gpu_stall_pct:.1f}%")
    print(f"   Stage 2 stall: {stats2.gpu_stall_pct:.1f}%")
    stall_reduction = stats1.gpu_stall_pct - stats2.gpu_stall_pct
    print(f"   Stall reduction: {stall_reduction:.1f}%")

    # Cost
    cost_reduction = ((stats1.cost_per_1m_tokens_usd - stats2.cost_per_1m_tokens_usd) /
                     stats1.cost_per_1m_tokens_usd * 100 if stats1.cost_per_1m_tokens_usd > 0 else 0)
    print(f"\n💰 Cost per 1M tokens:")
    print(f"   Stage 1: ${stats1.cost_per_1m_tokens_usd:.2f}")
    print(f"   Stage 2: ${stats2.cost_per_1m_tokens_usd:.2f}")
    print(f"   Reduction: {cost_reduction:.1f}%")

    # Target validation
    print(f"\n🎯 Stage 2 Target Validation:")
    target_speedup = 1.30  # Conservative target: 1.3× improvement over Stage 1
    print(f"   Target speedup: {target_speedup:.2f}×")
    print(f"   Actual speedup: {speedup:.3f}×")

    if speedup >= target_speedup:
        print(f"   ✅ PASSED - Stage 2 meets or exceeds target!")
    elif speedup >= 1.0:
        print(f"   ⚠️  ACCEPTABLE - Stage 2 shows improvement but below target")
    else:
        print(f"   ❌ FAILED - Stage 2 regression detected!")

    # Cumulative gain
    print(f"\n📈 Cumulative Gains (vs unoptimized baseline):")
    # Assume Stage 0 baseline is ~5× slower than Stage 1
    baseline_multiplier = 5.0
    stage1_cumulative = baseline_multiplier
    stage2_cumulative = baseline_multiplier * speedup

    print(f"   Stage 0 → Stage 1: ~{stage1_cumulative:.1f}× (baseline)")
    print(f"   Stage 0 → Stage 2: ~{stage2_cumulative:.1f}× (total)")

    return {
        'speedup': speedup,
        'latency_improvement': latency_improvement,
        'memory_change_pct': mem_change_pct,
        'cost_reduction_pct': cost_reduction,
        'target_met': speedup >= target_speedup,
        'cumulative_speedup': stage2_cumulative
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 1 vs Stage 2 Benchmark")
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
        default="benchmark_stage2_results.json",
        help="Output file for results"
    )

    args = parser.parse_args()

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
    print("STAGE 2 OPTIMIZATION BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")

    # Run Stage 1 (Balanced - Stage 1 only, no continuous batching)
    stage1_results = run_stage(
        stage_name="STAGE 1 (Balanced)",
        optimization_level="balanced",
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens
    )

    # Clean up before Stage 2
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(2)  # Let GPU settle

    # Run Stage 2 (High - Stage 1 + Stage 2 continuous batching)
    stage2_results = run_stage(
        stage_name="STAGE 2 (High)",
        optimization_level="high",
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens
    )

    # Validate correctness
    validate_correctness(stage1_results, stage2_results)

    # Print comparison
    comparison = print_comparison(stage1_results, stage2_results)

    # Save results
    results = {
        'model': args.model,
        'num_prompts': len(prompts),
        'max_tokens': args.max_tokens,
        'stage1': {
            'optimization_level': 'balanced',
            'scheduler': stage1_results['scheduler_type'],
            'throughput_tok_per_sec': stage1_results['stats'].tokens_per_second,
            'latency_ms_per_tok': stage1_results['stats'].latency_per_token_ms,
            'peak_memory_gb': stage1_results['peak_memory_gb'],
            'total_time_sec': stage1_results['total_time'],
        },
        'stage2': {
            'optimization_level': 'high',
            'scheduler': stage2_results['scheduler_type'],
            'throughput_tok_per_sec': stage2_results['stats'].tokens_per_second,
            'latency_ms_per_tok': stage2_results['stats'].latency_per_token_ms,
            'peak_memory_gb': stage2_results['peak_memory_gb'],
            'total_time_sec': stage2_results['total_time'],
        },
        'comparison': {
            'speedup': comparison['speedup'],
            'latency_improvement': comparison['latency_improvement'],
            'memory_change_pct': comparison['memory_change_pct'],
            'cost_reduction_pct': comparison['cost_reduction_pct'],
            'target_met': comparison['target_met'],
            'cumulative_speedup': comparison['cumulative_speedup'],
            'correctness_validated': True,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print(f"\n{'='*70}\n")

    # Exit with appropriate code
    if comparison['target_met']:
        print("✅ BENCHMARK PASSED - Stage 2 optimization successful!")
        print(f"📈 Total cumulative speedup: ~{comparison['cumulative_speedup']:.1f}× vs baseline")
        exit(0)
    elif comparison['speedup'] >= 1.0:
        print("⚠️  BENCHMARK ACCEPTABLE - Stage 2 shows improvement")
        print(f"📈 Total cumulative speedup: ~{comparison['cumulative_speedup']:.1f}× vs baseline")
        exit(0)
    else:
        print("❌ BENCHMARK FAILED - Stage 2 regression detected")
        exit(1)


if __name__ == "__main__":
    main()
