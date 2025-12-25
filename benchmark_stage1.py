#!/usr/bin/env python3
"""
Stage 0 vs Stage 1 Comparison Benchmark

Compares performance between:
- Stage 0: Conservative mode (Stage 1 optimizations disabled)
- Stage 1: Balanced mode (Stage 1 optimizations enabled)

Validates:
1. No regression in throughput (should improve by 1.2-1.3×)
2. Memory reduction (peak memory should decrease)
3. Output correctness (outputs must be identical)

Usage:
    python benchmark_stage1.py --model gpt2
    python benchmark_stage1.py --model meta-llama/Llama-2-7b-hf --max-tokens 512
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

    # Get KV cache stats
    kv_stats = None
    if model.kv_cache:
        kv_stats = model.kv_cache.get_stats()

    print(f"\n✓ {stage_name} complete:")
    print(f"    Throughput: {stats.tokens_per_second:.1f} tok/s")
    print(f"    Total time: {total_time:.2f}s")
    print(f"    Peak memory: {peak_memory_gb:.2f} GB")

    return {
        'outputs': outputs,
        'stats': stats,
        'total_time': total_time,
        'peak_memory_gb': peak_memory_gb,
        'kv_stats': kv_stats
    }


def validate_correctness(stage0_results, stage1_results):
    """Validate that Stage 0 and Stage 1 produce identical outputs"""
    print(f"\n{'='*70}")
    print("CORRECTNESS VALIDATION")
    print(f"{'='*70}")

    outputs_stage0 = stage0_results['outputs']
    outputs_stage1 = stage1_results['outputs']

    all_identical = True
    for i, (out0, out1) in enumerate(zip(outputs_stage0, outputs_stage1)):
        if out0 != out1:
            print(f"❌ Prompt {i} DIFFERS!")
            print(f"   Stage 0: {out0[:100]}...")
            print(f"   Stage 1: {out1[:100]}...")
            all_identical = False
        else:
            print(f"✓ Prompt {i}: Identical")

    if all_identical:
        print(f"\n✅ ALL OUTPUTS IDENTICAL - Stage 1 maintains correctness!")
    else:
        print(f"\n❌ CORRECTNESS FAILURE - Stage 1 changed outputs!")
        raise AssertionError("Stage 1 correctness validation failed")

    return all_identical


def print_comparison(stage0_results, stage1_results):
    """Print detailed comparison between Stage 0 and Stage 1"""
    print(f"\n{'='*70}")
    print("STAGE 0 vs STAGE 1 COMPARISON")
    print(f"{'='*70}")

    stats0 = stage0_results['stats']
    stats1 = stage1_results['stats']

    # Throughput
    speedup = stats1.tokens_per_second / stats0.tokens_per_second
    print(f"\n📊 Throughput:")
    print(f"   Stage 0: {stats0.tokens_per_second:.1f} tok/s")
    print(f"   Stage 1: {stats1.tokens_per_second:.1f} tok/s")
    print(f"   Speedup: {speedup:.3f}× {'✅' if speedup >= 1.0 else '❌'}")

    # Latency
    latency_improvement = stats0.latency_per_token_ms / stats1.latency_per_token_ms
    print(f"\n⏱️  Latency per token:")
    print(f"   Stage 0: {stats0.latency_per_token_ms:.2f} ms/tok")
    print(f"   Stage 1: {stats1.latency_per_token_ms:.2f} ms/tok")
    print(f"   Improvement: {latency_improvement:.3f}× {'✅' if latency_improvement >= 1.0 else '❌'}")

    # Memory
    mem0 = stage0_results['peak_memory_gb']
    mem1 = stage1_results['peak_memory_gb']
    mem_reduction_pct = ((mem0 - mem1) / mem0) * 100 if mem0 > 0 else 0.0

    print(f"\n💾 Peak Memory:")
    print(f"   Stage 0: {mem0:.2f} GB")
    print(f"   Stage 1: {mem1:.2f} GB")
    print(f"   Reduction: {mem_reduction_pct:.1f}% {'✅' if mem_reduction_pct >= 0 else '⚠️'}")

    # KV Cache allocation
    if stage0_results['kv_stats'] and stage1_results['kv_stats']:
        kv0 = stage0_results['kv_stats']
        kv1 = stage1_results['kv_stats']

        print(f"\n🗂️  KV Cache Blocks:")
        print(f"   Stage 0: {kv0.total_pages} blocks allocated, {kv0.used_pages} used ({kv0.utilization*100:.1f}%)")
        print(f"   Stage 1: {kv1.total_pages} blocks allocated, {kv1.used_pages} used ({kv1.utilization*100:.1f}%)")

        block_reduction = kv0.total_pages - kv1.total_pages
        block_reduction_pct = (block_reduction / kv0.total_pages) * 100 if kv0.total_pages > 0 else 0.0
        print(f"   Block reduction: {block_reduction} blocks ({block_reduction_pct:.1f}%)")

    # Target validation
    print(f"\n🎯 Stage 1 Target Validation:")
    target_speedup = 1.20  # Conservative target: 1.2× improvement
    print(f"   Target speedup: {target_speedup:.2f}×")
    print(f"   Actual speedup: {speedup:.3f}×")

    if speedup >= target_speedup:
        print(f"   ✅ PASSED - Stage 1 meets or exceeds target!")
    elif speedup >= 1.0:
        print(f"   ⚠️  ACCEPTABLE - Stage 1 shows improvement but below target")
    else:
        print(f"   ❌ FAILED - Stage 1 regression detected!")

    return {
        'speedup': speedup,
        'latency_improvement': latency_improvement,
        'memory_reduction_pct': mem_reduction_pct,
        'target_met': speedup >= target_speedup
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 0 vs Stage 1 Benchmark")
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
        default="benchmark_stage1_results.json",
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
    print("STAGE 1 OPTIMIZATION BENCHMARK")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")

    # Run Stage 0 (Conservative - Stage 1 disabled)
    stage0_results = run_stage(
        stage_name="STAGE 0 (Conservative)",
        optimization_level="conservative",
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens
    )

    # Clean up before Stage 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        time.sleep(2)  # Let GPU settle

    # Run Stage 1 (Balanced - Stage 1 enabled)
    stage1_results = run_stage(
        stage_name="STAGE 1 (Balanced)",
        optimization_level="balanced",
        model_name=args.model,
        prompts=prompts,
        max_tokens=args.max_tokens
    )

    # Validate correctness
    validate_correctness(stage0_results, stage1_results)

    # Print comparison
    comparison = print_comparison(stage0_results, stage1_results)

    # Save results
    results = {
        'model': args.model,
        'num_prompts': len(prompts),
        'max_tokens': args.max_tokens,
        'stage0': {
            'optimization_level': 'conservative',
            'throughput_tok_per_sec': stage0_results['stats'].tokens_per_second,
            'latency_ms_per_tok': stage0_results['stats'].latency_per_token_ms,
            'peak_memory_gb': stage0_results['peak_memory_gb'],
            'total_time_sec': stage0_results['total_time'],
        },
        'stage1': {
            'optimization_level': 'balanced',
            'throughput_tok_per_sec': stage1_results['stats'].tokens_per_second,
            'latency_ms_per_tok': stage1_results['stats'].latency_per_token_ms,
            'peak_memory_gb': stage1_results['peak_memory_gb'],
            'total_time_sec': stage1_results['total_time'],
        },
        'comparison': {
            'speedup': comparison['speedup'],
            'latency_improvement': comparison['latency_improvement'],
            'memory_reduction_pct': comparison['memory_reduction_pct'],
            'target_met': comparison['target_met'],
            'correctness_validated': True,
        }
    }

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print(f"\n{'='*70}\n")

    # Exit with appropriate code
    if comparison['target_met']:
        print("✅ BENCHMARK PASSED - Stage 1 optimization successful!")
        exit(0)
    elif comparison['speedup'] >= 1.0:
        print("⚠️  BENCHMARK ACCEPTABLE - Stage 1 shows improvement")
        exit(0)
    else:
        print("❌ BENCHMARK FAILED - Stage 1 regression detected")
        exit(1)


if __name__ == "__main__":
    main()
