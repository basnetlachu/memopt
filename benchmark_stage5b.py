#!/usr/bin/env python3
"""
Benchmark Stage 5b: Speculative Decoding

Compares Stage 2 (current best: 6.2x) vs Stage 5b (Speculative Decoding).

Stage 5b uses a small draft model to generate K candidate tokens,
then verifies them with the main model in one forward pass.

Expected speedup: 2-3x on top of Stage 2's 6.2x = 12-18x total from baseline

Usage:
    python benchmark_stage5b.py --model gpt2-xl --num-prompts 8 --max-tokens 256
"""

import time
import argparse
from typing import List, Dict
from memopt import OptimizedLLM


# Test prompts for benchmarking
TEST_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where technology has advanced beyond our wildest dreams,",
    "The key to solving climate change lies in",
    "Once upon a time in a distant galaxy,",
    "The most important lesson I learned was",
    "Scientists have discovered that",
    "In the year 2050, humanity will",
    "The secret to happiness is",
    "When exploring the depths of the ocean,",
    "The evolution of human consciousness",
    "In the realm of quantum physics,",
    "The ancient civilization left behind",
]


def benchmark_stage(
    model_name: str,
    optimization_level: str,
    num_prompts: int,
    max_tokens: int,
    enable_profiling: bool = False
) -> Dict:
    """Benchmark a single optimization stage."""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {optimization_level.upper()}")
    print(f"{'='*70}")

    # Initialize model
    print(f"Loading model: {model_name} with optimization: {optimization_level}")
    start_load = time.time()

    model = OptimizedLLM(
        model=model_name,
        optimization_level=optimization_level,
        enable_profiling=enable_profiling
    )

    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f}s\n")

    # Select prompts
    prompts = TEST_PROMPTS[:num_prompts]

    # Warmup
    print("Warming up...")
    model.generate(prompts[0], max_tokens=32)

    # Benchmark
    print(f"\nGenerating {num_prompts} responses ({max_tokens} tokens each)...")
    outputs = []
    total_tokens = 0

    start_time = time.time()

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{num_prompts}] Generating...", end='\r')
        output = model.generate(prompt, max_tokens=max_tokens)
        outputs.append(output)

        # Count tokens (approximate)
        total_tokens += max_tokens

    end_time = time.time()
    elapsed = end_time - start_time

    # Calculate metrics
    throughput = total_tokens / elapsed
    avg_latency = elapsed / num_prompts

    print(f"\n{'='*70}")
    print(f"Results for {optimization_level.upper()}:")
    print(f"{'='*70}")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} tokens/sec")
    print(f"Average latency: {avg_latency:.2f}s per request")
    print(f"Total tokens: {total_tokens}")

    # Show profiling stats if enabled
    if enable_profiling and hasattr(model, 'get_stats'):
        stats = model.get_stats()
        if stats:
            print(f"\n{'-'*70}")
            print("Performance Stats:")
            print(f"{'-'*70}")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

    return {
        'optimization_level': optimization_level,
        'elapsed_time': elapsed,
        'throughput': throughput,
        'avg_latency': avg_latency,
        'total_tokens': total_tokens,
        'outputs': outputs
    }


def compare_outputs(baseline_outputs: List[str], stage_outputs: List[str], stage_name: str):
    """Compare outputs between baseline and optimized stage."""
    print(f"\n{'='*70}")
    print(f"Correctness Check: {stage_name}")
    print(f"{'='*70}")

    mismatches = 0
    for i, (baseline, stage) in enumerate(zip(baseline_outputs, stage_outputs)):
        if baseline != stage:
            mismatches += 1
            print(f"⚠️  Output {i+1} differs:")
            print(f"  Baseline: {baseline[:100]}...")
            print(f"  {stage_name}: {stage[:100]}...")

    if mismatches == 0:
        print(f"✓ All outputs match baseline (100% correctness)")
    else:
        print(f"✗ {mismatches}/{len(baseline_outputs)} outputs differ")

    return mismatches == 0


def main():
    parser = argparse.ArgumentParser(description='Benchmark Stage 5b: Speculative Decoding')
    parser.add_argument('--model', type=str, default='gpt2-xl',
                      help='Model to benchmark (default: gpt2-xl)')
    parser.add_argument('--num-prompts', type=int, default=8,
                      help='Number of prompts to test (default: 8)')
    parser.add_argument('--max-tokens', type=int, default=256,
                      help='Max tokens per generation (default: 256)')
    parser.add_argument('--enable-profiling', action='store_true',
                      help='Enable detailed profiling')
    parser.add_argument('--skip-baseline', action='store_true',
                      help='Skip baseline benchmark (faster)')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("Stage 5b Benchmark: Speculative Decoding")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Profiling: {'Enabled' if args.enable_profiling else 'Disabled'}")
    print("="*70)

    results = []

    # Benchmark baseline (if not skipped)
    if not args.skip_baseline:
        baseline_results = benchmark_stage(
            args.model,
            'none',
            args.num_prompts,
            args.max_tokens,
            args.enable_profiling
        )
        results.append(baseline_results)
        baseline_throughput = baseline_results['throughput']
        baseline_outputs = baseline_results['outputs']
    else:
        print("\n⚠️  Skipping baseline (using Stage 2 as reference)")
        baseline_throughput = 37.4  # From FINAL_STATUS.md
        baseline_outputs = None

    # Benchmark Stage 2 (current best)
    stage2_results = benchmark_stage(
        args.model,
        'high',  # Stage 2
        args.num_prompts,
        args.max_tokens,
        args.enable_profiling
    )
    results.append(stage2_results)
    stage2_throughput = stage2_results['throughput']

    # Benchmark Stage 5b (Speculative Decoding)
    stage5b_results = benchmark_stage(
        args.model,
        'speculative',  # Stage 5b
        args.num_prompts,
        args.max_tokens,
        args.enable_profiling
    )
    results.append(stage5b_results)
    stage5b_throughput = stage5b_results['throughput']

    # Compare outputs for correctness
    if baseline_outputs is not None:
        print("\n" + "="*70)
        print("Correctness Verification")
        print("="*70)

        compare_outputs(baseline_outputs, stage2_results['outputs'], "Stage 2")
        compare_outputs(baseline_outputs, stage5b_results['outputs'], "Stage 5b")

    # Calculate speedup metrics
    speedup_stage2 = stage2_throughput / baseline_throughput
    speedup_stage5b = stage5b_throughput / baseline_throughput

    # Summary
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n{'Stage':<25} {'Throughput':<15} {'Speedup':<10} {'Status'}")
    print("-" * 70)

    for result in results:
        level = result['optimization_level']
        throughput = result['throughput']
        speedup = throughput / baseline_throughput

        # Format stage name
        if level == 'none':
            stage_name = "Baseline"
        elif level == 'high':
            stage_name = "Stage 2 (High)"
        elif level == 'speculative':
            stage_name = "Stage 5b (Speculative)"
        else:
            stage_name = level

        # Status indicator
        if level == 'none':
            status = "📊 Reference"
        elif level == 'high':
            status = "✅ Current Best"
        elif speedup > speedup_stage2:
            status = "🚀 NEW BEST!"
        else:
            status = "⚠️  Slower"

        print(f"{stage_name:<25} {throughput:>6.1f} tok/s    {speedup:>5.2f}x     {status}")
    improvement = (stage5b_throughput / stage2_throughput - 1) * 100

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    print(f"Stage 2 speedup: {speedup_stage2:.2f}x over baseline")
    print(f"Stage 5b speedup: {speedup_stage5b:.2f}x over baseline")
    print(f"Stage 5b vs Stage 2: {improvement:+.1f}% ({stage5b_throughput/stage2_throughput:.2f}x)")

    if stage5b_throughput > stage2_throughput:
        print(f"\n✅ Stage 5b is FASTER than Stage 2!")
        print(f"   Use 'speculative' preset for production")
    else:
        print(f"\n⚠️  Stage 5b is slower than Stage 2")
        print(f"   Continue using 'high' preset for production")

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
