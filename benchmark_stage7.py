#!/usr/bin/env python3
"""
Benchmark Stage 7: Flash Attention + Speculative Decoding

Tests the combined performance of:
- Stage 5b: Speculative Decoding (15.45x)
- Stage 7: Flash Attention 2 (2-3x additional)
- Combined target: 30-60x speedup 🚀

Usage:
    # Test baseline
    python benchmark_stage7.py --model gpt2-xl --level baseline

    # Test Stage 5b only (speculative decoding)
    python benchmark_stage7.py --model gpt2-xl --level speculative

    # Test Stage 7 (Stage 5b + Flash Attention) - BEST
    python benchmark_stage7.py --model gpt2-xl --level flash

    # Compare all levels
    python benchmark_stage7.py --model gpt2-xl --level all
"""

import argparse
import torch
import time
from typing import Dict, List
import sys

from memopt import OptimizedLLM
from transformers import AutoModelForCausalLM, AutoTokenizer


TEST_PROMPTS = [
    "The future of artificial intelligence is",
    "In a world where technology has advanced beyond our wildest dreams,",
    "The key to solving climate change lies in",
    "Once upon a time in a distant galaxy,",
    "The most important lesson I learned was",
    "Scientists have discovered that",
    "In the year 2050, humanity will",
    "The secret to happiness is",
]


def run_baseline(model_name: str, num_prompts: int, max_tokens: int) -> Dict:
    """Run baseline inference without optimizations."""
    print("\n" + "="*70)
    print("BASELINE: No Optimizations")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()

    # Benchmark
    prompts = TEST_PROMPTS[:num_prompts]
    print(f"\nGenerating {num_prompts} responses ({max_tokens} tokens each)...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    total_tokens = 0

    with torch.no_grad():
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{num_prompts}] Generating...", end='\r')
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
            total_tokens += max_tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed

    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        memory_gb = 0

    print(f"\n{'='*70}")
    print("BASELINE RESULTS:")
    print(f"{'='*70}")
    print(f"  Throughput:  {throughput:.1f} tok/s")
    print(f"  Total time:  {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    if torch.cuda.is_available():
        print(f"  Peak memory:  {memory_gb:.2f} GB")

    return {
        'name': 'Baseline',
        'throughput': throughput,
        'elapsed': elapsed,
        'memory_gb': memory_gb,
        'total_tokens': total_tokens
    }


def run_optimized(model_name: str, num_prompts: int, max_tokens: int, level: str) -> Dict:
    """Run with specified optimization level."""
    level_names = {
        'speculative': 'Stage 5b: Speculative Decoding',
        'flash': 'Stage 7: Flash Attention + Speculative'
    }

    print("\n" + "="*70)
    print(level_names.get(level, f"Optimization: {level}"))
    print("="*70)

    # Load optimized model
    model = OptimizedLLM(
        model=model_name,
        optimization_level=level,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Benchmark
    prompts = TEST_PROMPTS[:num_prompts]
    print(f"\nGenerating {num_prompts} responses ({max_tokens} tokens each)...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.time()
    total_tokens = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{num_prompts}] Generating...", end='\r')
        response = model.generate(prompt, max_tokens=max_tokens)
        total_tokens += max_tokens

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.time() - start_time
    throughput = total_tokens / elapsed

    if torch.cuda.is_available():
        memory_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        memory_gb = 0

    print(f"\n{'='*70}")
    print(f"{level_names.get(level, level.upper())} RESULTS:")
    print(f"{'='*70}")
    print(f"  Throughput:  {throughput:.1f} tok/s")
    print(f"  Total time:  {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    if torch.cuda.is_available():
        print(f"  Peak memory:  {memory_gb:.2f} GB")

    return {
        'name': level_names.get(level, level),
        'throughput': throughput,
        'elapsed': elapsed,
        'memory_gb': memory_gb,
        'total_tokens': total_tokens
    }


def print_comparison(results: List[Dict], baseline_throughput: float):
    """Print comparison table."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n{'Configuration':<40} {'Throughput':<15} {'Speedup':<10} {'Memory':<10}")
    print("-" * 70)

    for result in results:
        throughput = result['throughput']
        speedup = throughput / baseline_throughput
        memory = result['memory_gb']

        print(f"{result['name']:<40} {throughput:>8.1f} tok/s   {speedup:>6.2f}x    {memory:>5.2f} GB")

    print("="*70)

    # Highlight best result
    best = max(results, key=lambda x: x['throughput'])
    print(f"\n🚀 Best performance: {best['name']}")
    print(f"   {best['throughput']:.1f} tok/s ({best['throughput']/baseline_throughput:.2f}x speedup)")

    # Check if we hit target
    if best['throughput'] / baseline_throughput >= 30:
        print(f"\n✅ TARGET ACHIEVED: {best['throughput']/baseline_throughput:.1f}x speedup (target: 30-60x)")
    elif best['throughput'] / baseline_throughput >= 15:
        print(f"\n⚠️  Close to target: {best['throughput']/baseline_throughput:.1f}x speedup (target: 30-60x)")
        print(f"   Flash Attention 2 may provide additional 2-3x boost on GPU")
    else:
        print(f"\n⚠️  Below target: {best['throughput']/baseline_throughput:.1f}x speedup (target: 30-60x)")

    print("")


def main():
    parser = argparse.ArgumentParser(description='Benchmark Stage 7: Flash Attention + Speculative Decoding')
    parser.add_argument('--model', type=str, default='gpt2-xl',
                       help='Model to benchmark')
    parser.add_argument('--num-prompts', type=int, default=8,
                       help='Number of prompts to test')
    parser.add_argument('--max-tokens', type=int, default=256,
                       help='Max tokens per generation')
    parser.add_argument('--level', type=str, default='all',
                       choices=['baseline', 'speculative', 'flash', 'all'],
                       help='Optimization level to test')

    args = parser.parse_args()

    print("="*70)
    print("STAGE 7 BENCHMARK: Flash Attention + Speculative Decoding")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Tokens per prompt: {args.max_tokens}")
    print(f"Test level: {args.level}")
    print("="*70)

    results = []

    if args.level == 'all':
        # Test all levels
        baseline = run_baseline(args.model, args.num_prompts, args.max_tokens)
        results.append(baseline)

        speculative = run_optimized(args.model, args.num_prompts, args.max_tokens, 'speculative')
        results.append(speculative)

        flash = run_optimized(args.model, args.num_prompts, args.max_tokens, 'flash')
        results.append(flash)

        print_comparison(results, baseline['throughput'])

    elif args.level == 'baseline':
        result = run_baseline(args.model, args.num_prompts, args.max_tokens)
        print(f"\n✓ Baseline benchmark complete: {result['throughput']:.1f} tok/s")

    else:
        # Run baseline first for comparison
        baseline = run_baseline(args.model, args.num_prompts, args.max_tokens)
        result = run_optimized(args.model, args.num_prompts, args.max_tokens, args.level)

        speedup = result['throughput'] / baseline['throughput']
        print(f"\n{'='*70}")
        print(f"SPEEDUP: {speedup:.2f}x")
        print(f"{'='*70}")
        print(f"Baseline:  {baseline['throughput']:.1f} tok/s")
        print(f"Optimized: {result['throughput']:.1f} tok/s")
        print(f"Speedup:   {speedup:.2f}x")

        if speedup >= 30:
            print(f"\n✅ TARGET ACHIEVED: {speedup:.1f}x (target: 30-60x)")
        elif speedup >= 15:
            print(f"\n⚠️  Partial success: {speedup:.1f}x (target: 30-60x)")
        else:
            print(f"\n⚠️  Below target: {speedup:.1f}x (target: 30-60x)")

    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
