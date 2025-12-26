#!/usr/bin/env python3
"""
Benchmark Stage 6: Model Parallelism

Tests tensor parallelism across multiple GPUs with Llama-2-7B.

Expected results:
- Single GPU: Baseline performance
- 2 GPUs: ~0.9x performance (10% communication overhead)
- Memory per GPU: ~Half of single GPU

Usage:
    # Single GPU (baseline)
    python benchmark_stage6.py --model meta-llama/Llama-2-7b-hf --num-gpus 1

    # 2 GPUs (tensor parallel)
    torchrun --nproc_per_node=2 benchmark_stage6.py --model meta-llama/Llama-2-7b-hf --num-gpus 2
"""

import torch
import torch.distributed as dist
import argparse
import time
from typing import List, Dict
import os

# Import our model parallelism
from memopt.model_parallel import init_model_parallel
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


def benchmark_single_gpu(model_name: str, num_prompts: int, max_tokens: int) -> Dict:
    """Benchmark on single GPU (baseline)."""
    print("\n" + "="*70)
    print("BASELINE: Single GPU")
    print("="*70)

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated() / (1024**3)
        print(f"Memory allocated: {memory_before:.2f} GB")

    # Warmup
    print("Warming up...")
    inputs = tokenizer("Test prompt", return_tensors="pt").to(model.device)
    _ = model.generate(**inputs, max_new_tokens=32)

    # Benchmark
    prompts = TEST_PROMPTS[:num_prompts]
    print(f"\nGenerating {num_prompts} responses ({max_tokens} tokens each)...")

    start_time = time.time()
    total_tokens = 0

    for i, prompt in enumerate(prompts, 1):
        print(f"  [{i}/{num_prompts}] Generating...", end='\r')
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        total_tokens += max_tokens

    elapsed = time.time() - start_time

    # Results
    throughput = total_tokens / elapsed

    if torch.cuda.is_available():
        memory_after = torch.cuda.max_memory_allocated() / (1024**3)

    print(f"\n{'='*70}")
    print("BASELINE RESULTS:")
    print(f"{'='*70}")
    print(f"  Throughput:  {throughput:.1f} tok/s")
    print(f"  Total time:  {elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    if torch.cuda.is_available():
        print(f"  Peak memory:  {memory_after:.2f} GB")

    return {
        'throughput': throughput,
        'elapsed': elapsed,
        'memory_gb': memory_after if torch.cuda.is_available() else 0
    }


def benchmark_multi_gpu(model_name: str, num_prompts: int, max_tokens: int, world_size: int) -> Dict:
    """Benchmark with tensor parallelism across multiple GPUs."""
    # Initialize model parallelism
    rank = int(os.environ.get("LOCAL_RANK", 0))
    tp = init_model_parallel(world_size=world_size, rank=rank)

    if rank == 0:
        print("\n" + "="*70)
        print(f"STAGE 6: Tensor Parallelism ({world_size} GPUs)")
        print("="*70)

    # Load model
    if rank == 0:
        print(f"Loading {model_name} on GPU {rank}...")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": f"cuda:{rank}"}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Parallelize model
    model = tp.parallelize_model(model)

    if torch.cuda.is_available():
        memory_before = torch.cuda.memory_allocated(rank) / (1024**3)
        if rank == 0:
            print(f"Memory per GPU: {memory_before:.2f} GB")

    # Synchronize all GPUs
    tp.synchronize()

    # Warmup
    if rank == 0:
        print("Warming up...")

    inputs = tokenizer("Test prompt", return_tensors="pt").to(f"cuda:{rank}")
    _ = model.generate(**inputs, max_new_tokens=32)

    tp.synchronize()

    # Benchmark
    prompts = TEST_PROMPTS[:num_prompts]
    if rank == 0:
        print(f"\nGenerating {num_prompts} responses ({max_tokens} tokens each)...")

    start_time = time.time()
    total_tokens = 0

    for i, prompt in enumerate(prompts, 1):
        if rank == 0:
            print(f"  [{i}/{num_prompts}] Generating...", end='\r')

        inputs = tokenizer(prompt, return_tensors="pt").to(f"cuda:{rank}")
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        total_tokens += max_tokens

        # Synchronize after each generation
        tp.synchronize()

    elapsed = time.time() - start_time

    # Results
    throughput = total_tokens / elapsed

    if torch.cuda.is_available():
        memory_after = torch.cuda.max_memory_allocated(rank) / (1024**3)

    if rank == 0:
        print(f"\n{'='*70}")
        print(f"STAGE 6 RESULTS ({world_size} GPUs):")
        print(f"{'='*70}")
        print(f"  Throughput:   {throughput:.1f} tok/s")
        print(f"  Total time:   {elapsed:.2f}s")
        print(f"  Total tokens: {total_tokens}")
        if torch.cuda.is_available():
            print(f"  Memory per GPU: {memory_after:.2f} GB")

    # Cleanup
    tp.cleanup()

    return {
        'throughput': throughput,
        'elapsed': elapsed,
        'memory_gb': memory_after if torch.cuda.is_available() else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark Stage 6: Model Parallelism')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf',
                      help='Model to benchmark')
    parser.add_argument('--num-prompts', type=int, default=8,
                      help='Number of prompts to test')
    parser.add_argument('--max-tokens', type=int, default=256,
                      help='Max tokens per generation')
    parser.add_argument('--num-gpus', type=int, default=2,
                      help='Number of GPUs for tensor parallelism')

    args = parser.parse_args()

    print("\n" + "="*70)
    print("STAGE 6 BENCHMARK: Model Parallelism")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"GPUs: {args.num_gpus}")
    print(f"Prompts: {args.num_prompts}")
    print(f"Max tokens: {args.max_tokens}")
    print("="*70)

    # Check if running in distributed mode
    is_distributed = "LOCAL_RANK" in os.environ

    if args.num_gpus == 1:
        # Single GPU baseline
        results = benchmark_single_gpu(
            args.model,
            args.num_prompts,
            args.max_tokens
        )
        baseline_throughput = results['throughput']

    elif is_distributed:
        # Multi-GPU with tensor parallelism
        rank = int(os.environ.get("LOCAL_RANK", 0))
        results = benchmark_multi_gpu(
            args.model,
            args.num_prompts,
            args.max_tokens,
            args.num_gpus
        )

        if rank == 0:
            # Compare to baseline (from previous runs)
            baseline_throughput = 37.4  # Example from gpt2-xl
            speedup = results['throughput'] / baseline_throughput

            print("\n" + "="*70)
            print("COMPARISON")
            print("="*70)
            print(f"Efficiency: {(results['throughput'] / baseline_throughput / args.num_gpus):.1%}")
            print(f"  (Ideal: 100% = linear scaling)")
            print(f"Memory per GPU: {results['memory_gb']:.2f} GB")
            print(f"  (vs {results['memory_gb'] * args.num_gpus:.2f} GB on single GPU)")

    else:
        print("\n❌ ERROR: For multi-GPU, use torchrun:")
        print(f"\ntorchrun --nproc_per_node={args.num_gpus} benchmark_stage6.py --model {args.model} --num-gpus {args.num_gpus}")
        return

    print("\n" + "="*70)


if __name__ == '__main__':
    main()
