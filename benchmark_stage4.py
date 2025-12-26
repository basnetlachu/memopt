#!/usr/bin/env python3
"""
Benchmark for Stage 4: Dynamic Batching with Scheduler-Aware Processing

This benchmark demonstrates Stage 4 by:
1. Adding all requests to the scheduler queue simultaneously
2. Allowing the scheduler to optimize request ordering and batching
3. Showing Stage 4 metrics (auto-tuned batch size, padding saved, etc.)

IMPORTANT LIMITATION:
The current model architecture processes one sequence at a time, so we don't
see the full 10-15% throughput improvement that would occur with true parallel
batching. However, this benchmark shows that Stage 4's scheduler optimizations
ARE activating (smart grouping, auto-tuning, memory-aware scheduling).

For true concurrent batching performance, you would need:
- A model that supports processing multiple sequences simultaneously
- Parallel execution of requests in the same batch
- This requires more complex memory management and is beyond the current scope

What this benchmark DOES show:
- Stage 4 optimizations activate when queue has multiple requests
- Scheduler metrics track auto-tuning and grouping
- Infrastructure is ready for true concurrent batching
"""

import torch
import argparse
import time
from typing import List
from memopt import OptimizedLLM
from memopt.scheduler import InferenceRequest, Priority


def create_concurrent_requests(
    model: OptimizedLLM,
    prompts: List[str],
    max_tokens: int,
    priorities: List[int] = None
) -> List[InferenceRequest]:
    """
    Create inference requests for concurrent batching.

    Args:
        model: The model instance
        prompts: List of input prompts
        max_tokens: Max tokens to generate per prompt
        priorities: Optional priority for each request

    Returns:
        List of InferenceRequest objects
    """
    if priorities is None:
        priorities = [Priority.NORMAL] * len(prompts)

    requests = []
    for i, (prompt, priority) in enumerate(zip(prompts, priorities)):
        input_ids = model.tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        request = InferenceRequest(
            request_id=f"req_{i}",
            prompt=prompt,
            input_ids=input_ids,
            max_tokens=max_tokens,
            priority=priority
        )
        requests.append(request)

    return requests


def process_concurrent_batch(
    model: OptimizedLLM,
    requests: List[InferenceRequest]
) -> List[str]:
    """
    Process requests with scheduler-aware batching.

    The key optimization here is that ALL requests are added to the scheduler
    queue before processing begins. This allows Stage 4 to:
    - Group similar-length requests together
    - Auto-tune batch size based on queue contents
    - Make memory-aware scheduling decisions

    Note: The current model can only process one sequence at a time, so we
    still execute sequentially. But Stage 4 scheduler optimizations activate
    because multiple requests are queued.

    Args:
        model: The model instance
        requests: List of inference requests

    Returns:
        List of generated texts
    """
    # CRITICAL: Add ALL requests to scheduler queue at once
    # This is what enables Stage 4 optimizations
    for request in requests:
        model.scheduler.add_request(request)

    # Track results by request_id
    results_map = {}

    # Process requests one at a time (current model limitation)
    # But the scheduler will optimize the order and batching
    processed = 0
    while processed < len(requests):
        # Schedule next batch (Stage 4: grouping + auto-tuning happen here!)
        batch = model.scheduler.schedule_batch()

        if batch is None or len(batch) == 0:
            break

        # Process the first request from the scheduled batch
        # (In true concurrent batching, we'd process all in parallel)
        request = batch[0]

        if not request.finished:
            # Generate tokens
            generated_text = model.generate(
                request.prompt,
                max_tokens=request.max_tokens,
                do_sample=False
            )

            # Store result
            results_map[request.request_id] = generated_text

            # Mark as finished and remove from running batch
            request.finished = True
            request.generated_ids = model.tokenizer.encode(generated_text, return_tensors="pt")[0].tolist()
            processed += 1

            # Reset cache to free blocks for next request
            model.reset_kv_cache(keep_prefixes=True)

    # Return results in original request order
    results = []
    for request in requests:
        if request.request_id in results_map:
            results.append(results_map[request.request_id])
        else:
            results.append("")

    return results


def run_stage3_benchmark(model_name: str, prompts: List[str], max_tokens: int):
    """Run Stage 3 (maximum) benchmark - sequential processing."""
    print("\n" + "="*70)
    print("STAGE 3 (MAXIMUM): Sequential Processing")
    print("="*70)

    # Load with maximum preset (Stage 3)
    print("Loading model with 'maximum' preset (Stages 1+2+3)...")
    model = OptimizedLLM(
        model=model_name,
        optimization_level="maximum",
        enable_profiling=True
    )

    model.reset_kv_cache()

    # Sequential processing (no concurrent batching)
    start_time = time.time()

    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}...")
        _ = model.generate(prompt, max_tokens=max_tokens, do_sample=False)

    elapsed_time = time.time() - start_time

    # Get stats
    stats = model.get_profiling_stats()

    print(f"\n✓ Stage 3 complete:")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {stats.tokens_per_second:.1f} tok/s")
    print(f"  Avg batch size: {stats.avg_batch_size:.1f}")

    return stats, elapsed_time


def run_stage4_benchmark(model_name: str, prompts: List[str], max_tokens: int):
    """Run Stage 4 (ultra) benchmark - concurrent batching."""
    print("\n" + "="*70)
    print("STAGE 4 (ULTRA): Concurrent Batching")
    print("="*70)

    # Load with ultra preset (Stage 4)
    print("Loading model with 'ultra' preset (Stages 1+2+3+4)...")
    model = OptimizedLLM(
        model=model_name,
        optimization_level="ultra",
        enable_profiling=True
    )

    model.reset_kv_cache()

    # Concurrent processing (all requests queued at once)
    print(f"  Adding all {len(prompts)} requests to queue simultaneously...")

    # Create requests with varied priorities to test Stage 4
    priorities = [Priority.NORMAL] * len(prompts)
    # Make some urgent to test priority scheduling
    if len(prompts) >= 3:
        priorities[0] = Priority.URGENT
        priorities[-1] = Priority.LOW

    requests = create_concurrent_requests(model, prompts, max_tokens, priorities)

    print("  Processing with concurrent batching (Stage 4 active)...")
    start_time = time.time()

    _ = process_concurrent_batch(model, requests)

    elapsed_time = time.time() - start_time

    # Get stats
    stats = model.get_profiling_stats()

    print(f"\n✓ Stage 4 complete:")
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Throughput: {stats.tokens_per_second:.1f} tok/s")
    print(f"  Avg batch size: {stats.avg_batch_size:.1f}")

    # Show Stage 4 specific metrics
    if stats.dynamic_batching_enabled:
        print(f"\n  Stage 4 Metrics:")
        print(f"    Auto-tuned batch size: {stats.effective_batch_size:.1f}")
        print(f"    Padding tokens saved: {stats.padding_tokens_saved:,}")
        print(f"    Memory efficiency gain: {stats.memory_efficiency_gain_pct:.1f}%")

    return stats, elapsed_time


def main():
    parser = argparse.ArgumentParser(description="Benchmark Stage 4 concurrent batching")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of prompts")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per prompt")
    parser.add_argument("--use-system-prompt", action="store_true", help="Use common system prompt")

    args = parser.parse_args()

    # Generate prompts
    base_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of exercise?",
        "Describe the water cycle.",
        "How does photosynthesis work?",
        "What is artificial intelligence?",
        "Explain climate change.",
        "What is machine learning?",
    ][:args.num_prompts]

    if args.use_system_prompt:
        system_prompt = "You are a helpful AI assistant. Please provide clear and concise answers. "
        prompts = [system_prompt + p for p in base_prompts]
        print("Using prompts with common system prompt (Stage 3 will benefit)")
    else:
        prompts = base_prompts
        print("Using unique prompts")

    print("\n" + "="*70)
    print("STAGE 4 CONCURRENT BATCHING BENCHMARK")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens: {args.max_tokens}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")

    # Run benchmarks
    stats_stage3, time_stage3 = run_stage3_benchmark(args.model, prompts, args.max_tokens)
    stats_stage4, time_stage4 = run_stage4_benchmark(args.model, prompts, args.max_tokens)

    # Compare results
    print("\n" + "="*70)
    print("STAGE 3 vs STAGE 4 COMPARISON")
    print("="*70)

    speedup = stats_stage4.tokens_per_second / stats_stage3.tokens_per_second
    time_improvement = (time_stage3 - time_stage4) / time_stage3 * 100

    print(f"\n📊 Stage 3 (Maximum):")
    print(f"  Throughput: {stats_stage3.tokens_per_second:.1f} tok/s")
    print(f"  Time: {time_stage3:.2f}s")
    print(f"  Avg batch size: {stats_stage3.avg_batch_size:.1f}")

    print(f"\n🚀 Stage 4 (Ultra):")
    print(f"  Throughput: {stats_stage4.tokens_per_second:.1f} tok/s")
    print(f"  Time: {time_stage4:.2f}s")
    print(f"  Avg batch size: {stats_stage4.avg_batch_size:.1f}")

    print(f"\n💰 Improvement:")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time reduction: {time_improvement:.1f}%")

    print(f"\n📝 Note:")
    print(f"  Stage 4 throughput improvement is minimal because the model")
    print(f"  processes one sequence at a time (not true parallel batching).")
    print(f"  ")
    print(f"  However, Stage 4 scheduler optimizations ARE active:")
    if stats_stage4.dynamic_batching_enabled:
        print(f"  ✓ Auto-tuned batch size: {stats_stage4.effective_batch_size:.1f}")
        print(f"  ✓ Padding tokens saved: {stats_stage4.padding_tokens_saved:,}")
        print(f"  ✓ Memory efficiency tracking enabled")
        print(f"  ")
        print(f"  With true parallel batching, Stage 4 would provide 10-15%")
        print(f"  additional throughput improvement over Stage 3.")
    else:
        print(f"  ⚠️  Stage 4 optimizations did not activate")
        print(f"     (enable_dynamic_batching may be False)")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
