"""
Fleet-level throughput benchmark for trillion-token scale.

Measures tokens/GPU/day instead of per-request latency.
"""

import torch
import time
import argparse
from memopt import OptimizedLLM


def benchmark_fleet_throughput(
    model_name: str = "Qwen/Qwen2-7B",
    num_requests: int = 100,
    tokens_per_request: int = 500,
    shared_prefix_ratio: float = 0.7
):
    """
    Simulate fleet-level throughput.

    Args:
        model_name: Model to use
        num_requests: Number of requests to simulate
        tokens_per_request: Tokens to generate per request
        shared_prefix_ratio: % of requests with shared prefix (0.0-1.0)
    """
    print("=" * 80)
    print("FLEET-LEVEL THROUGHPUT BENCHMARK")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Requests: {num_requests}")
    print(f"  Tokens/request: {tokens_per_request}")
    print(f"  Shared prefix ratio: {shared_prefix_ratio:.1%}")

    # Initialize optimized model
    print(f"\nInitializing OptimizedLLM with trillion-token features...")
    model = OptimizedLLM(
        model=model_name,
        optimization_level="maximum",  # Enables all features
        max_kv_blocks=4096,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Create requests with shared prefixes
    shared_prefix = "You are a helpful AI assistant. Please answer the following question: "
    unique_suffix = [
        "What is artificial intelligence?",
        "Explain machine learning.",
        "What is deep learning?",
        "How do neural networks work?",
        "What is natural language processing?",
    ]

    requests = []
    num_shared = int(num_requests * shared_prefix_ratio)

    for i in range(num_requests):
        if i < num_shared:
            # Use shared prefix
            prompt = shared_prefix + unique_suffix[i % len(unique_suffix)]
        else:
            # Unique prompt
            prompt = f"Question {i}: " + unique_suffix[i % len(unique_suffix)]

        requests.append(prompt)

    print(f"\n  Created {num_requests} requests:")
    print(f"    - {num_shared} with shared prefix")
    print(f"    - {num_requests - num_shared} unique")

    # Benchmark
    print(f"\n🚀 Starting benchmark...")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated() / 1024**3

    start_time = time.time()
    total_tokens_generated = 0

    for i, prompt in enumerate(requests):
        if i % 10 == 0:
            print(f"  Processing request {i}/{num_requests}...", end='\r')

        try:
            response = model.generate(
                prompt,
                max_tokens=tokens_per_request,
                temperature=0.7
            )
            total_tokens_generated += tokens_per_request
        except Exception as e:
            print(f"\n  ⚠️  Request {i} failed: {e}")
            continue

    elapsed = time.time() - start_time

    # Calculate metrics
    tokens_per_sec = total_tokens_generated / elapsed
    tokens_per_day = tokens_per_sec * 86400

    # Get stats
    if hasattr(model, 'kv_cache') and model.kv_cache:
        kv_stats = model.kv_cache.get_stats()
    else:
        kv_stats = {}

    if hasattr(model, 'speculative_decoder') and model.speculative_decoder:
        spec_stats = model.speculative_decoder.get_stats()
    else:
        spec_stats = {}

    # Results
    print("\n" + "=" * 80)
    print("RESULTS - Fleet-Level Throughput")
    print("=" * 80)

    print(f"\n📊 Performance Metrics:")
    print(f"  Total requests: {num_requests}")
    print(f"  Total tokens: {total_tokens_generated:,}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Throughput: {tokens_per_sec:.1f} tokens/sec")
    print(f"  Daily capacity: {tokens_per_day:,.0f} tokens/day")

    if torch.cuda.is_available():
        end_memory = torch.cuda.memory_allocated() / 1024**3
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3

        print(f"\n💾 Memory Usage:")
        print(f"  Start: {start_memory:.2f} GB")
        print(f"  End: {end_memory:.2f} GB")
        print(f"  Peak: {peak_memory:.2f} GB")
        print(f"  Growth: {end_memory - start_memory:.2f} GB")

    # Trillion-token features
    print(f"\n🔧 Trillion-Token Feature Stats:")

    if 'sliding_window' in kv_stats:
        sw = kv_stats['sliding_window']
        print(f"  Sliding Window:")
        print(f"    - Window size: {sw['current_window_size']} tokens")
        print(f"    - Total evictions: {sw['total_evictions']:,}")
        print(f"    - Windows slid: {sw['total_windows_slid']}")

    if 'prefix_dedup' in kv_stats:
        pd = kv_stats['prefix_dedup']
        print(f"  Prefix Deduplication:")
        print(f"    - Hit rate: {pd['hit_rate']:.1%}")
        print(f"    - Tokens saved: {pd['total_tokens_saved']:,}")
        print(f"    - Cache size: {pd['current_cache_size']}")

    if 'memory_stats' in spec_stats:
        mem = spec_stats['memory_stats']
        print(f"  Memory Pressure:")
        print(f"    - Current: {mem['pressure']:.1%}")
        print(f"    - Free: {mem['free_gb']:.2f} GB")

    if 'adaptive_controller' in spec_stats:
        ac = spec_stats['adaptive_controller']
        print(f"  Adaptive Speculation:")
        print(f"    - Current K: {ac['current_k']}")
        print(f"    - Enabled: {ac['enabled']}")
        print(f"    - Acceptance rate: {ac['recent_acceptance_rate']:.1%}")

    # Estimate trillion-token timeline
    print(f"\n⏱️  Trillion-Token Timeline (single GPU):")

    trillion = 1_000_000_000_000
    days_for_trillion = trillion / tokens_per_day

    print(f"  1 trillion tokens: {days_for_trillion:,.0f} days ({days_for_trillion/365:.1f} years)")

    # Multi-GPU projection
    for num_gpus in [10, 100, 1000]:
        multi_days = days_for_trillion / num_gpus
        print(f"  1 trillion tokens ({num_gpus} GPUs): {multi_days:,.0f} days ({multi_days/365:.1f} years)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-7B")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--tokens-per-request", type=int, default=500)
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.7)

    args = parser.parse_args()

    benchmark_fleet_throughput(
        model_name=args.model,
        num_requests=args.num_requests,
        tokens_per_request=args.tokens_per_request,
        shared_prefix_ratio=args.shared_prefix_ratio
    )
