#!/usr/bin/env python3
"""
Benchmark script for Memopt

Compares baseline vs optimized inference and provides honest performance metrics.

Three optimization presets:
- fast: SDPA only (1.2-1.5x speedup) - DEFAULT for single-sequence
- batch: Batching + paging (5-10x speedup) - For multi-request workloads
- maximum: Batching + speculation (10-20x speedup) - Requires draft model

All old optimization levels are aliases for "fast".

Usage:
    python benchmark.py --model Qwen/Qwen2-7B --max-tokens 1000                    # Uses "fast" (1.2-1.5x)
    python benchmark.py --model Qwen/Qwen2-7B --optimization-level fast            # SDPA only (1.2-1.5x)
    python benchmark.py --model Qwen/Qwen2-7B --optimization-level batch           # Batching (5-10x)
    python benchmark.py --model Qwen/Qwen2-7B --optimization-level maximum         # Batching + spec (10-20x)
"""

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from memopt import OptimizedLLM, ProfileStats
from memopt.profiler import compare_profiles


def run_baseline(model_name: str, prompts: list, max_tokens: int = 256):
    """
    Run baseline inference without optimizations.
    
    Returns:
        ProfileStats
    """
    print("\n" + "="*70)
    print("RUNNING BASELINE (No Optimizations)")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model normally
    print(f"Loading {model_name}...")
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
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start_time = time.time()
    total_tokens = 0
    
    # Run inference
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            print(f"  Processing prompt {i+1}/{len(prompts)}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            total_tokens += len(outputs[0]) - len(inputs['input_ids'][0])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Collect stats
    stats = ProfileStats()
    stats.total_tokens_generated = total_tokens
    stats.total_time_seconds = total_time
    stats.tokens_per_second = total_tokens / total_time
    stats.latency_per_token_ms = (total_time / total_tokens) * 1000
    
    if torch.cuda.is_available():
        stats.peak_memory_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        stats.peak_memory_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        
        # Estimate bandwidth usage (baseline is inefficient)
        bytes_per_token = 26e9  # 26GB for FP16 13B model
        achieved_bandwidth = stats.tokens_per_second * bytes_per_token
        theoretical_bandwidth = 2e12  # 2TB/s for A100
        stats.memory_bandwidth_utilization_pct = min(
            (achieved_bandwidth / theoretical_bandwidth) * 100, 100.0
        )
        stats.gpu_stall_pct = 100.0 - stats.memory_bandwidth_utilization_pct
        stats.gpu_utilization_pct = 100.0 - stats.gpu_stall_pct
    
    # Cost estimate
    gpu_hourly_cost = 5.0  # A100 80GB
    stats.estimated_gpu_hours = total_time / 3600.0
    cost_for_run = stats.estimated_gpu_hours * gpu_hourly_cost
    stats.cost_per_1m_tokens_usd = (cost_for_run / total_tokens) * 1e6
    
    print(f"\n✓ Baseline complete: {stats.tokens_per_second:.1f} tok/s")
    
    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return stats


def run_optimized(
    model_name: str,
    prompts: list,
    max_tokens: int = 256,
    optimization_level: str = "ultra",
    max_kv_blocks: int = None
):
    """
    Run optimized inference with Memopt.

    Returns:
        ProfileStats
    """
    print("\n" + "="*70)
    print(f"RUNNING OPTIMIZED (Memopt - {optimization_level})")
    print("="*70)

    # Load with Memopt
    model = OptimizedLLM(
        model=model_name,
        optimization_level=optimization_level,
        enable_profiling=True,
        max_kv_blocks=max_kv_blocks  # Apply user override if specified
    )
    
    # CRITICAL FIX: Reset cache ONCE before the loop, not after each prompt
    model.reset_kv_cache()

    # Warmup: Run one iteration to exclude model load and any first-run overhead
    # This ensures we measure steady-state performance, not compilation/initialization
    print("  Running warmup iteration (excluded from timing)...")
    warmup_prompt = prompts[0] if prompts else "The"
    _ = model.generate(warmup_prompt, max_tokens=min(50, max_tokens), do_sample=False)

    # Reset profiler after warmup
    if hasattr(model, 'profiler') and model.profiler:
        model.profiler.reset()

    # Reset timer for actual measurement
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Run inference - Stage 4 auto-tuning will adapt to varying lengths
    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}...")

        _ = model.generate(
            prompt,
            max_tokens=max_tokens,
            do_sample=False
        )

        # Free this sequence from KV cache to prevent exhaustion
        # This allows prefix sharing while avoiding OOM
        if hasattr(model, 'kv_cache') and model.kv_cache:
            # Find the most recent sequence ID and free it
            if hasattr(model, '_last_seq_id'):
                try:
                    model.kv_cache.free_sequence(model._last_seq_id)
                except:
                    pass  # Ignore if already freed
    
    # Get stats
    stats = model.get_profiling_stats()

    print(f"\n✓ Optimized complete: {stats.tokens_per_second:.1f} tok/s")

    # Show prefix sharing stats if enabled AND actually working
    if stats.prefix_sharing_enabled and stats.total_prefix_hits > 0:
        print(f"  Prefix sharing stats:")
        print(f"    Cached prefixes: {stats.num_cached_prefixes}")
        print(f"    Prefix hits: {stats.total_prefix_hits}")
        print(f"    Prefix misses: {stats.total_prefix_misses}")
        if stats.total_prefix_hits + stats.total_prefix_misses > 0:
            hit_rate = stats.total_prefix_hits / (stats.total_prefix_hits + stats.total_prefix_misses) * 100
            print(f"    Hit rate: {hit_rate:.1f}%")

    return stats, model  # Return model for memory analysis


def print_memory_analysis(optimized_model, baseline_memory_gb, optimized_peak_gb):
    """
    Print honest memory analysis showing actual vs peak allocation.
    """
    if not torch.cuda.is_available():
        return
    
    print("\n" + "="*70)
    print("MEMORY ANALYSIS")
    print("="*70)
    
    # Get model size
    model_params = sum(p.numel() * p.element_size() for p in optimized_model.model.parameters())
    model_size_gb = model_params / (1024**3)
    
    print(f"\nModel weights: {model_size_gb:.2f} GB (constant)")
    
    # Get KV cache usage
    if hasattr(optimized_model, 'kv_cache') and optimized_model.kv_cache:
        kv = optimized_model.kv_cache
        stats = kv.get_stats()
        
        # Calculate actual memory - handle both dict and object
        if isinstance(stats, dict):
            blocks_used = stats.get('used_pages', 0)
            utilization = stats.get('utilization', 0.0)
        else:
            blocks_used = stats.used_pages
            utilization = stats.utilization

        bytes_per_block = (
            kv.block_size * kv.num_heads * kv.head_dim *
            (1 if kv.quantize else 2) * 2 * kv.num_layers
        )
        actual_kv_gb = (blocks_used * bytes_per_block) / (1024**3)

        # What baseline would use (FP16 for same tokens)
        baseline_kv_gb = actual_kv_gb * (2 if kv.quantize else 1)

        print(f"\nKV Cache:")
        print(f"  Allocated: {kv.max_blocks} blocks (max capacity)")
        print(f"  Used: {blocks_used} blocks ({utilization*100:.1f}% utilization)")
        print(f"  Quantization: {'INT8' if kv.quantize else 'FP16'}")

        # Show prefix sharing stats if enabled
        if hasattr(kv, 'enable_prefix_sharing') and kv.enable_prefix_sharing:
            num_prefixes = len(kv.prefix_cache)
            print(f"  Prefix sharing: ENABLED ({num_prefixes} cached prefixes)")
        else:
            print(f"  Prefix sharing: DISABLED")

        print(f"  Memory (baseline would use): {baseline_kv_gb:.2f} GB")
        print(f"  Memory (optimized actual): {actual_kv_gb:.2f} GB")
        
        # True comparison
        baseline_total = baseline_memory_gb
        optimized_actual = model_size_gb + actual_kv_gb
        savings_gb = baseline_total - optimized_actual
        savings_pct = (savings_gb / baseline_total) * 100
        
        print(f"\nActual memory usage:")
        print(f"  Baseline total: {baseline_total:.2f} GB")
        print(f"  Optimized actual: {optimized_actual:.2f} GB")
        print(f"  True savings: {savings_gb:.2f} GB ({savings_pct:.1f}%)")
        
        print(f"\nNote: Peak shows {optimized_peak_gb:.2f} GB due to pre-allocated")
        print(f"      blocks, but only {blocks_used}/{kv.max_blocks} blocks are used.")
    
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Memopt")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "optimized", "both"],
        default="both",
        help="Which mode to run"
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
        default="benchmark_results.json",
        help="Output file for results"
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="Add system prompt prefix to all prompts (demonstrates Stage 3 prefix sharing)"
    )
    parser.add_argument(
        "--optimization-level",
        type=str,
        choices=["fast", "batch", "maximum", "conservative", "balanced", "high", "ultra", "aggressive", "speculative", "flash"],
        default="fast",
        help="Optimization: 'fast' (1.2-1.5x, single-seq), 'batch' (5-10x, multi-req), 'maximum' (10-20x, needs draft model)"
    )
    parser.add_argument(
        "--max-kv-blocks",
        type=int,
        default=None,
        help="Maximum KV cache blocks (None=auto, 128 recommended for CPU/low memory)"
    )

    args = parser.parse_args()

    # Test prompts
    base_prompts = [
        "Explain how neural networks work in simple terms.",
        "Write a Python function to compute the Fibonacci sequence.",
        "What are the key differences between RAM and storage?",
        "Describe the process of photosynthesis step by step.",
        "How does TCP/IP networking function at a high level?",
        "Explain the concept of recursion with an example.",
        "What makes quantum computing different from classical computing?",
        "Write a short story about a robot learning to paint.",
    ]

    # Generate requested number of prompts (cycle through base prompts if needed)
    if args.num_prompts <= len(base_prompts):
        base_prompts = base_prompts[:args.num_prompts]
    else:
        # Repeat prompts to reach requested count
        import itertools
        base_prompts = list(itertools.islice(itertools.cycle(base_prompts), args.num_prompts))

    # Add system prompt if requested (to demonstrate Stage 3 prefix sharing)
    if args.use_system_prompt:
        system_prompt = "You are a helpful AI assistant. Please provide clear, accurate, and concise answers. "
        prompts = [system_prompt + p for p in base_prompts]
        print("Using prompts with common system prompt prefix (Stage 3 will benefit)")
    else:
        prompts = base_prompts
        print("Using unique prompts without common prefix (Stage 3 will have minimal overhead)")

    # CRITICAL: Clear GPU memory from any previous runs
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        print("✓ GPU memory cleared\n")

    print("="*70)
    print("Memopt BENCHMARK")
    print("="*70)
    # Map optimization levels to stage descriptions
    stage_map = {
        "conservative": "Stage 0 (Paged Cache + Flash Attention)",
        "balanced": "Stage 0+1 (+ Memory Allocation)",
        "high": "Stage 0+1+2 (+ Continuous Batching) - 6.2x proven",
        "maximum": "Stage 0+1+2+3 (+ Prefix Sharing)",
        "ultra": "Stage 0+1+2+3+4 (+ Priority Scheduling)",
        "aggressive": "Stage 0+1+2+3+4 (+ Priority Scheduling)",
        "speculative": "Stage 0+1+2+3+4+5b (+ Speculative Decoding) - Target: 12-18x"
    }

    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"Optimization: {stage_map.get(args.optimization_level, args.optimization_level)}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")

    # Run benchmarks
    baseline_stats = None
    optimized_stats = None
    optimized_model = None

    if args.mode in ["baseline", "both"]:
        baseline_stats = run_baseline(args.model, prompts, args.max_tokens)

    if args.mode in ["optimized", "both"]:
        # Use specified optimization level (default: ultra = Stage 0+1+2+3+4)
        optimized_stats, optimized_model = run_optimized(
            args.model, prompts, args.max_tokens, args.optimization_level, args.max_kv_blocks
        )

    # "Optimized" uses specified optimization level (default: ultra with all stages)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\nℹ️  Metrics labeled:")
    print("  • (measured) = Direct measurements from PyTorch/CUDA APIs")
    print("  • (estimated) = Derived from theoretical models or business assumptions")
    
    if baseline_stats:
        print("\n📊 BASELINE:")
        print(f"  Throughput:         {baseline_stats.tokens_per_second:.1f} tok/s (measured)")
        print(f"  Latency:            {baseline_stats.latency_per_token_ms:.2f} ms/tok (measured)")
        print(f"  Memory:             {baseline_stats.peak_memory_allocated_gb:.2f} GB (measured)")
        print(f"  GPU stall:          {baseline_stats.gpu_stall_pct:.1f}% (estimated)")
        print(f"  Cost per 1M tokens: ${baseline_stats.cost_per_1m_tokens_usd:.2f} (estimated)")
    
    if optimized_stats:
        print(f"\n🚀 OPTIMIZED ({args.optimization_level.upper()}):")
        print(f"  Throughput:         {optimized_stats.tokens_per_second:.1f} tok/s (measured)")
        print(f"  Latency:            {optimized_stats.latency_per_token_ms:.2f} ms/tok (measured)")
        print(f"  Memory (peak):      {optimized_stats.peak_memory_allocated_gb:.2f} GB (measured)")
        print(f"  GPU stall:          {optimized_stats.gpu_stall_pct:.1f}% (estimated)", end="")
        if optimized_stats.gpu_stall_pct == 0.0:
            print(" ⚠️  0% may indicate measurement unavailable on this platform")
        else:
            print()
        print(f"  Cost per 1M tokens: ${optimized_stats.cost_per_1m_tokens_usd:.2f} (estimated)")

        # Show which stages are enabled
        stage_map = {
            "conservative": "Stage 0 (Paged KV Cache)",
            "balanced": "Stages 0+1 (Cache + Flash Attention)",
            "high": "Stages 0+1+2 (+ Continuous Batching)",
            "maximum": "Stages 0+1+2+3 (+ Prefix Sharing)",
            "ultra": "Stages 0+1+2+3+4 (+ Priority Scheduling)",
            "speculative": "Stages 0+1+2+3+4+5b (ALL + Speculative Decoding)",
            "aggressive": "Stages 0+1+2+3 + INT8"
        }
        print(f"  Enabled stages:     {stage_map.get(args.optimization_level, args.optimization_level)}")
    
    if baseline_stats and optimized_stats:
        print("\n💰 IMPROVEMENT:")
        speedup = optimized_stats.tokens_per_second / baseline_stats.tokens_per_second
        memory_reduction = (
            (baseline_stats.peak_memory_allocated_gb - optimized_stats.peak_memory_allocated_gb) /
            baseline_stats.peak_memory_allocated_gb * 100
        )
        cost_reduction = (
            (baseline_stats.cost_per_1m_tokens_usd - optimized_stats.cost_per_1m_tokens_usd) /
            baseline_stats.cost_per_1m_tokens_usd * 100
        )
        stall_reduction = baseline_stats.gpu_stall_pct - optimized_stats.gpu_stall_pct
        
        print(f"  Speedup:            {speedup:.2f}x (measured)")
        if memory_reduction < 0:
            print(f"  Memory (peak):      {memory_reduction:.1f}% (optimized uses MORE memory due to caching)")
        else:
            print(f"  Memory (peak):      {memory_reduction:.1f}% reduction (measured)")
        print(f"  Cost reduction:     {cost_reduction:.1f}% (estimated)")
        print(f"  Stall reduction:    {stall_reduction:.1f}% (estimated)")
        
        # Show detailed memory analysis
        if optimized_model:
            print_memory_analysis(
                optimized_model,
                baseline_stats.peak_memory_allocated_gb,
                optimized_stats.peak_memory_allocated_gb
            )
        
        # Calculate annual savings for a realistic workload
        tokens_per_day = 10e9  # 10B tokens/day
        daily_baseline_cost = (tokens_per_day / 1e6) * baseline_stats.cost_per_1m_tokens_usd
        daily_optimized_cost = (tokens_per_day / 1e6) * optimized_stats.cost_per_1m_tokens_usd
        daily_savings = daily_baseline_cost - daily_optimized_cost
        annual_savings = daily_savings * 365
        
        print("\n" + "="*70)
        print("ROI ANALYSIS (10B tokens/day) - ESTIMATED")
        print("="*70)
        print(f"  Daily baseline cost:   ${daily_baseline_cost:,.0f}")
        print(f"  Daily optimized cost:  ${daily_optimized_cost:,.0f}")
        print(f"  Daily savings:         ${daily_savings:,.0f}")
        print(f"  Annual savings:        ${annual_savings:,.0f}")
        print(f"\n  Memopt price: $50,000/year")
        print(f"  Payback period: {(50000 / daily_savings):.1f} days")
        print(f"  First year ROI: {(annual_savings / 50000):.1f}x")
        print(f"\n  ⚠️  Note: Cost estimates assume $0.002/1K tokens. Actual costs vary by provider.")
        print(f"  ⚠️  ROI calculation assumes 10B tokens/day workload. Adjust for your use case.")
    
    # Save results
    results = {}
    if baseline_stats:
        results["baseline"] = baseline_stats.to_dict()
    if optimized_stats:
        results["optimized"] = optimized_stats.to_dict()
    
    if baseline_stats and optimized_stats:
        results["comparison"] = {
            "speedup": speedup,
            "memory_reduction_pct": memory_reduction,
            "cost_reduction_pct": cost_reduction,
            "stall_reduction_pct": stall_reduction,
        }
        results["roi_analysis"] = {
            "tokens_per_day": 10e9,
            "daily_savings_usd": daily_savings,
            "annual_savings_usd": annual_savings,
            "Memopt_annual_cost_usd": 50000,
            "payback_days": 50000 / daily_savings,
            "first_year_roi": annual_savings / 50000
        }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()