#!/usr/bin/env python3
"""
Benchmark script for MemOpt

Compares baseline vs optimized inference and provides customer-ready metrics.

Usage:
    python benchmark.py --model gpt2-large --mode both
    python benchmark.py --model meta-llama/Llama-2-7b-hf --mode both
    this file is not working, fucking files 
"""

import argparse
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import json

from memopt import OptimizedLLM, ProfileStats
from memopt.monitoring.profiler import compare_profiles


# GPU bandwidth lookup (measured values from spec sheets)
GPU_BANDWIDTHS = {
    "A100": 2.0e12,      # 2.0 TB/s (80GB)
    "H100": 3.35e12,     # 3.35 TB/s
    "V100": 900e9,       # 900 GB/s
    "A10": 600e9,        # 600 GB/s
    "RTX 3090": 936e9,   # 936 GB/s
    "RTX 4090": 1008e9,  # 1008 GB/s
    "RTX 3080": 760e9,   # 760 GB/s
    "RTX 4080": 717e9,   # 717 GB/s
    "RTX 3070": 448e9,   # 448 GB/s
    "RTX 4070": 504e9,   # 504 GB/s
    "T4": 320e9,         # 320 GB/s
}

# GPU hourly costs (AWS on-demand, as of 2024)
GPU_HOURLY_COSTS = {
    "A100": 5.0,    # $4.10-6.00 (using middle estimate)
    "H100": 8.0,    # $6.50-10.00
    "V100": 3.0,    # $2.48-4.00
    "A10": 1.5,     # $1.32-2.00
    "T4": 0.6,      # $0.526-0.90
    "RTX 3090": 1.2,  # Estimated (not AWS, local datacenter)
    "RTX 4090": 1.5,
    "RTX 3080": 0.8,
    "RTX 4080": 1.0,
    "RTX 3070": 0.5,
    "RTX 4070": 0.6,
}


def get_gpu_specs():
    """Get current GPU bandwidth and cost"""
    if not torch.cuda.is_available():
        return 1e12, 1.0  # Default fallback
    
    gpu_name = torch.cuda.get_device_name(0)
    
    # Find bandwidth
    bandwidth = 1e12  # Default 1 TB/s if unknown
    for gpu_key, bw in GPU_BANDWIDTHS.items():
        if gpu_key in gpu_name:
            bandwidth = bw
            break
    
    # Find cost
    cost = 1.0  # Default $1/hour if unknown
    for gpu_key, c in GPU_HOURLY_COSTS.items():
        if gpu_key in gpu_key:
            cost = c
            break
    
    return bandwidth, cost


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
    
    # Calculate actual model size for bandwidth estimation
    model_params = sum(p.numel() for p in model.parameters())
    model_size_gb = (model_params * 2) / (1024**3)  # FP16 = 2 bytes per param
    
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
        
        # Calculate bandwidth utilization based on ACTUAL model size
        bytes_per_token = model_size_gb * 1e9  # Approximate
        achieved_bandwidth = stats.tokens_per_second * bytes_per_token
        
        # Get actual GPU bandwidth
        theoretical_bandwidth, gpu_cost = get_gpu_specs()
        
        stats.memory_bandwidth_utilization_pct = min(
            (achieved_bandwidth / theoretical_bandwidth) * 100, 100.0
        )
        
        # GPU stall = time spent waiting on memory (rough estimate)
        # This is conservative - actual stalls may be higher
        stats.gpu_stall_pct = max(0.0, 100.0 - stats.memory_bandwidth_utilization_pct)
        stats.gpu_utilization_pct = 100.0 - stats.gpu_stall_pct
        
        # Store GPU info
        stats.gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_cost = 1.0
    
    # Cost estimate based on actual GPU
    stats.estimated_gpu_hours = total_time / 3600.0
    cost_for_run = stats.estimated_gpu_hours * gpu_cost
    stats.cost_per_1m_tokens_usd = (cost_for_run / total_tokens) * 1e6 if total_tokens > 0 else 0
    
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
    optimization_level: str = "high"
):
    """
    Run optimized inference with MemOpt.
    
    Returns:
        ProfileStats
    """
    print("\n" + "="*70)
    print(f"RUNNING OPTIMIZED (MemOpt - {optimization_level})")
    print("="*70)
    
    # Load with MemOpt
    model = OptimizedLLM(
        model=model_name,
        optimization_level=optimization_level,
        enable_profiling=True
    )
    
    # Run inference
    for i, prompt in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}...")
        
        _ = model.generate(
            prompt,
            max_tokens=max_tokens,
            do_sample=False
        )
        
        # Reset cache between unrelated prompts
        model.reset_kv_cache()
    
    # Get stats
    stats = model.get_profiling_stats()
    
    print(f"\n✓ Optimized complete: {stats.tokens_per_second:.1f} tok/s")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark MemOpt")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2-large",
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
        "--optimization-level",
        type=str,
        choices=["conservative", "balanced", "high", "aggressive"],
        default="high",
        help="Optimization level"
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
        "--tokens-per-day",
        type=float,
        default=10e9,
        help="Tokens per day for ROI calculation (default: 10 billion)"
    )
    parser.add_argument(
        "--revenue-share",
        type=float,
        default=0.35,
        help="Revenue share percentage (default: 0.35 = 35%%)"
    )
    
    args = parser.parse_args()
    
    # Test prompts
    prompts = [
        "Explain how neural networks work in simple terms.",
        "Write a Python function to compute the Fibonacci sequence.",
        "What are the key differences between RAM and storage?",
        "Describe the process of photosynthesis step by step.",
        "How does TCP/IP networking function at a high level?",
        "Explain the concept of recursion with an example.",
        "What makes quantum computing different from classical computing?",
        "Write a short story about a robot learning to paint.",
    ][:args.num_prompts]
    
    print("\n" + "="*70)
    print("MEMOPT BENCHMARK")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {args.max_tokens}")
    print(f"Optimization level: {args.optimization_level}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    else:
        print("\n⚠️  WARNING: CUDA not available, running on CPU (will be slow)")
    
    # Run benchmarks
    baseline_stats = None
    optimized_stats = None
    
    if args.mode in ["baseline", "both"]:
        baseline_stats = run_baseline(args.model, prompts, args.max_tokens)
    
    if args.mode in ["optimized", "both"]:
        optimized_stats = run_optimized(
            args.model, prompts, args.max_tokens, args.optimization_level
        )
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    if baseline_stats:
        print("\n📊 BASELINE:")
        print(f"  Throughput:         {baseline_stats.tokens_per_second:.1f} tok/s")
        print(f"  Latency:            {baseline_stats.latency_per_token_ms:.2f} ms/tok")
        print(f"  Memory:             {baseline_stats.peak_memory_allocated_gb:.2f} GB")
        if baseline_stats.gpu_stall_pct > 0:
            print(f"  GPU stall:          {baseline_stats.gpu_stall_pct:.1f}% (estimated)")
        print(f"  Cost per 1M tokens: ${baseline_stats.cost_per_1m_tokens_usd:.2f}")
    
    if optimized_stats:
        print("\n🚀 OPTIMIZED:")
        print(f"  Throughput:         {optimized_stats.tokens_per_second:.1f} tok/s")
        print(f"  Latency:            {optimized_stats.latency_per_token_ms:.2f} ms/tok")
        print(f"  Memory:             {optimized_stats.peak_memory_allocated_gb:.2f} GB")
        if optimized_stats.gpu_stall_pct > 0:
            print(f"  GPU stall:          {optimized_stats.gpu_stall_pct:.1f}% (estimated)")
        print(f"  Cost per 1M tokens: ${optimized_stats.cost_per_1m_tokens_usd:.2f}")
    
    if baseline_stats and optimized_stats:
        print("\n💰 IMPROVEMENT:")
        speedup = optimized_stats.tokens_per_second / baseline_stats.tokens_per_second
        
        memory_diff = optimized_stats.peak_memory_allocated_gb - baseline_stats.peak_memory_allocated_gb
        if memory_diff < 0:
            memory_reduction = abs(memory_diff) / baseline_stats.peak_memory_allocated_gb * 100
            print(f"  Speedup:            {speedup:.2f}x")
            print(f"  Memory reduction:   {memory_reduction:.1f}%")
        else:
            memory_increase = memory_diff / baseline_stats.peak_memory_allocated_gb * 100
            print(f"  Speedup:            {speedup:.2f}x")
            print(f"  Memory overhead:    +{memory_increase:.1f}%")
            print(f"    (One-time allocation for optimization structures)")
        
        cost_reduction = (
            (baseline_stats.cost_per_1m_tokens_usd - optimized_stats.cost_per_1m_tokens_usd) /
            baseline_stats.cost_per_1m_tokens_usd * 100
        )
        print(f"  Cost reduction:     {cost_reduction:.1f}%")
        
        if baseline_stats.gpu_stall_pct > 0 and optimized_stats.gpu_stall_pct >= 0:
            stall_reduction = baseline_stats.gpu_stall_pct - optimized_stats.gpu_stall_pct
            if stall_reduction > 0:
                print(f"  Stall reduction:    {stall_reduction:.1f}%")
        
        # Calculate ROI for customer's actual scenario
        tokens_per_day = args.tokens_per_day
        daily_baseline_cost = (tokens_per_day / 1e6) * baseline_stats.cost_per_1m_tokens_usd
        daily_optimized_cost = (tokens_per_day / 1e6) * optimized_stats.cost_per_1m_tokens_usd
        daily_savings = daily_baseline_cost - daily_optimized_cost
        annual_savings = daily_savings * 365
        
        # Revenue share pricing model
        revenue_share = args.revenue_share
        memopt_annual_cost = annual_savings * revenue_share
        customer_keeps = annual_savings * (1 - revenue_share)
        
        print(f"\n💵 ROI ANALYSIS ({tokens_per_day/1e9:.1f}B tokens/day):")
        print(f"  Daily baseline cost:      ${daily_baseline_cost:,.0f}")
        print(f"  Daily optimized cost:     ${daily_optimized_cost:,.0f}")
        print(f"  Daily savings:            ${daily_savings:,.0f}")
        print(f"  Annual savings:           ${annual_savings:,.0f}")
        print(f"\n  MemOpt cost ({revenue_share*100:.0f}% of savings): ${memopt_annual_cost:,.0f}/year")
        print(f"  Customer keeps ({(1-revenue_share)*100:.0f}%):      ${customer_keeps:,.0f}/year")
        print(f"\n  Customer ROI:             {(customer_keeps / memopt_annual_cost):.1f}x")
        print(f"  Payback period:           {(memopt_annual_cost / daily_savings):.1f} days")
        
        # Alternative: Fixed pricing comparison
        fixed_price = 50000
        if annual_savings > fixed_price:
            print(f"\n  Alternative fixed pricing: ${fixed_price:,}/year")
            print(f"  Fixed pricing ROI:         {(annual_savings / fixed_price):.1f}x")
            print(f"  Fixed payback:             {(fixed_price / daily_savings):.1f} days")
    
    # Save results
    results = {
        "model": args.model,
        "num_prompts": len(prompts),
        "max_tokens": args.max_tokens,
        "optimization_level": args.optimization_level,
    }
    
    if torch.cuda.is_available():
        results["gpu"] = torch.cuda.get_device_name(0)
    
    if baseline_stats:
        results["baseline"] = baseline_stats.to_dict()
    if optimized_stats:
        results["optimized"] = optimized_stats.to_dict()
    
    if baseline_stats and optimized_stats:
        results["comparison"] = {
            "speedup": float(speedup),
            "memory_change_gb": float(memory_diff),
            "cost_reduction_pct": float(cost_reduction),
        }
        results["roi_analysis"] = {
            "tokens_per_day": float(tokens_per_day),
            "daily_savings_usd": float(daily_savings),
            "annual_savings_usd": float(annual_savings),
            "revenue_share_pct": float(revenue_share * 100),
            "memopt_annual_cost_usd": float(memopt_annual_cost),
            "customer_keeps_usd": float(customer_keeps),
            "customer_roi": float(customer_keeps / memopt_annual_cost),
            "payback_days": float(memopt_annual_cost / daily_savings)
        }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()