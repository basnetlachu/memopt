#!/usr/bin/env python3
"""
Simple example of using MemOpt

This is what you'd show to a customer during a demo.
"""

from memopt import OptimizedLLM

def basic_example():
    """Most basic usage - just 3 lines."""
    print("\n" + "="*70)
    print("BASIC EXAMPLE - 3 Lines of Code")
    print("="*70 + "\n")
    
    # That's it - drop-in replacement for standard inference
    model = OptimizedLLM(
        model="meta-llama/Llama-2-7b-hf",
        optimization_level="high",
        enable_profiling=True
    )
    
    response = model.generate(
        "Explain how transformers work in machine learning",
        max_tokens=256
    )
    
    print("Response:", response[:200] + "...")
    
    # Show the metrics that matter
    model.print_profiling_stats()


def optimization_levels_example():
    """Show different optimization levels."""
    print("\n" + "="*70)
    print("OPTIMIZATION LEVELS")
    print("="*70 + "\n")
    
    levels = ["conservative", "balanced", "high", "aggressive"]
    
    prompt = "What is machine learning?"
    
    for level in levels:
        print(f"\n--- {level.upper()} ---")
        
        model = OptimizedLLM(
            model="meta-llama/Llama-2-7b-hf",
            optimization_level=level,
            enable_profiling=True
        )
        
        response = model.generate(prompt, max_tokens=50)
        stats = model.get_profiling_stats()
        
        print(f"Throughput: {stats.tokens_per_second:.1f} tok/s")
        print(f"Memory: {stats.peak_memory_allocated_gb:.2f} GB")
        print(f"Cost per 1M tokens: ${stats.cost_per_1m_tokens_usd:.2f}")


def batch_inference_example():
    """Show how to use with multiple prompts."""
    print("\n" + "="*70)
    print("BATCH INFERENCE")
    print("="*70 + "\n")
    
    model = OptimizedLLM(
        model="meta-llama/Llama-2-7b-hf",
        optimization_level="high",
        enable_profiling=True
    )
    
    prompts = [
        "What is Python?",
        "Explain recursion.",
        "What is TCP/IP?",
    ]
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt {i}: {prompt}")
        response = model.generate(prompt, max_tokens=100)
        print(f"Response: {response[:100]}...")
        
        # Important: reset cache between unrelated prompts
        model.reset_kv_cache()
    
    model.print_profiling_stats()


def cost_savings_demo():
    """
    Demo that shows clear ROI - this is what closes deals.
    """
    print("\n" + "="*70)
    print("COST SAVINGS DEMO")
    print("="*70 + "\n")
    
    print("Running baseline inference (no optimizations)...")
    # Simulate baseline - in real demo, you'd run actual baseline
    
    print("\nNow running with MemOpt optimizations...")
    model = OptimizedLLM(
        model="meta-llama/Llama-2-7b-hf",
        optimization_level="high",
        enable_profiling=True
    )
    
    test_prompt = "Explain the concept of memory bandwidth in GPUs and why it matters for AI inference."
    
    response = model.generate(test_prompt, max_tokens=300)
    
    print(f"\nGenerated {len(response.split())} words")
    
    stats = model.get_profiling_stats()
    
    print("\n" + "="*70)
    print("YOUR COST SAVINGS")
    print("="*70)
    
    # Baseline (typical values without optimization)
    baseline_cost_per_1m = 15.0
    baseline_throughput = 100.0
    baseline_gpu_stall = 75.0
    
    print(f"\nBASELINE (Without MemOpt):")
    print(f"  Throughput:         {baseline_throughput:.1f} tok/s")
    print(f"  GPU stall time:     {baseline_gpu_stall:.1f}%")
    print(f"  Cost per 1M tokens: ${baseline_cost_per_1m:.2f}")
    
    print(f"\nWITH MEMOPT:")
    print(f"  Throughput:         {stats.tokens_per_second:.1f} tok/s")
    print(f"  GPU stall time:     {stats.gpu_stall_pct:.1f}%")
    print(f"  Cost per 1M tokens: ${stats.cost_per_1m_tokens_usd:.2f}")
    
    speedup = stats.tokens_per_second / baseline_throughput
    cost_reduction = ((baseline_cost_per_1m - stats.cost_per_1m_tokens_usd) / 
                     baseline_cost_per_1m * 100)
    stall_reduction = baseline_gpu_stall - stats.gpu_stall_pct
    
    print(f"\nIMPROVEMENT:")
    print(f"  {speedup:.1f}x faster")
    print(f"  {cost_reduction:.1f}% cost reduction")
    print(f"  {stall_reduction:.1f}% less GPU idle time")
    
    # Calculate savings for customer's scale
    print(f"\n" + "="*70)
    print("YOUR ANNUAL SAVINGS")
    print("="*70)
    
    tokens_per_day = 10e9  # 10 billion tokens/day (adjust for customer)
    
    baseline_daily = (tokens_per_day / 1e6) * baseline_cost_per_1m
    optimized_daily = (tokens_per_day / 1e6) * stats.cost_per_1m_tokens_usd
    daily_savings = baseline_daily - optimized_daily
    
    print(f"\nAt your scale (10B tokens/day):")
    print(f"  Current daily cost:     ${baseline_daily:,.0f}")
    print(f"  With MemOpt:            ${optimized_daily:,.0f}")
    print(f"  Daily savings:          ${daily_savings:,.0f}")
    print(f"  Annual savings:         ${daily_savings * 365:,.0f}")
    
    print(f"\nMemOpt pricing: $50,000/year")
    print(f"Your ROI: {(daily_savings * 365 / 50000):.1f}x in year 1")
    print(f"Payback period: {(50000 / daily_savings):.0f} days")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    examples = {
        "basic": basic_example,
        "levels": optimization_levels_example,
        "batch": batch_inference_example,
        "demo": cost_savings_demo,
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    else:
        print("Usage: python example.py [basic|levels|batch|demo]")
        print("\nRunning cost savings demo (what you'd show customers)...")
        cost_savings_demo()
