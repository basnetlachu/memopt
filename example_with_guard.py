#!/usr/bin/env python3
"""
Example: Using MemOpt with Performance Guard

This example demonstrates production-safe usage with automatic
fallback when performance degrades.
"""

from memopt import OptimizedLLM
import time

print("="*70)
print("MEMOPT WITH PERFORMANCE GUARD - EXAMPLE")
print("="*70)

# Configuration
MODEL = "gpt2-xl"
PROMPT = "The future of artificial intelligence is"
MAX_TOKENS = 1000

# IMPORTANT: In production, measure this with your actual workload
# For this example, we'll use a reasonable estimate
BASELINE_THROUGHPUT = 31.0  # tok/s (measure with benchmark_performance.py)

print(f"\nConfiguration:")
print(f"  Model: {MODEL}")
print(f"  Prompt: '{PROMPT}'")
print(f"  Max tokens: {MAX_TOKENS}")
print(f"  Baseline throughput: {BASELINE_THROUGHPUT} tok/s")
print(f"  Safety threshold: {BASELINE_THROUGHPUT * 0.98:.1f} tok/s (98% of baseline)")

# Create model with guard enabled
print(f"\n{'='*70}")
print("LOADING MODEL WITH GUARD")
print("="*70)

model = OptimizedLLM(
    model=MODEL,
    optimization_level="speculative",
    enable_performance_guard=True,      # Enable safety guardrails
    baseline_throughput=BASELINE_THROUGHPUT,
    enable_profiling=False  # Disable profiling for accurate timing
)

print("\n✓ Model loaded with performance guard enabled")
print("  - Guard will monitor throughput and acceptance rate")
print("  - Auto-fallback if throughput drops below 98% of baseline")
print("  - Auto-fallback if acceptance rate drops below 85%")

# Generate text multiple times to see guard in action
print(f"\n{'='*70}")
print("GENERATING TEXT (GUARD MONITORING)")
print("="*70)

for i in range(5):
    print(f"\n--- Generation {i+1}/5 ---")

    start = time.time()
    response = model.generate(PROMPT, max_tokens=MAX_TOKENS, do_sample=False)
    elapsed = time.time() - start

    throughput = MAX_TOKENS / elapsed

    print(f"Time: {elapsed:.2f}s")
    print(f"Throughput: {throughput:.1f} tok/s")

    # Check guard status
    if model.performance_guard:
        stats = model.get_performance_guard_stats()

        print(f"\nGuard Status:")
        print(f"  Optimization level: {stats['optimization_level']}")
        print(f"  Current avg throughput: {stats['current_avg_throughput']:.1f} tok/s")
        print(f"  Throughput vs baseline: {stats['throughput_vs_baseline']:.2f}x")
        print(f"  Meets requirements: {'✅ YES' if stats['meets_requirements'] else '❌ NO'}")
        print(f"  Total measurements: {stats['total_measurements']}")
        print(f"  Total violations: {stats['total_violations']}")
        print(f"  Consecutive violations: {stats['consecutive_violations']}")
        print(f"  Total fallbacks: {stats['total_fallbacks']}")

        if stats['speculative_disabled']:
            print(f"  ⚠️  WARNING: Speculative decoding was disabled by guard!")

        if stats['cache_disabled']:
            print(f"  ⚠️  WARNING: KV cache was disabled by guard!")

    # Get speculative decoding stats
    if hasattr(model, 'speculative_decoder') and model.speculative_decoder:
        spec_stats = model.speculative_decoder.get_stats()
        if spec_stats['total_draft_tokens'] > 0:
            print(f"\nSpeculative Decoding:")
            print(f"  Acceptance rate: {spec_stats['acceptance_rate']:.1%}")
            print(f"  Theoretical speedup: {spec_stats['theoretical_speedup']:.2f}x")

# Final summary
print(f"\n{'='*70}")
print("FINAL SUMMARY")
print("="*70)

if model.performance_guard:
    final_stats = model.get_performance_guard_stats()

    print(f"\nPerformance Guard Results:")
    print(f"  Final optimization level: {final_stats['optimization_level']}")
    print(f"  Average throughput: {final_stats['current_avg_throughput']:.1f} tok/s")
    print(f"  Baseline throughput: {final_stats['baseline_throughput']:.1f} tok/s")
    print(f"  Speedup: {final_stats['throughput_vs_baseline']:.2f}x")
    print(f"  Total fallbacks: {final_stats['total_fallbacks']}")

    if final_stats['meets_requirements']:
        print(f"\n✅ SUCCESS: Performance requirements met!")
        print(f"   Throughput stayed above {BASELINE_THROUGHPUT * 0.98:.1f} tok/s threshold")
    else:
        print(f"\n⚠️  WARNING: Performance requirements not met")
        print(f"   Throughput dropped below {BASELINE_THROUGHPUT * 0.98:.1f} tok/s threshold")

    if final_stats['total_fallbacks'] == 0:
        print(f"\n🎉 EXCELLENT: No fallbacks needed - optimization worked great!")
    else:
        print(f"\n🛡️  SAFE: Guard triggered {final_stats['total_fallbacks']} fallback(s) to maintain performance")

print(f"\n{'='*70}")
print("For production deployment, use benchmark_performance.py to")
print("measure accurate baseline throughput with your workload.")
print("="*70)
