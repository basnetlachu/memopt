#!/usr/bin/env python3
"""
Quick test for adaptive caching implementation.
Tests that cache switching works correctly at different sequence lengths.
"""

import torch
from memopt import OptimizedLLM

print("="*70)
print("ADAPTIVE CACHING TEST")
print("="*70)

# Initialize model with speculative optimization
print("\nInitializing GPT-2 XL with speculative optimization...")
model = OptimizedLLM(
    model="gpt2-xl",
    optimization_level="speculative",
    device="cpu"  # Use CPU to avoid GPU memory issues
)

print("\n" + "="*70)
print("TEST 1: Short sequence (100 tokens) - should use NO cache")
print("="*70)

response = model.generate(
    "The future of artificial intelligence is",
    max_tokens=100
)

# Get speculative decoder stats
if model.speculative_decoder:
    stats = model.speculative_decoder.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache threshold: {stats['cache_threshold']} tokens")
    print(f"  No-cache tokens: {stats['no_cache_tokens']}")
    print(f"  Cached tokens: {stats['cached_tokens']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")

    if stats['no_cache_tokens'] > stats['cached_tokens']:
        print("  ✅ Correctly used NO cache for short sequence")
    else:
        print("  ❌ ERROR: Should have used NO cache")

print("\n" + "="*70)
print("TEST 2: Long sequence (600 tokens) - should use cache")
print("="*70)

# Reset stats
model.speculative_decoder.reset_stats()
model.speculative_decoder.no_cache_tokens = 0
model.speculative_decoder.cached_tokens = 0

response = model.generate(
    "In a world where technology advances rapidly",
    max_tokens=600
)

if model.speculative_decoder:
    stats = model.speculative_decoder.get_stats()
    print(f"\nCache Statistics:")
    print(f"  Cache threshold: {stats['cache_threshold']} tokens")
    print(f"  No-cache tokens: {stats['no_cache_tokens']}")
    print(f"  Cached tokens: {stats['cached_tokens']}")
    print(f"  Acceptance rate: {stats['acceptance_rate']:.2%}")

    if stats['cached_tokens'] > stats['no_cache_tokens']:
        print("  ✅ Correctly used cache for long sequence")
    else:
        print("  ❌ ERROR: Should have used cache")

print("\n" + "="*70)
print("✅ ADAPTIVE CACHING TESTS COMPLETE")
print("="*70)
