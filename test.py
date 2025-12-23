#!/usr/bin/env python3
"""
MemOpt Test Script
Verifies that the installation works correctly
"""

import sys
import torch

print("="*70)
print("MEMOPT INSTALLATION TEST")
print("="*70)

# Test 1: Import MemOpt
print("\n[1/5] Testing imports...")
try:
    from memopt import OptimizedLLM, ProfileStats
    print("✓ MemOpt imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MemOpt: {e}")
    sys.exit(1)

# Test 2: Check CUDA
print("\n[2/5] Checking CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("⚠️  CUDA not available, using CPU (will be slower)")
    device = "cpu"

# Test 3: Initialize model
print("\n[3/5] Initializing model...")
try:
    model = OptimizedLLM(
        model="gpt2",
        optimization_level="balanced",
        device=device,
        enable_profiling=True
    )
    print("✓ Model initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize model: {e}")
    sys.exit(1)

# Test 4: Generate text
print("\n[4/5] Generating text...")
try:
    output = model.generate(
        "Hello, this is a test",
        max_tokens=30,
        do_sample=False
    )
    print(f"✓ Text generated successfully")
    print(f"   Output: {output[:100]}...")
except Exception as e:
    print(f"❌ Failed to generate text: {e}")
    sys.exit(1)

# Test 5: Get stats
print("\n[5/5] Checking profiling...")
try:
    stats = model.get_profiling_stats()
    print(f"✓ Profiling works")
    print(f"   Throughput: {stats.tokens_per_second:.1f} tok/s")
    print(f"   Memory: {stats.peak_memory_allocated_gb:.2f} GB")
except Exception as e:
    print(f"❌ Failed to get stats: {e}")
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("🎉 ALL TESTS PASSED!")
print("="*70)
print("\nMemOpt is installed correctly and working.")
print("\nNext steps:")
print("  1. Run benchmark: python benchmark.py --model gpt2 --mode both")
print("  2. Try different optimization levels: conservative, balanced, high, aggressive")
print("  3. Test with your own models")
print("\n✨ Happy optimizing! ✨\n")