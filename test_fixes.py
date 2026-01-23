#!/usr/bin/env python3
"""
Quick test to verify the critical bug fixes.

Tests:
1. Request ID tracking works correctly
2. Token counting is non-zero
3. No KeyError in generate_batch()
4. No division by zero in metrics
"""

import sys
import traceback

def test_memopt_basic():
    """Test basic Memopt functionality"""
    print("="*70)
    print("TEST 1: Memopt Basic Generation")
    print("="*70)

    try:
        from memopt import OptimizedLLM

        model = OptimizedLLM(
            model="gpt2",
            optimization_level="batch",
            enable_profiling=True
        )

        prompts = [
            "The quick brown fox",
            "Hello world",
            "Test prompt"
        ]

        print(f"Processing {len(prompts)} prompts...")
        results = model.generate_batch(prompts, max_tokens=10, do_sample=False)

        # Verify results
        assert len(results) == len(prompts), f"Expected {len(prompts)} results, got {len(results)}"
        assert all(r is not None for r in results), "Some results are None"

        # Verify token counting
        stats = model.get_profiling_stats()
        print(f"\n✓ Generated {stats.total_tokens_generated} tokens")
        print(f"✓ Throughput: {stats.tokens_per_second:.1f} tok/s")

        assert stats.total_tokens_generated > 0, "Zero tokens generated!"
        assert stats.tokens_per_second > 0, "Zero throughput!"

        print("\n✅ TEST 1 PASSED: Memopt works correctly\n")
        return True

    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_benchmark_production():
    """Test production benchmark"""
    print("="*70)
    print("TEST 2: Production Benchmark (Memopt only)")
    print("="*70)

    try:
        import subprocess
        import json

        # Run benchmark with minimal config
        cmd = [
            "python", "benchmarks/benchmark_production.py",
            "--model", "gpt2",
            "--baseline", "memopt",
            "--num-prompts", "5",
            "--max-tokens", "10",
            "--output", "test_results.json"
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"\n❌ Benchmark failed with code {result.returncode}")
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            return False

        # Check output
        with open("test_results.json", "r") as f:
            data = json.load(f)

        if "results" in data and "memopt" in data["results"]:
            memopt_results = data["results"]["memopt"]
            tokens = memopt_results.get("total_tokens_generated", 0)
            throughput = memopt_results.get("tokens_per_second", 0)

            print(f"\n✓ Tokens generated: {tokens}")
            print(f"✓ Throughput: {throughput:.1f} tok/s")

            assert tokens > 0, "Zero tokens in benchmark results!"
            assert throughput > 0, "Zero throughput in benchmark results!"

            print("\n✅ TEST 2 PASSED: Benchmark works correctly\n")
            return True
        else:
            print(f"\n❌ No memopt results in output: {data}")
            return False

    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("CRITICAL BUG FIX VALIDATION")
    print("="*70 + "\n")

    results = []

    # Test 1: Basic Memopt
    results.append(("Memopt Basic", test_memopt_basic()))

    # Test 2: Benchmark
    results.append(("Benchmark", test_benchmark_production()))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED - Critical bugs are fixed!\n")
        sys.exit(0)
    else:
        print("\n⚠️  SOME TESTS FAILED - Review errors above\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
