"""
Correctness Tests for Memory Access Coalescing

These tests verify that coalescing produces identical outputs to baseline.
Critical for enterprise credibility - zero tolerance for correctness errors.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memopt.memory_coalescing import (
    MemoryAccessCoalescer,
    CoalescedKVCache,
    CoalescingStats
)


class TestMemoryCoalescingCorrectness:
    """Test that coalescing produces identical outputs to baseline."""

    def test_exact_match_simple(self):
        """Test that coalesced access returns identical tensors."""
        # Create test KV tensors
        k_original = torch.randn(1, 8, 128, 64, device="cpu")
        v_original = torch.randn(1, 8, 128, 64, device="cpu")

        storage = {(0, 0, 128): (k_original, v_original)}

        def fetch_fn(layer, start, end):
            return storage[(layer, start, end)]

        coalescer = MemoryAccessCoalescer()

        # First access - should fetch from storage
        k1, v1 = coalescer.get_kv(0, 0, 128, fetch_fn)
        assert torch.equal(k1, k_original), "First access should match original"
        assert torch.equal(v1, v_original), "First access should match original"

        # Second access - should hit cache
        k2, v2 = coalescer.get_kv(0, 0, 128, fetch_fn)
        assert torch.equal(k2, k_original), "Cached access should match original"
        assert torch.equal(v2, v_original), "Cached access should match original"
        assert torch.equal(k1, k2), "Cached should match first access"
        assert torch.equal(v1, v2), "Cached should match first access"

    def test_subset_reuse_correctness(self):
        """Test that reusing cached subset produces correct slice."""
        # Full sequence
        k_full = torch.randn(1, 8, 128, 64, device="cpu")
        v_full = torch.randn(1, 8, 128, 64, device="cpu")

        storage = {(0, 0, 128): (k_full, v_full)}

        def fetch_fn(layer, start, end):
            return storage.get((layer, start, end), (k_full[:, :, start:end, :], v_full[:, :, start:end, :]))

        coalescer = MemoryAccessCoalescer()

        # Cache full range
        k_cached, v_cached = coalescer.get_kv(0, 0, 128, fetch_fn)

        # Request subset - should slice from cache
        k_subset, v_subset = coalescer.get_kv(0, 0, 64, fetch_fn)

        # Verify subset matches manual slice
        k_expected = k_full[:, :, 0:64, :]
        v_expected = v_full[:, :, 0:64, :]

        assert torch.equal(k_subset, k_expected), "Subset should match manual slice"
        assert torch.equal(v_subset, v_expected), "Subset should match manual slice"

    def test_autoregressive_pattern_correctness(self):
        """
        Test autoregressive generation pattern (most common use case).

        Simulates: fetch[0:1], fetch[0:2], fetch[0:3], ...
        Each step should return correct prefix.
        """
        # Generate full sequence
        seq_len = 100
        k_full = torch.randn(1, 8, seq_len, 64, device="cpu")
        v_full = torch.randn(1, 8, seq_len, 64, device="cpu")

        def fetch_fn(layer, start, end):
            return (k_full[:, :, start:end, :], v_full[:, :, start:end, :])

        coalescer = MemoryAccessCoalescer()

        # Simulate autoregressive generation
        for i in range(1, seq_len):
            k_result, v_result = coalescer.get_kv(0, 0, i, fetch_fn)

            # Verify correctness
            k_expected = k_full[:, :, 0:i, :]
            v_expected = v_full[:, :, 0:i, :]

            assert torch.equal(k_result, k_expected), f"Step {i}: Keys mismatch"
            assert torch.equal(v_result, v_expected), f"Step {i}: Values mismatch"

    def test_coalesced_kv_cache_correctness(self):
        """Test full CoalescedKVCache against baseline."""
        num_layers = 4
        num_heads = 8
        head_dim = 64
        max_seq_len = 128
        batch_size = 2

        # Create baseline (no coalescing)
        baseline_cache = CoalescedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device="cpu",
            enable_coalescing=False
        )

        # Create optimized (with coalescing)
        optimized_cache = CoalescedKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            device="cpu",
            enable_coalescing=True
        )

        # Generate random KV updates
        torch.manual_seed(42)
        for layer in range(num_layers):
            for pos in range(50):
                for batch in range(batch_size):
                    new_k = torch.randn(num_heads, head_dim)
                    new_v = torch.randn(num_heads, head_dim)

                    # Update both caches
                    baseline_cache.update(layer, batch, new_k, new_v, pos)
                    optimized_cache.update(layer, batch, new_k, new_v, pos)

        # Verify all layers produce identical outputs
        for layer in range(num_layers):
            for batch in range(batch_size):
                k_baseline, v_baseline = baseline_cache.get(layer, batch)
                k_optimized, v_optimized = optimized_cache.get(layer, batch)

                assert torch.equal(k_baseline, k_optimized), f"Layer {layer} batch {batch}: Keys mismatch"
                assert torch.equal(v_baseline, v_optimized), f"Layer {layer} batch {batch}: Values mismatch"

    def test_multiple_layers_correctness(self):
        """Test correctness across multiple transformer layers."""
        num_layers = 12
        k_storage = {}
        v_storage = {}

        # Generate different KV for each layer
        for layer in range(num_layers):
            k_storage[layer] = torch.randn(1, 8, 128, 64, device="cpu")
            v_storage[layer] = torch.randn(1, 8, 128, 64, device="cpu")

        def fetch_fn(layer, start, end):
            return (
                k_storage[layer][:, :, start:end, :],
                v_storage[layer][:, :, start:end, :]
            )

        coalescer = MemoryAccessCoalescer()

        # Access different layers with overlapping ranges
        for layer in range(num_layers):
            k_result, v_result = coalescer.get_kv(layer, 0, 100, fetch_fn)

            # Verify correctness
            k_expected = k_storage[layer][:, :, 0:100, :]
            v_expected = v_storage[layer][:, :, 0:100, :]

            assert torch.equal(k_result, k_expected), f"Layer {layer}: Keys mismatch"
            assert torch.equal(v_result, v_expected), f"Layer {layer}: Values mismatch"


class TestCoalescingStatistics:
    """Test that statistics tracking is accurate."""

    def test_cache_hit_counting(self):
        """Verify cache hit/miss counting."""
        k_data = torch.randn(1, 8, 128, 64)
        v_data = torch.randn(1, 8, 128, 64)

        storage = {(0, 0, 128): (k_data, v_data)}

        def fetch_fn(layer, start, end):
            return storage[(layer, start, end)]

        coalescer = MemoryAccessCoalescer(enable_stats=True)

        # First access - cache miss
        coalescer.get_kv(0, 0, 128, fetch_fn)
        assert coalescer.stats.cache_misses == 1
        assert coalescer.stats.cache_hits == 0

        # Second access - cache hit
        coalescer.get_kv(0, 0, 128, fetch_fn)
        assert coalescer.stats.cache_misses == 1
        assert coalescer.stats.cache_hits == 1

        # Third access - cache hit
        coalescer.get_kv(0, 0, 128, fetch_fn)
        assert coalescer.stats.cache_misses == 1
        assert coalescer.stats.cache_hits == 2

    def test_bandwidth_savings_calculation(self):
        """Verify bandwidth savings are calculated correctly."""
        k_full = torch.randn(1, 8, 128, 64, dtype=torch.float16)
        v_full = torch.randn(1, 8, 128, 64, dtype=torch.float16)

        def fetch_fn(layer, start, end):
            return (k_full[:, :, start:end, :], v_full[:, :, start:end, :])

        coalescer = MemoryAccessCoalescer(enable_stats=True)

        # Cache full sequence
        coalescer.get_kv(0, 0, 128, fetch_fn)

        # Request subset - should save bandwidth
        coalescer.get_kv(0, 0, 64, fetch_fn)

        stats = coalescer.get_stats()
        assert stats.bytes_saved > 0, "Should save bytes by reusing cached data"
        assert stats.bandwidth_reduction_pct > 0, "Should show bandwidth reduction"

    def test_hit_rate_calculation(self):
        """Test hit rate percentage calculation."""
        k_data = torch.randn(1, 8, 128, 64)
        v_data = torch.randn(1, 8, 128, 64)

        storage = {(0, 0, 128): (k_data, v_data)}

        def fetch_fn(layer, start, end):
            return storage[(layer, start, end)]

        coalescer = MemoryAccessCoalescer(enable_stats=True)

        # 1 miss, 3 hits = 75% hit rate
        coalescer.get_kv(0, 0, 128, fetch_fn)  # miss
        coalescer.get_kv(0, 0, 128, fetch_fn)  # hit
        coalescer.get_kv(0, 0, 128, fetch_fn)  # hit
        coalescer.get_kv(0, 0, 128, fetch_fn)  # hit

        stats = coalescer.get_stats()
        assert stats.total_accesses == 4
        assert stats.hit_rate_pct == 75.0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_cache(self):
        """Test behavior with empty cache."""
        coalescer = MemoryAccessCoalescer()
        assert len(coalescer.access_cache) == 0

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        k_data = torch.randn(1, 8, 64, 64)
        v_data = torch.randn(1, 8, 64, 64)

        def fetch_fn(layer, start, end):
            return (k_data, v_data)

        coalescer = MemoryAccessCoalescer()
        coalescer.max_cache_entries = 5  # Small cache for testing

        # Fill cache beyond limit
        for i in range(10):
            coalescer.get_kv(i, 0, 64, fetch_fn)

        # Cache should be limited
        assert len(coalescer.access_cache) <= coalescer.max_cache_entries

    def test_reset_stats(self):
        """Test statistics reset."""
        k_data = torch.randn(1, 8, 128, 64)
        v_data = torch.randn(1, 8, 128, 64)

        storage = {(0, 0, 128): (k_data, v_data)}

        def fetch_fn(layer, start, end):
            return storage[(layer, start, end)]

        coalescer = MemoryAccessCoalescer(enable_stats=True)

        # Generate some activity
        coalescer.get_kv(0, 0, 128, fetch_fn)
        coalescer.get_kv(0, 0, 128, fetch_fn)

        # Reset
        coalescer.reset_stats()

        stats = coalescer.get_stats()
        assert stats.total_accesses == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0


if __name__ == "__main__":
    # Run tests
    print("Running Memory Coalescing Correctness Tests...\n")

    test_correctness = TestMemoryCoalescingCorrectness()
    test_stats = TestCoalescingStatistics()
    test_edge = TestEdgeCases()

    print("Testing correctness...")
    test_correctness.test_exact_match_simple()
    print("✅ Exact match test passed")

    test_correctness.test_subset_reuse_correctness()
    print("✅ Subset reuse test passed")

    test_correctness.test_autoregressive_pattern_correctness()
    print("✅ Autoregressive pattern test passed")

    test_correctness.test_coalesced_kv_cache_correctness()
    print("✅ Full KV cache test passed")

    test_correctness.test_multiple_layers_correctness()
    print("✅ Multiple layers test passed")

    print("\nTesting statistics...")
    test_stats.test_cache_hit_counting()
    print("✅ Cache hit counting test passed")

    test_stats.test_bandwidth_savings_calculation()
    print("✅ Bandwidth savings test passed")

    test_stats.test_hit_rate_calculation()
    print("✅ Hit rate calculation test passed")

    print("\nTesting edge cases...")
    test_edge.test_empty_cache()
    print("✅ Empty cache test passed")

    test_edge.test_cache_eviction()
    print("✅ Cache eviction test passed")

    test_edge.test_reset_stats()
    print("✅ Reset stats test passed")

    print("\n" + "="*70)
    print("ALL TESTS PASSED ✅")
    print("="*70)
    print("\nCoalescing optimization verified:")
    print("  ✅ Produces identical outputs to baseline")
    print("  ✅ Correctly tracks bandwidth savings")
    print("  ✅ Handles edge cases properly")
    print("\nSafe for production use.")
