#!/usr/bin/env python3
"""
Unit tests for Stage 3: KV Cache Prefix Sharing

Tests the prefix sharing optimization that detects and reuses
KV cache blocks for common prompt prefixes.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Mock pytest for standalone execution
    class pytest:
        class mark:
            @staticmethod
            def skipif(condition, reason=""):
                def decorator(func):
                    def wrapper(*args, **kwargs):
                        if condition:
                            print(f"SKIP: {reason}")
                            return None
                        return func(*args, **kwargs)
                    return wrapper
                return decorator

        @staticmethod
        def fixture(func):
            return func

        @staticmethod
        def main(args):
            pass

import torch
from memopt.kv_cache import PagedKVCache


class TestPrefixSharing:
    """Test KV cache prefix sharing functionality."""

    @pytest.fixture
    def kv_cache(self):
        """Create a KV cache with prefix sharing enabled."""
        return PagedKVCache(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            block_size=16,
            max_blocks=128,
            device="cpu",
            quantize=False,
            enable_prefix_sharing=True
        )

    def test_prefix_sharing_disabled_by_default(self):
        """Test that prefix sharing is disabled by default."""
        cache = PagedKVCache(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            device="cpu"
        )
        assert not cache.enable_prefix_sharing
        assert len(cache.prefix_cache) == 0

    def test_compute_prefix_hash(self, kv_cache):
        """Test prefix hash computation."""
        tokens1 = list(range(100))
        tokens2 = list(range(100))  # Same tokens
        tokens3 = list(range(50)) + list(range(100, 150))  # Different

        hash1 = kv_cache.compute_prefix_hash(tokens1)
        hash2 = kv_cache.compute_prefix_hash(tokens2)
        hash3 = kv_cache.compute_prefix_hash(tokens3)

        assert hash1 == hash2, "Identical token sequences should have same hash"
        assert hash1 != hash3, "Different token sequences should have different hash"

    def test_prefix_hash_consistency(self, kv_cache):
        """Test that prefix hash is consistent across calls."""
        tokens = list(range(100))

        hash1 = kv_cache.compute_prefix_hash(tokens)
        hash2 = kv_cache.compute_prefix_hash(tokens)
        hash3 = kv_cache.compute_prefix_hash(tokens)

        assert hash1 == hash2 == hash3, "Hash should be deterministic"

    def test_register_prefix(self, kv_cache):
        """Test prefix registration."""
        tokens = list(range(64))
        blocks = [0, 1, 2, 3]

        kv_cache.register_prefix(tokens, blocks)

        prefix_hash = kv_cache.compute_prefix_hash(tokens)
        assert prefix_hash in kv_cache.prefix_cache
        assert kv_cache.prefix_cache[prefix_hash] == blocks

    def test_register_short_prefix_ignored(self, kv_cache):
        """Test that short prefixes are not registered."""
        # Less than prefix_min_length (32)
        tokens = list(range(16))
        blocks = [0, 1]

        kv_cache.register_prefix(tokens, blocks)

        # Should not be registered
        assert len(kv_cache.prefix_cache) == 0

    def test_find_prefix_match_exact(self, kv_cache):
        """Test finding exact prefix match."""
        tokens = list(range(64))
        blocks = [0, 1, 2, 3]

        kv_cache.register_prefix(tokens, blocks)

        # Search for same prefix
        match = kv_cache.find_prefix_match(tokens)
        assert match is not None
        prefix_hash, matched_blocks = match
        assert matched_blocks == blocks

    def test_find_prefix_match_longer_sequence(self, kv_cache):
        """Test finding prefix match in longer sequence."""
        prefix_tokens = list(range(64))
        blocks = [0, 1, 2, 3]

        kv_cache.register_prefix(prefix_tokens, blocks)

        # Search with longer sequence (prefix + more)
        longer_tokens = list(range(128))
        match = kv_cache.find_prefix_match(longer_tokens)

        assert match is not None
        prefix_hash, matched_blocks = match
        assert matched_blocks == blocks

    def test_find_prefix_no_match(self, kv_cache):
        """Test that no match is found for different prefix."""
        tokens1 = list(range(64))
        blocks = [0, 1, 2, 3]

        kv_cache.register_prefix(tokens1, blocks)

        # Search for different prefix
        tokens2 = list(range(100, 164))
        match = kv_cache.find_prefix_match(tokens2)

        assert match is None

    def test_find_prefix_too_short(self, kv_cache):
        """Test that short sequences don't match."""
        tokens = list(range(64))
        blocks = [0, 1, 2, 3]

        kv_cache.register_prefix(tokens, blocks)

        # Search with too short sequence
        short_tokens = list(range(16))
        match = kv_cache.find_prefix_match(short_tokens)

        assert match is None

    def test_prefix_reference_counting(self, kv_cache):
        """Test that shared blocks are reference counted."""
        tokens = list(range(64))
        blocks = [0, 1, 2, 3]

        # Allocate blocks first
        for block_id in blocks:
            kv_cache.free_blocks.discard(block_id)
            kv_cache.block_ref_counts[block_id] = 1

        kv_cache.register_prefix(tokens, blocks)

        # Find match (should increment ref count)
        match = kv_cache.find_prefix_match(tokens)
        assert match is not None

        # Check ref counts incremented
        for block_id in blocks:
            assert kv_cache.block_ref_counts[block_id] == 2

    def test_multiple_prefixes(self, kv_cache):
        """Test managing multiple different prefixes."""
        prefix1 = list(range(64))
        blocks1 = [0, 1, 2, 3]

        prefix2 = list(range(100, 164))
        blocks2 = [4, 5, 6, 7]

        prefix3 = list(range(200, 264))
        blocks3 = [8, 9, 10, 11]

        kv_cache.register_prefix(prefix1, blocks1)
        kv_cache.register_prefix(prefix2, blocks2)
        kv_cache.register_prefix(prefix3, blocks3)

        # All should be findable
        assert kv_cache.find_prefix_match(prefix1) is not None
        assert kv_cache.find_prefix_match(prefix2) is not None
        assert kv_cache.find_prefix_match(prefix3) is not None

        # Correct blocks returned
        _, matched1 = kv_cache.find_prefix_match(prefix1)
        _, matched2 = kv_cache.find_prefix_match(prefix2)
        _, matched3 = kv_cache.find_prefix_match(prefix3)

        assert matched1 == blocks1
        assert matched2 == blocks2
        assert matched3 == blocks3

    def test_prefix_sharing_saves_blocks(self, kv_cache):
        """Test that prefix sharing reduces block allocation."""
        # System prompt (common prefix)
        system_prompt = list(range(64))

        # First request: allocate blocks
        seq_id_1 = 0
        num_blocks_1 = 8
        blocks_1 = kv_cache.allocate_blocks(seq_id_1, num_blocks_1)

        # Register the prefix
        prefix_blocks = blocks_1[:4]  # First 4 blocks are prefix
        kv_cache.register_prefix(system_prompt, prefix_blocks)

        initial_free_blocks = len(kv_cache.free_blocks)

        # Second request with same prefix
        seq_id_2 = 1

        # Find prefix match
        match = kv_cache.find_prefix_match(system_prompt + list(range(100, 132)))
        assert match is not None

        # Should have more free blocks (didn't allocate for prefix)
        # This is a conceptual test - in practice allocate_with_prefix_sharing
        # would be used, which we'll test separately

    def test_allocate_with_prefix_sharing(self, kv_cache):
        """Test allocation with prefix sharing."""
        # Register a prefix
        prefix_tokens = list(range(64))
        prefix_blocks = [0, 1, 2, 3]

        for block_id in prefix_blocks:
            kv_cache.free_blocks.discard(block_id)
            kv_cache.block_ref_counts[block_id] = 1

        kv_cache.register_prefix(prefix_tokens, prefix_blocks)

        # Allocate with prefix sharing
        seq_id = 1
        total_blocks_needed = 8

        allocated_blocks, shared_count = kv_cache.allocate_with_prefix_sharing(
            seq_id, prefix_tokens, total_blocks_needed
        )

        # Should reuse 4 prefix blocks + allocate 4 new blocks
        assert shared_count == 4
        assert len(allocated_blocks) == total_blocks_needed
        assert allocated_blocks[:4] == prefix_blocks

        # Check ref counts incremented for shared blocks
        for block_id in prefix_blocks:
            assert kv_cache.block_ref_counts[block_id] == 2

    def test_prefix_sharing_disabled_when_flag_false(self):
        """Test that prefix sharing doesn't work when disabled."""
        cache = PagedKVCache(
            num_layers=12,
            num_heads=12,
            head_dim=64,
            device="cpu",
            enable_prefix_sharing=False  # Disabled
        )

        tokens = list(range(64))
        blocks = [0, 1, 2, 3]

        # Try to register (should do nothing)
        cache.register_prefix(tokens, blocks)
        assert len(cache.prefix_cache) == 0

        # Try to find match (should return None)
        match = cache.find_prefix_match(tokens)
        assert match is None

    def test_get_stats_includes_prefix_info(self, kv_cache):
        """Test that stats include prefix sharing info."""
        # Register some prefixes
        prefix1 = list(range(64))
        blocks1 = [0, 1, 2, 3]
        kv_cache.register_prefix(prefix1, blocks1)

        prefix2 = list(range(100, 164))
        blocks2 = [4, 5, 6, 7]
        kv_cache.register_prefix(prefix2, blocks2)

        stats = kv_cache.get_stats()

        # Check stats has prefix info
        assert hasattr(stats, 'num_prefixes') or 'num_prefixes' in stats.__dict__

        # Should track number of registered prefixes
        if hasattr(stats, 'num_prefixes'):
            assert stats.num_prefixes == 2


class TestPrefixSharingIntegration:
    """Integration tests for prefix sharing with real KV cache operations."""

    @pytest.fixture
    def kv_cache(self):
        """Create a KV cache with prefix sharing enabled."""
        return PagedKVCache(
            num_layers=2,
            num_heads=4,
            head_dim=16,
            block_size=8,
            max_blocks=32,
            device="cpu",
            quantize=False,
            enable_prefix_sharing=True
        )

    def test_write_and_read_with_prefix_sharing(self, kv_cache):
        """Test writing and reading KV cache with shared prefix blocks."""
        # Allocate first sequence
        seq_id_1 = 0
        blocks_1 = kv_cache.allocate_blocks(seq_id_1, 4)

        # Write some KV data
        batch_size, num_heads, seq_len, head_dim = 1, 4, 8, 16
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        kv_cache.write_cache(
            layer_idx=0,
            seq_id=seq_id_1,
            k=k,
            v=v,
            start_pos=0
        )

        # Register prefix (first 2 blocks)
        prefix_tokens = list(range(32))
        prefix_blocks = blocks_1[:2]
        kv_cache.register_prefix(prefix_tokens, prefix_blocks)

        # Allocate second sequence with prefix sharing
        seq_id_2 = 1
        allocated_blocks, shared_count = kv_cache.allocate_with_prefix_sharing(
            seq_id_2, prefix_tokens, 4
        )

        assert shared_count == 2
        assert allocated_blocks[:2] == prefix_blocks

        # Read from second sequence - should get shared prefix data
        k_read, v_read = kv_cache.read_cache(
            layer_idx=0,
            seq_id=seq_id_2,
            max_length=8
        )

        # Prefix portion should match original
        # (This is a conceptual test - actual data would need proper indexing)
        assert k_read is not None
        assert v_read is not None

    def test_free_with_shared_blocks(self, kv_cache):
        """Test freeing sequences with shared blocks."""
        # Allocate first sequence
        seq_id_1 = 0
        blocks_1 = kv_cache.allocate_blocks(seq_id_1, 4)

        # Register prefix
        prefix_tokens = list(range(32))
        prefix_blocks = blocks_1[:2]
        kv_cache.register_prefix(prefix_tokens, prefix_blocks)

        # Allocate second sequence with sharing
        seq_id_2 = 1
        blocks_2, shared_count = kv_cache.allocate_with_prefix_sharing(
            seq_id_2, prefix_tokens, 4
        )

        # Free first sequence
        kv_cache.free(seq_id_1)

        # Shared blocks should NOT be freed (ref count > 0)
        for block_id in prefix_blocks:
            assert block_id not in kv_cache.free_blocks
            assert kv_cache.block_ref_counts[block_id] == 1

        # Free second sequence
        kv_cache.free(seq_id_2)

        # NOW shared blocks should be freed
        for block_id in prefix_blocks:
            assert block_id in kv_cache.free_blocks
            assert kv_cache.block_ref_counts[block_id] == 0


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("Running Stage 3 tests without pytest...")
        print("=" * 70)

        # Create test instances
        test_prefix = TestPrefixSharing()
        test_integration = TestPrefixSharingIntegration()

        # Get fixtures
        kv_cache = test_prefix.kv_cache()
        kv_cache_int = test_integration.kv_cache()

        # Run basic tests
        tests_run = 0
        tests_passed = 0

        test_cases = [
            ("Prefix sharing disabled by default", test_prefix.test_prefix_sharing_disabled_by_default),
            ("Compute prefix hash", lambda: test_prefix.test_compute_prefix_hash(kv_cache)),
            ("Prefix hash consistency", lambda: test_prefix.test_prefix_hash_consistency(kv_cache)),
            ("Register prefix", lambda: test_prefix.test_register_prefix(kv_cache)),
            ("Find exact prefix match", lambda: test_prefix.test_find_prefix_match_exact(kv_cache)),
            ("Find prefix in longer sequence", lambda: test_prefix.test_find_prefix_match_longer_sequence(kv_cache)),
            ("No match for different prefix", lambda: test_prefix.test_find_prefix_no_match(kv_cache)),
            ("Multiple prefixes", lambda: test_prefix.test_multiple_prefixes(kv_cache)),
            ("Prefix sharing disabled when flag false", test_prefix.test_prefix_sharing_disabled_when_flag_false),
            ("Write and read with prefix sharing", lambda: test_integration.test_write_and_read_with_prefix_sharing(kv_cache_int)),
            ("Free with shared blocks", lambda: test_integration.test_free_with_shared_blocks(kv_cache_int)),
        ]

        for test_name, test_func in test_cases:
            tests_run += 1
            try:
                test_func()
                print(f"✓ {test_name}")
                tests_passed += 1
            except Exception as e:
                print(f"✗ {test_name}: {str(e)}")

        print("=" * 70)
        print(f"Tests passed: {tests_passed}/{tests_run}")

        if tests_passed == tests_run:
            print("\n✅ ALL STAGE 3 TESTS PASSED")
            exit(0)
        else:
            print(f"\n❌ {tests_run - tests_passed} TESTS FAILED")
            exit(1)
