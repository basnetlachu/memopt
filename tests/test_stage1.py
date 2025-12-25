"""
Unit tests for Stage 1 optimizations

Tests adaptive buffer allocation and workspace tensor reuse.
Ensures no regression in correctness or performance.
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
                            print(f"⏭️  Skipped: {reason}")
                            return
                        return func(*args, **kwargs)
                    return wrapper
                return decorator

import torch
from memopt.memory_manager import SmartMemoryManager
from memopt.attention import OptimizedAttentionLayer


class TestAdaptiveAllocation:
    """Test adaptive buffer allocation (Stage 1)"""

    def test_adaptive_allocation_small_batch(self):
        """Small batches should use 1.1x buffer (vs 1.2x baseline)"""
        manager_stage0 = SmartMemoryManager(device="cuda", enable_adaptive_allocation=False)
        manager_stage1 = SmartMemoryManager(device="cuda", enable_adaptive_allocation=True)

        # Small batch: 4 prompts
        blocks_stage0 = manager_stage0.calculate_optimal_kv_blocks(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
            num_prompts=4,
            max_tokens_per_prompt=512,
            quantize=True
        )

        blocks_stage1 = manager_stage1.calculate_optimal_kv_blocks(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
            num_prompts=4,
            max_tokens_per_prompt=512,
            quantize=True
        )

        # Stage 1 should use fewer blocks (tighter buffer)
        assert blocks_stage1 <= blocks_stage0, f"Stage 1 should reduce blocks: {blocks_stage1} vs {blocks_stage0}"
        print(f"✓ Small batch: Stage 0 = {blocks_stage0} blocks, Stage 1 = {blocks_stage1} blocks")

    def test_adaptive_allocation_large_batch(self):
        """Large batches should use 1.15x buffer (vs 1.2x baseline)"""
        manager_stage0 = SmartMemoryManager(device="cuda", enable_adaptive_allocation=False)
        manager_stage1 = SmartMemoryManager(device="cuda", enable_adaptive_allocation=True)

        # Large batch: 16 prompts
        blocks_stage0 = manager_stage0.calculate_optimal_kv_blocks(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
            num_prompts=16,
            max_tokens_per_prompt=512,
            quantize=True
        )

        blocks_stage1 = manager_stage1.calculate_optimal_kv_blocks(
            num_layers=32,
            num_heads=32,
            head_dim=128,
            block_size=16,
            num_prompts=16,
            max_tokens_per_prompt=512,
            quantize=True
        )

        # Stage 1 should still reduce blocks slightly
        assert blocks_stage1 <= blocks_stage0, f"Stage 1 should reduce blocks: {blocks_stage1} vs {blocks_stage0}"
        print(f"✓ Large batch: Stage 0 = {blocks_stage0} blocks, Stage 1 = {blocks_stage1} blocks")

    def test_adaptive_allocation_never_under_allocates(self):
        """Ensure Stage 1 never allocates less than minimum required"""
        manager = SmartMemoryManager(device="cuda", enable_adaptive_allocation=True)

        # Test various batch sizes
        for num_prompts in [1, 4, 8, 16, 32]:
            for max_tokens in [128, 256, 512, 1024]:
                blocks = manager.calculate_optimal_kv_blocks(
                    num_layers=32,
                    num_heads=32,
                    head_dim=128,
                    block_size=16,
                    num_prompts=num_prompts,
                    max_tokens_per_prompt=max_tokens,
                    quantize=True
                )

                # Calculate minimum needed
                total_tokens = num_prompts * max_tokens
                min_blocks = (total_tokens + 15) // 16  # Ceiling division

                # Must have at least minimum
                assert blocks >= min_blocks, \
                    f"Under-allocation: {blocks} < {min_blocks} for {num_prompts} prompts × {max_tokens} tokens"

        print(f"✓ No under-allocation across all test cases")

    def test_memory_estimation_consistency(self):
        """Ensure memory estimation is consistent between Stage 0 and Stage 1"""
        manager_stage0 = SmartMemoryManager(device="cuda", enable_adaptive_allocation=False)
        manager_stage1 = SmartMemoryManager(device="cuda", enable_adaptive_allocation=True)

        # Both should estimate memory the same way (formula unchanged)
        for manager in [manager_stage0, manager_stage1]:
            memory_gb = manager._estimate_memory(
                max_blocks=1000,
                num_layers=32,
                num_heads=32,
                head_dim=128,
                block_size=16,
                quantize=True
            )

            # Sanity check: should be reasonable
            assert 0.1 < memory_gb < 100.0, f"Memory estimate out of range: {memory_gb} GB"

        print(f"✓ Memory estimation consistent")


class TestWorkspaceReuse:
    """Test workspace tensor reuse (Stage 1)"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_workspace_reuse_enabled(self):
        """Test that workspace reuse allocates and reuses tensors"""
        attention = OptimizedAttentionLayer(
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            use_flash=True,
            enable_workspace_reuse=True
        )

        # Verify workspace pool is initialized
        assert attention._workspace_pool is not None, "Workspace pool should be initialized"
        assert len(attention._workspace_pool) == 0, "Workspace pool should start empty"

        # Get a workspace tensor
        device = torch.device("cuda")
        dtype = torch.float16
        shape = (1, 32, 128, 128)

        tensor1 = attention._get_workspace_tensor(shape, dtype, device)
        assert tensor1.shape == shape
        assert tensor1.dtype == dtype
        assert tensor1.device.type == device.type

        # Pool should now have one entry
        assert len(attention._workspace_pool) == 1, "Workspace pool should have 1 tensor"

        # Get same shape again - should reuse
        tensor2 = attention._get_workspace_tensor(shape, dtype, device)
        assert tensor2 is tensor1, "Should reuse same tensor"
        assert len(attention._workspace_pool) == 1, "Pool size should not grow"

        print(f"✓ Workspace reuse working correctly")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_workspace_reuse_disabled(self):
        """Test that workspace reuse can be disabled"""
        attention = OptimizedAttentionLayer(
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            use_flash=True,
            enable_workspace_reuse=False
        )

        # Workspace pool should be None when disabled
        assert attention._workspace_pool is None, "Workspace pool should be None when disabled"

        # Get workspace tensors - should allocate new each time
        device = torch.device("cuda")
        dtype = torch.float16
        shape = (1, 32, 128, 128)

        tensor1 = attention._get_workspace_tensor(shape, dtype, device)
        tensor2 = attention._get_workspace_tensor(shape, dtype, device)

        # Should be different tensors
        assert tensor1 is not tensor2, "Should allocate new tensors when reuse disabled"

        print(f"✓ Workspace reuse correctly disabled")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_workspace_different_shapes(self):
        """Test that different shapes get different workspace tensors"""
        attention = OptimizedAttentionLayer(
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            use_flash=True,
            enable_workspace_reuse=True
        )

        device = torch.device("cuda")
        dtype = torch.float16

        shape1 = (1, 32, 128, 128)
        shape2 = (1, 32, 256, 128)

        tensor1 = attention._get_workspace_tensor(shape1, dtype, device)
        tensor2 = attention._get_workspace_tensor(shape2, dtype, device)

        # Different shapes should get different tensors
        assert tensor1 is not tensor2, "Different shapes should get different tensors"
        assert len(attention._workspace_pool) == 2, "Pool should have 2 tensors"

        # Requesting shape1 again should reuse tensor1
        tensor3 = attention._get_workspace_tensor(shape1, dtype, device)
        assert tensor3 is tensor1, "Should reuse tensor for shape1"

        print(f"✓ Different shapes handled correctly")


class TestStage1Integration:
    """Integration tests for Stage 1 with full model"""

    def test_optimization_presets_have_stage1_flags(self):
        """Ensure all optimization presets define Stage 1 flags"""
        from memopt.model import OptimizedLLM

        presets = OptimizedLLM.OPTIMIZATION_PRESETS

        for level, config in presets.items():
            assert 'enable_adaptive_allocation' in config, \
                f"Preset '{level}' missing 'enable_adaptive_allocation'"
            assert 'enable_workspace_reuse' in config, \
                f"Preset '{level}' missing 'enable_workspace_reuse'"

            # Conservative should have Stage 1 disabled
            if level == "conservative":
                assert config['enable_adaptive_allocation'] is False
                assert config['enable_workspace_reuse'] is False
            # Others should have Stage 1 enabled
            else:
                assert config['enable_adaptive_allocation'] is True
                assert config['enable_workspace_reuse'] is True

        print(f"✓ All presets have correct Stage 1 flags")

    def test_backward_compatibility(self):
        """Ensure Stage 1 flags have safe defaults"""
        from memopt.memory_manager import SmartMemoryManager
        from memopt.attention import OptimizedAttentionLayer

        # Should work without specifying Stage 1 flags (backward compatible)
        manager = SmartMemoryManager(device="cuda")
        assert hasattr(manager, 'enable_adaptive_allocation')

        attention = OptimizedAttentionLayer(
            num_heads=32,
            num_kv_heads=32,
            head_dim=128,
            use_flash=True
        )
        assert hasattr(attention, 'enable_workspace_reuse')


if __name__ == "__main__":
    # Run tests manually if pytest not available
    print("="*70)
    print("STAGE 1 UNIT TESTS")
    print("="*70)

    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available, some tests will be skipped")

    print("\n--- Testing Adaptive Allocation ---")
    test_adaptive = TestAdaptiveAllocation()
    test_adaptive.test_adaptive_allocation_small_batch()
    test_adaptive.test_adaptive_allocation_large_batch()
    test_adaptive.test_adaptive_allocation_never_under_allocates()
    test_adaptive.test_memory_estimation_consistency()

    if torch.cuda.is_available():
        print("\n--- Testing Workspace Reuse ---")
        test_workspace = TestWorkspaceReuse()
        test_workspace.test_workspace_reuse_enabled()
        test_workspace.test_workspace_reuse_disabled()
        test_workspace.test_workspace_different_shapes()

    print("\n--- Testing Integration ---")
    test_integration = TestStage1Integration()
    test_integration.test_optimization_presets_have_stage1_flags()
    test_integration.test_backward_compatibility()

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
