#!/usr/bin/env python3
"""
Unit tests for Stage 4: Priority Scheduling

Tests the priority-aware scheduler that enables dynamic batching
and request prioritization.
"""

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from memopt.scheduler import Priority, InferenceRequest, ContinuousBatchScheduler
import torch


class TestPriorityScheduling:
    """Test priority scheduling functionality."""

    def test_priority_constants(self):
        """Test that priority constants are defined correctly."""
        assert Priority.URGENT == 0
        assert Priority.HIGH == 1
        assert Priority.NORMAL == 2
        assert Priority.LOW == 3
        assert Priority.BACKGROUND == 4
        print("✓ Priority constants defined correctly")

    def test_request_default_priority(self):
        """Test that requests have correct default priority."""
        req = InferenceRequest(
            request_id="test_1",
            prompt="Test prompt",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_tokens=10
        )
        assert req.priority == Priority.NORMAL
        print("✓ Request default priority is NORMAL")

    def test_request_custom_priority(self):
        """Test setting custom priority."""
        req = InferenceRequest(
            request_id="test_2",
            prompt="Urgent request",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_tokens=10,
            priority=Priority.URGENT
        )
        assert req.priority == Priority.URGENT
        print("✓ Custom priority set correctly")

    def test_scheduler_priority_ordering(self):
        """Test that scheduler processes higher priority requests first."""
        scheduler = ContinuousBatchScheduler(max_batch_size=8, device="cpu")

        # Add requests in mixed priority order
        low_req = InferenceRequest(
            request_id="low",
            prompt="Low priority",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_tokens=10,
            priority=Priority.LOW
        )
        urgent_req = InferenceRequest(
            request_id="urgent",
            prompt="Urgent priority",
            input_ids=torch.tensor([[4, 5, 6]]),
            max_tokens=10,
            priority=Priority.URGENT
        )
        normal_req = InferenceRequest(
            request_id="normal",
            prompt="Normal priority",
            input_ids=torch.tensor([[7, 8, 9]]),
            max_tokens=10,
            priority=Priority.NORMAL
        )

        # Add in non-priority order
        scheduler.add_request(low_req)
        scheduler.add_request(urgent_req)
        scheduler.add_request(normal_req)

        # Schedule batch - urgent should be first
        batch = scheduler.schedule_batch()
        assert batch is not None
        assert len(batch) > 0
        assert batch[0].request_id == "urgent", "Urgent request should be processed first"
        print("✓ Scheduler respects priority ordering")

    def test_batch_size_limit(self):
        """Test that batch size is limited."""
        scheduler = ContinuousBatchScheduler(max_batch_size=3, device="cpu")

        # Add 5 requests
        for i in range(5):
            req = InferenceRequest(
                request_id=f"req_{i}",
                prompt=f"Request {i}",
                input_ids=torch.tensor([[1, 2, 3]]),
                max_tokens=10,
                priority=Priority.NORMAL
            )
            scheduler.add_request(req)

        # Schedule batch - should return max 3
        batch = scheduler.schedule_batch()
        assert batch is not None
        assert len(batch) <= 3, f"Batch size should be <= 3, got {len(batch)}"
        print(f"✓ Batch size limited to {len(batch)} (max: 3)")

    def test_queue_depth(self):
        """Test queue depth tracking."""
        scheduler = ContinuousBatchScheduler(max_batch_size=2, device="cpu")

        # Add 4 requests
        for i in range(4):
            req = InferenceRequest(
                request_id=f"req_{i}",
                prompt=f"Request {i}",
                input_ids=torch.tensor([[1, 2]]),
                max_tokens=5,
                priority=Priority.NORMAL
            )
            scheduler.add_request(req)

        initial_depth = scheduler.get_queue_depth()
        assert initial_depth == 4, f"Expected queue depth 4, got {initial_depth}"

        # Schedule batch (should take 2 requests)
        batch = scheduler.schedule_batch()
        remaining_depth = scheduler.get_queue_depth()
        assert remaining_depth == 2, f"Expected queue depth 2 after scheduling, got {remaining_depth}"
        print("✓ Queue depth tracking works")


class TestDynamicBatching:
    """Test Stage 4 dynamic batching optimizations."""

    def test_batch_size_auto_tuning(self):
        """Test that batch size auto-tunes based on sequence length."""
        scheduler = ContinuousBatchScheduler(
            max_batch_size=32,
            enable_dynamic_batching=True,
            device="cpu"
        )

        # Simulate short sequences (should use max batch size)
        scheduler._avg_seq_len_history = [50, 60, 70, 80]
        optimal_size = scheduler._compute_dynamic_batch_size()
        assert optimal_size == 32, f"Short seqs should use max batch size, got {optimal_size}"
        print(f"✓ Short sequences: batch size = {optimal_size}")

        # Simulate long sequences (should reduce batch size)
        scheduler._avg_seq_len_history = [600, 700, 800, 900]
        optimal_size = scheduler._compute_dynamic_batch_size()
        assert optimal_size <= 8, f"Long seqs should use small batch size, got {optimal_size}"
        print(f"✓ Long sequences: batch size = {optimal_size}")

        # Simulate medium sequences (should interpolate)
        scheduler._avg_seq_len_history = [256, 300, 280, 320]
        optimal_size = scheduler._compute_dynamic_batch_size()
        assert 8 < optimal_size < 32, f"Medium seqs should use medium batch size, got {optimal_size}"
        print(f"✓ Medium sequences: batch size = {optimal_size}")

    def test_memory_estimation_with_reuse(self):
        """Test improved memory estimation with prefix sharing."""
        scheduler = ContinuousBatchScheduler(
            max_batch_size=16,
            enable_dynamic_batching=True,
            enable_affinity=True,
            device="cpu"
        )

        req = InferenceRequest(
            request_id="test",
            prompt="Common system prompt for testing",
            input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
            max_tokens=100
        )

        # Memory estimate without prefix sharing
        scheduler.enable_affinity = False
        mem_without = scheduler._estimate_memory_with_reuse(req, [])

        # Memory estimate with prefix sharing potential
        scheduler.enable_affinity = True
        scheduler.affinity_groups["Common system prompt for testing"[:50]] = ["req1"]
        mem_with = scheduler._estimate_memory_with_reuse(req, [])

        # Should save memory with prefix sharing
        assert mem_with < mem_without, "Prefix sharing should reduce memory estimate"
        savings_pct = ((mem_without - mem_with) / mem_without) * 100
        print(f"✓ Memory estimation with prefix sharing: {savings_pct:.1f}% savings")

    def test_grouping_tracks_savings(self):
        """Test that smart grouping tracks padding savings."""
        scheduler = ContinuousBatchScheduler(
            max_batch_size=8,
            enable_dynamic_batching=True,
            device="cpu"
        )

        # Add requests with similar lengths
        for i in range(4):
            req = InferenceRequest(
                request_id=f"req_{i}",
                prompt=f"Request {i}",
                input_ids=torch.tensor([[1] * (50 + i * 5)]),  # 50, 55, 60, 65 tokens
                max_tokens=50,
                priority=Priority.NORMAL
            )
            scheduler.add_request(req)

        # Schedule batch
        batch = scheduler.schedule_batch()
        assert batch is not None
        assert len(batch) == 4

        # Grouping savings should be tracked
        initial_savings = scheduler._grouping_savings
        assert initial_savings >= 0, "Should track grouping savings"
        print(f"✓ Smart grouping tracks {initial_savings} tokens saved")


class TestStage4Integration:
    """Integration tests for Stage 4 with model."""

    def test_ultra_preset_exists(self):
        """Test that 'ultra' optimization preset exists."""
        from memopt.model import OptimizedLLM

        presets = OptimizedLLM.OPTIMIZATION_PRESETS
        assert "ultra" in presets, "'ultra' preset should exist for Stage 4"
        print("✓ 'ultra' preset exists")

    def test_ultra_preset_config(self):
        """Test that 'ultra' preset has Stage 4 features."""
        from memopt.model import OptimizedLLM

        ultra_config = OptimizedLLM.OPTIMIZATION_PRESETS["ultra"]

        # Check Stage 4 features
        assert ultra_config.get("enable_priority_scheduling", False), "Should enable priority scheduling"
        assert ultra_config.get("enable_dynamic_batching", False), "Should enable dynamic batching"
        assert ultra_config.get("max_batch_size", 0) == 32, "Should have larger batch size"

        # Check all previous stages enabled
        assert ultra_config.get("enable_adaptive_allocation", False), "Stage 1 should be enabled"
        assert ultra_config.get("use_continuous_batching", False), "Stage 2 should be enabled"
        assert ultra_config.get("enable_prefix_sharing", False), "Stage 3 should be enabled"

        print("✓ 'ultra' preset configured correctly")

    def test_generate_with_priority_method_exists(self):
        """Test that generate_with_priority method exists."""
        from memopt.model import OptimizedLLM

        # Check method exists
        assert hasattr(OptimizedLLM, 'generate_with_priority'), "generate_with_priority method should exist"
        print("✓ generate_with_priority method exists")

    def test_scheduler_has_dynamic_batching_flag(self):
        """Test that scheduler receives dynamic batching flag."""
        from memopt.model import OptimizedLLM

        # This would require model loading, so just check the parameter exists
        scheduler = ContinuousBatchScheduler(enable_dynamic_batching=True, device="cpu")
        assert hasattr(scheduler, 'enable_dynamic_batching'), "Scheduler should have enable_dynamic_batching"
        assert scheduler.enable_dynamic_batching == True, "Should be enabled when passed"
        print("✓ Scheduler accepts enable_dynamic_batching parameter")


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("Running Stage 4 tests without pytest...")
        print("=" * 70)

        # Create test instances
        test_priority = TestPriorityScheduling()
        test_dynamic = TestDynamicBatching()
        test_integration = TestStage4Integration()

        # Run tests
        tests_run = 0
        tests_passed = 0

        test_cases = [
            ("Priority constants", test_priority.test_priority_constants),
            ("Request default priority", test_priority.test_request_default_priority),
            ("Request custom priority", test_priority.test_request_custom_priority),
            ("Scheduler priority ordering", test_priority.test_scheduler_priority_ordering),
            ("Batch size limit", test_priority.test_batch_size_limit),
            ("Queue depth tracking", test_priority.test_queue_depth),
            ("Batch size auto-tuning", test_dynamic.test_batch_size_auto_tuning),
            ("Memory estimation with reuse", test_dynamic.test_memory_estimation_with_reuse),
            ("Grouping tracks savings", test_dynamic.test_grouping_tracks_savings),
            ("Ultra preset exists", test_integration.test_ultra_preset_exists),
            ("Ultra preset config", test_integration.test_ultra_preset_config),
            ("generate_with_priority exists", test_integration.test_generate_with_priority_method_exists),
            ("Scheduler has dynamic batching flag", test_integration.test_scheduler_has_dynamic_batching_flag),
        ]

        for test_name, test_func in test_cases:
            tests_run += 1
            try:
                test_func()
                tests_passed += 1
            except Exception as e:
                print(f"✗ {test_name}:")
                import traceback
                traceback.print_exc()

        print("=" * 70)
        print(f"Tests passed: {tests_passed}/{tests_run}")

        if tests_passed == tests_run:
            print(f"\n✅ ALL STAGE 4 TESTS PASSED ({tests_passed}/{tests_run})")
            exit(0)
        else:
            print(f"\n❌ {tests_run - tests_passed} TESTS FAILED")
            exit(1)
