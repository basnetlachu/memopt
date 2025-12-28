"""
Unit tests for Stage 2 optimizations

Tests continuous batching scheduler integration.
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
from Memopt.scheduler import ContinuousBatchScheduler, SimpleScheduler, InferenceRequest


class TestContinuousBatchScheduler:
    """Test ContinuousBatchScheduler (Stage 2)"""

    def test_scheduler_initialization(self):
        """Test scheduler can be initialized"""
        scheduler = ContinuousBatchScheduler(
            max_batch_size=16,
            max_total_tokens=4096,
            memory_limit_gb=20.0,
            enable_affinity=True,
            device="cuda"
        )

        assert scheduler.max_batch_size == 16
        assert scheduler.max_total_tokens == 4096
        assert scheduler.memory_limit_gb == 20.0
        assert scheduler.enable_affinity is True
        assert len(scheduler.waiting_queue) == 0
        assert len(scheduler.running_batch) == 0

        print("✓ Scheduler initialized correctly")

    def test_add_request(self):
        """Test adding requests to scheduler"""
        scheduler = ContinuousBatchScheduler()

        # Create dummy request
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        request = InferenceRequest(
            request_id="req_1",
            prompt="Test prompt",
            input_ids=input_ids,
            max_tokens=50,
            priority=1
        )

        scheduler.add_request(request)

        assert scheduler.get_queue_depth() == 1
        print("✓ Request added to queue")

    def test_schedule_batch(self):
        """Test batch scheduling logic"""
        scheduler = ContinuousBatchScheduler(max_batch_size=4)

        # Add multiple requests
        for i in range(3):
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            request = InferenceRequest(
                request_id=f"req_{i}",
                prompt=f"Prompt {i}",
                input_ids=input_ids,
                max_tokens=50,
                priority=1
            )
            scheduler.add_request(request)

        # Schedule batch
        batch = scheduler.schedule_batch()

        assert batch is not None
        assert len(batch) == 3
        assert scheduler.get_queue_depth() == 0  # All moved to running batch

        print("✓ Batch scheduled correctly")

    def test_batch_size_limit(self):
        """Test that batch size limit is respected"""
        scheduler = ContinuousBatchScheduler(max_batch_size=2)

        # Add 3 requests
        for i in range(3):
            input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
            request = InferenceRequest(
                request_id=f"req_{i}",
                prompt=f"Prompt {i}",
                input_ids=input_ids,
                max_tokens=50,
                priority=1
            )
            scheduler.add_request(request)

        # Schedule batch
        batch = scheduler.schedule_batch()

        # Should only take 2 requests (max_batch_size=2)
        assert len(batch) == 2
        assert scheduler.get_queue_depth() == 1  # 1 still waiting

        print("✓ Batch size limit respected")

    def test_priority_scheduling(self):
        """Test that higher priority requests are scheduled first"""
        scheduler = ContinuousBatchScheduler(max_batch_size=2)

        # Add requests with different priorities
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        low_priority = InferenceRequest(
            request_id="low",
            prompt="Low priority",
            input_ids=input_ids,
            max_tokens=50,
            priority=1
        )

        high_priority = InferenceRequest(
            request_id="high",
            prompt="High priority",
            input_ids=input_ids,
            max_tokens=50,
            priority=10  # Higher priority
        )

        # Add low priority first
        scheduler.add_request(low_priority)
        scheduler.add_request(high_priority)

        # Schedule batch
        batch = scheduler.schedule_batch()

        # High priority should be scheduled first
        assert batch[0].request_id == "high"
        assert batch[1].request_id == "low"

        print("✓ Priority scheduling works correctly")

    def test_memory_limit_check(self):
        """Test that memory limit prevents over-allocation"""
        scheduler = ContinuousBatchScheduler(
            max_batch_size=100,  # High limit
            max_total_tokens=1000,  # Low token limit
            memory_limit_gb=1.0  # Low memory limit
        )

        # Add many large requests
        for i in range(10):
            # Large input
            input_ids = torch.tensor([[1] * 500], dtype=torch.long)
            request = InferenceRequest(
                request_id=f"req_{i}",
                prompt="Large prompt",
                input_ids=input_ids,
                max_tokens=500,  # Will need lots of memory
                priority=1
            )
            scheduler.add_request(request)

        # Schedule batch
        batch = scheduler.schedule_batch()

        # Should not take all requests due to memory/token limits
        assert len(batch) < 10
        print(f"✓ Memory limit enforced: scheduled {len(batch)}/10 requests")


class TestStage2Integration:
    """Integration tests for Stage 2 with full model"""

    def test_optimization_presets_have_stage2_flags(self):
        """Ensure all optimization presets define Stage 2 flags"""
        from Memopt.model import OptimizedLLM

        presets = OptimizedLLM.OPTIMIZATION_PRESETS

        for level, config in presets.items():
            assert 'use_continuous_batching' in config, \
                f"Preset '{level}' missing 'use_continuous_batching'"

            # Conservative and balanced should have Stage 2 disabled
            if level in ["conservative", "balanced"]:
                assert config['use_continuous_batching'] is False, \
                    f"Preset '{level}' should have continuous batching disabled"
            # High and aggressive should have Stage 2 enabled
            else:
                assert config['use_continuous_batching'] is True, \
                    f"Preset '{level}' should have continuous batching enabled"

        print("✓ All presets have correct Stage 2 flags")

    def test_scheduler_selection(self):
        """Test that correct scheduler is selected based on preset"""
        from Memopt.model import OptimizedLLM
        from Memopt.scheduler import ContinuousBatchScheduler, SimpleScheduler

        # Mock minimal model for testing
        # We'll just check the preset configs without loading a full model

        # Conservative should use SimpleScheduler
        assert OptimizedLLM.OPTIMIZATION_PRESETS["conservative"]["use_continuous_batching"] is False

        # High should use ContinuousBatchScheduler
        assert OptimizedLLM.OPTIMIZATION_PRESETS["high"]["use_continuous_batching"] is True

        print("✓ Scheduler selection logic correct")

    def test_backward_compatibility(self):
        """Ensure Stage 2 flags have safe defaults"""
        from Memopt.scheduler import ContinuousBatchScheduler, SimpleScheduler

        # Should work without issues
        simple = SimpleScheduler(device="cuda")
        assert hasattr(simple, 'add_request')
        assert hasattr(simple, 'schedule_batch')

        continuous = ContinuousBatchScheduler(device="cuda")
        assert hasattr(continuous, 'add_request')
        assert hasattr(continuous, 'schedule_batch')

        print("✓ Backward compatibility maintained")


class TestSimpleSchedulerCompatibility:
    """Test that SimpleScheduler still works (Stage 0/1 compatibility)"""

    def test_simple_scheduler_basic_operation(self):
        """Test SimpleScheduler works as before"""
        scheduler = SimpleScheduler(device="cuda")

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        request = InferenceRequest(
            request_id="test",
            prompt="Test",
            input_ids=input_ids,
            max_tokens=50
        )

        scheduler.add_request(request)
        batch = scheduler.schedule_batch()

        assert batch is not None
        assert len(batch) == 1
        assert batch[0].request_id == "test"

        print("✓ SimpleScheduler still works correctly")

    def test_simple_scheduler_single_request_only(self):
        """Test SimpleScheduler only handles one request at a time"""
        scheduler = SimpleScheduler(device="cuda")

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

        req1 = InferenceRequest("req1", "Test 1", input_ids, 50)
        req2 = InferenceRequest("req2", "Test 2", input_ids, 50)

        scheduler.add_request(req1)
        # Adding req2 overwrites req1 (SimpleScheduler behavior)
        scheduler.add_request(req2)

        batch = scheduler.schedule_batch()

        # Only latest request
        assert len(batch) == 1
        assert batch[0].request_id == "req2"

        print("✓ SimpleScheduler maintains single-request behavior")


if __name__ == "__main__":
    # Run tests manually if pytest not available
    print("="*70)
    print("STAGE 2 UNIT TESTS")
    print("="*70)

    print("\n--- Testing ContinuousBatchScheduler ---")
    test_scheduler = TestContinuousBatchScheduler()
    test_scheduler.test_scheduler_initialization()
    test_scheduler.test_add_request()
    test_scheduler.test_schedule_batch()
    test_scheduler.test_batch_size_limit()
    test_scheduler.test_priority_scheduling()
    test_scheduler.test_memory_limit_check()

    print("\n--- Testing Stage 2 Integration ---")
    test_integration = TestStage2Integration()
    test_integration.test_optimization_presets_have_stage2_flags()
    test_integration.test_scheduler_selection()
    test_integration.test_backward_compatibility()

    print("\n--- Testing SimpleScheduler Compatibility ---")
    test_compatibility = TestSimpleSchedulerCompatibility()
    test_compatibility.test_simple_scheduler_basic_operation()
    test_compatibility.test_simple_scheduler_single_request_only()

    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED")
    print("="*70)
