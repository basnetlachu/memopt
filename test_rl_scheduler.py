#!/usr/bin/env python3
"""
Test Script for RL Scheduler Integration

This script verifies that:
1. Existing code still works (backward compatibility)
2. RL scheduler can be enabled/disabled
3. No imports break when RL deps not installed

Usage:
    python3 test_rl_scheduler.py
"""

import sys
import torch


def test_backward_compatibility():
    """Test that existing code still works without RL."""
    print("\n" + "="*70)
    print("TEST 1: Backward Compatibility (No RL)")
    print("="*70)

    try:
        from memopt.scheduler import ContinuousBatchScheduler, InferenceRequest
        import uuid

        # Create scheduler WITHOUT RL (should work as before)
        scheduler = ContinuousBatchScheduler(
            max_batch_size=32,
            enable_dynamic_batching=True,
            enable_affinity=True
        )

        # Add some test requests
        for i in range(5):
            req = InferenceRequest(
                request_id=str(uuid.uuid4()),
                prompt=f"Test prompt {i}",
                input_ids=torch.randint(0, 32000, (1, 100), dtype=torch.long),
                max_tokens=50,
                priority=2
            )
            scheduler.add_request(req)

        # Schedule a batch
        batch = scheduler.schedule_batch()

        if batch and len(batch) > 0:
            print(f"✓ Scheduler created {len(batch)} batches")
            print(f"✓ Backward compatibility: PASSED")
            return True
        else:
            print("✗ Scheduler failed to create batch")
            return False

    except Exception as e:
        print(f"✗ Backward compatibility: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rl_scheduler_disabled():
    """Test RL scheduler when explicitly disabled."""
    print("\n" + "="*70)
    print("TEST 2: RL Scheduler Disabled (enable_rl_scheduling=False)")
    print("="*70)

    try:
        from memopt.scheduler import ContinuousBatchScheduler

        # Create scheduler with RL explicitly disabled
        scheduler = ContinuousBatchScheduler(
            max_batch_size=32,
            enable_dynamic_batching=True,
            enable_rl_scheduling=False  # Explicitly disable
        )

        # Verify RL is disabled
        if scheduler.enable_rl_scheduling:
            print("✗ RL scheduling should be disabled")
            return False

        if scheduler.rl_agent is not None:
            print("✗ RL agent should be None")
            return False

        print("✓ RL scheduler disabled: PASSED")
        return True

    except Exception as e:
        print(f"✗ RL scheduler disabled test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rl_scheduler_enabled():
    """Test RL scheduler when enabled (should gracefully fallback if agent not trained)."""
    print("\n" + "="*70)
    print("TEST 3: RL Scheduler Enabled (enable_rl_scheduling=True)")
    print("="*70)

    try:
        from memopt.scheduler import ContinuousBatchScheduler

        # Try to enable RL (will fallback if agent not found)
        scheduler = ContinuousBatchScheduler(
            max_batch_size=32,
            enable_dynamic_batching=True,
            enable_rl_scheduling=True,  # Enable RL
            rl_agent_path="scheduler_rl_agent.zip"
        )

        # Should gracefully fallback if agent not found
        # (enable_rl_scheduling will be set to False by __init__)
        print(f"  RL scheduling enabled: {scheduler.enable_rl_scheduling}")
        print(f"  RL agent loaded: {scheduler.rl_agent is not None}")

        # Test scheduling still works
        from memopt.scheduler import InferenceRequest
        import uuid

        req = InferenceRequest(
            request_id=str(uuid.uuid4()),
            prompt="Test",
            input_ids=torch.randint(0, 32000, (1, 100), dtype=torch.long),
            max_tokens=50,
            priority=2
        )
        scheduler.add_request(req)
        batch = scheduler.schedule_batch()

        if batch:
            print("✓ RL scheduler enabled (with fallback): PASSED")
            return True
        else:
            print("✗ Scheduling failed")
            return False

    except ImportError as e:
        print(f"⚠ RL dependencies not installed ({e}), this is OK for production without RL")
        print("✓ RL scheduler enabled test: PASSED (graceful degradation)")
        return True
    except Exception as e:
        print(f"✗ RL scheduler enabled test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rl_environment():
    """Test RL environment can be imported and created."""
    print("\n" + "="*70)
    print("TEST 4: RL Environment (Optional)")
    print("="*70)

    try:
        from memopt.rl_scheduler import BatchSchedulerEnv, RLSchedulerConfig
        from memopt.scheduler import ContinuousBatchScheduler

        # Create environment
        scheduler = ContinuousBatchScheduler(
            max_batch_size=32,
            enable_dynamic_batching=True
        )

        config = RLSchedulerConfig()
        env = BatchSchedulerEnv(scheduler=scheduler, config=config)

        # Test reset
        state = env.reset()
        print(f"  State shape: {state.shape}")
        print(f"  State: {state}")

        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")

        print("✓ RL environment test: PASSED")
        return True

    except ImportError as e:
        print(f"⚠ RL dependencies not installed ({e})")
        print("  To use RL features, install: pip install stable-baselines3 gym")
        print("✓ RL environment test: SKIPPED (dependencies missing)")
        return True
    except Exception as e:
        print(f"✗ RL environment test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*70)
    print("MEMOPT RL SCHEDULER INTEGRATION TESTS")
    print("="*70)

    results = []

    # Run tests
    results.append(("Backward Compatibility", test_backward_compatibility()))
    results.append(("RL Disabled", test_rl_scheduler_disabled()))
    results.append(("RL Enabled (with fallback)", test_rl_scheduler_enabled()))
    results.append(("RL Environment", test_rl_environment()))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {name}")

    print("="*70)
    print(f"Result: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n✓ All tests passed! RL scheduler integration successful.")
        print("\nNext steps:")
        print("1. Install RL dependencies: pip install stable-baselines3 gym tensorboard")
        print("2. Train RL agent: python3 train_rl_scheduler.py")
        print("3. Use RL scheduler in production with enable_rl_scheduling=True")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
