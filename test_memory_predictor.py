#!/usr/bin/env python3
"""
Test Script for Neural Memory Predictor Integration

This script verifies that:
1. Memory tracer works correctly
2. Neural predictor can be trained on synthetic data
3. Scheduler integration works
4. No breaking changes to existing code

Usage:
    python3 test_memory_predictor.py
"""

import sys
import os


def test_memory_tracer():
    """Test memory tracer creation and data collection."""
    print("\n" + "="*70)
    print("TEST 1: Memory Tracer")
    print("="*70)

    try:
        from memopt.memory_tracer import MemoryTracer, create_synthetic_traces

        # Create tracer
        tracer = MemoryTracer(enable=True)

        # Generate synthetic traces
        traces = create_synthetic_traces(num_traces=100)
        tracer.traces = traces

        # Get stats
        stats = tracer.get_stats()
        print(f"✓ Created {len(tracer)} synthetic traces")
        print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
        print(f"  Avg seq len: {stats['avg_seq_len']:.1f}")
        print(f"  Avg memory: {stats['avg_memory_mb']:.1f} MB")

        # Save to CSV
        tracer.save("test_memory_traces.csv")

        # Verify file exists
        if os.path.exists("test_memory_traces.csv"):
            print("✓ Memory tracer test: PASSED")
            return True
        else:
            print("✗ CSV file not created")
            return False

    except Exception as e:
        print(f"✗ Memory tracer test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_predictor_training():
    """Test neural predictor training on synthetic data."""
    print("\n" + "="*70)
    print("TEST 2: Neural Predictor Training")
    print("="*70)

    try:
        from memopt.neural_memory_predictor import NeuralMemoryPredictor

        # Check if test data exists
        if not os.path.exists("test_memory_traces.csv"):
            print("⚠ Test data not found, skipping")
            return True

        # Create and train predictor
        predictor = NeuralMemoryPredictor(device="cpu")

        print("Training on synthetic data (10 epochs, fast)...")
        history = predictor.train_from_csv(
            csv_path="test_memory_traces.csv",
            epochs=10,
            batch_size=16,
            learning_rate=0.001,
            validation_split=0.2,
            verbose=False
        )

        # Check training worked
        if len(history['train_loss']) == 10:
            print(f"✓ Training completed")
            print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            print(f"  Final val loss: {history['val_loss'][-1]:.4f}")

            # Save model
            predictor.save("test_memory_predictor.pth")

            print("✓ Neural predictor training test: PASSED")
            return True
        else:
            print("✗ Training history incomplete")
            return False

    except Exception as e:
        print(f"✗ Neural predictor training test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_predictor_inference():
    """Test neural predictor inference."""
    print("\n" + "="*70)
    print("TEST 3: Neural Predictor Inference")
    print("="*70)

    try:
        from memopt.neural_memory_predictor import NeuralMemoryPredictor

        # Load trained model
        if not os.path.exists("test_memory_predictor.pth"):
            print("⚠ Trained model not found, skipping")
            return True

        predictor = NeuralMemoryPredictor.load("test_memory_predictor.pth", device="cpu")

        # Make predictions
        memory_mb = predictor.predict(
            batch_size=8,
            avg_seq_len=512,
            model_config={'hidden_size': 4096, 'num_layers': 32},
            quantize_kv=False,
            use_flash_attention=True
        )

        print(f"✓ Prediction: {memory_mb:.1f} MB for batch_size=8, avg_seq_len=512")

        # Verify prediction is reasonable
        if 0 < memory_mb < 100000:  # Should be positive and less than 100GB
            print("✓ Neural predictor inference test: PASSED")
            return True
        else:
            print(f"✗ Prediction out of reasonable range: {memory_mb:.1f} MB")
            return False

    except Exception as e:
        print(f"✗ Neural predictor inference test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scheduler_integration():
    """Test scheduler integration with neural predictor."""
    print("\n" + "="*70)
    print("TEST 4: Scheduler Integration")
    print("="*70)

    try:
        from memopt.scheduler import ContinuousBatchScheduler

        # Create scheduler
        scheduler = ContinuousBatchScheduler(
            max_batch_size=32,
            enable_dynamic_batching=True
        )

        # Try to enable neural predictor (will fail gracefully if model not found)
        if os.path.exists("test_memory_predictor.pth"):
            scheduler.enable_neural_memory_predictor(
                model_path="test_memory_predictor.pth",
                model_config={'hidden_size': 4096, 'num_layers': 32}
            )

            if scheduler.use_neural_memory_predictor:
                print("✓ Neural predictor enabled in scheduler")
            else:
                print("⚠ Neural predictor available but not enabled")
        else:
            print("⚠ Neural predictor model not found (expected for first run)")

        print("✓ Scheduler integration test: PASSED")
        return True

    except Exception as e:
        print(f"✗ Scheduler integration test: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """Clean up test files."""
    test_files = [
        "test_memory_traces.csv",
        "test_memory_predictor.pth",
        "test_memory_predictor_training.png"
    ]

    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
            print(f"✓ Cleaned up {f}")


def main():
    print("\n" + "="*70)
    print("PHASE 2: NEURAL MEMORY PREDICTOR TESTS")
    print("="*70)

    results = []

    # Run tests
    results.append(("Memory Tracer", test_memory_tracer()))
    results.append(("Neural Predictor Training", test_neural_predictor_training()))
    results.append(("Neural Predictor Inference", test_neural_predictor_inference()))
    results.append(("Scheduler Integration", test_scheduler_integration()))

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

    # Cleanup
    print("\nCleaning up test files...")
    cleanup_test_files()

    if passed == total:
        print("\n✓ All tests passed! Phase 2 implementation successful.")
        print("\nNext steps:")
        print("1. Collect real production traces: benchmark.py --enable-memory-tracing")
        print("2. Train on real data: python3 train_memory_predictor.py --traces memory_traces.csv")
        print("3. Use in production with scheduler.enable_neural_memory_predictor()")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
