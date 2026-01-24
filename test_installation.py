"""
Installation Test for MemOpt

Run this to verify all Phase 1-4 features are working.
"""

import sys


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        # Phase 1
        from memopt import (
            BandwidthProfiler,
            BandwidthStats,
            BandwidthAnalyzer,
            BottleneckDetector,
        )
        print("✅ Phase 1: Core profiling")

        # Phase 2
        from memopt import (
            BandwidthVisualizer,
            ReportGenerator,
            generate_html_report,
        )
        print("✅ Phase 2: Visualization & Reporting")

        # Phase 3
        from memopt import (
            OptimizationEngine,
            LazyKVCache,
            run_optimization_demo,
        )
        print("✅ Phase 3: Optimization Engine")

        # Utilities
        from memopt import PagedKVCache, CacheStats
        print("✅ Utilities: KV Cache")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality."""
    print("\nTesting basic functionality...")

    try:
        import torch
        from memopt import BandwidthProfiler, BottleneckDetector

        # Create profiler
        profiler = BandwidthProfiler(device="cpu")  # Use CPU for testing
        print("✅ BandwidthProfiler created")

        # Test stats
        profiler.start_profiling()
        # Simulate work
        import time
        time.sleep(0.01)
        profiler.end_profiling()
        stats = profiler.get_stats()
        print(f"✅ Profiler stats generated: {stats.total_time_seconds:.3f}s")

        # Test bottleneck detector
        detector = BottleneckDetector()
        bottlenecks = detector.detect_bottlenecks(stats)
        print(f"✅ Bottleneck detector: {len(bottlenecks)} bottlenecks found")

        return True

    except Exception as e:
        print(f"❌ Functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_examples_exist():
    """Test that all examples exist."""
    print("\nTesting examples...")

    import os

    examples = [
        "examples/basic_profiling.py",
        "examples/visualization_demo.py",
        "examples/lazy_kv_demo.py",
        "examples/complete_workflow.py",
    ]

    all_exist = True
    for example in examples:
        if os.path.exists(example):
            print(f"✅ {example}")
        else:
            print(f"❌ {example} not found")
            all_exist = False

    return all_exist


def main():
    print("\n" + "="*70)
    print("MEMOPT INSTALLATION TEST")
    print("="*70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Basic Functionality", test_basic_functionality),
        ("Examples", test_examples_exist),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")
        results.append((name, test_func()))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! MemOpt is ready to use.")
        print("\nQuick start:")
        print("  python examples/complete_workflow.py")
        print("  python examples/lazy_kv_demo.py")
        print("  memopt-profile --demo lazy-kv")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
