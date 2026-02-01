"""
Phase 4 Tests - Production Hardening

Tests for:
- Session persistence
- GPU profiles
- Multi-GPU support
- API versioning
"""

import json
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from memopt.profiler.session_persistence import SessionPersistence
from memopt.profiler.gpu_profiles import get_gpu_profile, apply_gpu_profile, get_profile_summary
from memopt.profiler.multi_gpu import (
    RankLocalProfiler,
    MetricsAggregator,
    PerGPUMetrics,
    get_local_rank,
    is_distributed,
)
from memopt.profiler import api


class SimpleModel(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden * 4)
        self.fc2 = nn.Linear(hidden * 4, hidden)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def test_gpu_profile():
    """Test GPU profile detection and application."""
    print("\n" + "=" * 60)
    print("TEST: GPU Profile")
    print("=" * 60)

    profile = get_gpu_profile()
    print(f"Detected GPU: {profile.name}")
    print(f"Compute Capability: {profile.compute_capability}")
    print(f"L2 Cache: {profile.l2_cache_mb:.1f} MB")
    print(f"Memory Bandwidth: {profile.memory_bandwidth_gbps:.0f} GB/s")
    print(f"TF32 Enabled: {profile.enable_tf32}")
    print(f"Preferred Compile Mode: {profile.preferred_compile_mode}")

    # Apply profile
    apply_gpu_profile(profile)
    print("\nProfile applied successfully")

    # Check settings
    if torch.cuda.is_available():
        tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        tf32_cudnn = torch.backends.cudnn.allow_tf32
        cudnn_bench = torch.backends.cudnn.benchmark
        print(f"TF32 matmul: {tf32_matmul}, TF32 cudnn: {tf32_cudnn}, cudnn benchmark: {cudnn_bench}")

    print("\n" + get_profile_summary())
    return True


def test_session_persistence():
    """Test session save/load functionality."""
    print("\n" + "=" * 60)
    print("TEST: Session Persistence")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        persistence = SessionPersistence(Path(tmpdir))

        # Create a mock session
        from memopt.profiler.adaptive_optimizer import (
            OptimizationSession,
            OptimizationResult,
            OptimizationStatus,
            PerformanceMetrics,
        )
        from memopt.profiler.traffic_attribution import OptimizationCandidate, OptimizationType

        session = OptimizationSession(
            session_id="test_session_0",
            start_time=1000.0,
        )

        # Add a mock result
        candidate = OptimizationCandidate(
            optimization_type=OptimizationType.CACHE_RESIDENCY,
            target="cudnn_tf32",
            description="Test optimization",
            expected_traffic_reduction_pct=5.0,
            expected_speedup=1.1,
        )

        result = OptimizationResult(
            candidate=candidate,
            status=OptimizationStatus.COMMITTED,
            baseline_time_ms=10.0,
            optimized_time_ms=8.0,
            actual_speedup=1.25,
            semantics_verified=True,
            baseline_metrics=PerformanceMetrics(
                mean_time_ms=10.0,
                median_time_ms=10.0,
                std_time_ms=0.5,
            ),
            optimized_metrics=PerformanceMetrics(
                mean_time_ms=8.0,
                median_time_ms=8.0,
                std_time_ms=0.4,
            ),
        )
        session.add_result(result)

        # Save session
        filepath = persistence.save_session(session, model_name="test_model")
        print(f"Session saved to: {filepath}")

        # List sessions
        sessions = persistence.list_sessions()
        print(f"Found {len(sessions)} session(s)")
        assert len(sessions) == 1

        # Load session
        loaded = persistence.load_session("test_session_0")
        assert loaded is not None
        print(f"Loaded session: {loaded['session_id']}")
        print(f"Total speedup: {loaded['total_speedup']:.3f}x")
        print(f"Validation: {loaded.get('validation', {})}")

        # Check detailed metrics are saved
        result_data = loaded["results"][0]
        assert "baseline_metrics" in result_data
        assert "validation" in loaded
        print(f"Baseline metrics saved: {result_data['baseline_metrics'] is not None}")

        # Get best session
        best = persistence.get_best_session("test_model")
        assert best is not None
        print(f"Best session speedup: {best['total_speedup']:.3f}x")

    print("\nSession persistence test PASSED")
    return True


def test_multi_gpu_profiler():
    """Test per-GPU profiling."""
    print("\n" + "=" * 60)
    print("TEST: Multi-GPU Profiler")
    print("=" * 60)

    # Create rank-local profiler
    profiler = RankLocalProfiler(device_id=0)
    print(f"Device ID: {profiler.device_id}")
    print(f"Device Name: {profiler.device_name}")

    # Record some metrics
    profiler.record_baseline(time_ms=10.0, memory_bytes=1024 * 1024 * 100)
    profiler.record_optimized(time_ms=8.0, memory_bytes=1024 * 1024 * 80)
    profiler.record_stalls(memory_stall_pct=45.0, compute_stall_pct=10.0)
    profiler.record_bandwidth(bytes_transferred=1024 * 1024 * 1000, time_seconds=0.01)

    metrics = profiler.get_metrics()
    print(f"Recorded speedup: {metrics.speedup:.3f}x")
    print(f"Memory stall: {metrics.memory_stall_pct:.1f}%")
    print(f"Bandwidth: {metrics.achieved_bandwidth_gbps:.1f} GB/s")

    # Test serialization
    data = profiler.to_dict()
    assert "speedup" in data
    assert "memory_stall_pct" in data
    print(f"Serialized {len(data)} fields")

    print("\nMulti-GPU profiler test PASSED")
    return True


def test_metrics_aggregation():
    """Test multi-GPU metrics aggregation."""
    print("\n" + "=" * 60)
    print("TEST: Metrics Aggregation")
    print("=" * 60)

    aggregator = MetricsAggregator()

    # Simulate metrics from 2 GPUs
    metrics1 = PerGPUMetrics(
        device_id=0,
        device_name="GPU 0",
        total_dram_bytes=1024 * 1024 * 1024,
        memory_stall_pct=40.0,
        speedup=1.25,
        achieved_bandwidth_gbps=800,
    )

    metrics2 = PerGPUMetrics(
        device_id=1,
        device_name="GPU 1",
        total_dram_bytes=1024 * 1024 * 1024 * 2,
        memory_stall_pct=50.0,
        speedup=1.15,  # Bottleneck
        achieved_bandwidth_gbps=750,
    )

    aggregator.add_metrics(metrics1)
    aggregator.add_metrics(metrics2)

    agg = aggregator.get_aggregated()
    print(f"Total GPUs: {agg.num_gpus}")
    print(f"Total DRAM: {agg.total_dram_bytes / 1e9:.2f} GB")
    print(f"Avg memory stall: {agg.avg_memory_stall_pct:.1f}%")
    print(f"Total speedup (bottleneck-limited): {agg.total_speedup:.3f}x")
    print(f"Bottleneck GPU: {agg.bottleneck_gpu}")

    assert agg.total_speedup == 1.15  # Limited by slowest GPU
    assert agg.bottleneck_gpu == 1

    print("\n" + aggregator.get_summary())
    print("\nMetrics aggregation test PASSED")
    return True


def test_api_versioning():
    """Test API versioning and stability."""
    print("\n" + "=" * 60)
    print("TEST: API Versioning")
    print("=" * 60)

    print(f"API Version: {api.__version__}")
    print(f"API Contract: {api.__api_version__}")

    # Test version check
    assert api.check_version("0.1.0") == True
    assert api.check_version("0.4.0") == True
    print("Version checks passed")

    # Test stable API
    gpu_info = api.get_gpu_info()
    print(f"GPU Info: {gpu_info['name']}")

    # Test exports
    assert "optimize" in api.__all__
    assert "profile" in api.__all__
    assert "attribute" in api.__all__
    print(f"Exported {len(api.__all__)} API functions")

    print("\nAPI versioning test PASSED")
    return True


def test_full_optimization_with_persistence():
    """Test full optimization pipeline with session persistence."""
    print("\n" + "=" * 60)
    print("TEST: Full Optimization with Persistence")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("SKIPPED: No GPU available")
        return True

    device = torch.device("cuda")
    model = SimpleModel(hidden=512).to(device)
    sample = torch.randn(8, 512).to(device)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Use API to create optimizer with persistence
        optimizer = api.create_optimizer(
            use_fallbacks=True,
            auto_apply_gpu_profile=True,
            persistence_dir=Path(tmpdir),
        )

        print(f"GPU profile: {optimizer.gpu_profile.name if optimizer.gpu_profile else 'None'}")

        # Run optimization
        from memopt.profiler.traffic_attribution import TrafficAttributor

        attributor = TrafficAttributor()
        attributor.analyze_model(model, sample)
        candidates = attributor.get_optimization_candidates()

        session = optimizer.optimize(
            model=model,
            candidates=candidates,
            input_fn=lambda: torch.randn(8, 512).to(device),
            num_warmup=3,
            num_measure=10,
        )

        print(f"Session ID: {session.session_id}")
        print(f"Committed: {session.committed_count}")
        print(f"Rolled back: {session.rollback_count}")
        print(f"Total speedup: {session.total_speedup:.3f}x")

        # Verify session was saved
        persistence = SessionPersistence(Path(tmpdir))
        sessions = persistence.list_sessions()
        print(f"Saved sessions: {len(sessions)}")
        assert len(sessions) >= 1

        # Load and verify
        loaded = persistence.load_session(session.session_id)
        assert loaded is not None
        assert "validation" in loaded
        print(f"Session validation status: {loaded['validation']}")

    print("\nFull optimization with persistence test PASSED")
    return True


def main():
    """Run all Phase 4 tests."""
    print("=" * 60)
    print("PHASE 4 TESTS - Production Hardening")
    print("=" * 60)

    tests = [
        ("GPU Profile", test_gpu_profile),
        ("Session Persistence", test_session_persistence),
        ("Multi-GPU Profiler", test_multi_gpu_profiler),
        ("Metrics Aggregation", test_metrics_aggregation),
        ("API Versioning", test_api_versioning),
        ("Full Optimization with Persistence", test_full_optimization_with_persistence),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, "PASSED" if passed else "FAILED"))
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        icon = "✓" if status == "PASSED" else "✗"
        print(f"  [{icon}] {name}: {status}")

    passed = sum(1 for _, s in results if s == "PASSED")
    print(f"\n{passed}/{len(results)} tests passed")


if __name__ == "__main__":
    main()
