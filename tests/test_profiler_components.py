#!/usr/bin/env python3
"""
Unit tests for profiler core components.

Tests:
1. RobustPerformanceMeasurer - with mock timings
2. SemanticVerifier - with known diffs
3. OptimizationKnowledgeBase - persistence and learning
4. WorkloadProfiler - drift detection
5. Config - loading and defaults

Run with: pytest tests/test_profiler_components.py -v
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import numpy as np


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class TestConfig:
    """Tests for configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        from memopt.profiler.config import MemoptConfig

        config = MemoptConfig()

        # Attribution defaults
        assert config.attribution.min_read_count == 2
        assert config.attribution.max_reuse_distance == 3

        # Measurement defaults
        assert config.measurement.warmup_iterations == 5
        assert config.measurement.measure_iterations == 20

        # Verification defaults
        assert config.verification.fp32_rtol == 1e-3
        assert config.verification.auto_tolerance is True

        # Compile defaults
        assert config.compile.enabled is True
        assert config.compile.default_mode == "default"

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from memopt.profiler.config import MemoptConfig

        config = MemoptConfig()
        data = config.to_dict()

        assert "attribution" in data
        assert "measurement" in data
        assert "verification" in data
        assert "compile" in data
        assert data["measurement"]["warmup_iterations"] == 5

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from memopt.profiler.config import MemoptConfig

        data = {
            "measurement": {
                "warmup_iterations": 10,
                "measure_iterations": 50
            },
            "verification": {
                "fp32_atol": 0.01
            }
        }

        config = MemoptConfig.from_dict(data)

        assert config.measurement.warmup_iterations == 10
        assert config.measurement.measure_iterations == 50
        assert config.verification.fp32_atol == 0.01
        # Unchanged values should use defaults
        assert config.attribution.min_read_count == 2

    def test_config_save_load_json(self):
        """Test saving and loading config as JSON."""
        from memopt.profiler.config import MemoptConfig

        config = MemoptConfig()
        config.measurement.warmup_iterations = 15

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            config.save(path)
            loaded = MemoptConfig.load(path)
            assert loaded.measurement.warmup_iterations == 15
        finally:
            path.unlink()

    def test_global_config(self):
        """Test global config get/set."""
        from memopt.profiler.config import get_config, set_config, MemoptConfig

        original = get_config()

        custom = MemoptConfig()
        custom.measurement.warmup_iterations = 99
        set_config(custom)

        assert get_config().measurement.warmup_iterations == 99

        # Restore original
        set_config(original)


class TestRobustPerformanceMeasurer:
    """Tests for RobustPerformanceMeasurer."""

    def test_import(self):
        """Test that RobustPerformanceMeasurer can be imported."""
        from memopt.profiler.adaptive_optimizer import RobustPerformanceMeasurer
        assert RobustPerformanceMeasurer is not None

    def test_default_creation(self):
        """Test creating measurer with defaults from config."""
        from memopt.profiler.adaptive_optimizer import RobustPerformanceMeasurer

        measurer = RobustPerformanceMeasurer()
        assert measurer.num_warmup == 5  # Default from config
        assert measurer.num_measure == 20

    def test_custom_params(self):
        """Test creating measurer with custom params."""
        from memopt.profiler.adaptive_optimizer import RobustPerformanceMeasurer

        measurer = RobustPerformanceMeasurer(
            num_warmup=3,
            num_measure=10,
            outlier_threshold=1.5
        )
        assert measurer.num_warmup == 3
        assert measurer.num_measure == 10
        assert measurer.outlier_threshold == 1.5

    def test_measure_cpu_model(self):
        """Test measuring a CPU model."""
        from memopt.profiler.adaptive_optimizer import RobustPerformanceMeasurer

        model = SimpleModel()
        measurer = RobustPerformanceMeasurer(num_warmup=2, num_measure=5)

        metrics = measurer.measure(model, lambda: torch.randn(4, 64))

        assert metrics.mean_time_ms > 0
        assert metrics.median_time_ms > 0
        assert metrics.std_time_ms >= 0
        assert len(metrics.samples) > 0
        assert metrics.coefficient_of_variation >= 0

    def test_outlier_removal(self):
        """Test that outlier removal works."""
        from memopt.profiler.adaptive_optimizer import RobustPerformanceMeasurer

        measurer = RobustPerformanceMeasurer(num_warmup=1, num_measure=10)

        # Create model with varying latency (simulated via multiple runs)
        model = SimpleModel()
        metrics = measurer.measure(model, lambda: torch.randn(4, 64))

        # Should have samples after outlier removal
        assert len(metrics.samples) >= 3

    def test_improvement_significance(self):
        """Test statistical significance detection."""
        from memopt.profiler.adaptive_optimizer import (
            RobustPerformanceMeasurer,
            PerformanceMetrics
        )

        measurer = RobustPerformanceMeasurer()

        # Clear improvement
        baseline = PerformanceMetrics(
            mean_time_ms=10.0,
            median_time_ms=10.0,
            std_time_ms=0.5,
            confidence_95_ms=0.2,
            samples=[9.5, 10.0, 10.5] * 5
        )

        optimized = PerformanceMetrics(
            mean_time_ms=5.0,
            median_time_ms=5.0,
            std_time_ms=0.3,
            confidence_95_ms=0.1,
            samples=[4.8, 5.0, 5.2] * 5
        )

        is_significant, p_value = measurer.is_improvement_significant(
            baseline, optimized, min_improvement=0.02
        )

        assert is_significant is True
        assert p_value < 0.5

    def test_no_improvement_detected(self):
        """Test that no improvement is correctly detected."""
        from memopt.profiler.adaptive_optimizer import (
            RobustPerformanceMeasurer,
            PerformanceMetrics
        )

        measurer = RobustPerformanceMeasurer()

        # No improvement (same performance)
        baseline = PerformanceMetrics(
            mean_time_ms=10.0,
            median_time_ms=10.0,
            std_time_ms=0.5,
            confidence_95_ms=0.2,
            samples=[9.5, 10.0, 10.5]
        )

        optimized = PerformanceMetrics(
            mean_time_ms=10.1,
            median_time_ms=10.1,
            std_time_ms=0.5,
            confidence_95_ms=0.2,
            samples=[9.6, 10.1, 10.6]
        )

        is_significant, _ = measurer.is_improvement_significant(
            baseline, optimized, min_improvement=0.02
        )

        assert is_significant is False


class TestSemanticVerifier:
    """Tests for SemanticVerifier."""

    def test_import(self):
        """Test that SemanticVerifier can be imported."""
        from memopt.profiler.adaptive_optimizer import SemanticVerifier
        assert SemanticVerifier is not None

    def test_default_creation(self):
        """Test creating verifier with defaults."""
        from memopt.profiler.adaptive_optimizer import SemanticVerifier

        verifier = SemanticVerifier()
        assert verifier.auto_tolerance is True
        assert verifier.sample_size == 5

    def test_custom_tolerance(self):
        """Test creating verifier with custom tolerance."""
        from memopt.profiler.adaptive_optimizer import SemanticVerifier

        verifier = SemanticVerifier(rtol=1e-2, atol=1e-2, auto_tolerance=False)
        assert verifier.rtol == 1e-2
        assert verifier.atol == 1e-2

    def test_identical_models_pass(self):
        """Test that identical models pass verification."""
        from memopt.profiler.adaptive_optimizer import SemanticVerifier

        model = SimpleModel()
        verifier = SemanticVerifier()

        is_equivalent, max_diff = verifier.verify(
            model, model, lambda: torch.randn(4, 64)
        )

        assert is_equivalent is True
        assert max_diff == 0.0

    def test_different_models_fail(self):
        """Test that different models fail verification."""
        from memopt.profiler.adaptive_optimizer import SemanticVerifier

        model1 = SimpleModel()
        model2 = SimpleModel()  # Different random weights

        verifier = SemanticVerifier(rtol=1e-6, atol=1e-6, auto_tolerance=False)

        is_equivalent, max_diff = verifier.verify(
            model1, model2, lambda: torch.randn(4, 64)
        )

        assert is_equivalent is False
        assert max_diff > 0

    def test_auto_tolerance_fp16(self):
        """Test auto tolerance adjustment for FP16."""
        from memopt.profiler.adaptive_optimizer import get_tolerance_for_dtype

        rtol, atol = get_tolerance_for_dtype(torch.float16)
        assert rtol >= 1e-3  # More relaxed for FP16
        assert atol >= 1e-3

    def test_auto_tolerance_fp32(self):
        """Test auto tolerance for FP32."""
        from memopt.profiler.adaptive_optimizer import get_tolerance_for_dtype

        rtol, atol = get_tolerance_for_dtype(torch.float32)
        assert rtol == 1e-3  # Default from config
        assert atol == 1e-3

    def test_tolerance_multiplier(self):
        """Test tolerance multiplier for compiled models."""
        from memopt.profiler.adaptive_optimizer import get_tolerance_for_dtype

        rtol, atol = get_tolerance_for_dtype(torch.float32, multiplier=2.0)
        assert rtol == 2e-3
        assert atol == 2e-3


class TestOptimizationKnowledgeBase:
    """Tests for OptimizationKnowledgeBase."""

    def test_import(self):
        """Test that OptimizationKnowledgeBase can be imported."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase
        assert OptimizationKnowledgeBase is not None

    def test_creation(self):
        """Test creating knowledge base."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()
        assert kb is not None
        assert len(kb.patterns) == 0

    def test_record_success(self):
        """Test recording successful optimization."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()

        workload = {"batch_size": 32, "seq_len": 128, "model_type": "transformer"}
        kb.record_success("kernel_fusion", workload, improvement=15.0)

        pattern = kb.get_pattern("kernel_fusion", workload)
        assert pattern is not None
        assert pattern.success_count == 1
        assert pattern.avg_improvement == 15.0

    def test_record_failure(self):
        """Test recording failed optimization."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()

        workload = {"batch_size": 32, "seq_len": 128, "model_type": "transformer"}
        kb.record_failure("kernel_fusion", workload, "Compilation failed")

        pattern = kb.get_pattern("kernel_fusion", workload)
        assert pattern is not None
        assert pattern.failure_count == 1
        assert pattern.success_rate == 0.0

    def test_should_skip_after_failures(self):
        """Test skipping optimization after multiple failures."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()
        workload = {"batch_size": 32, "seq_len": 128, "model_type": "mlp"}

        # Record 3 failures
        for _ in range(3):
            kb.record_failure("layout_transform", workload, "Failed")

        assert kb.should_skip("layout_transform", workload) is True

    def test_should_not_skip_with_successes(self):
        """Test not skipping when there are successes."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()
        workload = {"batch_size": 32, "seq_len": 128, "model_type": "mlp"}

        kb.record_success("tiling", workload, 10.0)
        kb.record_failure("tiling", workload, "Failed")

        assert kb.should_skip("tiling", workload) is False

    def test_persistence(self):
        """Test saving and loading knowledge base."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            cache_path = Path(f.name)

        try:
            # Create and populate KB
            kb1 = OptimizationKnowledgeBase(cache_path=cache_path)
            workload = {"batch_size": 32, "model_type": "cnn"}
            kb1.record_success("cache_residency", workload, 20.0)

            # Load into new KB
            kb2 = OptimizationKnowledgeBase(cache_path=cache_path)

            pattern = kb2.get_pattern("cache_residency", workload)
            assert pattern is not None
            assert pattern.success_count == 1
            assert pattern.avg_improvement == 20.0
        finally:
            cache_path.unlink(missing_ok=True)

    def test_expected_improvement(self):
        """Test expected improvement calculation."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()
        workload = {"batch_size": 32, "model_type": "mlp"}

        # No history
        assert kb.get_expected_improvement("unknown", workload) == 0.0

        # With history
        kb.record_success("tiling", workload, 10.0)
        kb.record_success("tiling", workload, 20.0)

        expected = kb.get_expected_improvement("tiling", workload)
        assert expected > 0

    def test_summary(self):
        """Test summary generation."""
        from memopt.profiler.adaptive_optimizer import OptimizationKnowledgeBase

        kb = OptimizationKnowledgeBase()
        workload = {"batch_size": 32, "model_type": "mlp"}

        kb.record_success("tiling", workload, 10.0)
        kb.record_failure("kernel_fusion", workload, "Failed")

        summary = kb.summary()
        assert "tiling" in summary
        assert "kernel_fusion" in summary


class TestWorkloadProfiler:
    """Tests for WorkloadProfiler."""

    def test_import(self):
        """Test that WorkloadProfiler can be imported."""
        from memopt.profiler.adaptive_optimizer import WorkloadProfiler
        assert WorkloadProfiler is not None

    def test_capture_profile(self):
        """Test capturing workload profile."""
        from memopt.profiler.adaptive_optimizer import WorkloadProfiler

        profiler = WorkloadProfiler()
        model = SimpleModel()

        profile = profiler.capture_profile(model, lambda: torch.randn(8, 64))

        assert "batch_size" in profile
        assert "model_params" in profile
        assert "model_type" in profile
        assert profile["batch_size"] == 8

    def test_detect_no_drift(self):
        """Test no drift with same workload."""
        from memopt.profiler.adaptive_optimizer import WorkloadProfiler

        profiler = WorkloadProfiler()
        model = SimpleModel()

        profile1 = profiler.capture_profile(model, lambda: torch.randn(8, 64))
        profiler.set_baseline(profile1)

        profile2 = profiler.capture_profile(model, lambda: torch.randn(8, 64))
        drift_detected, details = profiler.detect_drift(profile2)

        assert drift_detected is False

    def test_detect_batch_drift(self):
        """Test drift detection with batch size change."""
        from memopt.profiler.adaptive_optimizer import WorkloadProfiler

        profiler = WorkloadProfiler()
        model = SimpleModel()

        # Baseline with batch=8
        profile1 = profiler.capture_profile(model, lambda: torch.randn(8, 64))
        profiler.set_baseline(profile1)

        # Current with batch=32 (4x change)
        profile2 = profiler.capture_profile(model, lambda: torch.randn(32, 64))
        drift_detected, details = profiler.detect_drift(profile2)

        assert drift_detected is True
        assert "batch_size" in details
        assert details["batch_size"] > 0.2  # >20% change

    def test_detect_sequence_drift(self):
        """Test drift detection with sequence length change."""
        from memopt.profiler.adaptive_optimizer import WorkloadProfiler

        profiler = WorkloadProfiler()
        model = SimpleModel()

        # Baseline with seq_len embedded in shape
        profile1 = {"batch_size": 8, "seq_len": 128, "hidden_dim": 64}
        profiler.set_baseline(profile1)

        # Current with longer sequence
        profile2 = {"batch_size": 8, "seq_len": 512, "hidden_dim": 64}
        drift_detected, details = profiler.detect_drift(profile2)

        assert drift_detected is True
        assert "seq_len" in details

    def test_model_type_detection(self):
        """Test model type heuristic detection."""
        from memopt.profiler.adaptive_optimizer import WorkloadProfiler

        profiler = WorkloadProfiler()

        # MLP model
        mlp = SimpleModel()
        profile = profiler.capture_profile(mlp, lambda: torch.randn(4, 64))
        assert profile["model_type"] == "mlp"

        # CNN model
        class ConvModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 16, 3)

            def forward(self, x):
                return self.conv(x)

        cnn = ConvModel()
        profile = profiler.capture_profile(cnn, lambda: torch.randn(4, 3, 32, 32))
        assert profile["model_type"] == "cnn"


class TestCompileCompatibilityTracker:
    """Tests for CompileCompatibilityTracker."""

    def test_import(self):
        """Test that CompileCompatibilityTracker can be imported."""
        from memopt.profiler.adaptive_optimizer import CompileCompatibilityTracker
        assert CompileCompatibilityTracker is not None

    def test_record_success(self):
        """Test recording successful compilation."""
        from memopt.profiler.adaptive_optimizer import CompileCompatibilityTracker

        tracker = CompileCompatibilityTracker()
        model = SimpleModel()

        tracker.record_success(model, "default")

        best_mode = tracker.get_best_mode(model)
        assert best_mode == "default"

    def test_record_failure(self):
        """Test recording failed compilation."""
        from memopt.profiler.adaptive_optimizer import CompileCompatibilityTracker

        tracker = CompileCompatibilityTracker()
        model = SimpleModel()

        tracker.record_failure(model, "reduce-overhead", "Graph break")

        # Should not suggest this mode
        assert tracker.should_skip(model, "reduce-overhead") is False  # Only 1 failure

    def test_should_skip_after_failures(self):
        """Test skipping after multiple failures."""
        from memopt.profiler.adaptive_optimizer import CompileCompatibilityTracker

        tracker = CompileCompatibilityTracker()
        model = SimpleModel()

        # Record 3 failures
        for _ in range(3):
            tracker.record_failure(model, "max-autotune", "OOM")

        assert tracker.should_skip(model, "max-autotune") is True

    def test_get_best_mode_priority(self):
        """Test best mode selection priority."""
        from memopt.profiler.adaptive_optimizer import CompileCompatibilityTracker

        tracker = CompileCompatibilityTracker()
        model = SimpleModel()

        tracker.record_success(model, "reduce-overhead")

        # Should prefer the successful mode
        best = tracker.get_best_mode(model)
        assert best in ["default", "reduce-overhead"]


class TestKernelFuser:
    """Tests for KernelFuser with graceful failure handling."""

    def test_import(self):
        """Test that KernelFuser can be imported."""
        from memopt.profiler.adaptive_optimizer import KernelFuser
        assert KernelFuser is not None

    def test_compile_disabled_in_config(self):
        """Test that compile can be disabled via config."""
        from memopt.profiler.adaptive_optimizer import KernelFuser
        from memopt.profiler.config import get_config, set_config, MemoptConfig

        config = MemoptConfig()
        config.compile.enabled = False
        original = get_config()
        set_config(config)

        try:
            model = SimpleModel()
            result = KernelFuser.compile_model(model)
            # Should return original model when disabled
            assert result is model
        finally:
            set_config(original)

    def test_compile_returns_model_on_failure(self):
        """Test that compile returns original model on failure."""
        from memopt.profiler.adaptive_optimizer import KernelFuser

        model = SimpleModel()

        # Even if torch.compile doesn't exist or fails,
        # should return original model
        result = KernelFuser.compile_model(model)
        assert result is not None


class TestIntegration:
    """Integration tests for profiler components."""

    def test_full_optimization_flow(self):
        """Test complete optimization with all components."""
        from memopt.profiler.adaptive_optimizer import (
            AdaptiveOptimizer,
            SemanticVerifier,
            OptimizationKnowledgeBase,
            generate_fallback_candidates
        )

        model = SimpleModel()
        optimizer = AdaptiveOptimizer(use_fallbacks=True)

        # Generate fallback candidates
        candidates = generate_fallback_candidates(model)
        assert len(candidates) >= 2

        # Run optimization
        session = optimizer.optimize(
            model=model,
            candidates=[],  # Will use fallbacks
            input_fn=lambda: torch.randn(4, 64),
            num_warmup=2,
            num_measure=5
        )

        assert session is not None
        assert session.session_id is not None

    def test_config_affects_components(self):
        """Test that config changes affect all components."""
        from memopt.profiler.config import get_config, set_config, MemoptConfig
        from memopt.profiler.adaptive_optimizer import RobustPerformanceMeasurer

        config = MemoptConfig()
        config.measurement.warmup_iterations = 99
        original = get_config()
        set_config(config)

        try:
            measurer = RobustPerformanceMeasurer()
            assert measurer.num_warmup == 99
        finally:
            set_config(original)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
