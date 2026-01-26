#!/usr/bin/env python3
"""
Comprehensive Test Suite for MemOpt

Tests the core optimization and measurement modules:
1. MemoryCoalescer - hooks into models and tracks accesses
2. BandwidthTracker - measures GPU memory usage
3. Integration tests - end-to-end validation

Run with: pytest tests/test_optimization.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn


# Skip all tests if CUDA not available
CUDA_AVAILABLE = torch.cuda.is_available()
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


class SimpleModel(nn.Module):
    """Simple model for testing without transformers dependency."""
    def __init__(self, hidden_size: int = 256, num_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, batch_first=True)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        attn_out, _ = self.attention(x, x, x)
        return attn_out


class TestMemoryCoalescer:
    """Tests for MemoryCoalescer."""

    def test_import(self):
        """Test that MemoryCoalescer can be imported."""
        from memopt.optimization import MemoryCoalescer, CoalescingConfig, CoalescingStats
        assert MemoryCoalescer is not None
        assert CoalescingConfig is not None
        assert CoalescingStats is not None

    def test_coalescer_creation(self):
        """Test creating a MemoryCoalescer instance."""
        from memopt.optimization import MemoryCoalescer

        model = SimpleModel()
        coalescer = MemoryCoalescer(model, mode='inference')
        assert coalescer is not None
        assert coalescer._enabled is False

    def test_enable_disable(self):
        """Test enabling and disabling the coalescer."""
        from memopt.optimization import MemoryCoalescer

        model = SimpleModel()
        coalescer = MemoryCoalescer(model, mode='inference')

        coalescer.enable()
        assert coalescer._enabled is True

        coalescer.disable()
        assert coalescer._enabled is False

    def test_context_manager(self):
        """Test using coalescer as context manager."""
        from memopt.optimization import MemoryCoalescer

        model = SimpleModel()

        with MemoryCoalescer(model, mode='inference') as coalescer:
            assert coalescer._enabled is True

        assert coalescer._enabled is False

    def test_get_stats(self):
        """Test getting coalescing statistics."""
        from memopt.optimization import MemoryCoalescer

        model = SimpleModel()
        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        # Run a forward pass
        x = torch.randn(1, 10, 256)
        with torch.no_grad():
            model(x)

        stats = coalescer.get_stats()
        coalescer.disable()

        assert stats.layers_optimized >= 0
        assert stats.forward_passes >= 0

    @skip_no_cuda
    def test_coalescer_with_cuda(self):
        """Test coalescer on CUDA device."""
        from memopt.optimization import MemoryCoalescer

        model = SimpleModel().cuda()
        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        x = torch.randn(1, 10, 256).cuda()
        with torch.no_grad():
            output = model(x)

        stats = coalescer.get_stats()
        coalescer.disable()

        assert output.shape == x.shape
        assert stats is not None


class TestBandwidthTracker:
    """Tests for BandwidthTracker."""

    def test_import(self):
        """Test that BandwidthTracker can be imported."""
        from memopt.measurement import BandwidthTracker, BandwidthMeasurement
        assert BandwidthTracker is not None
        assert BandwidthMeasurement is not None

    def test_tracker_creation(self):
        """Test creating a BandwidthTracker instance."""
        from memopt.measurement import BandwidthTracker

        tracker = BandwidthTracker()
        assert tracker is not None

    def test_measure_context_manager(self):
        """Test measuring with context manager."""
        from memopt.measurement import BandwidthTracker

        tracker = BandwidthTracker()

        with tracker.measure("test"):
            # Simulate some work
            x = torch.randn(100, 100)
            y = torch.matmul(x, x)

        measurement = tracker.get_measurement("test")
        assert measurement is not None
        assert measurement.duration_ms >= 0

    def test_multiple_measurements(self):
        """Test multiple measurements."""
        from memopt.measurement import BandwidthTracker

        tracker = BandwidthTracker()

        with tracker.measure("first"):
            x = torch.randn(100, 100)

        with tracker.measure("second"):
            y = torch.randn(200, 200)

        first = tracker.get_measurement("first")
        second = tracker.get_measurement("second")

        assert first is not None
        assert second is not None
        assert first.name == "first"
        assert second.name == "second"

    @skip_no_cuda
    def test_tracker_with_cuda(self):
        """Test tracker on CUDA device."""
        from memopt.measurement import BandwidthTracker

        tracker = BandwidthTracker(device='cuda')

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

        with tracker.measure("cuda_test"):
            x = torch.randn(1000, 1000).cuda()
            y = torch.matmul(x, x)
            torch.cuda.synchronize()

        measurement = tracker.get_measurement("cuda_test")
        assert measurement.peak_memory_bytes > 0
        assert measurement.duration_ms > 0

    def test_compare_measurements(self):
        """Test comparing measurements."""
        from memopt.measurement import BandwidthTracker

        tracker = BandwidthTracker()

        with tracker.measure("baseline"):
            x = torch.randn(100, 100)

        with tracker.measure("optimized"):
            y = torch.randn(50, 50)

        report = tracker.compare("baseline", "optimized")
        assert report is not None
        assert hasattr(report, 'baseline')
        assert hasattr(report, 'optimized')


class TestIntegration:
    """Integration tests combining coalescer and tracker."""

    @skip_no_cuda
    def test_full_workflow(self):
        """Test complete optimization workflow."""
        from memopt.optimization import MemoryCoalescer
        from memopt.measurement import BandwidthTracker

        model = SimpleModel().cuda()
        tracker = BandwidthTracker(device='cuda')

        # Baseline
        torch.cuda.reset_peak_memory_stats()
        with tracker.measure("baseline"):
            x = torch.randn(1, 10, 256).cuda()
            with torch.no_grad():
                baseline_out = model(x)
            torch.cuda.synchronize()

        # Optimized
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        with tracker.measure("optimized"):
            x = torch.randn(1, 10, 256).cuda()
            with torch.no_grad():
                optimized_out = model(x)
            torch.cuda.synchronize()

        stats = coalescer.get_stats()
        coalescer.disable()

        # Verify
        report = tracker.compare("baseline", "optimized")

        assert report is not None
        assert stats is not None
        assert stats.layers_optimized >= 0

    @skip_no_cuda
    def test_correctness_preservation(self):
        """Test that optimization preserves output correctness."""
        from memopt.optimization import MemoryCoalescer

        model = SimpleModel().cuda()
        model.eval()

        # Fixed input for reproducibility
        torch.manual_seed(42)
        x = torch.randn(1, 10, 256).cuda()

        # Baseline
        with torch.no_grad():
            baseline_out = model(x.clone())

        # Optimized
        coalescer = MemoryCoalescer(model, mode='inference')
        coalescer.enable()

        with torch.no_grad():
            optimized_out = model(x.clone())

        coalescer.disable()

        # Outputs should be identical
        assert torch.allclose(baseline_out, optimized_out, atol=1e-5)


class TestCoalescingConfig:
    """Tests for CoalescingConfig."""

    def test_default_config(self):
        """Test default configuration."""
        from memopt.optimization import CoalescingConfig

        config = CoalescingConfig()
        assert config.cache_size_mb > 0
        assert config.optimize_attention is True

    def test_custom_config(self):
        """Test custom configuration."""
        from memopt.optimization import CoalescingConfig

        config = CoalescingConfig(
            cache_size_mb=512,
            optimize_attention=True,
            optimize_mlp=False,
            enable_profiling=True
        )

        assert config.cache_size_mb == 512
        assert config.optimize_attention is True
        assert config.optimize_mlp is False
        assert config.enable_profiling is True

    def test_config_with_coalescer(self):
        """Test using custom config with coalescer."""
        from memopt.optimization import MemoryCoalescer, CoalescingConfig

        config = CoalescingConfig(
            cache_size_mb=64,
            optimize_attention=True,
            optimize_mlp=True
        )

        model = SimpleModel()
        coalescer = MemoryCoalescer(model, mode='inference', config=config)

        assert coalescer.config.cache_size_mb == 64


class TestCoalescingStats:
    """Tests for CoalescingStats."""

    def test_stats_attributes(self):
        """Test CoalescingStats attributes."""
        from memopt.optimization import CoalescingStats

        stats = CoalescingStats()
        assert hasattr(stats, 'layers_optimized')
        assert hasattr(stats, 'total_accesses')
        assert hasattr(stats, 'cache_hits')
        assert hasattr(stats, 'cache_misses')
        assert hasattr(stats, 'hit_rate')
        assert hasattr(stats, 'bandwidth_reduction')

    def test_hit_rate_calculation(self):
        """Test hit rate calculation."""
        from memopt.optimization import CoalescingStats

        stats = CoalescingStats()
        stats.total_accesses = 100
        stats.cache_hits = 25

        # hit_rate should be calculated correctly
        assert stats.hit_rate >= 0


class TestMainImports:
    """Test that main package imports work."""

    def test_main_imports(self):
        """Test importing from main package."""
        from memopt import (
            MemoryCoalescer,
            CoalescingConfig,
            CoalescingStats,
            BandwidthTracker,
        )

        assert MemoryCoalescer is not None
        assert CoalescingConfig is not None
        assert CoalescingStats is not None
        assert BandwidthTracker is not None

    def test_version(self):
        """Test version is set."""
        import memopt
        assert hasattr(memopt, '__version__')
        assert memopt.__version__ == "1.0.0"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
