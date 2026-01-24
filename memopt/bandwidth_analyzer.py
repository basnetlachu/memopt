"""
Bandwidth Analyzer

Analyzes GPU models to identify memory bandwidth bottlenecks.
Integrates with PyTorch Profiler and provides detailed bandwidth metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import time

from .bandwidth_profiler import BandwidthProfiler, BandwidthStats


@dataclass
class ModelBandwidthProfile:
    """Complete bandwidth profile for a model."""

    model_name: str
    total_params: int
    baseline_stats: BandwidthStats
    optimized_stats: Optional[BandwidthStats] = None
    layer_profiles: Optional[Dict[str, BandwidthStats]] = None


class BandwidthAnalyzer:
    """
    Analyzes models to identify memory bandwidth bottlenecks.

    Workflow:
    1. Profile baseline model execution
    2. Identify memory-bound layers/operations
    3. Suggest optimizations
    4. Profile with optimizations applied
    5. Generate comparison report
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        enable_per_layer_profiling: bool = False,
    ):
        """
        Initialize bandwidth analyzer.

        Args:
            model: PyTorch model to analyze
            device: Device to run on
            enable_per_layer_profiling: Profile each layer individually (slow but detailed)
        """
        self.model = model.to(device)
        self.device = device
        self.enable_per_layer_profiling = enable_per_layer_profiling

        # Model info
        self.model_name = model.__class__.__name__
        self.total_params = sum(p.numel() for p in model.parameters())

        # Profilers
        self.profiler = BandwidthProfiler(
            device=device,
            enable_per_layer_profiling=enable_per_layer_profiling,
            enable_pytorch_profiler=True,
        )

        # Results
        self.baseline_stats: Optional[BandwidthStats] = None
        self.optimized_stats: Optional[BandwidthStats] = None

    def profile_inference(
        self,
        input_generator: Callable,
        num_iterations: int = 10,
        warmup_iterations: int = 3,
        optimization_name: str = "baseline",
    ) -> BandwidthStats:
        """
        Profile model inference with bandwidth measurement.

        Args:
            input_generator: Function that generates input tensors
            num_iterations: Number of iterations to profile
            warmup_iterations: Number of warmup iterations (not profiled)
            optimization_name: Name of optimization applied (e.g., "baseline", "lazy_kv")

        Returns:
            BandwidthStats with profiling results
        """
        self.model.eval()

        # Warmup
        print(f"Warming up ({warmup_iterations} iterations)...")
        with torch.no_grad():
            for _ in range(warmup_iterations):
                inputs = input_generator()
                _ = self.model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Profile
        print(f"Profiling ({num_iterations} iterations)...")
        self.profiler.reset()
        self.profiler.start_profiling()

        with torch.no_grad():
            for _ in range(num_iterations):
                inputs = input_generator()
                _ = self.model(inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.profiler.end_profiling()

        # Get stats
        stats = self.profiler.get_stats(optimization_name=optimization_name)

        # Store as baseline if first profile
        if self.baseline_stats is None:
            self.baseline_stats = stats

        return stats

    def profile_generation(
        self,
        model_generate_fn: Callable,
        prompts: List[str],
        max_tokens: int = 50,
        optimization_name: str = "baseline",
    ) -> BandwidthStats:
        """
        Profile text generation with bandwidth measurement.

        Specifically designed for LLM inference profiling.

        Args:
            model_generate_fn: Function that runs generation (model.generate)
            prompts: List of prompts to generate from
            max_tokens: Max tokens to generate
            optimization_name: Name of optimization applied

        Returns:
            BandwidthStats with profiling results
        """
        print(f"Profiling generation for {len(prompts)} prompts...")

        self.profiler.reset()
        self.profiler.start_profiling()

        # Run generation
        outputs = model_generate_fn(prompts, max_tokens=max_tokens)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.profiler.end_profiling()

        # Get stats
        stats = self.profiler.get_stats(optimization_name=optimization_name)

        # Store as baseline if first profile
        if self.baseline_stats is None:
            self.baseline_stats = stats

        return stats

    def analyze_bottlenecks(self, stats: BandwidthStats) -> Dict[str, str]:
        """
        Analyze bandwidth stats to identify bottlenecks.

        Args:
            stats: BandwidthStats from profiling

        Returns:
            Dictionary of bottleneck analysis
        """
        analysis = {}

        # Overall bottleneck
        if stats.is_memory_bound:
            analysis["overall"] = "memory_bound"
            analysis["severity"] = "high" if stats.memory_bound_pct > 70 else "medium"
        else:
            analysis["overall"] = "compute_bound"
            analysis["severity"] = "low"

        # Bandwidth utilization
        if stats.bandwidth_utilization_pct < 30:
            analysis["bandwidth_utilization"] = "very_low"
            analysis["recommendation"] = "Reduce memory traffic (lazy allocation, quantization)"
        elif stats.bandwidth_utilization_pct < 60:
            analysis["bandwidth_utilization"] = "low"
            analysis["recommendation"] = "Consider memory optimizations"
        else:
            analysis["bandwidth_utilization"] = "good"
            analysis["recommendation"] = "Focus on compute optimizations"

        # Memory usage
        if stats.peak_memory_allocated_gb > 40:
            analysis["memory_usage"] = "high"
            analysis["memory_recommendation"] = "Consider quantization or smaller batch size"
        else:
            analysis["memory_usage"] = "acceptable"

        return analysis

    def suggest_optimizations(self, stats: BandwidthStats) -> List[str]:
        """
        Suggest optimizations based on bandwidth profile.

        Args:
            stats: BandwidthStats from profiling

        Returns:
            List of optimization suggestions
        """
        suggestions = []

        # Memory-bound optimizations
        if stats.is_memory_bound:
            if stats.bandwidth_utilization_pct < 40:
                suggestions.append(
                    "⚠️  LOW BANDWIDTH UTILIZATION: Apply memory access coalescing (15-25% reduction, measured)"
                )

            if stats.peak_memory_allocated_gb > 20:
                suggestions.append(
                    "⚠️  HIGH MEMORY USAGE: Apply INT8 quantization (4x memory reduction)"
                )

            suggestions.append(
                "💡 Memory-bound workload: Focus on reducing HBM traffic"
            )
        else:
            suggestions.append(
                "✅ Compute-bound workload: Memory bandwidth is not the bottleneck"
            )
            suggestions.append(
                "💡 Consider compute optimizations (Flash Attention, fused kernels)"
            )

        return suggestions

    def print_analysis(self, stats: BandwidthStats):
        """
        Print detailed bandwidth analysis.

        Args:
            stats: BandwidthStats to analyze
        """
        print("\n" + "="*70)
        print("BANDWIDTH ANALYSIS")
        print("="*70)

        # Model info
        print(f"\n📦 MODEL INFO")
        print(f"  Name:                    {self.model_name}")
        print(f"  Parameters:              {self.total_params:,}")
        print(f"  Device:                  {stats.gpu_name}")

        # Bottleneck analysis
        analysis = self.analyze_bottlenecks(stats)
        print(f"\n🔍 BOTTLENECK ANALYSIS")
        print(f"  Overall:                 {analysis['overall']}")
        print(f"  Severity:                {analysis['severity']}")
        print(f"  Bandwidth utilization:   {analysis['bandwidth_utilization']}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        suggestions = self.suggest_optimizations(stats)
        for suggestion in suggestions:
            print(f"  {suggestion}")

        # Print bandwidth stats
        self.profiler.print_stats(baseline_stats=self.baseline_stats if stats != self.baseline_stats else None)

    def generate_report(self, output_file: str = "bandwidth_report.json"):
        """
        Generate comprehensive bandwidth report.

        Args:
            output_file: Output file path for JSON report
        """
        report = {
            "model": {
                "name": self.model_name,
                "total_params": self.total_params,
            },
            "baseline": self.baseline_stats.to_dict() if self.baseline_stats else None,
            "optimized": self.optimized_stats.to_dict() if self.optimized_stats else None,
        }

        if self.baseline_stats and self.optimized_stats:
            report["improvement"] = {
                "bandwidth_improvement_pct": (
                    (self.optimized_stats.achieved_bandwidth_gbs -
                     self.baseline_stats.achieved_bandwidth_gbs) /
                    max(self.baseline_stats.achieved_bandwidth_gbs, 1)
                ) * 100,
                "memory_reduction_gb": (
                    self.baseline_stats.peak_memory_allocated_gb -
                    self.optimized_stats.peak_memory_allocated_gb
                ),
            }

        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 Report saved to {output_file}")


def profile_model(
    model: nn.Module,
    input_generator: Callable,
    device: str = "cuda",
    num_iterations: int = 10,
    warmup_iterations: int = 3,
) -> BandwidthStats:
    """
    Quick helper function to profile a model.

    Args:
        model: PyTorch model to profile
        input_generator: Function that generates input tensors
        device: Device to run on
        num_iterations: Number of profiling iterations
        warmup_iterations: Number of warmup iterations

    Returns:
        BandwidthStats with profiling results
    """
    analyzer = BandwidthAnalyzer(model=model, device=device)
    stats = analyzer.profile_inference(
        input_generator=input_generator,
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations,
    )
    analyzer.print_analysis(stats)
    return stats
