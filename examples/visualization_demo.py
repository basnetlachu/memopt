"""
Visualization & Reporting Demo

Demonstrates Phase 2 features:
- ASCII charts and visualizations
- HTML report generation
- Dashboard creation
- Comparison visualizations
"""

import torch
import torch.nn as nn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memopt.bandwidth_analyzer import BandwidthAnalyzer
from memopt.bandwidth_profiler import BandwidthStats
from memopt.bottleneck_detector import BottleneckDetector
from memopt.visualizer import BandwidthVisualizer, print_dashboard
from memopt.report_generator import generate_html_report


class SimpleTransformer(nn.Module):
    """Simple transformer model for demonstration."""

    def __init__(self, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embedding = nn.Embedding(10000, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10000)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x


def main():
    """Run visualization and reporting demo."""

    print("\n" + "="*70)
    print("MEMOPT PHASE 2 - Visualization & Reporting Demo")
    print("="*70)

    # Check CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  CUDA not available. Running on CPU.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"\n✅ Using GPU: {torch.cuda.get_device_name(0)}")

    # Create model
    print("\n📦 Creating model...")
    model = SimpleTransformer(d_model=512, nhead=8, num_layers=6)
    model = model.to(device)

    # Input generator
    def input_generator():
        return torch.randint(0, 10000, (8, 128), device=device)

    # Profile the model
    print("\n🔍 Profiling model...")
    analyzer = BandwidthAnalyzer(model=model, device=device)

    stats = analyzer.profile_inference(
        input_generator=input_generator,
        num_iterations=20,
        warmup_iterations=5,
        optimization_name="baseline",
    )

    # Detect bottlenecks
    print("\n🔍 Detecting bottlenecks...")
    detector = BottleneckDetector()
    bottlenecks = detector.detect_bottlenecks(stats)

    # ==================================================================
    # PHASE 2 FEATURE 1: ASCII Visualizations
    # ==================================================================
    print("\n" + "="*70)
    print("FEATURE 1: ASCII Visualizations")
    print("="*70)

    viz = BandwidthVisualizer(width=60)

    # Bandwidth chart
    print(viz.draw_bandwidth_chart(stats))

    # Memory usage
    print(viz.draw_memory_usage(stats))

    # Bottleneck chart
    print(viz.draw_bottleneck_chart(bottlenecks))

    # Utilization gauge
    print("\n" + "="*60)
    print("UTILIZATION GAUGES")
    print("="*60)
    print(viz.draw_utilization_gauge(stats.bandwidth_utilization_pct, "Bandwidth"))
    print(viz.draw_utilization_gauge(stats.peak_memory_allocated_gb / 40.0 * 100, "Memory"))

    # ==================================================================
    # PHASE 2 FEATURE 2: Dashboard
    # ==================================================================
    print("\n" + "="*70)
    print("FEATURE 2: Comprehensive Dashboard")
    print("="*70)

    print_dashboard(stats, bottlenecks)

    # ==================================================================
    # PHASE 2 FEATURE 3: Comparison (Simulate optimized version)
    # ==================================================================
    print("\n" + "="*70)
    print("FEATURE 3: Baseline vs Optimized Comparison")
    print("="*70)

    # Simulate optimized stats (30% bandwidth improvement)
    optimized_stats = BandwidthStats(
        total_time_seconds=stats.total_time_seconds * 0.75,
        peak_memory_allocated_gb=stats.peak_memory_allocated_gb * 0.7,
        peak_memory_reserved_gb=stats.peak_memory_reserved_gb * 0.7,
        hbm_read_gb=stats.hbm_read_gb * 0.65,
        hbm_write_gb=stats.hbm_write_gb * 0.65,
        total_hbm_traffic_gb=stats.total_hbm_traffic_gb * 0.65,
        achieved_bandwidth_gbs=stats.achieved_bandwidth_gbs * 1.3,
        gpu_name=stats.gpu_name,
        theoretical_bandwidth_gbs=stats.theoretical_bandwidth_gbs,
        bandwidth_utilization_pct=stats.bandwidth_utilization_pct * 1.3,
        is_memory_bound=False,
        memory_bound_pct=stats.memory_bound_pct * 0.5,
        optimization_applied="lazy_kv_cache",
    )

    print(viz.draw_comparison_chart(stats, optimized_stats))

    # Show improvement summary
    detector.print_comparison(stats, optimized_stats)

    # ==================================================================
    # PHASE 2 FEATURE 4: HTML Report Generation
    # ==================================================================
    print("\n" + "="*70)
    print("FEATURE 4: HTML Report Generation")
    print("="*70)

    html_file = generate_html_report(
        stats=optimized_stats,
        bottlenecks=bottlenecks,
        baseline_stats=stats,
        model_name="SimpleTransformer (6 layers)",
        output_file="bandwidth_report.html",
    )

    print(f"\n✅ HTML report generated: {html_file}")
    print(f"   Open with: open {html_file}")

    # ==================================================================
    # PHASE 2 FEATURE 5: Timeline Visualization (Example)
    # ==================================================================
    print("\n" + "="*70)
    print("FEATURE 5: Execution Timeline")
    print("="*70)

    # Simulate timeline events
    timeline_events = [
        ("Embedding", 0.01, stats.achieved_bandwidth_gbs * 0.8),
        ("Layer 1", 0.02, stats.achieved_bandwidth_gbs * 0.9),
        ("Layer 2", 0.03, stats.achieved_bandwidth_gbs * 0.95),
        ("Layer 3", 0.04, stats.achieved_bandwidth_gbs * 1.0),
        ("Layer 4", 0.05, stats.achieved_bandwidth_gbs * 0.92),
        ("Layer 5", 0.06, stats.achieved_bandwidth_gbs * 0.88),
        ("Layer 6", 0.07, stats.achieved_bandwidth_gbs * 0.85),
        ("Output", 0.08, stats.achieved_bandwidth_gbs * 0.7),
    ]

    print(viz.draw_timeline(timeline_events, "LAYER-BY-LAYER BANDWIDTH"))

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "="*70)
    print("PHASE 2 DEMO COMPLETE")
    print("="*70)

    print("\n✅ Features demonstrated:")
    print("   1. ASCII bandwidth charts")
    print("   2. Memory usage visualization")
    print("   3. Bottleneck breakdown")
    print("   4. Utilization gauges")
    print("   5. Comprehensive dashboard")
    print("   6. Baseline vs optimized comparison")
    print("   7. HTML report generation")
    print("   8. Execution timeline")

    print(f"\n📊 Files generated:")
    print(f"   • {html_file} (HTML report)")

    print("\n💡 Next steps:")
    print("   • Open the HTML report in your browser")
    print("   • Phase 3: Implement lazy KV cache optimization")
    print("   • Validate 15-25% bandwidth reduction (hardware-validated)")

    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
