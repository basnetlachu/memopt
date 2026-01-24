"""
Bandwidth Visualizer

Creates terminal-based visualizations (ASCII charts) for bandwidth profiling results.
Provides clear, actionable insights through visual representations.
"""

from typing import Dict, List, Tuple, Optional
import math

from .bandwidth_profiler import BandwidthStats
from .bottleneck_detector import Bottleneck, BottleneckSeverity


class BandwidthVisualizer:
    """
    Creates ASCII-based visualizations for bandwidth profiling.

    Supports:
    - Bar charts for bandwidth utilization
    - Timeline charts for bottleneck progression
    - Comparison charts (baseline vs optimized)
    - Memory usage heatmaps
    """

    def __init__(self, width: int = 60):
        """
        Initialize visualizer.

        Args:
            width: Chart width in characters (default 60)
        """
        self.width = width

    def draw_bar_chart(
        self,
        data: Dict[str, float],
        title: str,
        max_value: Optional[float] = None,
        unit: str = "",
    ) -> str:
        """
        Draw a horizontal bar chart.

        Args:
            data: Dictionary of label -> value
            title: Chart title
            max_value: Maximum value for scale (auto if None)
            unit: Unit string (e.g., "GB/s", "%")

        Returns:
            ASCII bar chart as string
        """
        if not data:
            return f"\n{title}\n(No data)\n"

        # Calculate max value for scaling
        if max_value is None:
            max_value = max(data.values())

        if max_value == 0:
            max_value = 1  # Avoid division by zero

        # Find longest label for alignment
        max_label_len = max(len(str(label)) for label in data.keys())

        # Build chart
        lines = [f"\n{title}", "=" * self.width]

        for label, value in data.items():
            # Calculate bar length
            bar_width = int((value / max_value) * (self.width - max_label_len - 15))
            bar_width = max(0, bar_width)  # Ensure non-negative

            # Create bar
            bar = "█" * bar_width

            # Format value
            value_str = f"{value:.1f}{unit}"

            # Create line
            line = f"{label:<{max_label_len}} │ {bar} {value_str}"
            lines.append(line)

        lines.append("=" * self.width)

        return "\n".join(lines)

    def draw_bandwidth_chart(self, stats: BandwidthStats) -> str:
        """
        Draw bandwidth utilization chart.

        Args:
            stats: BandwidthStats to visualize

        Returns:
            ASCII chart showing bandwidth metrics
        """
        data = {
            "Achieved": stats.achieved_bandwidth_gbs,
            "Theoretical": stats.theoretical_bandwidth_gbs,
        }

        return self.draw_bar_chart(
            data=data,
            title="📊 BANDWIDTH COMPARISON",
            max_value=stats.theoretical_bandwidth_gbs,
            unit=" GB/s",
        )

    def draw_utilization_gauge(self, percentage: float, label: str) -> str:
        """
        Draw a utilization gauge (0-100%).

        Args:
            percentage: Utilization percentage (0-100)
            label: Label for the gauge

        Returns:
            ASCII gauge visualization
        """
        gauge_width = self.width - 20
        filled = int((percentage / 100.0) * gauge_width)
        empty = gauge_width - filled

        # Color indicators (using characters)
        if percentage >= 80:
            fill_char = "█"  # High utilization (good for compute)
            status = "🟢 Excellent"
        elif percentage >= 60:
            fill_char = "▓"
            status = "🟡 Good"
        elif percentage >= 40:
            fill_char = "▒"
            status = "🟠 Moderate"
        else:
            fill_char = "░"
            status = "🔴 Low"

        gauge = fill_char * filled + "░" * empty

        return f"{label}: [{gauge}] {percentage:.1f}% - {status}"

    def draw_memory_usage(self, stats: BandwidthStats) -> str:
        """
        Draw memory usage visualization.

        Args:
            stats: BandwidthStats with memory info

        Returns:
            ASCII memory usage chart
        """
        lines = [
            "\n💾 MEMORY USAGE",
            "=" * self.width,
        ]

        # Memory allocated bar
        gpu_memory = 40.0  # Assume 40GB GPU (conservative)
        mem_pct = (stats.peak_memory_allocated_gb / gpu_memory) * 100

        gauge = self.draw_utilization_gauge(mem_pct, "Memory Allocated")
        lines.append(gauge)
        lines.append(f"   {stats.peak_memory_allocated_gb:.2f} GB / {gpu_memory:.0f} GB")

        # HBM traffic
        lines.append("")
        lines.append("HBM Traffic Breakdown:")
        traffic_data = {
            "Reads": stats.hbm_read_gb,
            "Writes": stats.hbm_write_gb,
        }

        max_traffic = stats.total_hbm_traffic_gb
        for label, value in traffic_data.items():
            if max_traffic > 0:
                bar_width = int((value / max_traffic) * (self.width - 30))
                bar = "█" * bar_width
                lines.append(f"   {label:<8} │ {bar} {value:.2f} GB")

        lines.append("=" * self.width)

        return "\n".join(lines)

    def draw_bottleneck_chart(self, bottlenecks: List[Bottleneck]) -> str:
        """
        Visualize detected bottlenecks.

        Args:
            bottlenecks: List of detected bottlenecks

        Returns:
            ASCII bottleneck visualization
        """
        if not bottlenecks:
            return "\n🟢 No significant bottlenecks detected!\n"

        lines = [
            "\n🔍 BOTTLENECK BREAKDOWN",
            "=" * self.width,
        ]

        severity_emojis = {
            BottleneckSeverity.CRITICAL: "🔴",
            BottleneckSeverity.HIGH: "🟠",
            BottleneckSeverity.MEDIUM: "🟡",
            BottleneckSeverity.LOW: "🟢",
            BottleneckSeverity.NONE: "⚪",
        }

        for i, bottleneck in enumerate(bottlenecks, 1):
            emoji = severity_emojis.get(bottleneck.severity, "⚪")
            lines.append(f"\n{emoji} {bottleneck.type.value.upper()}")
            lines.append(f"   Severity: {bottleneck.severity.value}")

            if bottleneck.impact_pct > 0:
                # Draw impact bar
                impact_bar_width = int((bottleneck.impact_pct / 100.0) * (self.width - 25))
                impact_bar = "█" * impact_bar_width
                lines.append(f"   Impact:   [{impact_bar:<{self.width-25}}] {bottleneck.impact_pct:.1f}%")

            if bottleneck.estimated_speedup > 1.1:
                lines.append(f"   Potential speedup: {bottleneck.estimated_speedup:.2f}x")

        lines.append("\n" + "=" * self.width)

        return "\n".join(lines)

    def draw_comparison_chart(
        self,
        baseline_stats: BandwidthStats,
        optimized_stats: BandwidthStats,
    ) -> str:
        """
        Draw side-by-side comparison of baseline vs optimized.

        Args:
            baseline_stats: Baseline bandwidth stats
            optimized_stats: Optimized bandwidth stats

        Returns:
            ASCII comparison chart
        """
        lines = [
            "\n📊 BASELINE vs OPTIMIZED COMPARISON",
            "=" * self.width,
        ]

        # Metrics to compare
        metrics = [
            ("Bandwidth", baseline_stats.achieved_bandwidth_gbs, optimized_stats.achieved_bandwidth_gbs, "GB/s", True),
            ("Utilization", baseline_stats.bandwidth_utilization_pct, optimized_stats.bandwidth_utilization_pct, "%", True),
            ("Memory", baseline_stats.peak_memory_allocated_gb, optimized_stats.peak_memory_allocated_gb, "GB", False),
            ("HBM Traffic", baseline_stats.total_hbm_traffic_gb, optimized_stats.total_hbm_traffic_gb, "GB", False),
        ]

        for name, base_val, opt_val, unit, higher_better in metrics:
            # Calculate improvement
            if base_val > 0:
                if higher_better:
                    improvement = ((opt_val - base_val) / base_val) * 100
                    symbol = "↑" if improvement > 0 else "↓"
                else:
                    improvement = ((base_val - opt_val) / base_val) * 100
                    symbol = "↓" if improvement > 0 else "↑"
            else:
                improvement = 0
                symbol = "→"

            # Draw bars
            max_val = max(base_val, opt_val)
            if max_val > 0:
                base_bar_width = int((base_val / max_val) * (self.width // 2 - 15))
                opt_bar_width = int((opt_val / max_val) * (self.width // 2 - 15))
            else:
                base_bar_width = 0
                opt_bar_width = 0

            lines.append(f"\n{name}:")
            lines.append(f"  Baseline:  {'█' * base_bar_width} {base_val:.2f} {unit}")
            lines.append(f"  Optimized: {'█' * opt_bar_width} {opt_val:.2f} {unit}")
            lines.append(f"  Change:    {symbol} {abs(improvement):.1f}%")

        lines.append("\n" + "=" * self.width)

        return "\n".join(lines)

    def draw_timeline(
        self,
        events: List[Tuple[str, float, float]],
        title: str = "EXECUTION TIMELINE",
    ) -> str:
        """
        Draw execution timeline showing bandwidth over time.

        Args:
            events: List of (label, start_time, bandwidth) tuples
            title: Timeline title

        Returns:
            ASCII timeline visualization
        """
        if not events:
            return f"\n{title}\n(No timeline data)\n"

        lines = [
            f"\n{title}",
            "=" * self.width,
        ]

        # Find max bandwidth for scaling
        max_bandwidth = max(bw for _, _, bw in events)
        if max_bandwidth == 0:
            max_bandwidth = 1

        # Time axis
        max_time = max(t for _, t, _ in events)
        time_scale = self.width / max_time if max_time > 0 else 1

        for label, time, bandwidth in events:
            # Calculate bar height (bandwidth)
            bar_height = int((bandwidth / max_bandwidth) * 10)

            # Calculate position on timeline
            pos = int(time * time_scale)

            # Create timeline bar
            timeline = [" "] * self.width
            for i in range(pos, min(pos + 5, self.width)):
                timeline[i] = "█"

            lines.append(f"{label:<20} │ {''.join(timeline)} {bandwidth:.1f} GB/s")

        lines.append("=" * self.width)
        lines.append(f"{'Time (s)':>20} │ 0{' ' * (self.width - 10)}{max_time:.2f}")

        return "\n".join(lines)

    def create_summary_dashboard(
        self,
        stats: BandwidthStats,
        bottlenecks: Optional[List[Bottleneck]] = None,
    ) -> str:
        """
        Create comprehensive dashboard with all visualizations.

        Args:
            stats: BandwidthStats to visualize
            bottlenecks: Optional list of bottlenecks

        Returns:
            Complete dashboard as string
        """
        dashboard = [
            "\n" + "="*70,
            "BANDWIDTH PROFILING DASHBOARD",
            "="*70,
        ]

        # GPU info
        dashboard.append(f"\n🖥️  GPU: {stats.gpu_name}")
        dashboard.append(f"    Theoretical bandwidth: {stats.theoretical_bandwidth_gbs:.0f} GB/s")

        # Bandwidth utilization gauge
        dashboard.append(f"\n{self.draw_utilization_gauge(stats.bandwidth_utilization_pct, 'Bandwidth Utilization')}")

        # Bandwidth chart
        dashboard.append(self.draw_bandwidth_chart(stats))

        # Memory usage
        dashboard.append(self.draw_memory_usage(stats))

        # Bottlenecks
        if bottlenecks:
            dashboard.append(self.draw_bottleneck_chart(bottlenecks))

        # Summary stats
        dashboard.append(f"\n📈 SUMMARY STATISTICS")
        dashboard.append("=" * self.width)
        dashboard.append(f"  Total time:           {stats.total_time_seconds:.3f}s")
        dashboard.append(f"  Achieved bandwidth:   {stats.achieved_bandwidth_gbs:.1f} GB/s")
        dashboard.append(f"  Utilization:          {stats.bandwidth_utilization_pct:.1f}%")
        dashboard.append(f"  Memory allocated:     {stats.peak_memory_allocated_gb:.2f} GB")
        dashboard.append(f"  HBM traffic:          {stats.total_hbm_traffic_gb:.2f} GB")

        bottleneck_status = "Memory-bound ❌" if stats.is_memory_bound else "Compute-bound ✅"
        dashboard.append(f"  Bottleneck status:    {bottleneck_status}")

        dashboard.append("\n" + "="*70 + "\n")

        return "\n".join(dashboard)


def print_bandwidth_chart(stats: BandwidthStats):
    """
    Quick helper to print bandwidth chart.

    Args:
        stats: BandwidthStats to visualize
    """
    viz = BandwidthVisualizer()
    print(viz.draw_bandwidth_chart(stats))


def print_comparison(baseline_stats: BandwidthStats, optimized_stats: BandwidthStats):
    """
    Quick helper to print comparison chart.

    Args:
        baseline_stats: Baseline stats
        optimized_stats: Optimized stats
    """
    viz = BandwidthVisualizer()
    print(viz.draw_comparison_chart(baseline_stats, optimized_stats))


def print_dashboard(stats: BandwidthStats, bottlenecks: Optional[List[Bottleneck]] = None):
    """
    Quick helper to print full dashboard.

    Args:
        stats: BandwidthStats to visualize
        bottlenecks: Optional bottlenecks to show
    """
    viz = BandwidthVisualizer()
    print(viz.create_summary_dashboard(stats, bottlenecks))
