"""
Report Generator

Generates professional HTML and PDF reports for bandwidth profiling results.
Suitable for sharing with stakeholders and customers.
"""

from typing import Dict, List, Optional
from datetime import datetime
import json

from .bandwidth_profiler import BandwidthStats
from .bottleneck_detector import Bottleneck, BottleneckSeverity, BottleneckType
from .hardware_validator import HardwareValidator


class ReportGenerator:
    """
    Generates professional reports in HTML and JSON formats.

    Reports include:
    - Executive summary
    - Detailed bandwidth metrics
    - Bottleneck analysis
    - Optimization recommendations
    - Before/after comparisons
    """

    def __init__(self):
        """Initialize report generator."""
        self.report_data = {}

    def generate_html_report(
        self,
        stats: BandwidthStats,
        bottlenecks: Optional[List[Bottleneck]] = None,
        baseline_stats: Optional[BandwidthStats] = None,
        model_name: str = "Model",
        output_file: str = "bandwidth_report.html",
    ) -> str:
        """
        Generate comprehensive HTML report.

        Args:
            stats: BandwidthStats to report
            bottlenecks: Optional list of bottlenecks
            baseline_stats: Optional baseline for comparison
            model_name: Name of the model being profiled
            output_file: Output HTML file path

        Returns:
            Path to generated HTML file
        """
        html = self._build_html_report(stats, bottlenecks, baseline_stats, model_name)

        with open(output_file, 'w') as f:
            f.write(html)

        return output_file

    def _build_html_report(
        self,
        stats: BandwidthStats,
        bottlenecks: Optional[List[Bottleneck]],
        baseline_stats: Optional[BandwidthStats],
        model_name: str,
    ) -> str:
        """Build HTML report content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MemOpt Bandwidth Profiling Report - {model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f5f5f5;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        .header {{
            border-bottom: 3px solid #2563eb;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}

        .header h1 {{
            color: #1e40af;
            font-size: 32px;
            margin-bottom: 10px;
        }}

        .header .meta {{
            color: #666;
            font-size: 14px;
        }}

        .section {{
            margin: 30px 0;
        }}

        .section h2 {{
            color: #1e40af;
            font-size: 24px;
            margin-bottom: 15px;
            border-bottom: 2px solid #e5e7eb;
            padding-bottom: 10px;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .metric-card {{
            background: #f9fafb;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #2563eb;
        }}

        .metric-card .label {{
            color: #666;
            font-size: 14px;
            margin-bottom: 5px;
        }}

        .metric-card .value {{
            font-size: 28px;
            font-weight: bold;
            color: #1e40af;
        }}

        .metric-card .unit {{
            font-size: 18px;
            color: #666;
        }}

        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e5e7eb;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3b82f6, #2563eb);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 14px;
        }}

        .bottleneck-card {{
            background: #fff;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 20px;
            margin: 15px 0;
        }}

        .bottleneck-card.critical {{
            border-left: 4px solid #dc2626;
        }}

        .bottleneck-card.high {{
            border-left: 4px solid #f59e0b;
        }}

        .bottleneck-card.medium {{
            border-left: 4px solid #eab308;
        }}

        .bottleneck-card.low {{
            border-left: 4px solid #22c55e;
        }}

        .recommendation {{
            background: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}

        .recommendation h4 {{
            color: #1e40af;
            margin-bottom: 5px;
        }}

        .comparison-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        .comparison-table th,
        .comparison-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e5e7eb;
        }}

        .comparison-table th {{
            background: #f9fafb;
            font-weight: 600;
            color: #374151;
        }}

        .improvement-positive {{
            color: #22c55e;
            font-weight: bold;
        }}

        .improvement-negative {{
            color: #dc2626;
            font-weight: bold;
        }}

        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            text-align: center;
            color: #666;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 MemOpt Bandwidth Profiling Report</h1>
            <div class="meta">
                <p><strong>Model:</strong> {model_name}</p>
                <p><strong>Generated:</strong> {timestamp}</p>
                <p><strong>GPU:</strong> {stats.gpu_name}</p>
            </div>
        </div>

        {self._generate_methodology_disclaimer(stats)}

        <!-- Executive Summary -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="label">Achieved Bandwidth</div>
                    <div class="value">{stats.achieved_bandwidth_gbs:.1f} <span class="unit">GB/s</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Utilization</div>
                    <div class="value">{stats.bandwidth_utilization_pct:.1f} <span class="unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">Memory Allocated</div>
                    <div class="value">{stats.peak_memory_allocated_gb:.2f} <span class="unit">GB</span></div>
                </div>
                <div class="metric-card">
                    <div class="label">HBM Traffic</div>
                    <div class="value">{stats.total_hbm_traffic_gb:.2f} <span class="unit">GB</span></div>
                </div>
            </div>

            <p style="margin-top: 20px;">
                <strong>Bottleneck Status:</strong>
                {'<span style="color: #dc2626;">Memory-bound ❌</span>' if stats.is_memory_bound else '<span style="color: #22c55e;">Compute-bound ✅</span>'}
            </p>

            <div class="progress-bar">
                <div class="progress-fill" style="width: {stats.bandwidth_utilization_pct}%;">
                    {stats.bandwidth_utilization_pct:.1f}% Bandwidth Utilization
                </div>
            </div>
        </div>

        <!-- Detailed Metrics -->
        <div class="section">
            <h2>📈 Detailed Metrics</h2>
            <table class="comparison-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>Total Execution Time</td>
                    <td>{stats.total_time_seconds:.3f} seconds</td>
                </tr>
                <tr>
                    <td>Theoretical Bandwidth</td>
                    <td>{stats.theoretical_bandwidth_gbs:.0f} GB/s</td>
                </tr>
                <tr>
                    <td>Achieved Bandwidth</td>
                    <td>{stats.achieved_bandwidth_gbs:.1f} GB/s</td>
                </tr>
                <tr>
                    <td>HBM Reads</td>
                    <td>{stats.hbm_read_gb:.2f} GB</td>
                </tr>
                <tr>
                    <td>HBM Writes</td>
                    <td>{stats.hbm_write_gb:.2f} GB</td>
                </tr>
                <tr>
                    <td>Peak Memory</td>
                    <td>{stats.peak_memory_allocated_gb:.2f} GB</td>
                </tr>
                <tr>
                    <td>Memory Stall Time</td>
                    <td>{stats.memory_bound_pct:.1f}%</td>
                </tr>
            </table>
        </div>

        {self._generate_bottleneck_section_html(bottlenecks) if bottlenecks else ''}

        {self._generate_comparison_section_html(baseline_stats, stats) if baseline_stats else ''}

        <div class="footer">
            <p>Generated by <strong>MemOpt</strong> - GPU Memory Bandwidth Profiling Platform</p>
            <p>For more information, visit <a href="https://github.com/yourusername/memopt">github.com/memopt</a></p>
        </div>
    </div>
</body>
</html>"""

        return html

    def _generate_methodology_disclaimer(self, stats: BandwidthStats) -> str:
        """Generate methodology disclaimer section."""
        # Check if hardware validation data exists
        hardware_validated = hasattr(stats, 'hardware_validated') and stats.hardware_validated

        if hardware_validated:
            validation_method = getattr(stats, 'validation_method', 'hardware_counters')
            hardware_dram_gb = getattr(stats, 'hardware_dram_gb', 0)

            return f'''
        <div class="section" style="background: #ecfdf5; border-left: 4px solid #22c55e; padding: 15px; margin: 20px 0;">
            <h3 style="color: #15803d; margin-bottom: 10px;">✅ Hardware-Validated Measurements</h3>
            <p style="margin: 5px 0;">
                <strong>Validation Method:</strong> {validation_method}<br>
                <strong>Actual DRAM Traffic:</strong> {hardware_dram_gb:.2f} GB (measured by GPU hardware counters)<br>
                <strong>Methodology:</strong> Bandwidth measurements validated against NVIDIA GPU hardware performance counters.
                These are direct measurements from the GPU memory controller, not estimates.
            </p>
        </div>
'''
        else:
            return '''
        <div class="section" style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 15px; margin: 20px 0;">
            <h3 style="color: #92400e; margin-bottom: 10px;">⚠️ Measurement Methodology</h3>
            <p style="margin: 5px 0;">
                <strong>Source:</strong> PyTorch Profiler estimates (lower-bound, directionally accurate)<br>
                <strong>Limitation:</strong> Not validated against GPU hardware counters<br>
                <strong>Recommendation:</strong> For hardware-validated measurements, install
                <a href="https://developer.nvidia.com/nsight-compute">Nsight Compute</a> and run with
                <code>--validate-hardware</code> flag.
            </p>
            <p style="margin: 10px 0 5px 0; font-size: 14px; color: #92400e;">
                <strong>Note:</strong> These measurements provide directionally accurate insights for optimization
                but should not be used for absolute performance claims without hardware validation.
            </p>
        </div>
'''

    def _generate_bottleneck_section_html(self, bottlenecks: List[Bottleneck]) -> str:
        """Generate HTML for bottleneck section."""
        if not bottlenecks:
            return ""

        severity_class = {
            BottleneckSeverity.CRITICAL: "critical",
            BottleneckSeverity.HIGH: "high",
            BottleneckSeverity.MEDIUM: "medium",
            BottleneckSeverity.LOW: "low",
        }

        severity_emoji = {
            BottleneckSeverity.CRITICAL: "🔴",
            BottleneckSeverity.HIGH: "🟠",
            BottleneckSeverity.MEDIUM: "🟡",
            BottleneckSeverity.LOW: "🟢",
        }

        html = '<div class="section"><h2>🔍 Bottleneck Analysis</h2>'

        for bottleneck in bottlenecks:
            severity = bottleneck.severity
            emoji = severity_emoji.get(severity, "⚪")
            css_class = severity_class.get(severity, "low")

            html += f'''
            <div class="bottleneck-card {css_class}">
                <h3>{emoji} {bottleneck.type.value.upper()} - {severity.value.title()}</h3>
                <p><strong>Description:</strong> {bottleneck.description}</p>
                {f'<p><strong>Impact:</strong> {bottleneck.impact_pct:.1f}% of execution time wasted</p>' if bottleneck.impact_pct > 0 else ''}
                {f'<p><strong>Potential Speedup:</strong> {bottleneck.estimated_speedup:.2f}x</p>' if bottleneck.estimated_speedup > 1.1 else ''}

                <h4 style="margin-top: 15px;">Recommended Optimizations:</h4>
                <ul style="margin-left: 20px;">
'''

            for opt in bottleneck.suggested_optimizations:
                html += f'<li>{opt}</li>'

            html += '''
                </ul>
            </div>
'''

        html += '</div>'
        return html

    def _generate_comparison_section_html(
        self,
        baseline_stats: BandwidthStats,
        optimized_stats: BandwidthStats,
    ) -> str:
        """Generate HTML for baseline vs optimized comparison."""
        html = '<div class="section"><h2>⚖️ Baseline vs Optimized Comparison</h2>'

        metrics = [
            ("Bandwidth", baseline_stats.achieved_bandwidth_gbs, optimized_stats.achieved_bandwidth_gbs, "GB/s", True),
            ("Utilization", baseline_stats.bandwidth_utilization_pct, optimized_stats.bandwidth_utilization_pct, "%", True),
            ("Memory", baseline_stats.peak_memory_allocated_gb, optimized_stats.peak_memory_allocated_gb, "GB", False),
            ("HBM Traffic", baseline_stats.total_hbm_traffic_gb, optimized_stats.total_hbm_traffic_gb, "GB", False),
        ]

        html += '<table class="comparison-table">'
        html += '<tr><th>Metric</th><th>Baseline</th><th>Optimized</th><th>Change</th></tr>'

        for name, base_val, opt_val, unit, higher_better in metrics:
            if base_val > 0:
                if higher_better:
                    improvement = ((opt_val - base_val) / base_val) * 100
                else:
                    improvement = ((base_val - opt_val) / base_val) * 100

                if improvement > 0:
                    change_class = "improvement-positive"
                    symbol = "↑" if higher_better else "↓"
                else:
                    change_class = "improvement-negative"
                    symbol = "↓" if higher_better else "↑"
            else:
                improvement = 0
                change_class = ""
                symbol = "→"

            html += f'''
            <tr>
                <td>{name}</td>
                <td>{base_val:.2f} {unit}</td>
                <td>{opt_val:.2f} {unit}</td>
                <td class="{change_class}">{symbol} {abs(improvement):.1f}%</td>
            </tr>
'''

        html += '</table></div>'
        return html

    def generate_json_report(
        self,
        stats: BandwidthStats,
        bottlenecks: Optional[List[Bottleneck]] = None,
        baseline_stats: Optional[BandwidthStats] = None,
        model_name: str = "Model",
        output_file: str = "bandwidth_report.json",
    ) -> str:
        """
        Generate JSON report for programmatic access.

        Args:
            stats: BandwidthStats to report
            bottlenecks: Optional list of bottlenecks
            baseline_stats: Optional baseline for comparison
            model_name: Name of the model
            output_file: Output JSON file path

        Returns:
            Path to generated JSON file
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "bandwidth_stats": stats.to_dict(),
        }

        if bottlenecks:
            report["bottlenecks"] = [
                {
                    "type": b.type.value,
                    "severity": b.severity.value,
                    "description": b.description,
                    "impact_pct": b.impact_pct,
                    "estimated_speedup": b.estimated_speedup,
                    "suggested_optimizations": b.suggested_optimizations,
                }
                for b in bottlenecks
            ]

        if baseline_stats:
            report["baseline_stats"] = baseline_stats.to_dict()
            report["comparison"] = {
                "bandwidth_improvement_pct": (
                    (stats.achieved_bandwidth_gbs - baseline_stats.achieved_bandwidth_gbs) /
                    max(baseline_stats.achieved_bandwidth_gbs, 1)
                ) * 100,
                "memory_reduction_gb": (
                    baseline_stats.peak_memory_allocated_gb - stats.peak_memory_allocated_gb
                ),
            }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        return output_file


def generate_html_report(
    stats: BandwidthStats,
    bottlenecks: Optional[List[Bottleneck]] = None,
    baseline_stats: Optional[BandwidthStats] = None,
    model_name: str = "Model",
    output_file: str = "bandwidth_report.html",
) -> str:
    """
    Quick helper to generate HTML report.

    Args:
        stats: BandwidthStats to report
        bottlenecks: Optional bottlenecks
        baseline_stats: Optional baseline for comparison
        model_name: Model name
        output_file: Output file path

    Returns:
        Path to generated HTML file
    """
    generator = ReportGenerator()
    return generator.generate_html_report(stats, bottlenecks, baseline_stats, model_name, output_file)
