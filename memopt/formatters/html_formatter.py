"""
HTML Report Generator for memopt

Generates professional, shareable HTML reports with:
- Inline CSS (no external dependencies)
- Responsive design
- Syntax-highlighted code snippets
- Visual priority indicators
- ROI summary section
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import html


class HTMLReportGenerator:
    """
    Generate professional HTML optimization reports.

    Creates self-contained HTML documents with inline CSS
    for easy sharing and viewing.
    """

    # CSS styles embedded in the document
    CSS_STYLES = """
        :root {
            --primary-color: #2563eb;
            --success-color: #16a34a;
            --warning-color: #d97706;
            --danger-color: #dc2626;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: #e2e8f0;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .header .subtitle {
            opacity: 0.9;
            font-size: 1rem;
        }

        .card {
            background: var(--card-bg);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }

        .card h2 {
            font-size: 1.25rem;
            margin-bottom: 1rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card h3 {
            font-size: 1rem;
            margin: 1rem 0 0.5rem 0;
            color: var(--text-secondary);
        }

        .roi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .roi-stat {
            background: var(--bg-color);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }

        .roi-stat .value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--success-color);
        }

        .roi-stat .label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }

        .recommendation {
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .recommendation.priority-high {
            border-left: 4px solid var(--danger-color);
        }

        .recommendation.priority-medium {
            border-left: 4px solid var(--warning-color);
        }

        .recommendation.priority-low {
            border-left: 4px solid var(--success-color);
        }

        .recommendation-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }

        .recommendation-title {
            font-weight: 600;
            font-size: 1rem;
        }

        .priority-badge {
            font-size: 0.75rem;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            text-transform: uppercase;
        }

        .priority-badge.high {
            background: #fef2f2;
            color: var(--danger-color);
        }

        .priority-badge.medium {
            background: #fffbeb;
            color: var(--warning-color);
        }

        .priority-badge.low {
            background: #f0fdf4;
            color: var(--success-color);
        }

        .impact-badge {
            background: #eff6ff;
            color: var(--primary-color);
            font-size: 0.875rem;
            font-weight: 600;
            padding: 0.25rem 0.75rem;
            border-radius: 6px;
            margin-left: 0.5rem;
        }

        .code-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-top: 1rem;
        }

        @media (max-width: 768px) {
            .code-comparison {
                grid-template-columns: 1fr;
            }
        }

        .code-block {
            background: #1e293b;
            color: #e2e8f0;
            border-radius: 8px;
            padding: 1rem;
            font-family: 'SF Mono', Consolas, Monaco, 'Andale Mono', monospace;
            font-size: 0.875rem;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-word;
        }

        .code-block-header {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        .code-before .code-block-header {
            color: var(--danger-color);
        }

        .code-after .code-block-header {
            color: var(--success-color);
        }

        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }

        .summary-table th,
        .summary-table td {
            text-align: left;
            padding: 0.75rem;
            border-bottom: 1px solid var(--border-color);
        }

        .summary-table th {
            font-weight: 600;
            color: var(--text-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .summary-table tr:last-child td {
            border-bottom: none;
        }

        .footer {
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--border-color);
        }

        .access-pattern {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.5rem;
        }

        .pattern-tag {
            background: var(--bg-color);
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 0.25rem 0.75rem;
            font-size: 0.875rem;
        }

        .pattern-tag.issue {
            background: #fef2f2;
            border-color: #fecaca;
            color: var(--danger-color);
        }
    """

    def __init__(self):
        """Initialize HTML report generator."""
        pass

    def generate(
        self,
        phase2_report: Any,
        roi_report: Optional[Any] = None,
        model_name: Optional[str] = None,
        title: Optional[str] = None
    ) -> str:
        """
        Generate complete HTML report.

        Args:
            phase2_report: Phase2Report from optimization synthesis
            roi_report: Optional ROIReport from business calculator
            model_name: Optional model name
            title: Optional custom title

        Returns:
            Complete HTML document as string
        """
        model = model_name or getattr(phase2_report, 'kernel_name', 'Model')
        report_title = title or f"Memory Optimization Report: {model}"

        html_parts = [
            self._generate_doctype(),
            self._generate_head(report_title),
            '<body>',
            '<div class="container">',
            self._generate_header(report_title, model),
        ]

        # ROI Section (if available)
        if roi_report:
            html_parts.append(self._generate_roi_section(roi_report))

        # Impact Score Summary
        if hasattr(phase2_report, 'impact_score'):
            html_parts.append(self._generate_impact_section(phase2_report.impact_score))

        # Access Patterns Summary
        if hasattr(phase2_report, 'access_patterns'):
            html_parts.append(self._generate_access_patterns_section(phase2_report.access_patterns))

        # Recommendations
        if hasattr(phase2_report, 'recommendations') and phase2_report.recommendations:
            html_parts.append(self._generate_recommendations_section(phase2_report.recommendations))

        # Optimization Candidates Summary
        if hasattr(phase2_report, 'optimization_candidates') and phase2_report.optimization_candidates:
            html_parts.append(self._generate_candidates_section(phase2_report.optimization_candidates))

        html_parts.extend([
            self._generate_footer(),
            '</div>',
            '</body>',
            '</html>'
        ])

        return '\n'.join(html_parts)

    def _generate_doctype(self) -> str:
        """Generate HTML doctype and opening."""
        return '<!DOCTYPE html>\n<html lang="en">'

    def _generate_head(self, title: str) -> str:
        """Generate HTML head section with embedded CSS."""
        escaped_title = html.escape(title)
        return f'''<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{escaped_title}</title>
    <style>
{self.CSS_STYLES}
    </style>
</head>'''

    def _generate_header(self, title: str, model: str) -> str:
        """Generate page header."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f'''<div class="header">
    <h1>{html.escape(title)}</h1>
    <div class="subtitle">Generated by memopt | {timestamp}</div>
</div>'''

    def _generate_roi_section(self, roi_report: Any) -> str:
        """Generate ROI summary section."""
        monthly = getattr(roi_report, 'monthly_cost_savings', 0)
        annual = getattr(roi_report, 'annual_cost_savings', 0)
        speedup = getattr(roi_report, 'total_speedup_pct', 0)
        hours = getattr(roi_report, 'monthly_hours_saved', 0)

        return f'''<div class="card">
    <h2>ROI Analysis</h2>
    <div class="roi-grid">
        <div class="roi-stat">
            <div class="value">{speedup:.1f}%</div>
            <div class="label">Total Speedup</div>
        </div>
        <div class="roi-stat">
            <div class="value">{hours:,.0f}</div>
            <div class="label">GPU Hours Saved/Month</div>
        </div>
        <div class="roi-stat">
            <div class="value">${monthly:,.0f}</div>
            <div class="label">Monthly Savings</div>
        </div>
        <div class="roi-stat">
            <div class="value">${annual:,.0f}</div>
            <div class="label">Annual Savings</div>
        </div>
    </div>
</div>'''

    def _generate_impact_section(self, impact_score: Any) -> str:
        """Generate impact score section."""
        kernel = getattr(impact_score, 'kernel_name', 'Unknown')
        priority = getattr(impact_score, 'priority', 'MEDIUM')
        recoverable = getattr(impact_score, 'recoverable_gpu_time_pct', 0)
        time_weight = getattr(impact_score, 'time_weight_pct', 0)
        inefficiency = getattr(impact_score, 'inefficiency_factor', 0) * 100

        priority_class = priority.lower()

        return f'''<div class="card">
    <h2>Impact Analysis</h2>
    <table class="summary-table">
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Kernel</td>
            <td>{html.escape(str(kernel))}</td>
        </tr>
        <tr>
            <td>Priority</td>
            <td><span class="priority-badge {priority_class}">{priority}</span></td>
        </tr>
        <tr>
            <td>GPU Time Weight</td>
            <td>{time_weight:.1f}%</td>
        </tr>
        <tr>
            <td>Inefficiency Factor</td>
            <td>{inefficiency:.1f}%</td>
        </tr>
        <tr>
            <td>Recoverable GPU Time</td>
            <td>{recoverable:.1f}%</td>
        </tr>
    </table>
</div>'''

    def _generate_access_patterns_section(self, access_patterns: Any) -> str:
        """Generate access patterns summary section."""
        html_content = '<div class="card"><h2>Access Pattern Analysis</h2>'

        # Coalescing info
        if hasattr(access_patterns, 'coalescing'):
            coal = access_patterns.coalescing
            efficiency = getattr(coal, 'efficiency_pct', 0)
            issue_class = 'issue' if efficiency < 80 else ''
            html_content += f'''
    <h3>Memory Coalescing</h3>
    <div class="access-pattern">
        <span class="pattern-tag {issue_class}">Efficiency: {efficiency:.1f}%</span>
    </div>'''

        # Redundant fetch info
        if hasattr(access_patterns, 'redundant_fetch'):
            rf = access_patterns.redundant_fetch
            ratio = getattr(rf, 'redundancy_ratio', 0)
            issue_class = 'issue' if ratio > 1.5 else ''
            html_content += f'''
    <h3>Cache Utilization</h3>
    <div class="access-pattern">
        <span class="pattern-tag {issue_class}">Redundancy Ratio: {ratio:.2f}x</span>
    </div>'''

        # Cache thrashing info
        if hasattr(access_patterns, 'cache_thrashing'):
            ct = access_patterns.cache_thrashing
            if hasattr(ct, 'is_thrashing') and ct.is_thrashing:
                html_content += '''
    <h3>Cache Behavior</h3>
    <div class="access-pattern">
        <span class="pattern-tag issue">Cache Thrashing Detected</span>
    </div>'''

        html_content += '</div>'
        return html_content

    def _generate_recommendations_section(self, recommendations: List[Any]) -> str:
        """Generate recommendations section with code snippets."""
        html_content = '<div class="card"><h2>Recommendations</h2>'

        for rec in recommendations:
            title = getattr(rec, 'title', 'Optimization')
            priority = getattr(rec, 'priority', 'MEDIUM')
            impact = getattr(rec, 'estimated_impact_pct', 0)
            code_before = getattr(rec, 'code_before', '')
            code_after = getattr(rec, 'code_after', '')
            full_text = getattr(rec, 'full_text', '')

            priority_class = priority.lower()

            html_content += f'''
    <div class="recommendation priority-{priority_class}">
        <div class="recommendation-header">
            <span class="recommendation-title">{html.escape(title)}</span>
            <div>
                <span class="priority-badge {priority_class}">{priority}</span>
                <span class="impact-badge">{impact:.0f}% improvement</span>
            </div>
        </div>'''

            # Add description if available
            if full_text:
                # Extract first paragraph as description
                desc_lines = full_text.split('\n')
                for line in desc_lines:
                    line = line.strip()
                    if line and not line.startswith(('│', '┌', '└', '─', '├')):
                        html_content += f'<p>{html.escape(line)}</p>'
                        break

            # Add code comparison
            if code_before or code_after:
                html_content += '<div class="code-comparison">'

                if code_before:
                    html_content += f'''
        <div class="code-before">
            <div class="code-block-header">Before</div>
            <div class="code-block">{html.escape(code_before)}</div>
        </div>'''

                if code_after:
                    html_content += f'''
        <div class="code-after">
            <div class="code-block-header">After</div>
            <div class="code-block">{html.escape(code_after)}</div>
        </div>'''

                html_content += '</div>'

            html_content += '</div>'

        html_content += '</div>'
        return html_content

    def _generate_candidates_section(self, candidates: List[Any]) -> str:
        """Generate optimization candidates summary table."""
        html_content = '''<div class="card">
    <h2>All Optimization Candidates</h2>
    <table class="summary-table">
        <tr>
            <th>Optimization</th>
            <th>Priority</th>
            <th>Expected Impact</th>
            <th>Confidence</th>
        </tr>'''

        for candidate in candidates:
            opt_type = getattr(candidate, 'optimization_type', 'Unknown')
            if hasattr(opt_type, 'value'):
                opt_name = opt_type.value.replace('_', ' ').title()
            else:
                opt_name = str(opt_type).replace('_', ' ').title()

            priority = getattr(candidate, 'priority', 'MEDIUM')
            impact = getattr(candidate, 'expected_impact_pct', 0)
            confidence = getattr(candidate, 'confidence', 0) * 100
            priority_class = priority.lower()

            html_content += f'''
        <tr>
            <td>{html.escape(opt_name)}</td>
            <td><span class="priority-badge {priority_class}">{priority}</span></td>
            <td>{impact:.1f}%</td>
            <td>{confidence:.0f}%</td>
        </tr>'''

        html_content += '''
    </table>
</div>'''
        return html_content

    def _generate_footer(self) -> str:
        """Generate page footer."""
        return '''<div class="footer">
    <p>Generated by memopt - GPU Memory Optimization Platform</p>
    <p>For more information, visit the documentation or run <code>memopt --help</code></p>
</div>'''
