"""
ROI Calculator for GPU Memory Optimization

Calculates return on investment metrics based on optimization results,
GPU costs, and projected time savings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


@dataclass
class ROIReport:
    """
    ROI analysis report for optimization recommendations.

    Contains cost savings projections based on speedup percentages,
    GPU costs, and fleet size.
    """

    # Input parameters
    total_speedup_pct: float              # Combined speedup from all optimizations
    gpu_cost_per_hour: float              # Cost per GPU hour in dollars
    gpu_count: int                        # Number of GPUs in fleet

    # Calculated metrics
    monthly_hours_saved: float            # GPU hours saved per month
    monthly_cost_savings: float           # Dollar savings per month
    annual_cost_savings: float            # Dollar savings per year

    # Per-optimization breakdown
    optimization_breakdown: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    calculated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: Optional[str] = None

    def __str__(self) -> str:
        """Generate formatted ROI summary."""
        lines = [
            "",
            "=" * 60,
            "ROI ANALYSIS",
            "=" * 60,
            "",
            f"Total Speedup: {self.total_speedup_pct:.1f}%",
            f"GPU Fleet Size: {self.gpu_count} GPUs",
            f"GPU Cost: ${self.gpu_cost_per_hour:.2f}/hour",
            "",
            "-" * 60,
            "PROJECTED SAVINGS",
            "-" * 60,
            f"Monthly GPU Hours Saved: {self.monthly_hours_saved:,.1f} hours",
            f"Monthly Cost Savings: ${self.monthly_cost_savings:,.2f}",
            f"Annual Cost Savings: ${self.annual_cost_savings:,.2f}",
            "",
        ]

        if self.optimization_breakdown:
            lines.extend([
                "-" * 60,
                "BREAKDOWN BY OPTIMIZATION",
                "-" * 60,
            ])
            for opt in self.optimization_breakdown:
                lines.append(
                    f"  {opt['name']:<30} "
                    f"{opt['speedup_pct']:>5.1f}% -> "
                    f"${opt['monthly_savings']:>10,.2f}/mo"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_speedup_pct': self.total_speedup_pct,
            'gpu_cost_per_hour': self.gpu_cost_per_hour,
            'gpu_count': self.gpu_count,
            'monthly_hours_saved': self.monthly_hours_saved,
            'monthly_cost_savings': self.monthly_cost_savings,
            'annual_cost_savings': self.annual_cost_savings,
            'optimization_breakdown': self.optimization_breakdown,
            'calculated_at': self.calculated_at,
            'model_name': self.model_name,
        }


class ROICalculator:
    """
    Calculate ROI metrics for GPU memory optimization.

    Converts speedup percentages to dollar savings based on
    GPU fleet size and hourly costs.

    Calculation methodology:
    - Speedup translates to reduced GPU time for same workload
    - Hours saved = (speedup_pct / 100) * base_hours * gpu_count
    - Cost savings = hours_saved * gpu_cost_per_hour

    Example:
        A 20% speedup means completing work in 80% of original time,
        saving 20% of GPU hours.
    """

    # Default assumptions
    DEFAULT_GPU_COST_PER_HOUR = 3.00      # Conservative cloud GPU pricing
    DEFAULT_OPERATING_HOURS_PER_MONTH = 720  # 24/7 operation
    DEFAULT_GPU_UTILIZATION = 0.80        # 80% average utilization

    def __init__(
        self,
        gpu_cost_per_hour: float = DEFAULT_GPU_COST_PER_HOUR,
        gpu_count: int = 1,
        operating_hours_per_month: float = DEFAULT_OPERATING_HOURS_PER_MONTH,
        utilization_factor: float = DEFAULT_GPU_UTILIZATION
    ):
        """
        Initialize ROI calculator with fleet parameters.

        Args:
            gpu_cost_per_hour: Cost per GPU hour in dollars
            gpu_count: Number of GPUs in the fleet
            operating_hours_per_month: Hours GPUs run per month (default 720 = 24/7)
            utilization_factor: Average GPU utilization (0-1, default 0.8)
        """
        self.gpu_cost_per_hour = gpu_cost_per_hour
        self.gpu_count = gpu_count
        self.operating_hours_per_month = operating_hours_per_month
        self.utilization_factor = utilization_factor

    def calculate_from_phase2_report(
        self,
        phase2_report: Any,
        model_name: Optional[str] = None
    ) -> ROIReport:
        """
        Calculate ROI from a Phase 2 optimization report.

        Args:
            phase2_report: Phase2Report from optimization synthesis
            model_name: Optional model name for the report

        Returns:
            ROIReport with calculated savings
        """
        # Extract optimization candidates and their impacts
        breakdown = []
        total_speedup = 0.0

        if hasattr(phase2_report, 'optimization_candidates'):
            for candidate in phase2_report.optimization_candidates:
                speedup = getattr(candidate, 'expected_impact_pct', 0.0)
                total_speedup += speedup

                # Calculate savings for this optimization
                monthly_hours = self._calculate_hours_saved(speedup)
                monthly_savings = monthly_hours * self.gpu_cost_per_hour

                opt_type = getattr(candidate, 'optimization_type', 'unknown')
                if hasattr(opt_type, 'value'):
                    opt_name = opt_type.value
                else:
                    opt_name = str(opt_type)

                breakdown.append({
                    'name': opt_name,
                    'description': getattr(candidate, 'description', ''),
                    'speedup_pct': speedup,
                    'priority': getattr(candidate, 'priority', 'MEDIUM'),
                    'monthly_hours_saved': monthly_hours,
                    'monthly_savings': monthly_savings,
                })

        # Also check impact_score for recoverable GPU time
        if hasattr(phase2_report, 'impact_score'):
            impact = phase2_report.impact_score
            recoverable = getattr(impact, 'recoverable_gpu_time_pct', 0.0)
            if recoverable > total_speedup:
                total_speedup = recoverable

        return self.calculate(total_speedup, breakdown, model_name)

    def calculate(
        self,
        total_speedup_pct: float,
        breakdown: Optional[List[Dict[str, Any]]] = None,
        model_name: Optional[str] = None
    ) -> ROIReport:
        """
        Calculate ROI from a speedup percentage.

        Args:
            total_speedup_pct: Total expected speedup percentage
            breakdown: Optional per-optimization breakdown
            model_name: Optional model name

        Returns:
            ROIReport with calculated savings
        """
        monthly_hours_saved = self._calculate_hours_saved(total_speedup_pct)
        monthly_cost_savings = monthly_hours_saved * self.gpu_cost_per_hour
        annual_cost_savings = monthly_cost_savings * 12

        return ROIReport(
            total_speedup_pct=total_speedup_pct,
            gpu_cost_per_hour=self.gpu_cost_per_hour,
            gpu_count=self.gpu_count,
            monthly_hours_saved=monthly_hours_saved,
            monthly_cost_savings=monthly_cost_savings,
            annual_cost_savings=annual_cost_savings,
            optimization_breakdown=breakdown or [],
            model_name=model_name,
        )

    def _calculate_hours_saved(self, speedup_pct: float) -> float:
        """
        Calculate GPU hours saved per month from speedup percentage.

        A speedup of X% means completing the same work in (100-X)% of the time,
        thus saving X% of GPU hours.
        """
        # Effective hours being used for compute
        effective_hours = (
            self.operating_hours_per_month *
            self.gpu_count *
            self.utilization_factor
        )

        # Hours saved = speedup percentage of effective hours
        hours_saved = effective_hours * (speedup_pct / 100.0)

        return hours_saved

    def calculate_breakeven(
        self,
        implementation_cost: float,
        total_speedup_pct: float
    ) -> Dict[str, Any]:
        """
        Calculate break-even timeline for optimization investment.

        Args:
            implementation_cost: One-time cost to implement optimizations
            total_speedup_pct: Expected speedup percentage

        Returns:
            Dict with breakeven analysis
        """
        roi = self.calculate(total_speedup_pct)

        if roi.monthly_cost_savings <= 0:
            return {
                'breakeven_months': float('inf'),
                'breakeven_possible': False,
                'monthly_savings': 0,
                'implementation_cost': implementation_cost,
            }

        breakeven_months = implementation_cost / roi.monthly_cost_savings

        return {
            'breakeven_months': breakeven_months,
            'breakeven_possible': True,
            'monthly_savings': roi.monthly_cost_savings,
            'implementation_cost': implementation_cost,
            'first_year_net_savings': roi.annual_cost_savings - implementation_cost,
        }

    def compare_scenarios(
        self,
        scenarios: List[Dict[str, float]]
    ) -> List[ROIReport]:
        """
        Compare multiple optimization scenarios.

        Args:
            scenarios: List of dicts with 'name' and 'speedup_pct' keys

        Returns:
            List of ROIReports for comparison
        """
        reports = []
        for scenario in scenarios:
            report = self.calculate(
                total_speedup_pct=scenario.get('speedup_pct', 0),
                model_name=scenario.get('name', 'Unnamed Scenario')
            )
            reports.append(report)
        return reports
