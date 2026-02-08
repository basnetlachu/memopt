"""
JSON Exporter for memopt

Exports optimization reports in machine-readable JSON format
for integration with CI/CD, dashboards, and automation tools.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class JSONExporter:
    """
    Export optimization reports as structured JSON.

    Produces machine-readable output suitable for:
    - CI/CD pipeline integration
    - Dashboard ingestion
    - API responses
    - Automated decision making
    """

    def __init__(self, pretty: bool = True):
        """
        Initialize JSON exporter.

        Args:
            pretty: Whether to format JSON with indentation
        """
        self.pretty = pretty

    def export(
        self,
        phase2_report: Any,
        roi_report: Optional[Any] = None,
        model_name: Optional[str] = None,
        include_metadata: bool = True
    ) -> str:
        """
        Export complete analysis to JSON string.

        Args:
            phase2_report: Phase2Report from optimization synthesis
            roi_report: Optional ROIReport from business calculator
            model_name: Optional model name
            include_metadata: Whether to include export metadata

        Returns:
            JSON string
        """
        data = self.to_dict(phase2_report, roi_report, model_name, include_metadata)
        return self._serialize(data)

    def to_dict(
        self,
        phase2_report: Any,
        roi_report: Optional[Any] = None,
        model_name: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Convert reports to dictionary structure.

        Args:
            phase2_report: Phase2Report from optimization synthesis
            roi_report: Optional ROIReport from business calculator
            model_name: Optional model name
            include_metadata: Whether to include export metadata

        Returns:
            Dictionary ready for JSON serialization
        """
        result: Dict[str, Any] = {}

        # Metadata
        if include_metadata:
            result['metadata'] = {
                'exported_at': datetime.now().isoformat(),
                'format_version': '1.0',
                'generator': 'memopt',
            }

        # Model info
        result['model'] = {
            'name': model_name or getattr(phase2_report, 'kernel_name', 'unknown'),
        }

        # Impact score
        if hasattr(phase2_report, 'impact_score'):
            result['impact_score'] = self._serialize_impact_score(phase2_report.impact_score)

        # Access patterns
        if hasattr(phase2_report, 'access_patterns'):
            result['access_patterns'] = self._serialize_access_patterns(phase2_report.access_patterns)

        # Optimization candidates
        if hasattr(phase2_report, 'optimization_candidates'):
            result['optimization_candidates'] = [
                self._serialize_candidate(c)
                for c in phase2_report.optimization_candidates
            ]

        # Recommendations
        if hasattr(phase2_report, 'recommendations'):
            result['recommendations'] = [
                self._serialize_recommendation(r)
                for r in phase2_report.recommendations
            ]

        # ROI analysis
        if roi_report:
            result['roi_analysis'] = self._serialize_roi(roi_report)

        # Summary
        result['summary'] = self._generate_summary(phase2_report, roi_report)

        return result

    def _serialize(self, data: Dict[str, Any]) -> str:
        """Serialize dictionary to JSON string."""
        if self.pretty:
            return json.dumps(data, indent=2, default=self._json_default)
        return json.dumps(data, default=self._json_default)

    def _json_default(self, obj: Any) -> Any:
        """Handle non-serializable objects."""
        if isinstance(obj, Enum):
            return obj.value
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return str(obj)

    def _serialize_impact_score(self, impact: Any) -> Dict[str, Any]:
        """Serialize ImpactScore to dict."""
        return {
            'kernel_name': getattr(impact, 'kernel_name', 'unknown'),
            'base_impact_score': getattr(impact, 'base_impact_score', 0),
            'final_impact_score': getattr(impact, 'final_impact_score', 0),
            'time_weight_pct': getattr(impact, 'time_weight_pct', 0),
            'inefficiency_factor': getattr(impact, 'inefficiency_factor', 0),
            'dram_traffic_gb': getattr(impact, 'dram_traffic_gb', 0),
            'total_expected_improvement_pct': getattr(impact, 'total_expected_improvement_pct', 0),
            'recoverable_gpu_time_pct': getattr(impact, 'recoverable_gpu_time_pct', 0),
            'num_optimizations_applicable': getattr(impact, 'num_optimizations_applicable', 0),
            'priority': getattr(impact, 'priority', 'MEDIUM'),
        }

    def _serialize_access_patterns(self, patterns: Any) -> Dict[str, Any]:
        """Serialize AccessPatternReport to dict."""
        result: Dict[str, Any] = {}

        if hasattr(patterns, 'coalescing'):
            coal = patterns.coalescing
            result['coalescing'] = {
                'efficiency_pct': getattr(coal, 'efficiency_pct', 0),
                'transactions_actual': getattr(coal, 'transactions_actual', 0),
                'transactions_ideal': getattr(coal, 'transactions_ideal', 0),
                'strided_accesses': getattr(coal, 'strided_accesses', 0),
                'random_accesses': getattr(coal, 'random_accesses', 0),
            }

        if hasattr(patterns, 'redundant_fetch'):
            rf = patterns.redundant_fetch
            result['redundant_fetch'] = {
                'redundancy_ratio': getattr(rf, 'redundancy_ratio', 0),
                'unique_bytes': getattr(rf, 'unique_bytes', 0),
                'total_bytes_fetched': getattr(rf, 'total_bytes_fetched', 0),
                'wasted_bandwidth_pct': getattr(rf, 'wasted_bandwidth_pct', 0),
            }

        if hasattr(patterns, 'cache_thrashing'):
            ct = patterns.cache_thrashing
            result['cache_thrashing'] = {
                'is_thrashing': getattr(ct, 'is_thrashing', False),
                'l2_hit_rate': getattr(ct, 'l2_hit_rate', 0),
                'l1_hit_rate': getattr(ct, 'l1_hit_rate', 0),
                'working_set_mb': getattr(ct, 'working_set_mb', 0),
            }

        if hasattr(patterns, 'overall_pattern'):
            result['overall_pattern'] = str(getattr(patterns, 'overall_pattern', 'unknown'))

        return result

    def _serialize_candidate(self, candidate: Any) -> Dict[str, Any]:
        """Serialize OptimizationCandidate to dict."""
        opt_type = getattr(candidate, 'optimization_type', 'unknown')
        if hasattr(opt_type, 'value'):
            opt_type_str = opt_type.value
        else:
            opt_type_str = str(opt_type)

        return {
            'rule_name': getattr(candidate, 'rule_name', ''),
            'optimization_type': opt_type_str,
            'description': getattr(candidate, 'description', ''),
            'expected_impact_pct': getattr(candidate, 'expected_impact_pct', 0),
            'option1_action': getattr(candidate, 'option1_action', ''),
            'option2_recommendation': getattr(candidate, 'option2_recommendation', ''),
            'option3_kernel': getattr(candidate, 'option3_kernel', None),
            'priority': getattr(candidate, 'priority', 'MEDIUM'),
            'confidence': getattr(candidate, 'confidence', 0),
        }

    def _serialize_recommendation(self, rec: Any) -> Dict[str, Any]:
        """Serialize FormattedRecommendation to dict."""
        opt_type = getattr(rec, 'optimization_type', 'unknown')
        if hasattr(opt_type, 'value'):
            opt_type_str = opt_type.value
        else:
            opt_type_str = str(opt_type)

        return {
            'title': getattr(rec, 'title', ''),
            'full_text': getattr(rec, 'full_text', ''),
            'code_before': getattr(rec, 'code_before', ''),
            'code_after': getattr(rec, 'code_after', ''),
            'estimated_impact_pct': getattr(rec, 'estimated_impact_pct', 0),
            'priority': getattr(rec, 'priority', 'MEDIUM'),
            'applies_to_kernel': getattr(rec, 'applies_to_kernel', ''),
            'optimization_type': opt_type_str,
        }

    def _serialize_roi(self, roi: Any) -> Dict[str, Any]:
        """Serialize ROIReport to dict."""
        if hasattr(roi, 'to_dict'):
            return roi.to_dict()

        return {
            'total_speedup_pct': getattr(roi, 'total_speedup_pct', 0),
            'gpu_cost_per_hour': getattr(roi, 'gpu_cost_per_hour', 0),
            'gpu_count': getattr(roi, 'gpu_count', 1),
            'monthly_hours_saved': getattr(roi, 'monthly_hours_saved', 0),
            'monthly_cost_savings': getattr(roi, 'monthly_cost_savings', 0),
            'annual_cost_savings': getattr(roi, 'annual_cost_savings', 0),
            'optimization_breakdown': getattr(roi, 'optimization_breakdown', []),
            'calculated_at': getattr(roi, 'calculated_at', ''),
            'model_name': getattr(roi, 'model_name', None),
        }

    def _generate_summary(
        self,
        phase2_report: Any,
        roi_report: Optional[Any]
    ) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary: Dict[str, Any] = {
            'total_optimizations': 0,
            'high_priority_count': 0,
            'medium_priority_count': 0,
            'low_priority_count': 0,
            'total_expected_improvement_pct': 0,
        }

        if hasattr(phase2_report, 'optimization_candidates'):
            candidates = phase2_report.optimization_candidates
            summary['total_optimizations'] = len(candidates)

            for c in candidates:
                priority = getattr(c, 'priority', 'MEDIUM')
                if priority == 'HIGH':
                    summary['high_priority_count'] += 1
                elif priority == 'MEDIUM':
                    summary['medium_priority_count'] += 1
                else:
                    summary['low_priority_count'] += 1

                summary['total_expected_improvement_pct'] += getattr(c, 'expected_impact_pct', 0)

        if hasattr(phase2_report, 'impact_score'):
            impact = phase2_report.impact_score
            summary['priority'] = getattr(impact, 'priority', 'MEDIUM')
            summary['recoverable_gpu_time_pct'] = getattr(impact, 'recoverable_gpu_time_pct', 0)

        if roi_report:
            summary['monthly_cost_savings'] = getattr(roi_report, 'monthly_cost_savings', 0)
            summary['annual_cost_savings'] = getattr(roi_report, 'annual_cost_savings', 0)

        return summary

    def export_minimal(self, phase2_report: Any) -> str:
        """
        Export minimal summary for quick checks.

        Returns just key metrics without full details.
        """
        summary = self._generate_summary(phase2_report, None)

        if hasattr(phase2_report, 'impact_score'):
            impact = phase2_report.impact_score
            summary['kernel_name'] = getattr(impact, 'kernel_name', 'unknown')

        return self._serialize(summary)
