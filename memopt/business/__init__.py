"""
memopt.business - Business Intelligence and ROI Calculation

Provides ROI calculations, cost analysis, and business metrics
for GPU memory optimization decisions.
"""

from .roi_calculator import ROICalculator, ROIReport

__all__ = ['ROICalculator', 'ROIReport']
