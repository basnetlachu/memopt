"""
memopt drift/regression alert system.

Detects when previously-optimized GPU processes degrade
below their post-optimization utilization baseline.

Usage:
    from memopt.alerts import DriftDetector, AlertStore, AlertNotifier, DriftAlert
"""
from .alert_store import AlertStore, DriftAlert

__all__ = [
    "AlertStore",
    "DriftAlert",
]
