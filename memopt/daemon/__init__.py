"""
memopt daemon module - Background GPU monitoring and optimization service.
"""

from .daemon_service import MemoptDaemon, DaemonConfig
from .process_monitor import ProcessMonitor, GPUProcess, GPUState
from .scheduler import SafeScheduler, OptimizationTask, OptimizationState
from .reporter import DashboardReporter, get_reporter, init_reporter

# Zero-touch scan/apply API
from .scanner import GPUScanner
from .scanner import GPUProcess as ScannedGPUProcess
from .process_inspector import ProcessInspector, ProcessProfile
from .report import ScanReporter
from .apply import ApplyEngine, ApplyResult

__all__ = [
    "MemoptDaemon",
    "DaemonConfig",
    "ProcessMonitor",
    "GPUProcess",
    "GPUState",
    "SafeScheduler",
    "OptimizationTask",
    "OptimizationState",
    "DashboardReporter",
    "get_reporter",
    "init_reporter",
    # Scan/apply
    "GPUScanner",
    "ScannedGPUProcess",
    "ProcessInspector",
    "ProcessProfile",
    "ScanReporter",
    "ApplyEngine",
    "ApplyResult",
    # Zero-touch daemon + ROI
    "ZeroTouchDaemon",
    "ZeroDaemonConfig",
    "OptimizationEvent",
    "ROICalculator",
    "ROIEstimate",
    # Control plane reporter
    "ControlPlaneReporter",
]

# Zero-touch daemon + ROI
from .zero_touch import ZeroTouchDaemon, OptimizationEvent
from .zero_touch import DaemonConfig as ZeroDaemonConfig
from .roi_calculator import ROICalculator, ROIEstimate
# Control plane reporter
from .reporter import ControlPlaneReporter
