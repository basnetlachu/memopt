"""
memopt daemon module - Background GPU monitoring and optimization service.
"""

from .daemon_service import MemoptDaemon, DaemonConfig
from .process_monitor import ProcessMonitor, GPUProcess, GPUState
from .scheduler import SafeScheduler, OptimizationTask, OptimizationState
from .reporter import DashboardReporter, get_reporter, init_reporter

# Zero-touch scan API. The apply / zero-touch / ROI helpers were
# removed in commit c5705a3 ("optimizations deleted……") and are not
# being revived (per docs/pillar_revival_plan.md M7). Their __all__
# entries have been dropped accordingly.
from .scanner import GPUScanner
from .scanner import GPUProcess as ScannedGPUProcess
from .process_inspector import ProcessInspector, ProcessProfile
from .report import ScanReporter

# Control plane reporter
from .reporter import ControlPlaneReporter

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
    # Scan
    "GPUScanner",
    "ScannedGPUProcess",
    "ProcessInspector",
    "ProcessProfile",
    "ScanReporter",
    # Control plane reporter
    "ControlPlaneReporter",
]
