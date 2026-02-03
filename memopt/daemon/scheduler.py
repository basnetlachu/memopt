"""
Safe optimization scheduler.
Manages when and how to apply optimizations to GPU workloads.
"""

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from enum import Enum

from .process_monitor import ProcessMonitor, GPUProcess, GPUState

logger = logging.getLogger("memopt.daemon")


class OptimizationState(Enum):
    """State of an optimization attempt."""
    PENDING = "pending"
    PROFILING = "profiling"
    READY = "ready"  # Profiled, ready to optimize
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class OptimizationTask:
    """Represents a pending optimization task."""
    pid: int
    process_name: str
    gpu_index: int
    state: OptimizationState = OptimizationState.PENDING
    memory_mb: float = 0.0
    profile_result: Optional[Dict] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class SafeScheduler:
    """
    Scheduler that ensures optimizations are applied safely.

    Safety rules:
    1. Only optimize processes that have been stable for N checks
    2. Only optimize when GPU utilization is below threshold
    3. Never optimize during memory pressure
    4. Only one optimization at a time per GPU
    5. Back off if optimization fails
    """

    def __init__(
        self,
        monitor: ProcessMonitor,
        utilization_threshold: float = 80.0,
        memory_threshold: float = 85.0,
        cooldown_seconds: float = 60.0,
        max_retries: int = 3
    ):
        """
        Args:
            monitor: ProcessMonitor instance
            utilization_threshold: Max GPU utilization % for optimization
            memory_threshold: Max memory usage % for optimization
            cooldown_seconds: Wait time after failed optimization
            max_retries: Max retry attempts per process
        """
        self.monitor = monitor
        self.utilization_threshold = utilization_threshold
        self.memory_threshold = memory_threshold
        self.cooldown_seconds = cooldown_seconds
        self.max_retries = max_retries

        self._tasks: Dict[int, OptimizationTask] = {}
        self._active_gpu: Optional[int] = None  # GPU currently being optimized
        self._failed_attempts: Dict[int, int] = {}  # PID -> failure count
        self._last_failure: Optional[datetime] = None

    def add_task(self, proc: GPUProcess) -> OptimizationTask:
        """Add a process to the optimization queue."""
        if proc.pid in self._tasks:
            return self._tasks[proc.pid]

        task = OptimizationTask(
            pid=proc.pid,
            process_name=proc.name,
            gpu_index=proc.gpu_index,
            memory_mb=proc.memory_used_mb
        )
        self._tasks[proc.pid] = task
        logger.info(f"Added optimization task for PID {proc.pid} ({proc.name})")
        return task

    def remove_task(self, pid: int):
        """Remove a task from the queue."""
        if pid in self._tasks:
            del self._tasks[pid]

    def get_next_task(self) -> Optional[OptimizationTask]:
        """
        Get the next task that is safe to execute.

        Returns None if no tasks are safe to run.
        """
        # Check cooldown from last failure
        if self._last_failure:
            elapsed = (datetime.now() - self._last_failure).total_seconds()
            if elapsed < self.cooldown_seconds:
                logger.debug(f"In cooldown, {self.cooldown_seconds - elapsed:.0f}s remaining")
                return None

        # Check if we're already optimizing
        if self._active_gpu is not None:
            return None

        # Get current GPU states
        gpu_states = self.monitor.get_gpu_states()
        gpu_state_map = {s.index: s for s in gpu_states}

        # Find a safe task to run
        for task in self._tasks.values():
            if task.state not in (OptimizationState.PENDING, OptimizationState.READY):
                continue

            # Check failure count
            if self._failed_attempts.get(task.pid, 0) >= self.max_retries:
                task.state = OptimizationState.SKIPPED
                task.error = "Max retries exceeded"
                continue

            # Check GPU safety
            gpu_state = gpu_state_map.get(task.gpu_index)
            if gpu_state is None:
                continue

            if not self._is_gpu_safe(gpu_state):
                continue

            # Check process is still running
            stable = self.monitor.get_stable_processes()
            if task.pid not in stable:
                task.state = OptimizationState.SKIPPED
                task.error = "Process no longer stable"
                continue

            return task

        return None

    def _is_gpu_safe(self, state: GPUState) -> bool:
        """Check if GPU is safe for optimization."""
        # Check utilization
        if state.utilization_pct > self.utilization_threshold:
            logger.debug(
                f"GPU {state.index} utilization {state.utilization_pct:.0f}% > "
                f"threshold {self.utilization_threshold:.0f}%"
            )
            return False

        # Check memory
        memory_pct = (state.memory_used_mb / state.memory_total_mb) * 100
        if memory_pct > self.memory_threshold:
            logger.debug(
                f"GPU {state.index} memory {memory_pct:.0f}% > "
                f"threshold {self.memory_threshold:.0f}%"
            )
            return False

        return True

    def start_optimization(self, task: OptimizationTask):
        """Mark a task as started."""
        task.state = OptimizationState.OPTIMIZING
        task.started_at = datetime.now()
        self._active_gpu = task.gpu_index
        logger.info(f"Starting optimization for PID {task.pid}")

    def complete_optimization(self, task: OptimizationTask, success: bool, error: str = None):
        """Mark a task as completed."""
        task.completed_at = datetime.now()
        self._active_gpu = None

        if success:
            task.state = OptimizationState.COMPLETED
            logger.info(f"Optimization completed for PID {task.pid}")
        else:
            task.state = OptimizationState.FAILED
            task.error = error
            self._failed_attempts[task.pid] = self._failed_attempts.get(task.pid, 0) + 1
            self._last_failure = datetime.now()
            logger.warning(f"Optimization failed for PID {task.pid}: {error}")

    def get_stats(self) -> Dict:
        """Get scheduler statistics."""
        completed = sum(1 for t in self._tasks.values()
                        if t.state == OptimizationState.COMPLETED)
        failed = sum(1 for t in self._tasks.values()
                     if t.state == OptimizationState.FAILED)
        pending = sum(1 for t in self._tasks.values()
                      if t.state in (OptimizationState.PENDING, OptimizationState.READY))

        return {
            "total_tasks": len(self._tasks),
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "active_gpu": self._active_gpu,
            "in_cooldown": self._last_failure is not None and
                          (datetime.now() - self._last_failure).total_seconds() < self.cooldown_seconds
        }

    def cleanup_stale(self, max_age_hours: int = 24):
        """Remove old completed/failed tasks."""
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = []

        for pid, task in self._tasks.items():
            if task.state in (OptimizationState.COMPLETED, OptimizationState.FAILED,
                              OptimizationState.SKIPPED):
                if task.completed_at and task.completed_at < cutoff:
                    to_remove.append(pid)

        for pid in to_remove:
            del self._tasks[pid]

        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} stale tasks")
