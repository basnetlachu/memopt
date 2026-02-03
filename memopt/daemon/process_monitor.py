"""
Process monitor for GPU workloads.
Uses pynvml or nvidia-smi fallback to detect running GPU processes.
"""

import subprocess
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger("memopt.daemon")


@dataclass
class GPUProcess:
    """Represents a process using GPU memory."""
    pid: int
    name: str
    gpu_index: int
    memory_used_mb: float
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    stable_count: int = 0  # How many checks it's been stable


@dataclass
class GPUState:
    """Current state of a GPU."""
    index: int
    name: str
    memory_total_mb: float
    memory_used_mb: float
    memory_free_mb: float
    utilization_pct: float
    temperature_c: int
    processes: List[GPUProcess] = field(default_factory=list)


class ProcessMonitor:
    """
    Monitor GPU processes and detect stable workloads.

    Uses pynvml if available, falls back to nvidia-smi parsing.
    """

    def __init__(self, stability_threshold: int = 3):
        """
        Args:
            stability_threshold: Number of consecutive checks before a process
                                 is considered stable (safe to profile).
        """
        self.stability_threshold = stability_threshold
        self._nvml_available = False
        self._tracked_processes: Dict[int, GPUProcess] = {}
        self._init_nvml()

    def _init_nvml(self):
        """Try to initialize NVML."""
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_available = True
            self._nvml = pynvml
            logger.info("NVML initialized successfully")
        except ImportError:
            logger.warning("pynvml not available, using nvidia-smi fallback")
        except Exception as e:
            logger.warning(f"NVML init failed: {e}, using nvidia-smi fallback")

    def get_gpu_count(self) -> int:
        """Get number of GPUs."""
        if self._nvml_available:
            return self._nvml.nvmlDeviceGetCount()

        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split("\n")
            return len(lines)
        except Exception:
            return 0

    def get_gpu_states(self) -> List[GPUState]:
        """Get current state of all GPUs."""
        if self._nvml_available:
            return self._get_gpu_states_nvml()
        return self._get_gpu_states_smi()

    def _get_gpu_states_nvml(self) -> List[GPUState]:
        """Get GPU states using NVML."""
        states = []
        device_count = self._nvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = self._nvml.nvmlDeviceGetHandleByIndex(i)
            name = self._nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
            util = self._nvml.nvmlDeviceGetUtilizationRates(handle)

            try:
                temp = self._nvml.nvmlDeviceGetTemperature(
                    handle, self._nvml.NVML_TEMPERATURE_GPU
                )
            except Exception:
                temp = 0

            # Get processes
            processes = []
            try:
                procs = self._nvml.nvmlDeviceGetComputeRunningProcesses(handle)
                for proc in procs:
                    try:
                        proc_name = self._get_process_name(proc.pid)
                        processes.append(GPUProcess(
                            pid=proc.pid,
                            name=proc_name,
                            gpu_index=i,
                            memory_used_mb=proc.usedGpuMemory / (1024 * 1024)
                        ))
                    except Exception:
                        pass
            except Exception:
                pass

            states.append(GPUState(
                index=i,
                name=name,
                memory_total_mb=mem_info.total / (1024 * 1024),
                memory_used_mb=mem_info.used / (1024 * 1024),
                memory_free_mb=mem_info.free / (1024 * 1024),
                utilization_pct=util.gpu,
                temperature_c=temp,
                processes=processes
            ))

        return states

    def _get_gpu_states_smi(self) -> List[GPUState]:
        """Get GPU states using nvidia-smi (fallback)."""
        states = []

        try:
            # Get GPU info
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    idx = int(parts[0])
                    states.append(GPUState(
                        index=idx,
                        name=parts[1],
                        memory_total_mb=float(parts[2]),
                        memory_used_mb=float(parts[3]),
                        memory_free_mb=float(parts[4]),
                        utilization_pct=float(parts[5]),
                        temperature_c=int(parts[6]),
                        processes=[]
                    ))

            # Get processes
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-compute-apps=pid,gpu_uuid,used_memory,process_name",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10
            )

            # Map UUIDs to indices
            uuid_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            uuid_to_idx = {}
            for line in uuid_result.stdout.strip().split("\n"):
                if "," in line:
                    idx, uuid = line.split(",", 1)
                    uuid_to_idx[uuid.strip()] = int(idx.strip())

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    pid = int(parts[0])
                    uuid = parts[1]
                    mem = float(parts[2])
                    name = parts[3]

                    gpu_idx = uuid_to_idx.get(uuid, 0)
                    for state in states:
                        if state.index == gpu_idx:
                            state.processes.append(GPUProcess(
                                pid=pid,
                                name=name,
                                gpu_index=gpu_idx,
                                memory_used_mb=mem
                            ))
                            break

        except Exception as e:
            logger.error(f"nvidia-smi failed: {e}")

        return states

    def _get_process_name(self, pid: int) -> str:
        """Get process name from PID."""
        try:
            with open(f"/proc/{pid}/comm", "r") as f:
                return f.read().strip()
        except Exception:
            try:
                result = subprocess.run(
                    ["ps", "-p", str(pid), "-o", "comm="],
                    capture_output=True, text=True, timeout=2
                )
                return result.stdout.strip() or f"pid:{pid}"
            except Exception:
                return f"pid:{pid}"

    def update_tracking(self, gpu_states: List[GPUState]) -> Dict[int, GPUProcess]:
        """
        Update tracked processes and return stable ones.

        Returns:
            Dict of PID -> GPUProcess for processes that are stable
            (have been running consistently for stability_threshold checks).
        """
        now = datetime.now()
        current_pids = set()

        # Update existing and add new
        for state in gpu_states:
            for proc in state.processes:
                current_pids.add(proc.pid)

                if proc.pid in self._tracked_processes:
                    tracked = self._tracked_processes[proc.pid]
                    tracked.last_seen = now
                    tracked.memory_used_mb = proc.memory_used_mb
                    tracked.stable_count += 1
                else:
                    proc.first_seen = now
                    proc.last_seen = now
                    proc.stable_count = 1
                    self._tracked_processes[proc.pid] = proc

        # Remove processes that are no longer running
        stale_pids = set(self._tracked_processes.keys()) - current_pids
        for pid in stale_pids:
            del self._tracked_processes[pid]

        # Return stable processes
        return {
            pid: proc
            for pid, proc in self._tracked_processes.items()
            if proc.stable_count >= self.stability_threshold
        }

    def get_stable_processes(self) -> Dict[int, GPUProcess]:
        """Get currently stable processes (convenience method)."""
        gpu_states = self.get_gpu_states()
        return self.update_tracking(gpu_states)

    def is_safe_to_profile(self, gpu_index: int) -> bool:
        """
        Check if it's safe to profile on a given GPU.

        Returns True if:
        - GPU utilization is not too high (< 90%)
        - Memory usage is stable
        - No new processes have started recently
        """
        states = self.get_gpu_states()

        for state in states:
            if state.index == gpu_index:
                # Check utilization
                if state.utilization_pct > 90:
                    return False

                # Check if processes are stable
                for proc in state.processes:
                    if proc.pid in self._tracked_processes:
                        if self._tracked_processes[proc.pid].stable_count < 2:
                            return False

                return True

        return False

    def shutdown(self):
        """Cleanup NVML if initialized."""
        if self._nvml_available:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
