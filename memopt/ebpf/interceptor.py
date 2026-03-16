
"""
CUDA Kernel Interceptor — eBPF uprobe + /proc fallback.

On systems with BCC installed (apt-get install python3-bcc bpfcc-tools):
  Attaches uprobes to cuLaunchKernel and cuMemcpyAsync in libcuda.so.
  Receives every kernel launch event via BPF perf buffer.
  Detects suboptimal kernels by their grid/block dimension patterns.

On systems WITHOUT BCC (the current production server):
  Falls back to /proc + nvidia-smi polling.
  Same interface, coarser granularity.

Both paths:
  - Are safe to run as a daemon thread.
  - Expose get_stats() for the control plane API.
  - Call on_suboptimal_kernel() callback when a bad kernel is found.
"""

import ctypes
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("memopt.ebpf")


@dataclass
class KernelLaunchEvent:
    pid:              int
    tid:              int
    timestamp_ns:     int
    func_ptr:         int
    grid_dim:         tuple     # (x, y, z)
    block_dim:        tuple     # (x, y, z)
    shared_mem_bytes: int
    comm:             str
    is_memcpy:        bool = False


@dataclass
class SuboptimalKernelDetection:
    pid:              int
    func_ptr:         int
    reason:           str       # "standard_attention", "unoptimized_matmul", etc.
    launch_count:     int
    recommendation:   str
    estimated_speedup: float


class CUDAKernelInterceptor:
    """
    Attaches eBPF probes to CUDA driver functions (BCC path) or
    falls back to /proc + nvidia-smi polling (no-BCC path).

    Usage:
        interceptor = CUDAKernelInterceptor()
        interceptor.on_suboptimal_kernel = my_callback
        interceptor.start(pids=[1234])
        # ... run daemon ...
        interceptor.stop()
    """

    # Known suboptimal kernel signatures — detected by block dimensions.
    # Standard O(n²) attention is typically launched with (16,16,1) or (32,32,1)
    # blocks and grid size growing with sequence length.
    SUBOPTIMAL_PATTERNS = {
        "standard_attention": {
            "description":    "Standard O(n²) attention — Flash Attention available",
            "recommendation": "Replace with flash_attn_func from flash-attn package",
            "speedup":        2.5,
            "block_patterns": [(16, 16, 1), (32, 32, 1)],
        },
        "naive_softmax": {
            "description":    "Non-fused softmax — fused version available",
            "recommendation": "Use torch.nn.functional.softmax with torch.compile",
            "speedup":        1.3,
            "block_patterns": [(256, 1, 1), (512, 1, 1)],
        },
    }

    def __init__(self):
        self._bcc_available   = self._check_bcc()
        self._running         = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._bpf             = None

        # Event storage
        self.launch_events:   deque = deque(maxlen=10000)
        self.kernel_profiles: Dict[int, Dict] = defaultdict(lambda: {
            "count": 0, "total_shared_mem": 0, "block_dims": []
        })
        self.detections:      List[SuboptimalKernelDetection] = []
        self.monitored_pids:  set = set()

        # Callback — set by caller to receive detections
        self.on_suboptimal_kernel: Optional[Callable] = None

    def _check_bcc(self) -> bool:
        """Check if BCC (BPF Compiler Collection) is importable."""
        try:
            import bcc  # noqa: F401
            return True
        except ImportError:
            logger.info(
                "BCC not available (no python3-bcc). "
                "Install: apt-get install python3-bcc bpfcc-tools linux-headers-$(uname -r). "
                "Falling back to /proc + nvidia-smi monitoring."
            )
            return False

    def _find_libcuda(self) -> Optional[str]:
        """Locate libcuda.so on this system."""
        candidates = [
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so",
            "/usr/local/cuda/lib64/libcuda.so",
            "/usr/lib/libcuda.so.1",
        ]

        # Try ldconfig first (most reliable)
        try:
            result = subprocess.run(
                ["ldconfig", "-p"], capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.splitlines():
                if "libcuda.so" in line and "=>" in line:
                    path = line.split("=>")[-1].strip()
                    if os.path.exists(path):
                        return path
        except Exception:
            pass

        for c in candidates:
            if os.path.exists(c):
                return c

        return None

    def start(self, pids: Optional[List[int]] = None) -> bool:
        """
        Start interception.

        Args:
            pids: PIDs to monitor, or None for all CUDA processes.

        Returns:
            True  = eBPF active (BCC path)
            False = /proc fallback active (no BCC or no root)
        """
        if pids:
            self.monitored_pids = set(pids)

        if not self._bcc_available:
            logger.info("BCC unavailable — starting /proc fallback monitor")
            self._start_proc_fallback(pids)
            return False

        if os.geteuid() != 0:
            logger.warning("eBPF requires root. Falling back to /proc monitor.")
            self._start_proc_fallback(pids)
            return False

        libcuda_path = self._find_libcuda()
        if not libcuda_path:
            logger.warning("libcuda.so not found. Falling back to /proc monitor.")
            self._start_proc_fallback(pids)
            return False

        logger.info("Attaching eBPF probes to %s", libcuda_path)

        try:
            from bcc import BPF

            bpf_source_path = Path(__file__).parent / "cuda_probe.c"
            with open(bpf_source_path) as f:
                bpf_source = f.read()

            self._bpf = BPF(text=bpf_source)

            self._bpf.attach_uprobe(
                name=libcuda_path,
                sym="cuLaunchKernel",
                fn_name="probe_cuLaunchKernel",
            )
            logger.info("Attached uprobe to cuLaunchKernel")

            try:
                self._bpf.attach_uprobe(
                    name=libcuda_path,
                    sym="cuMemcpyAsync",
                    fn_name="probe_cuMemcpyAsync",
                )
                logger.info("Attached uprobe to cuMemcpyAsync")
            except Exception as e:
                logger.debug("cuMemcpyAsync probe failed (non-critical): %s", e)

            self._bpf["cuda_launches"].open_perf_buffer(self._handle_bpf_event)

            if pids:
                for pid in pids:
                    self._bpf["monitored_pids"][ctypes.c_uint32(pid)] = \
                        ctypes.c_uint32(1)

            self._running = True
            self._monitor_thread = threading.Thread(
                target=self._bpf_poll_loop,
                daemon=True,
                name="memopt-ebpf",
            )
            self._monitor_thread.start()

            logger.info(
                "eBPF interception active — monitoring: %s",
                pids or "all CUDA processes",
            )
            return True

        except Exception as e:
            logger.error("eBPF attach failed: %s — using /proc fallback", e)
            self._bpf = None
            self._start_proc_fallback(pids)
            return False

    # ── BPF event handler ────────────────────────────────────────────────────

    def _handle_bpf_event(self, cpu, data, size):
        """Called by BPF perf buffer for every intercepted CUDA call."""
        try:
            class _RawEvent(ctypes.Structure):
                _fields_ = [
                    ("pid",              ctypes.c_uint32),
                    ("tid",              ctypes.c_uint32),
                    ("timestamp_ns",     ctypes.c_uint64),
                    ("func_ptr",         ctypes.c_uint64),
                    ("grid_dim_x",       ctypes.c_uint32),
                    ("grid_dim_y",       ctypes.c_uint32),
                    ("grid_dim_z",       ctypes.c_uint32),
                    ("block_dim_x",      ctypes.c_uint32),
                    ("block_dim_y",      ctypes.c_uint32),
                    ("block_dim_z",      ctypes.c_uint32),
                    ("shared_mem_bytes", ctypes.c_uint32),
                    ("comm",             ctypes.c_char * 16),
                ]

            raw = ctypes.cast(data, ctypes.POINTER(_RawEvent)).contents

            event = KernelLaunchEvent(
                pid=raw.pid,
                tid=raw.tid,
                timestamp_ns=raw.timestamp_ns,
                func_ptr=raw.func_ptr,
                grid_dim=(raw.grid_dim_x, raw.grid_dim_y, raw.grid_dim_z),
                block_dim=(raw.block_dim_x, raw.block_dim_y, raw.block_dim_z),
                shared_mem_bytes=raw.shared_mem_bytes,
                comm=raw.comm.decode("utf-8", errors="replace").rstrip("\x00"),
                is_memcpy=(raw.func_ptr == 0xDEADBEEF),
            )

            self.launch_events.append(event)

            profile = self.kernel_profiles[event.func_ptr]
            profile["count"] += 1
            profile["total_shared_mem"] += event.shared_mem_bytes
            if event.block_dim not in profile["block_dims"]:
                profile["block_dims"].append(event.block_dim)

            if profile["count"] % 100 == 0:
                self._check_suboptimal(event, profile)

        except Exception as e:
            logger.debug("BPF event handling error: %s", e)

    def _check_suboptimal(self, event: KernelLaunchEvent, profile: Dict):
        """Detect suboptimal patterns from accumulated kernel data."""
        for pattern_name, pattern in self.SUBOPTIMAL_PATTERNS.items():
            if event.block_dim in pattern["block_patterns"]:
                if any(d.func_ptr == event.func_ptr for d in self.detections):
                    continue  # already reported

                detection = SuboptimalKernelDetection(
                    pid=event.pid,
                    func_ptr=event.func_ptr,
                    reason=pattern_name,
                    launch_count=profile["count"],
                    recommendation=pattern["recommendation"],
                    estimated_speedup=pattern["speedup"],
                )
                self.detections.append(detection)

                logger.warning(
                    "SUBOPTIMAL KERNEL: PID %d func=0x%x pattern=%s "
                    "launches=%d fix=%s",
                    event.pid, event.func_ptr, pattern_name,
                    profile["count"], pattern["recommendation"],
                )

                if self.on_suboptimal_kernel:
                    self.on_suboptimal_kernel(detection)

    def _bpf_poll_loop(self):
        """BPF perf buffer polling loop."""
        logger.info("eBPF poll loop started")
        while self._running:
            try:
                self._bpf.perf_buffer_poll(timeout=100)
            except Exception as e:
                logger.error("eBPF poll error: %s", e)
                time.sleep(1)

    # ── /proc fallback ───────────────────────────────────────────────────────

    def _start_proc_fallback(self, pids: Optional[List[int]]):
        """Start /proc + nvidia-smi fallback monitor thread."""
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._proc_monitor_loop,
            args=(pids,),
            daemon=True,
            name="memopt-proc-monitor",
        )
        self._monitor_thread.start()
        logger.debug("Proc fallback monitor thread started")

    def _proc_monitor_loop(self, pids: Optional[List[int]]):
        """
        /proc-based fallback.
        Polls nvidia-smi every 5 seconds for VRAM usage per process.
        Coarser than eBPF — no kernel-level visibility — but always works.
        """
        while self._running:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.strip().split(",")
                    if len(parts) != 2:
                        continue
                    try:
                        pid = int(parts[0].strip())
                    except ValueError:
                        continue

                    if pids and pid not in pids:
                        continue

                    event = KernelLaunchEvent(
                        pid=pid, tid=pid,
                        timestamp_ns=time.time_ns(),
                        func_ptr=0,
                        grid_dim=(0, 0, 0), block_dim=(0, 0, 0),
                        shared_mem_bytes=0, comm="unknown",
                    )
                    self.launch_events.append(event)

            except Exception as e:
                logger.debug("Proc monitor poll error: %s", e)

            # Sleep in small increments so stop() responds quickly
            for _ in range(10):
                if not self._running:
                    return
                time.sleep(0.5)

    # ── Public API ───────────────────────────────────────────────────────────

    def add_pid(self, pid: int):
        """Add a PID to the monitored set at runtime."""
        self.monitored_pids.add(pid)
        if self._bpf:
            try:
                self._bpf["monitored_pids"][ctypes.c_uint32(pid)] = \
                    ctypes.c_uint32(1)
            except Exception as e:
                logger.debug("BPF pid add failed: %s", e)

    def remove_pid(self, pid: int):
        """Remove a PID from the monitored set."""
        self.monitored_pids.discard(pid)
        if self._bpf:
            try:
                del self._bpf["monitored_pids"][ctypes.c_uint32(pid)]
            except Exception:
                pass

    def get_stats(self) -> dict:
        """Return current interception statistics (safe to call anytime)."""
        return {
            "ebpf_active":        self._bpf is not None,
            "monitored_pids":     list(self.monitored_pids),
            "total_launches":     len(self.launch_events),
            "unique_kernels":     len(self.kernel_profiles),
            "detections":         len(self.detections),
            "suboptimal_kernels": [
                {
                    "pid":     d.pid,
                    "reason":  d.reason,
                    "launches": d.launch_count,
                    "speedup": d.estimated_speedup,
                    "fix":     d.recommendation,
                }
                for d in self.detections
            ],
        }

    def stop(self):
        """Stop interception and detach eBPF probes."""
        self._running = False

        if self._bpf:
            try:
                self._bpf.cleanup()
                logger.info("eBPF probes detached")
            except Exception as e:
                logger.debug("eBPF cleanup error: %s", e)

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
