"""
Hardware Abstraction Layer (HAL).

Two complementary surfaces:

1. `HAL` / `get_hal()` — Phase 9 HAL for runtime hardware detection.
   Answers "what GPU backend is on this node?" across NVIDIA, AMD,
   and CPU-only hosts. Detection runs once, is cached, and never
   raises.

2. `backend` / `tiers` / `tier_names` / `get_backend()` — legacy
   module-level singletons consumed by `tier_manager` and the
   public `memopt.vmm` package. These resolve to a concrete
   backend class (`CUDABackend`, `ROCmBackend`, `UnifiedBackend`)
   and expose memory-tier metadata.

Everything else in VMM imports from here. No other VMM module
should touch `torch.cuda`, `torch.version.hip`, `pynvml`, or
`rocm-smi` directly.

Detection order (HAL):
    1. NVIDIA via pynvml (most reliable)
    2. NVIDIA via torch.cuda (fallback when pynvml absent)
    3. AMD ROCm via /opt/rocm + rocm-smi, or torch HIP build
    4. Default: CPU_ONLY
"""
from __future__ import annotations

import enum
import logging
import os
import threading
from typing import List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Phase 9 HAL: runtime hardware detection
# ─────────────────────────────────────────────


class HardwareBackend(enum.Enum):
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM    = "amd_rocm"
    CPU_ONLY    = "cpu_only"
    UNKNOWN     = "unknown"


class GPUInfo:
    """Information about one GPU device."""

    def __init__(self):
        self.index:       int = 0
        self.name:        str = "unknown"
        self.total_bytes: int = 0
        self.free_bytes:  int = 0
        self.compute_cap: str = "0.0"
        self.backend: HardwareBackend = HardwareBackend.UNKNOWN

    def to_dict(self) -> dict:
        return {
            "index":       self.index,
            "name":        self.name,
            "total_gb":    round(self.total_bytes / 1e9, 2),
            "free_gb":     round(self.free_bytes / 1e9, 2),
            "compute_cap": self.compute_cap,
            "backend":     self.backend.value,
        }


_hal_instance: Optional["HAL"] = None
_init_lock = threading.Lock()


class HAL:
    """
    Hardware Abstraction Layer.

    Provides a unified runtime view over NVIDIA CUDA, AMD ROCm, and
    CPU-only backends. All operations are no-ops when the matching
    hardware is absent. Never raises from public methods.
    """

    def __init__(self):
        self._backend: HardwareBackend = HardwareBackend.UNKNOWN
        self._gpus: List[GPUInfo] = []
        self._detect()

    @property
    def backend(self) -> HardwareBackend:
        return self._backend

    @property
    def gpu_count(self) -> int:
        return len(self._gpus)

    @property
    def total_hbm_bytes(self) -> int:
        return sum(g.total_bytes for g in self._gpus)

    @property
    def free_hbm_bytes(self) -> int:
        return sum(g.free_bytes for g in self._gpus)

    @property
    def gpus(self) -> List[GPUInfo]:
        return list(self._gpus)

    @property
    def tier_names(self) -> List[str]:
        """Memory tier names for the active backend.
        Mirrors the module-level `tier_names` so callers can read it
        through any HAL instance without importing the module symbol."""
        return list(tier_names)

    def is_gpu_available(self) -> bool:
        return self._backend in (
            HardwareBackend.NVIDIA_CUDA,
            HardwareBackend.AMD_ROCM,
        )

    def stats(self) -> dict:
        return {
            "backend":      self._backend.value,
            "gpu_count":    self.gpu_count,
            "total_hbm_gb": round(self.total_hbm_bytes / 1e9, 2),
            "free_hbm_gb":  round(self.free_hbm_bytes / 1e9, 2),
            "gpus":         [g.to_dict() for g in self._gpus],
        }

    # ── Detection ─────────────────────────────────────────────────────

    def _detect(self) -> None:
        if self._try_detect_nvidia_nvml():
            return
        if self._try_detect_nvidia_torch():
            return
        if self._try_detect_amd_rocm():
            return
        self._backend = HardwareBackend.CPU_ONLY
        logger.info("HAL: No GPU detected. Running in CPU-only mode.")

    def _try_detect_nvidia_nvml(self) -> bool:
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                if count == 0:
                    return False

                gpus: List[GPUInfo] = []
                for i in range(count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode("utf-8")

                    gpu = GPUInfo()
                    gpu.index = i
                    gpu.name = name
                    gpu.total_bytes = int(info.total)
                    gpu.free_bytes = int(info.free)
                    gpu.backend = HardwareBackend.NVIDIA_CUDA
                    try:
                        major, minor = \
                            pynvml.nvmlDeviceGetCudaComputeCapability(
                                handle)
                        gpu.compute_cap = f"{major}.{minor}"
                    except Exception:
                        pass
                    gpus.append(gpu)
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass

            self._gpus = gpus
            self._backend = HardwareBackend.NVIDIA_CUDA
            logger.info(
                "HAL: NVIDIA CUDA detected. %d GPU(s) via pynvml.",
                len(gpus))
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug("HAL: pynvml detection failed: %s", e)
            return False

    def _try_detect_nvidia_torch(self) -> bool:
        try:
            import torch
            # Torch compiled with HIP also reports
            # torch.cuda.is_available() = True; defer to AMD detector.
            if getattr(torch.version, "hip", None):
                return False
            if not torch.cuda.is_available():
                return False
            count = torch.cuda.device_count()
            if count == 0:
                return False

            gpus: List[GPUInfo] = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                gpu = GPUInfo()
                gpu.index = i
                gpu.name = props.name
                gpu.total_bytes = int(props.total_memory)
                gpu.backend = HardwareBackend.NVIDIA_CUDA
                gpu.compute_cap = f"{props.major}.{props.minor}"
                try:
                    torch.cuda.set_device(i)
                    free, _total = torch.cuda.mem_get_info(i)
                    gpu.free_bytes = int(free)
                except Exception:
                    gpu.free_bytes = gpu.total_bytes
                gpus.append(gpu)

            self._gpus = gpus
            self._backend = HardwareBackend.NVIDIA_CUDA
            logger.info(
                "HAL: NVIDIA CUDA detected. %d GPU(s) via torch.",
                count)
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug("HAL: torch CUDA detection failed: %s", e)
            return False

    def _try_detect_amd_rocm(self) -> bool:
        """
        Detect AMD ROCm GPUs.

        REQUIRES AMD HARDWARE TO VALIDATE end-to-end. On machines
        without ROCm or without AMD GPUs this returns False safely.
        """
        rocm_path = os.getenv("ROCM_PATH", "/opt/rocm")

        gpus: List[GPUInfo] = []
        if os.path.exists(rocm_path):
            gpus = self._query_rocm_smi(rocm_path)
        if not gpus:
            gpus = self._query_amd_torch()
        if not gpus:
            return False

        self._gpus = gpus
        self._backend = HardwareBackend.AMD_ROCM
        logger.info("HAL: AMD ROCm detected. %d GPU(s).", len(gpus))
        return True

    def _query_rocm_smi(self, rocm_path: str) -> List[GPUInfo]:
        import subprocess
        rocm_smi = os.path.join(rocm_path, "bin", "rocm-smi")
        if not os.path.exists(rocm_smi):
            return []
        try:
            result = subprocess.run(
                [rocm_smi, "--showmeminfo", "vram", "--csv"],
                capture_output=True, text=True, timeout=10.0)
            if result.returncode != 0:
                return []
            return self._parse_rocm_smi_csv(result.stdout)
        except Exception as e:
            logger.debug("HAL: rocm-smi failed: %s", e)
            return []

    @staticmethod
    def _parse_rocm_smi_csv(csv_output: str) -> List[GPUInfo]:
        """
        Parse `rocm-smi --showmeminfo vram --csv` output.

        Expected header columns:
            device, VRAM Total Memory (B), VRAM Used Memory (B)

        Returns empty list on parse error. Static so it can be
        invoked from a bare HAL instance during testing.
        """
        lines = csv_output.strip().split("\n")
        if len(lines) < 2:
            return []

        gpus: List[GPUInfo] = []
        for i, line in enumerate(lines[1:]):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                total = int(parts[1])
                used = int(parts[2])
            except (ValueError, IndexError):
                continue

            gpu = GPUInfo()
            gpu.index = i
            gpu.name = parts[0]
            gpu.total_bytes = total
            gpu.free_bytes = max(0, total - used)
            gpu.backend = HardwareBackend.AMD_ROCM
            gpus.append(gpu)
        return gpus

    def _query_amd_torch(self) -> List[GPUInfo]:
        try:
            import torch
            hip_version = getattr(torch.version, "hip", None)
            if hip_version is None:
                return []
            if not torch.cuda.is_available():
                return []
            count = torch.cuda.device_count()
            if count == 0:
                return []

            gpus: List[GPUInfo] = []
            for i in range(count):
                props = torch.cuda.get_device_properties(i)
                gpu = GPUInfo()
                gpu.index = i
                gpu.name = props.name
                gpu.total_bytes = int(props.total_memory)
                gpu.free_bytes = int(props.total_memory)
                gpu.backend = HardwareBackend.AMD_ROCM
                gpus.append(gpu)
            return gpus
        except Exception:
            return []


def get_hal() -> HAL:
    """
    Return the process-wide HAL singleton, creating it on first call.
    Thread-safe via module-level lock. Never raises; falls back to
    a minimal CPU-only HAL if construction fails.
    """
    global _hal_instance
    if _hal_instance is not None:
        return _hal_instance
    with _init_lock:
        if _hal_instance is None:
            try:
                _hal_instance = HAL()
            except Exception as e:
                logger.error("HAL init failed: %s", e)
                inst = HAL.__new__(HAL)
                inst._backend = HardwareBackend.CPU_ONLY
                inst._gpus = []
                _hal_instance = inst
    return _hal_instance


def reset_hal() -> None:
    """Reset the HAL singleton — for testing only."""
    global _hal_instance
    _hal_instance = None


# ─────────────────────────────────────────────
# Legacy backend singletons (unchanged contract)
# ─────────────────────────────────────────────


def _detect_backend():
    try:
        import torch
        if torch.cuda.is_available():
            if getattr(torch.version, "hip", None) is not None:
                # AMD ROCm hardware detected. The user-facing ROCmBackend
                # at backends/rocm_backend.py is currently a stub that
                # raises NotImplementedError on construction. Don't let
                # that crash module import — log a warning and fall back
                # to the unified (CPU-capable) backend so the rest of the
                # system stays usable.
                try:
                    from .backends.rocm_backend import ROCmBackend
                    b = ROCmBackend()
                    logger.info("VMM HAL: ROCm backend (AMD GPU detected)")
                    return b
                except NotImplementedError:
                    logger.warning(
                        "VMM HAL: AMD ROCm hardware detected but the "
                        "ROCm backend is a stub. Falling back to "
                        "unified/CPU mode. See "
                        "memopt/vmm/backends/rocm_backend.py for "
                        "contribution instructions."
                    )
            else:
                from .backends.cuda_backend import CUDABackend
                b = CUDABackend()
                logger.info("VMM HAL: CUDA backend (NVIDIA GPU detected)")
                return b
    except ImportError:
        pass

    from .backends.unified_backend import UnifiedBackend
    b = UnifiedBackend()
    logger.info("VMM HAL: Unified backend (CPU-only or Apple Silicon)")
    return b


# Module-level singleton — instantiated once at first import
backend = _detect_backend()
tiers = backend.detect_tiers()
tier_names: List[str] = [t.name for t in tiers]


def get_backend():
    """Return the active legacy backend. Prefer `backend` directly."""
    return backend


__all__ = [
    # Phase 9 HAL
    "HAL", "HardwareBackend", "GPUInfo", "get_hal", "reset_hal",
    # Legacy contract
    "backend", "tiers", "tier_names", "get_backend",
]
