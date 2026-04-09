"""
Universal Memory Profile — detect all available memory tiers on the
current hardware and return a structured profile.

Works on any platform: CUDA (Ampere/Hopper/Blackwell), ROCm, Apple
Silicon, plain Linux.  All hardware probes are wrapped in try/except
so this module never raises.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ── GPU bandwidth look-up table (GB/s) ───────────────────────────────────

GPU_SPECS: dict[str, float] = {
    "NVIDIA A100-SXM4-80GB": 2000.0,
    "NVIDIA A100-SXM4-40GB": 1555.0,
    "NVIDIA A100 80GB PCIe": 1935.0,
    "NVIDIA H100 SXM5 80GB": 3350.0,
    "NVIDIA H100 PCIe": 2000.0,
    "NVIDIA RTX 4090": 1008.0,
    "NVIDIA RTX PRO 6000 Blackwell": 1700.0,
    "NVIDIA B200": 8000.0,
}


# ── Dataclasses ───────────────────────────────────────────────────────────

@dataclass
class MemoryTier:
    name: str               # "hbm" | "dram" | "nvme" | "cxl" | "remote_hbm"
    capacity_gb: float
    bandwidth_gbs: float    # theoretical peak GB/s
    latency_us: float       # theoretical latency in microseconds
    is_available: bool
    device_path: str        # empty string if not applicable


@dataclass
class UniversalMemoryProfile:
    tiers: List[MemoryTier]
    total_capacity_gb: float
    architecture: str       # "cuda_ampere" | "cuda_hopper" | "cuda_blackwell"
                            # | "cuda_unknown" | "rocm" | "cpu" | "unknown"
    device_name: str
    compute_capability: Tuple[int, ...] = field(default_factory=tuple)
    supports_rdma: bool = False
    supports_cxl: bool = False
    detected_at: float = 0.0


# ── Detection ─────────────────────────────────────────────────────────────

def detect_universal_profile() -> UniversalMemoryProfile:
    """Probe hardware and return a UniversalMemoryProfile.  Never raises."""

    all_tiers: list[MemoryTier] = []
    architecture = "unknown"
    device_name = ""
    compute_capability: Tuple[int, ...] = ()
    supports_rdma = False
    supports_cxl = False

    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    # ── HBM tier ──────────────────────────────────────────────────────
    if cuda_available:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            capacity_gb = props.total_memory / 1e9
            device_name = props.name
            bw = GPU_SPECS.get(device_name, 2000.0)
            all_tiers.append(MemoryTier(
                name="hbm",
                capacity_gb=capacity_gb,
                bandwidth_gbs=bw,
                latency_us=1.0,
                is_available=True,
                device_path="",
            ))
        except Exception:
            logger.debug("universal_profile: HBM detection failed", exc_info=True)
    # (if no CUDA, HBM tier is simply not added)

    # ── DRAM tier ─────────────────────────────────────────────────────
    dram_capacity = 16.0  # safe default
    try:
        import psutil
        mem = psutil.virtual_memory()
        dram_capacity = mem.total / 1e9
    except ImportError:
        try:
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        dram_capacity = kb / 1e6  # kB → GB
                        break
        except Exception:
            pass
    except Exception:
        pass

    all_tiers.append(MemoryTier(
        name="dram",
        capacity_gb=dram_capacity,
        bandwidth_gbs=50.0,
        latency_us=80.0,
        is_available=True,
        device_path="",
    ))

    # ── NVMe tier ─────────────────────────────────────────────────────
    nvme_dir = os.environ.get("MEMOPT_NVME_DIR", "")
    if nvme_dir and os.path.isdir(nvme_dir):
        try:
            usage = shutil.disk_usage(nvme_dir)
            all_tiers.append(MemoryTier(
                name="nvme",
                capacity_gb=usage.free / 1e9,
                bandwidth_gbs=14.0,
                latency_us=100_000.0,
                is_available=True,
                device_path=nvme_dir,
            ))
        except Exception:
            logger.debug("universal_profile: NVMe detection failed", exc_info=True)

    # ── CXL tier ──────────────────────────────────────────────────────
    try:
        cxl_path = Path("/sys/bus/cxl/devices/")
        if cxl_path.is_dir() and any(cxl_path.iterdir()):
            all_tiers.append(MemoryTier(
                name="cxl",
                capacity_gb=0.0,
                bandwidth_gbs=500.0,
                latency_us=5.0,
                is_available=True,
                device_path=str(cxl_path),
            ))
            supports_cxl = True
    except OSError:
        pass
    except Exception:
        logger.debug("universal_profile: CXL detection failed", exc_info=True)

    # ── Remote HBM tier ───────────────────────────────────────────────
    node_hosts = os.environ.get("MEMOPT_NODE_HOSTS", "")
    if node_hosts:
        all_tiers.append(MemoryTier(
            name="remote_hbm",
            capacity_gb=0.0,
            bandwidth_gbs=25.0,
            latency_us=5.0,
            is_available=True,
            device_path="",
        ))

    # ── Architecture detection ────────────────────────────────────────
    if cuda_available:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            cc = (props.major, props.minor)
            compute_capability = cc
            if cc >= (12, 0):
                architecture = "cuda_blackwell"
            elif cc >= (9, 0):
                architecture = "cuda_hopper"
            elif cc >= (8, 0):
                architecture = "cuda_ampere"
            else:
                architecture = "cuda_unknown"
        except Exception:
            architecture = "cuda_unknown"
    else:
        try:
            import torch
            if getattr(torch.version, "hip", None) is not None:
                architecture = "rocm"
            else:
                architecture = "cpu"
        except ImportError:
            architecture = "cpu"

    # ── RDMA detection ────────────────────────────────────────────────
    try:
        result = subprocess.run(
            ["ibv_devinfo"], capture_output=True, timeout=2, text=True,
        )
        if result.returncode == 0 and "PORT_ACTIVE" in result.stdout:
            supports_rdma = True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # ── Build profile ─────────────────────────────────────────────────
    # Only available tiers, ordered by latency ascending (fastest first)
    available = [t for t in all_tiers if t.is_available]
    available.sort(key=lambda t: t.latency_us)

    return UniversalMemoryProfile(
        tiers=available,
        total_capacity_gb=sum(t.capacity_gb for t in available),
        architecture=architecture,
        device_name=device_name,
        compute_capability=compute_capability,
        supports_rdma=supports_rdma,
        supports_cxl=supports_cxl,
        detected_at=time.time(),
    )
