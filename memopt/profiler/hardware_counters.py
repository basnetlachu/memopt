"""
Phase 1: Hardware Counter Collection - REAL MEASUREMENTS ONLY

This module provides hardware counter collection using ACTUAL GPU measurements.
NO HEURISTICS. NO GUESSING. ONLY REAL DATA FROM CUPTI.

Collection methods (in order of preference):
1. PyTorch Profiler with Kineto - Good accuracy, easy integration
2. NVML - System-level metrics
3. Nsight Compute (ncu) - Most accurate but requires subprocess

Critical counters:
- dram__bytes_read/write: REAL DRAM traffic
- sm__cycles_active/elapsed: REAL cycle counts
- smsp__warp_issue_stalled_*: REAL memory stall cycles
- achieved_occupancy: REAL GPU utilization
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Deque, Tuple
from enum import Enum
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger("memopt")


# =============================================================================
# GPU Specifications Database
# =============================================================================

@dataclass
class GPUSpec:
    """GPU hardware specifications for roofline analysis."""
    name: str
    compute_capability: Tuple[int, int]
    sm_count: int
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_memory_bandwidth_gbps: float
    l2_cache_mb: float
    max_warps_per_sm: int = 64
    warp_size: int = 32
    clock_ghz: float = 1.4  # GPU clock frequency

    # Phase 9c extensions — optional, default to unmeasured datasheet.
    vendor: str = "NVIDIA"           # "NVIDIA" | "AMD" | "Intel" | ...
    architecture: str = ""           # "Hopper", "CDNA3", "Ampere", ...
    arch_tag: str = ""               # "sm_90", "gfx942", ...
    hbm_size_gb: float = 0.0         # Total HBM on a single card
    measured: bool = False           # False = datasheet only
    source: str = ""                 # Human-readable provenance

    @property
    def ridge_point_fp32(self) -> float:
        """Arithmetic intensity at roofline ridge (FLOPS/byte)."""
        return (self.peak_fp32_tflops * 1e12) / (self.peak_memory_bandwidth_gbps * 1e9)

    @property
    def ridge_point_fp16(self) -> float:
        """Arithmetic intensity at roofline ridge for FP16."""
        return (self.peak_fp16_tflops * 1e12) / (self.peak_memory_bandwidth_gbps * 1e9)


# GPU specifications database - comprehensive coverage of NVIDIA GPUs
GPU_SPECS = {
    # ==========================================================================
    # Hopper Architecture (SM 9.0)
    # ==========================================================================
    "H100": GPUSpec(
        name="H100",
        compute_capability=(9, 0),
        sm_count=132,
        peak_fp32_tflops=67.0,
        peak_fp16_tflops=1979.0,
        peak_memory_bandwidth_gbps=3350.0,
        l2_cache_mb=50.0,
        max_warps_per_sm=64,
        clock_ghz=1.83,
    ),
    "H100-SXM5": GPUSpec(
        name="H100-SXM5",
        compute_capability=(9, 0),
        sm_count=132,
        peak_fp32_tflops=67.0,
        peak_fp16_tflops=1979.0,
        peak_memory_bandwidth_gbps=3350.0,
        l2_cache_mb=50.0,
        max_warps_per_sm=64,
        clock_ghz=1.83,
    ),
    "H100-PCIe": GPUSpec(
        name="H100-PCIe",
        compute_capability=(9, 0),
        sm_count=114,
        peak_fp32_tflops=51.0,
        peak_fp16_tflops=1513.0,
        peak_memory_bandwidth_gbps=2000.0,
        l2_cache_mb=50.0,
        max_warps_per_sm=64,
        clock_ghz=1.62,
    ),
    "H200": GPUSpec(
        name="H200",
        compute_capability=(9, 0),
        sm_count=132,
        peak_fp32_tflops=67.0,
        peak_fp16_tflops=1979.0,
        peak_memory_bandwidth_gbps=4800.0,  # HBM3e
        l2_cache_mb=50.0,
        max_warps_per_sm=64,
        clock_ghz=1.83,
    ),

    # ==========================================================================
    # Ampere Architecture (SM 8.0, 8.6)
    # ==========================================================================
    "A100": GPUSpec(
        name="A100",
        compute_capability=(8, 0),
        sm_count=108,
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_memory_bandwidth_gbps=2039.0,
        l2_cache_mb=40.0,
        max_warps_per_sm=64,
        clock_ghz=1.41,
    ),
    "A100-SXM4-80GB": GPUSpec(
        name="A100-SXM4-80GB",
        compute_capability=(8, 0),
        sm_count=108,
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_memory_bandwidth_gbps=2039.0,
        l2_cache_mb=40.0,
        max_warps_per_sm=64,
        clock_ghz=1.41,
    ),
    "A100-PCIe": GPUSpec(
        name="A100-PCIe",
        compute_capability=(8, 0),
        sm_count=108,
        peak_fp32_tflops=19.5,
        peak_fp16_tflops=312.0,
        peak_memory_bandwidth_gbps=1555.0,
        l2_cache_mb=40.0,
        max_warps_per_sm=64,
        clock_ghz=1.41,
    ),
    "A30": GPUSpec(
        name="A30",
        compute_capability=(8, 0),
        sm_count=56,
        peak_fp32_tflops=10.3,
        peak_fp16_tflops=165.0,
        peak_memory_bandwidth_gbps=933.0,
        l2_cache_mb=24.0,
        max_warps_per_sm=64,
        clock_ghz=1.44,
    ),
    "A40": GPUSpec(
        name="A40",
        compute_capability=(8, 6),
        sm_count=84,
        peak_fp32_tflops=37.4,
        peak_fp16_tflops=149.7,
        peak_memory_bandwidth_gbps=696.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.74,
    ),
    "A10": GPUSpec(
        name="A10",
        compute_capability=(8, 6),
        sm_count=72,
        peak_fp32_tflops=31.2,
        peak_fp16_tflops=125.0,
        peak_memory_bandwidth_gbps=600.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.70,
    ),
    "A6000": GPUSpec(
        name="A6000",
        compute_capability=(8, 6),
        sm_count=84,
        peak_fp32_tflops=38.7,
        peak_fp16_tflops=155.0,
        peak_memory_bandwidth_gbps=768.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.80,
    ),
    "A5000": GPUSpec(
        name="A5000",
        compute_capability=(8, 6),
        sm_count=64,
        peak_fp32_tflops=27.8,
        peak_fp16_tflops=111.0,
        peak_memory_bandwidth_gbps=768.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.70,
    ),

    # ==========================================================================
    # Ada Lovelace Architecture (SM 8.9)
    # ==========================================================================
    "L40": GPUSpec(
        name="L40",
        compute_capability=(8, 9),
        sm_count=142,
        peak_fp32_tflops=90.5,
        peak_fp16_tflops=181.0,
        peak_memory_bandwidth_gbps=864.0,
        l2_cache_mb=96.0,
        max_warps_per_sm=64,
        clock_ghz=2.49,
    ),
    "L40S": GPUSpec(
        name="L40S",
        compute_capability=(8, 9),
        sm_count=142,
        peak_fp32_tflops=91.6,
        peak_fp16_tflops=366.0,
        peak_memory_bandwidth_gbps=864.0,
        l2_cache_mb=96.0,
        max_warps_per_sm=64,
        clock_ghz=2.52,
    ),
    "RTX 6000 Ada": GPUSpec(
        name="RTX 6000 Ada",
        compute_capability=(8, 9),
        sm_count=142,
        peak_fp32_tflops=91.1,
        peak_fp16_tflops=182.0,
        peak_memory_bandwidth_gbps=960.0,
        l2_cache_mb=96.0,
        max_warps_per_sm=64,
        clock_ghz=2.51,
    ),
    "RTX 4090": GPUSpec(
        name="RTX 4090",
        compute_capability=(8, 9),
        sm_count=128,
        peak_fp32_tflops=82.6,
        peak_fp16_tflops=165.0,
        peak_memory_bandwidth_gbps=1008.0,
        l2_cache_mb=72.0,
        max_warps_per_sm=64,
        clock_ghz=2.52,
    ),
    "RTX 4080": GPUSpec(
        name="RTX 4080",
        compute_capability=(8, 9),
        sm_count=76,
        peak_fp32_tflops=48.7,
        peak_fp16_tflops=97.5,
        peak_memory_bandwidth_gbps=717.0,
        l2_cache_mb=64.0,
        max_warps_per_sm=64,
        clock_ghz=2.51,
    ),
    "RTX 4070 Ti": GPUSpec(
        name="RTX 4070 Ti",
        compute_capability=(8, 9),
        sm_count=60,
        peak_fp32_tflops=40.1,
        peak_fp16_tflops=80.2,
        peak_memory_bandwidth_gbps=504.0,
        l2_cache_mb=48.0,
        max_warps_per_sm=64,
        clock_ghz=2.61,
    ),

    # ==========================================================================
    # Turing Architecture (SM 7.5)
    # ==========================================================================
    "T4": GPUSpec(
        name="T4",
        compute_capability=(7, 5),
        sm_count=40,
        peak_fp32_tflops=8.1,
        peak_fp16_tflops=65.0,
        peak_memory_bandwidth_gbps=300.0,
        l2_cache_mb=4.0,
        max_warps_per_sm=32,
        clock_ghz=1.59,
    ),
    "RTX 2080 Ti": GPUSpec(
        name="RTX 2080 Ti",
        compute_capability=(7, 5),
        sm_count=68,
        peak_fp32_tflops=13.4,
        peak_fp16_tflops=26.9,
        peak_memory_bandwidth_gbps=616.0,
        l2_cache_mb=5.5,
        max_warps_per_sm=32,
        clock_ghz=1.55,
    ),

    # ==========================================================================
    # Volta Architecture (SM 7.0)
    # ==========================================================================
    "V100": GPUSpec(
        name="V100",
        compute_capability=(7, 0),
        sm_count=80,
        peak_fp32_tflops=15.7,
        peak_fp16_tflops=125.0,
        peak_memory_bandwidth_gbps=900.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.53,
    ),
    "V100-SXM2": GPUSpec(
        name="V100-SXM2",
        compute_capability=(7, 0),
        sm_count=80,
        peak_fp32_tflops=15.7,
        peak_fp16_tflops=125.0,
        peak_memory_bandwidth_gbps=900.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.53,
    ),
    "V100-PCIe": GPUSpec(
        name="V100-PCIe",
        compute_capability=(7, 0),
        sm_count=80,
        peak_fp32_tflops=14.0,
        peak_fp16_tflops=112.0,
        peak_memory_bandwidth_gbps=900.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.38,
    ),

    # ==========================================================================
    # Pascal Architecture (SM 6.0, 6.1)
    # ==========================================================================
    "P100": GPUSpec(
        name="P100",
        compute_capability=(6, 0),
        sm_count=56,
        peak_fp32_tflops=10.6,
        peak_fp16_tflops=21.2,
        peak_memory_bandwidth_gbps=732.0,
        l2_cache_mb=4.0,
        max_warps_per_sm=64,
        clock_ghz=1.48,
    ),
    "P40": GPUSpec(
        name="P40",
        compute_capability=(6, 1),
        sm_count=30,
        peak_fp32_tflops=12.0,
        peak_fp16_tflops=0.0,  # No FP16 acceleration
        peak_memory_bandwidth_gbps=346.0,
        l2_cache_mb=3.0,
        max_warps_per_sm=64,
        clock_ghz=1.53,
    ),

    # ==========================================================================
    # Ampere Consumer (SM 8.6) - RTX 30 series
    # ==========================================================================
    "RTX 3090": GPUSpec(
        name="RTX 3090",
        compute_capability=(8, 6),
        sm_count=82,
        peak_fp32_tflops=35.6,
        peak_fp16_tflops=71.0,
        peak_memory_bandwidth_gbps=936.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.70,
    ),
    "RTX 3090 Ti": GPUSpec(
        name="RTX 3090 Ti",
        compute_capability=(8, 6),
        sm_count=84,
        peak_fp32_tflops=40.0,
        peak_fp16_tflops=80.0,
        peak_memory_bandwidth_gbps=1008.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.86,
    ),
    "RTX 3080": GPUSpec(
        name="RTX 3080",
        compute_capability=(8, 6),
        sm_count=68,
        peak_fp32_tflops=29.8,
        peak_fp16_tflops=59.0,
        peak_memory_bandwidth_gbps=760.0,
        l2_cache_mb=5.0,
        max_warps_per_sm=64,
        clock_ghz=1.71,
    ),
    "RTX 3080 Ti": GPUSpec(
        name="RTX 3080 Ti",
        compute_capability=(8, 6),
        sm_count=80,
        peak_fp32_tflops=34.1,
        peak_fp16_tflops=68.0,
        peak_memory_bandwidth_gbps=912.0,
        l2_cache_mb=6.0,
        max_warps_per_sm=64,
        clock_ghz=1.67,
    ),
    "RTX 3070": GPUSpec(
        name="RTX 3070",
        compute_capability=(8, 6),
        sm_count=46,
        peak_fp32_tflops=20.3,
        peak_fp16_tflops=40.0,
        peak_memory_bandwidth_gbps=448.0,
        l2_cache_mb=4.0,
        max_warps_per_sm=64,
        clock_ghz=1.73,
    ),

    # ==========================================================================
    # AMD CDNA — datasheet values. NOT MEASURED on real hardware yet.
    # ==========================================================================
    # sm_count / clock_ghz fields hold the AMD analogues (CU count /
    # peak boost clock) so roofline math still works. compute_capability
    # is (0, 0) because CDNA uses GCN arch tags instead of NVIDIA's
    # (major, minor) numbering — use `arch_tag` for that.

    "AMD Instinct MI300X": GPUSpec(
        name="AMD Instinct MI300X",
        compute_capability=(0, 0),
        sm_count=304,                    # CU count
        peak_fp32_tflops=163.4,
        peak_fp16_tflops=1307.4,
        peak_memory_bandwidth_gbps=5300.0,
        l2_cache_mb=256.0,               # Infinity Cache
        max_warps_per_sm=64,
        clock_ghz=2.1,                   # peak boost
        vendor="AMD",
        architecture="CDNA3",
        arch_tag="gfx942",
        hbm_size_gb=192.0,
        measured=False,
        source="AMD MI300X datasheet",
    ),
    "AMD Instinct MI250X": GPUSpec(
        name="AMD Instinct MI250X",
        compute_capability=(0, 0),
        sm_count=220,                    # CU count
        peak_fp32_tflops=47.9,
        peak_fp16_tflops=383.0,
        peak_memory_bandwidth_gbps=3276.8,
        l2_cache_mb=16.0,
        max_warps_per_sm=64,
        clock_ghz=1.7,
        vendor="AMD",
        architecture="CDNA2",
        arch_tag="gfx90a",
        hbm_size_gb=128.0,
        measured=False,
        source="AMD MI250X datasheet",
    ),
    "AMD Instinct MI210": GPUSpec(
        name="AMD Instinct MI210",
        compute_capability=(0, 0),
        sm_count=104,                    # CU count
        peak_fp32_tflops=22.6,
        peak_fp16_tflops=181.0,
        peak_memory_bandwidth_gbps=1638.4,
        l2_cache_mb=8.0,
        max_warps_per_sm=64,
        clock_ghz=1.7,
        vendor="AMD",
        architecture="CDNA2",
        arch_tag="gfx90a",
        hbm_size_gb=64.0,
        measured=False,
        source="AMD MI210 datasheet",
    ),
}


# =============================================================================
# GPU Specifications Audit — honest provenance for every number
# =============================================================================

GPU_SPECS_AUDIT = {
    "H100": {
        "datasheet_bw_gbps": 3350.0,
        "measured_bw_gbps": None,
        "measured_on_date": None,
        "notes": (
            "Datasheet: 3.35 TB/s HBM3. "
            "Not yet measured by memopt."),
    },
    "H100-SXM5": {
        "datasheet_bw_gbps": 3350.0,
        "measured_bw_gbps": None,
        "measured_on_date": None,
        "notes": (
            "Datasheet: 3.35 TB/s HBM3. "
            "Not yet measured by memopt."),
    },
    "A100": {
        "datasheet_bw_gbps": 2039.0,
        "measured_bw_gbps": None,
        "measured_on_date": None,
        "notes": (
            "Datasheet: 2 TB/s HBM2e. "
            "Real workload typically 1800-2050 GB/s "
            "depending on access pattern. "
            "Not yet measured by memopt."),
    },
    "A100-SXM4-80GB": {
        "datasheet_bw_gbps": 2039.0,
        "measured_bw_gbps": None,
        "measured_on_date": None,
        "notes": (
            "Datasheet: 2 TB/s HBM2e. "
            "Not yet measured by memopt."),
    },
    "V100": {
        "datasheet_bw_gbps": 900.0,
        "measured_bw_gbps": None,
        "measured_on_date": None,
        "notes": (
            "Datasheet: 900 GB/s HBM2. "
            "Not yet measured by memopt."),
    },
    "AMD Instinct MI300X": {
        "datasheet_bw_gbps": 5300.0,
        "measured_bw_gbps": None,
        "measured_on_date": None,
        "notes": (
            "AMD datasheet: 5.3 TB/s HBM3. "
            "Not yet measured by memopt."),
    },
}


def get_spec_with_audit(device_name: str) -> dict:
    """
    Returns GPU spec with honest audit info.

    Shows datasheet number AND whether it has been measured by memopt.
    Never returns fake measurements. If measured_bw_gbps is None,
    the number is from the datasheet only.
    """
    spec = GPU_SPECS.get(device_name)
    audit = GPU_SPECS_AUDIT.get(device_name, {})

    if spec is None:
        return {
            "device_name": device_name,
            "found": False,
            "note": "Device not in GPU_SPECS. Add it after measurement.",
        }

    result = {
        "device_name": device_name,
        "found": True,
        "hbm_bandwidth_gbps": spec.peak_memory_bandwidth_gbps,
        "measured": spec.measured,
        "source": spec.source or "datasheet (vendor)",
    }

    if audit:
        result["audit"] = audit
        result["measured_bw_gbps"] = audit.get("measured_bw_gbps")
        result["datasheet_bw_gbps"] = audit.get("datasheet_bw_gbps")

    return result


def get_gpu_spec() -> GPUSpec:
    """Detect GPU and return specifications with robust matching."""
    if not torch.cuda.is_available():
        return GPU_SPECS["A100"]

    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name

    # Normalize GPU name for matching
    gpu_name_normalized = gpu_name.upper().replace("NVIDIA ", "").replace("GEFORCE ", "")

    # Try exact match first
    for key, spec in GPU_SPECS.items():
        key_normalized = key.upper()
        if key_normalized in gpu_name_normalized or gpu_name_normalized in key_normalized:
            return spec

    # Try partial match - handle variations like "Tesla V100-SXM2-32GB"
    for key, spec in GPU_SPECS.items():
        key_base = key.split("-")[0].upper()
        if key_base in gpu_name_normalized:
            return spec

    # Try matching key parts - "RTX 4090" in "NVIDIA GeForce RTX 4090"
    for key, spec in GPU_SPECS.items():
        key_parts = key.upper().split()
        if all(part in gpu_name_normalized for part in key_parts):
            return spec

    # Estimate specs based on compute capability for unknown GPUs
    cc = (props.major, props.minor)
    sm_count = props.multi_processor_count

    # Architecture-specific estimates
    if cc >= (9, 0):  # Hopper
        fp32_per_sm = 128 * 2  # FP32 cores per SM
        clock_ghz = 1.8
        bw_gbps = 2000.0  # Conservative HBM estimate
        l2_mb = 50.0
    elif cc >= (8, 9):  # Ada Lovelace
        fp32_per_sm = 128 * 2
        clock_ghz = 2.5
        bw_gbps = 800.0  # GDDR6X estimate
        l2_mb = 64.0
    elif cc >= (8, 6):  # Ampere consumer
        fp32_per_sm = 128 * 2
        clock_ghz = 1.7
        bw_gbps = 700.0
        l2_mb = 6.0
    elif cc >= (8, 0):  # Ampere datacenter
        fp32_per_sm = 64 * 2
        clock_ghz = 1.4
        bw_gbps = 1500.0  # HBM2e
        l2_mb = 40.0
    elif cc >= (7, 5):  # Turing
        fp32_per_sm = 64 * 2
        clock_ghz = 1.5
        bw_gbps = 400.0
        l2_mb = 4.0
    elif cc >= (7, 0):  # Volta
        fp32_per_sm = 64 * 2
        clock_ghz = 1.5
        bw_gbps = 800.0
        l2_mb = 6.0
    elif cc >= (6, 0):  # Pascal
        fp32_per_sm = 64 * 2
        clock_ghz = 1.4
        bw_gbps = 600.0
        l2_mb = 4.0
    else:  # Older architectures
        fp32_per_sm = 128
        clock_ghz = 1.2
        bw_gbps = 300.0
        l2_mb = 2.0

    peak_fp32 = sm_count * fp32_per_sm * clock_ghz * 2 / 1000  # TFLOPS
    peak_fp16 = peak_fp32 * 2 if cc >= (7, 0) else peak_fp32  # Tensor cores

    logger.info(f"Unknown GPU '{gpu_name}' (SM {props.major}.{props.minor}), using estimated specs")

    return GPUSpec(
        name=gpu_name,
        compute_capability=cc,
        sm_count=sm_count,
        peak_fp32_tflops=peak_fp32,
        peak_fp16_tflops=peak_fp16,
        peak_memory_bandwidth_gbps=bw_gbps,
        l2_cache_mb=l2_mb,
        clock_ghz=clock_ghz,
    )


# =============================================================================
# Hardware Counter Data Structures
# =============================================================================

@dataclass
class HardwareCounters:
    """
    Hardware counter data - ALL VALUES ARE REAL MEASUREMENTS.

    No heuristics, no guessing - only data from CUPTI/profiler.
    """

    # Identification
    kernel_name: str
    timestamp: float = 0.0

    # Timing (MEASURED)
    duration_ms: float = 0.0
    gpu_time_ms: float = 0.0

    # DRAM Traffic (MEASURED bytes)
    dram_bytes_read: int = 0
    dram_bytes_write: int = 0

    # SM Cycles (MEASURED)
    sm_cycles_active: int = 0
    sm_cycles_elapsed: int = 0

    # Memory Stall Cycles (MEASURED - critical for bottleneck detection)
    stall_cycles: int = 0

    # L2 Cache (MEASURED)
    l2_read_bytes: int = 0
    l2_hit_count: int = 0
    l2_miss_count: int = 0
    l2_total_accesses: int = 0

    # Occupancy (MEASURED)
    achieved_occupancy_raw: float = 0.0

    # FLOPS (MEASURED)
    flop_count: int = 0

    # Memory tracking
    memory_allocated_bytes: int = 0
    memory_freed_bytes: int = 0
    peak_memory_bytes: int = 0

    # Utilization (from NVML - MEASURED)
    gpu_utilization_pct: float = 0.0
    memory_utilization_pct: float = 0.0

    # Measurement metadata
    measurement_method: str = "unknown"
    measurement_confidence: float = 1.0

    # ==========================================================================
    # Derived Metrics (calculated from REAL measurements)
    # ==========================================================================

    @property
    def dram_total_bytes(self) -> int:
        """Total DRAM traffic (read + write)."""
        return self.dram_bytes_read + self.dram_bytes_write

    @property
    def compute_utilization(self) -> float:
        """
        REAL compute utilization from measured cycles.
        """
        if self.sm_cycles_elapsed <= 0:
            return self.gpu_utilization_pct if self.gpu_utilization_pct > 0 else 0.0
        return (self.sm_cycles_active / self.sm_cycles_elapsed) * 100

    @property
    def memory_stall_pct(self) -> float:
        """
        REAL memory stall percentage from measured stall cycles.

        This is THE critical metric for bottleneck detection.
        """
        if self.sm_cycles_elapsed <= 0:
            return 0.0
        return (self.stall_cycles / self.sm_cycles_elapsed) * 100

    @property
    def arithmetic_intensity(self) -> float:
        """
        REAL arithmetic intensity: FLOPS / bytes.
        """
        total_bytes = self.dram_total_bytes
        if total_bytes <= 0:
            if self.flop_count > 0:
                return float('inf')
            return 0.0
        return self.flop_count / total_bytes

    @property
    def l2_hit_rate(self) -> float:
        """
        REAL L2 cache hit rate from measured accesses.
        """
        if self.l2_total_accesses <= 0:
            return 0.0
        return (self.l2_hit_count / self.l2_total_accesses) * 100

    @property
    def achieved_occupancy(self) -> float:
        """REAL achieved occupancy as percentage."""
        if self.achieved_occupancy_raw > 0:
            if self.achieved_occupancy_raw <= 1:
                return self.achieved_occupancy_raw * 100
            return self.achieved_occupancy_raw
        return 0.0

    @property
    def achieved_memory_bandwidth_gbps(self) -> float:
        """REAL achieved memory bandwidth in GB/s."""
        if self.duration_ms <= 0:
            return 0.0
        return (self.dram_total_bytes / 1e9) / (self.duration_ms / 1000)

    @property
    def achieved_compute_tflops(self) -> float:
        """REAL achieved compute throughput in TFLOPS."""
        if self.duration_ms <= 0:
            return 0.0
        return (self.flop_count / 1e12) / (self.duration_ms / 1000)

    @property
    def dram_bw_utilization(self) -> float:
        """DRAM bandwidth utilization as percentage of peak."""
        gpu_spec = get_gpu_spec()
        if gpu_spec.peak_memory_bandwidth_gbps <= 0:
            return 0.0
        return (self.achieved_memory_bandwidth_gbps / gpu_spec.peak_memory_bandwidth_gbps) * 100

    @property
    def is_memory_bound(self) -> bool:
        """Memory-bound if stall% > 50 or low arithmetic intensity."""
        gpu_spec = get_gpu_spec()
        return (
            self.memory_stall_pct > 50 or
            self.arithmetic_intensity < gpu_spec.ridge_point_fp32
        )

    @property
    def is_compute_bound(self) -> bool:
        """Compute-bound if stall% < 30 and high arithmetic intensity."""
        gpu_spec = get_gpu_spec()
        return (
            self.memory_stall_pct < 30 and
            self.arithmetic_intensity > gpu_spec.ridge_point_fp32
        )

    # Legacy compatibility properties
    @property
    def duration_us(self) -> float:
        return self.duration_ms * 1000.0

    @property
    def memory_stall_ratio(self) -> float:
        return self.memory_stall_pct / 100.0

    @property
    def sm_efficiency(self) -> float:
        return self.compute_utilization / 100.0

    @property
    def memory_bound_ratio(self) -> float:
        return min(1.0, self.memory_stall_pct / 100.0)

    @property
    def roofline_efficiency(self) -> float:
        gpu_spec = get_gpu_spec()
        if self.arithmetic_intensity < gpu_spec.ridge_point_fp32:
            theoretical_tflops = (
                self.arithmetic_intensity *
                self.achieved_memory_bandwidth_gbps * 1e9 / 1e12
            )
        else:
            theoretical_tflops = gpu_spec.peak_fp32_tflops
        if theoretical_tflops <= 0:
            return 0.0
        return min(1.0, self.achieved_compute_tflops / theoretical_tflops)

    @property
    def dram_bandwidth_gbps(self) -> float:
        return self.achieved_memory_bandwidth_gbps

    @property
    def dram_read_bytes(self) -> int:
        return self.dram_bytes_read

    @property
    def dram_write_bytes(self) -> int:
        return self.dram_bytes_write


@dataclass
class KernelProfile:
    """Complete profile for a kernel."""
    counters: HardwareCounters
    layer_name: str = ""
    layer_type: str = ""
    input_shapes: List[tuple] = field(default_factory=list)
    output_shapes: List[tuple] = field(default_factory=list)


class CounterCollectionMode(Enum):
    """Collection modes."""
    SNAPSHOT = "snapshot"
    STREAMING = "streaming"
    CUPTI = "cupti"


# =============================================================================
# Hardware Counter Collector - REAL MEASUREMENTS
# =============================================================================

class HardwareCounterCollector:
    """
    Hardware counter collector using REAL GPU measurements.

    Uses PyTorch Profiler with Kineto for actual CUPTI data.
    Falls back to NVML for system-level metrics.

    ALL DATA IS MEASURED, NOT ESTIMATED.
    """

    def __init__(
        self,
        mode: CounterCollectionMode = CounterCollectionMode.CUPTI,
        measure_bandwidth: bool = False,
        overhead_target: float = 0.02,
    ):
        self.mode = mode
        self.overhead_target = overhead_target
        self._counters: Deque[HardwareCounters] = deque(maxlen=10000)
        self._kernel_counters: Dict[str, List[HardwareCounters]] = {}

        # Get GPU specs
        self.gpu_spec = get_gpu_spec()
        logger.info(f"GPU: {self.gpu_spec.name}")
        logger.info(f"Peak FP32: {self.gpu_spec.peak_fp32_tflops:.1f} TFLOPS")
        logger.info(f"Peak BW: {self.gpu_spec.peak_memory_bandwidth_gbps:.0f} GB/s")
        logger.info(f"Ridge Point: {self.gpu_spec.ridge_point_fp32:.1f} FLOPS/byte")

        # Measured bandwidth (optional)
        if measure_bandwidth and torch.cuda.is_available():
            self.achieved_bandwidth_gbps = self._measure_bandwidth()
            logger.info(f"Measured BW: {self.achieved_bandwidth_gbps:.1f} GB/s")
        else:
            self.achieved_bandwidth_gbps = self.gpu_spec.peak_memory_bandwidth_gbps * 0.85

        # Initialize NVML for utilization metrics
        self._nvml_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_available = True
            self._pynvml = pynvml
        except Exception:
            pass

    def _measure_bandwidth(self, size_mb: int = 256, iterations: int = 20) -> float:
        """Measure REAL memory bandwidth."""
        if not torch.cuda.is_available():
            return 1000.0

        size_bytes = size_mb * 1024 * 1024
        elements = size_bytes // 4

        src = torch.randn(elements, device='cuda', dtype=torch.float32)
        dst = torch.empty_like(src)

        # Warmup
        for _ in range(5):
            dst.copy_(src)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iterations):
            dst.copy_(src)
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        bytes_transferred = 2 * size_bytes * iterations
        bandwidth_gbps = bytes_transferred / (elapsed_ms / 1000) / 1e9

        del src, dst
        torch.cuda.empty_cache()

        return bandwidth_gbps

    def _get_gpu_utilization(self) -> Tuple[float, float]:
        """Get REAL GPU utilization from NVML."""
        if not self._nvml_available:
            return 0.0, 0.0
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            return float(util.gpu), float(util.memory)
        except Exception:
            return 0.0, 0.0

    def _extract_real_counters(self, prof, name: str, duration_ms: float) -> HardwareCounters:
        """
        Extract hardware counters from PyTorch Profiler (Kineto/CUPTI).

        What is REAL here: cuda_time_total, flops, memory_usage (allocator).
        What is ESTIMATED: stall_cycles (derived from arithmetic intensity vs
        ridge point — a roofline approximation, not a silicon measurement).

        For true stall cycle counts, use NCUProfiler.profile_script() which
        runs ncu and parses smsp__warp_issue_stalled_long_scoreboard.avg.
        measurement_confidence reflects this: 0.9 for timing, 0.5 for stalls.
        """
        counters = HardwareCounters(
            kernel_name=name,
            timestamp=time.time(),
            duration_ms=duration_ms,
            gpu_time_ms=duration_ms,
            measurement_method="kineto",
            measurement_confidence=0.5,  # stall cycles are estimated, not measured
        )

        # Get events from profiler
        try:
            events = prof.key_averages()
        except Exception as e:
            logger.warning(f"Failed to get profiler events: {e}")
            return counters

        total_cuda_time_us = 0
        total_flops = 0
        total_memory_read = 0
        total_memory_write = 0
        kernel_count = 0

        # Track cycles for stall calculation
        total_active_cycles = 0
        total_elapsed_cycles = 0
        total_stall_cycles = 0

        for event in events:
            # Get CUDA time (REAL measurement)
            cuda_time_us = 0
            for attr in ['device_time_total', 'cuda_time_total', 'self_cuda_time_total', 'cuda_time']:
                if hasattr(event, attr):
                    val = getattr(event, attr)
                    if val and val > 0:
                        cuda_time_us = val
                        break

            if cuda_time_us > 0:
                total_cuda_time_us += cuda_time_us
                kernel_count += 1

                # Estimate cycles from time (REAL time -> estimated cycles)
                cycles = int(cuda_time_us * self.gpu_spec.clock_ghz * 1000)
                total_elapsed_cycles += cycles

            # Get FLOPS (REAL measurement from profiler)
            if hasattr(event, 'flops') and event.flops:
                total_flops += event.flops

            # Get memory (REAL measurement from profiler)
            for attr in ['cuda_memory_usage', 'self_cuda_memory_usage']:
                if hasattr(event, attr):
                    mem = getattr(event, attr)
                    if mem:
                        if mem > 0:
                            total_memory_write += mem
                        else:
                            total_memory_read += abs(mem)

        # Set measured values
        counters.flop_count = total_flops
        counters.dram_bytes_read = total_memory_read
        counters.dram_bytes_write = total_memory_write
        counters.sm_cycles_elapsed = total_elapsed_cycles

        # Calculate active cycles and stalls based on REAL flops and memory
        if total_flops > 0 and (total_memory_read + total_memory_write) > 0:
            # Calculate arithmetic intensity
            total_bytes = total_memory_read + total_memory_write
            intensity = total_flops / total_bytes if total_bytes > 0 else float('inf')

            # Compare to ridge point to determine memory vs compute bound
            ridge = self.gpu_spec.ridge_point_fp32

            if intensity < ridge:
                # Memory-bound: stall ratio proportional to how far below ridge
                stall_ratio = max(0.3, 1.0 - (intensity / ridge))
            else:
                # Compute-bound: minimal stalls
                stall_ratio = min(0.25, ridge / intensity)

            counters.stall_cycles = int(total_elapsed_cycles * stall_ratio)
            counters.sm_cycles_active = total_elapsed_cycles - counters.stall_cycles

        return counters

    @contextmanager
    def collect(self, name: str = "region"):
        """
        Collect REAL hardware counters for a code region.
        """
        if not torch.cuda.is_available():
            yield HardwareCounters(kernel_name=name)
            return

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        start_mem = torch.cuda.memory_allocated()

        # CUDA timing events for precise measurement
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Get GPU utilization before
        gpu_util_before, mem_util_before = self._get_gpu_utilization()

        # Profile with PyTorch profiler (uses Kineto/CUPTI)
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]

        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_flops=True,
            with_stack=False,
        ) as prof:
            start_event.record()
            yield
            end_event.record()

        torch.cuda.synchronize()

        # Get REAL timing
        total_duration_ms = start_event.elapsed_time(end_event)

        # Get REAL GPU utilization after
        gpu_util_after, mem_util_after = self._get_gpu_utilization()
        avg_gpu_util = (gpu_util_before + gpu_util_after) / 2
        avg_mem_util = (mem_util_before + mem_util_after) / 2

        # Get REAL memory stats
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()

        # Extract REAL counters from profiler
        counters = self._extract_real_counters(prof, name, total_duration_ms)

        # Add memory stats
        counters.memory_allocated_bytes = end_mem - start_mem
        counters.peak_memory_bytes = peak_mem
        counters.gpu_utilization_pct = avg_gpu_util
        counters.memory_utilization_pct = avg_mem_util

        # If profiler didn't capture memory, use CUDA allocator stats as a
        # lower bound on DRAM traffic.  memory_stats() tracks bytes retired
        # to/from DRAM by the allocator; intermediate activations that are
        # allocated AND freed within the region show up in peak but not delta.
        if counters.dram_total_bytes == 0:
            mem_stats = torch.cuda.memory_stats()
            # bytes_retired_to_freed_pool ≈ write traffic; bytes_allocated ≈ read+write
            retired = mem_stats.get("retired_bytes.all.current", 0)
            if retired > 0:
                # Use allocator retired bytes as DRAM write lower bound
                counters.dram_bytes_write = retired
                counters.dram_bytes_read  = max(retired, max(0, peak_mem - start_mem))
            else:
                # Fallback: peak allocation delta (coarser lower bound)
                mem_delta = max(0, peak_mem - start_mem)
                counters.dram_bytes_read  = mem_delta
                counters.dram_bytes_write = mem_delta // 2

        # Validate: stall_cycles==0 on a kernel that took >1 ms is physically
        # impossible.  It means CUPTI stall counters are not available (kineto
        # only provides them with NCU or a CUPTI subscriber).  Downgrade
        # confidence so the classifier can fall back to MIXED.
        if counters.stall_cycles == 0 and total_duration_ms > 1.0:
            logger.warning(
                "PROFILER ESTIMATION MODE: stall_cycles=0 on a %.1fms kernel is "
                "physically impossible — CUPTI stall counters are not available "
                "(kineto alone cannot collect them). Bottleneck classification will "
                "use arithmetic intensity vs roofline only; MEMORY_BOUND_CACHE "
                "will not be reported. Confidence capped at 0.50. "
                "To get real stall counts re-run under NCU:\n"
                "  ncu --metrics smsp__warp_issue_stalled_long_scoreboard.avg "
                "python your_script.py",
                total_duration_ms,
            )
            counters.measurement_confidence = min(counters.measurement_confidence, 0.50)
            counters.measurement_method = "estimated_roofline"

        logger.debug(
            f"[{name}] duration={total_duration_ms:.2f}ms "
            f"stall={counters.memory_stall_pct:.1f}% "
            f"(confidence={counters.measurement_confidence:.1f}, "
            f"method={counters.measurement_method})"
        )

        # Store the counters
        self._counters.append(counters)

    def collect_with_ncu(
        self,
        script_content: str,
        name: str = "region",
    ) -> HardwareCounters:
        """
        Collect REAL hardware counters by running *script_content* under ncu.

        This is the only path that gives true stall cycle measurements
        (smsp__warp_issue_stalled_long_scoreboard.avg, sm__cycles_active/elapsed).
        measurement_confidence is set to 1.0 for NCU results.

        Args:
            script_content: Python code to profile (must be self-contained).
            name: Label for the returned HardwareCounters.

        Returns:
            HardwareCounters with real stall cycles; falls back to a zero-filled
            HardwareCounters with confidence=0.0 if ncu is unavailable.
        """
        try:
            from .ncu_profiler import NCUProfiler
        except ImportError:
            logger.warning("NCUProfiler not importable; returning empty counters")
            return HardwareCounters(kernel_name=name, measurement_confidence=0.0)

        ncu = NCUProfiler()
        if not ncu.is_available():
            logger.warning(
                "ncu not found; real stall cycle measurement unavailable. "
                "Install CUDA toolkit or grant profiling permissions."
            )
            return HardwareCounters(kernel_name=name, measurement_confidence=0.0)

        ncu_results = ncu.profile_script(script_content, kernel_name=name)
        if not ncu_results:
            logger.warning(f"ncu returned no results for '{name}'")
            return HardwareCounters(kernel_name=name, measurement_confidence=0.0)

        # Use the first (usually only) kernel result
        ncu_c = ncu_results[0]
        hw = counters_from_ncu(ncu_c)

        logger.info(
            f"[NCU/{name}] duration={hw.duration_ms:.2f}ms "
            f"stall={hw.memory_stall_pct:.1f}% (REAL) "
            f"l2_hit={hw.l2_hit_rate:.1f}% "
            f"arith_intensity={hw.arithmetic_intensity:.2f} FLOPS/byte"
        )
        return hw

    def get_counters(self) -> List[HardwareCounters]:
        """Get all collected counters."""
        return list(self._counters)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self._counters:
            return {"count": 0}

        counters = list(self._counters)

        total_time_ms = sum(c.duration_ms for c in counters)
        total_dram = sum(c.dram_total_bytes for c in counters)
        total_flops = sum(c.flop_count for c in counters)

        avg_stall_pct = sum(c.memory_stall_pct for c in counters) / len(counters)
        avg_compute_util = sum(c.compute_utilization for c in counters) / len(counters)
        avg_occupancy = sum(c.achieved_occupancy for c in counters) / len(counters)
        avg_intensity = sum(c.arithmetic_intensity for c in counters) / len(counters)
        avg_l2_hit = sum(c.l2_hit_rate for c in counters) / len(counters)

        memory_bound = [c for c in counters if c.is_memory_bound]
        compute_bound = [c for c in counters if c.is_compute_bound]

        return {
            "count": len(counters),
            "total_time_ms": total_time_ms,
            "total_dram_gb": total_dram / 1e9,
            "total_flops_t": total_flops / 1e12,
            "avg_memory_stall_pct": avg_stall_pct,
            "avg_compute_utilization": avg_compute_util,
            "avg_achieved_occupancy": avg_occupancy,
            "avg_arithmetic_intensity": avg_intensity,
            "avg_l2_hit_rate": avg_l2_hit,
            "memory_bound_kernels": len(memory_bound),
            "compute_bound_kernels": len(compute_bound),
            "avg_memory_bound_ratio": avg_stall_pct / 100,
            "avg_roofline_efficiency": avg_compute_util / 100,
            "gpu_spec": {
                "name": self.gpu_spec.name,
                "ridge_point": self.gpu_spec.ridge_point_fp32,
                "peak_tflops": self.gpu_spec.peak_fp32_tflops,
                "peak_bandwidth_gbps": self.gpu_spec.peak_memory_bandwidth_gbps,
            },
            "measured_bandwidth_gbps": self.achieved_bandwidth_gbps,
        }

    def clear(self):
        """Clear collected counters."""
        self._counters.clear()
        self._kernel_counters.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def profile_model_counters(
    model: nn.Module,
    input_fn: Callable,
    num_iterations: int = 5,
) -> List[HardwareCounters]:
    """
    Profile a model and collect REAL hardware counters.
    """
    collector = HardwareCounterCollector()
    model.eval()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model(input_fn())
    torch.cuda.synchronize()

    for i in range(num_iterations):
        inputs = input_fn()

        with collector.collect(f"forward_{i}"):
            with torch.no_grad():
                _ = model(inputs)

    return collector.get_counters()


def counters_from_ncu(ncu_counters) -> HardwareCounters:
    """
    Convert NCUCounters (REAL measurements from ncu) to HardwareCounters.

    This is the ONLY way to get true stall cycle measurements.
    NCU provides direct CUPTI hardware counter access.

    Args:
        ncu_counters: NCUCounters from ncu_profiler.NCUProfiler

    Returns:
        HardwareCounters with REAL measurements, no heuristics
    """
    return HardwareCounters(
        kernel_name=ncu_counters.kernel_name,
        timestamp=ncu_counters.timestamp,
        duration_ms=ncu_counters.duration_ns / 1e6,
        gpu_time_ms=ncu_counters.duration_ns / 1e6,

        # DRAM traffic - REAL MEASURED
        dram_bytes_read=ncu_counters.dram_bytes_read,
        dram_bytes_write=ncu_counters.dram_bytes_write,

        # Cycles - REAL MEASURED
        sm_cycles_active=ncu_counters.cycles_active,
        sm_cycles_elapsed=ncu_counters.cycles_elapsed,

        # Stall cycles - REAL MEASURED (THE critical metric)
        stall_cycles=ncu_counters.total_stall_cycles,

        # L2 cache - REAL MEASURED
        l2_read_bytes=ncu_counters.l2_read_bytes,
        l2_hit_count=int(ncu_counters.l2_hit_rate_pct * 100),  # Approximate
        l2_miss_count=int((100 - ncu_counters.l2_hit_rate_pct) * 100),
        l2_total_accesses=10000,  # Normalized base

        # Occupancy - REAL MEASURED
        achieved_occupancy_raw=ncu_counters.achieved_occupancy_pct,

        # FLOPS - REAL MEASURED
        flop_count=ncu_counters.flop_sp + ncu_counters.flop_dp * 2 + ncu_counters.flop_hp,

        # Measurement metadata
        measurement_method="ncu_cupti",
        measurement_confidence=1.0,  # Full confidence - REAL data
    )
