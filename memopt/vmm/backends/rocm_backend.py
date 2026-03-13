"""
AMD ROCm/HIP backend — MI300X, MI250X.
Only imported when torch.version.hip is not None.

ROCm exposes the same torch.cuda namespace at the Python level.
We subclass CUDABackend and only override detect_tiers() to reflect
MI300X's unified HBM3 pool (192 GB, ~5.3 TB/s).
"""
from __future__ import annotations
from .cuda_backend import CUDABackend, MemoryTier


class ROCmBackend(CUDABackend):
    """
    ROCm is API-compatible with CUDA at the torch level.
    Only tier characteristics differ from CUDABackend.
    """

    def detect_tiers(self) -> list[MemoryTier]:
        """MI300X-calibrated tier specs. Falls back gracefully on MI250."""
        import psutil, shutil, torch
        _, total = torch.cuda.mem_get_info()
        dram = psutil.virtual_memory().total
        _, nvme_total, _ = shutil.disk_usage("/tmp")
        return [
            MemoryTier("hbm",  total,      latency_us=0.8,       bandwidth_gbps=5300.0),
            MemoryTier("dram", dram,        latency_us=80.0,      bandwidth_gbps=50.0),
            MemoryTier("nvme", nvme_total,  latency_us=100_000.0, bandwidth_gbps=14.0),
        ]
