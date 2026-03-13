"""
NVIDIA CUDA backend — H100, A100, RTX series.
Only imported when torch.cuda.is_available() is True.

torch is imported lazily inside each method so this module can be
imported on machines without torch (e.g. test collection on CI).
"""
from __future__ import annotations
import os
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch  # only for type hints; not imported at runtime


@dataclass
class MemoryTier:
    name: str
    capacity_bytes: int
    latency_us: float
    bandwidth_gbps: float


class CUDABackend:
    """
    Wraps CUDA memory ops. All public methods match the protocol
    defined in hal.py — no extra required arguments.
    """

    def __init__(self) -> None:
        import torch
        # Two dedicated streams, created once, reused forever.
        # Never use torch.cuda.current_stream() — that is the compute stream.
        self._copy_stream = torch.cuda.Stream()  # HBM <-> DRAM
        self._nvme_stream = torch.cuda.Stream()  # NVMe staging hops

    def detect_tiers(self) -> list[MemoryTier]:
        """Return the available memory tiers on this GPU node."""
        import torch, psutil, shutil
        _, total = torch.cuda.mem_get_info()
        dram = psutil.virtual_memory().total
        _, nvme_total, _ = shutil.disk_usage("/tmp")
        return [
            MemoryTier("hbm",  total,      latency_us=1.0,       bandwidth_gbps=3350.0),
            MemoryTier("dram", dram,        latency_us=80.0,      bandwidth_gbps=50.0),
            MemoryTier("nvme", nvme_total,  latency_us=100_000.0, bandwidth_gbps=14.0),
        ]

    def allocate(self, size_bytes: int, tier: str) -> Union["torch.Tensor", str]:
        """
        Allocate size_bytes on the requested tier.
        Returns a tensor for hbm/dram, or a file path string for nvme.
        """
        import torch
        if tier == "hbm":
            return torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        if tier == "dram":
            return torch.empty(size_bytes, dtype=torch.uint8, device="cpu").pin_memory()
        if tier == "nvme":
            # NVMe blocks are file-backed. Pinned memory is not applicable here.
            # Pinning is only needed for DRAM tensors that will be DMA'd to the GPU.
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".vmm_block")
            # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
            # TODO Phase 2: replace with GPUDirect Storage (GDS) for direct NVMe→HBM DMA at 14 GB/s
            f.write(b"\x00" * size_bytes)
            f.flush()
            f.close()
            return f.name
        raise ValueError(f"Unknown tier: {tier!r}")

    def free(self, handle: Union["torch.Tensor", str], tier: str) -> None:
        """Release memory back to the allocator."""
        if tier in ("hbm", "dram"):
            del handle
        elif tier == "nvme" and isinstance(handle, str) and os.path.exists(handle):
            os.unlink(handle)

    @staticmethod
    def _gds_available() -> bool:
        """Check if NVIDIA GPUDirect Storage (cuFile) is available."""
        try:
            import importlib
            importlib.import_module("cufile")
            return True
        except ImportError:
            return False

    def async_copy(
        self,
        src: Union["torch.Tensor", str],
        dst: Union["torch.Tensor", str],
        prefer_nvme_stream: bool = False,
    ) -> object:
        """Non-blocking DMA copy. Returns the CUDA stream used."""
        import torch, threading
        s = self._nvme_stream if prefer_nvme_stream else self._copy_stream
        with torch.cuda.stream(s):
            if isinstance(src, str):
                # NVMe → tensor
                if self._gds_available() and dst.is_cuda:
                    import cufile
                    def _gds_read():
                        cf = cufile.CuFile(src, "r")
                        cf.read(dst, src_file_offset=0)
                        cf.close()
                    threading.Thread(target=_gds_read, daemon=True).start()
                else:
                    # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
                    # TODO Phase 2: replace with GPUDirect Storage (GDS) for direct NVMe→HBM DMA at 14 GB/s
                    def _read():
                        with open(src, "rb") as f:
                            data = f.read(dst.nbytes)
                        dst.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8), non_blocking=True)
                    threading.Thread(target=_read, daemon=True).start()
            elif isinstance(dst, str):
                # tensor → NVMe
                if self._gds_available() and src.is_cuda:
                    import cufile
                    def _gds_write():
                        cf = cufile.CuFile(dst, "w")
                        cf.write(src, file_offset=0)
                        cf.close()
                    threading.Thread(target=_gds_write, daemon=True).start()
                else:
                    # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
                    def _write():
                        cpu = src.cpu() if src.is_cuda else src
                        with open(dst, "wb") as f:
                            f.write(cpu.numpy().tobytes())
                    threading.Thread(target=_write, daemon=True).start()
            else:
                # tensor → tensor (HBM <-> DRAM)
                dst.copy_(src, non_blocking=True)
        return s

    def record_event(self, stream: "torch.cuda.Stream") -> "torch.cuda.Event":
        """Record an event on stream. Use event.synchronize() to wait only for this copy."""
        import torch
        event = torch.cuda.Event()
        event.record(stream)
        return event

    def current_hbm_used_bytes(self) -> int:
        """Return bytes currently allocated on the GPU."""
        import torch
        return torch.cuda.memory_allocated()

    def synchronize(self) -> None:
        """Block until all CUDA streams have completed."""
        import torch
        torch.cuda.synchronize()

    def _load_to_tensor(self, handle: Union["torch.Tensor", str]) -> "torch.Tensor":
        import torch
        if isinstance(handle, str):
            with open(handle, "rb") as f:
                data = f.read()
            return torch.frombuffer(bytearray(data), dtype=torch.uint8)
        return handle
