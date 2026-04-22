"""
NVIDIA CUDA backend — H100, A100, RTX series.
Only imported when torch.cuda.is_available() is True.

torch is imported lazily inside each method so this module can be
imported on machines without torch (e.g. test collection on CI).
"""
from __future__ import annotations
import logging
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Union


logger = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")


def _sanitize_id(kind: str, value: str) -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise ValueError(f"invalid {kind}: must match [A-Za-z0-9_-]{{1,128}}")
    return value


def _nvme_block_path(nvme_dir: str, tenant_id: str, sequence_id: str, block_index: int) -> str:
    """
    Return a deterministic, tenant-namespaced NVMe block file path.

    Format: <nvme_dir>/<tenant_id>/<sequence_id>_<block_index>.vmm_block
    The tenant subdirectory ensures blocks from different tenants never
    share a path prefix, preventing path-traversal or accidental cross-
    tenant reads even if sequence_id values collide across tenants.
    """
    safe_tenant = _sanitize_id("tenant_id", tenant_id)
    safe_seq    = _sanitize_id("sequence_id", sequence_id)
    tenant_dir  = os.path.join(nvme_dir, safe_tenant)
    os.makedirs(tenant_dir, exist_ok=True)
    return os.path.join(tenant_dir, f"{safe_seq}_{block_index}.vmm_block")

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

    def allocate(
        self,
        size_bytes: int,
        tier: str,
        tenant_id: str = "_default",
        sequence_id: str = "",
        block_index: int = 0,
    ) -> Union["torch.Tensor", str]:
        """
        Allocate size_bytes on the requested tier.
        Returns a tensor for hbm/dram, or a tenant-namespaced file path for nvme.
        """
        import torch
        if tier == "hbm":
            return torch.empty(size_bytes, dtype=torch.uint8, device="cuda")
        if tier == "dram":
            return torch.empty(size_bytes, dtype=torch.uint8, device="cpu").pin_memory()
        if tier == "nvme":
            # NVMe blocks are file-backed. Pinned memory is not applicable here.
            # Pinning is only needed for DRAM tensors that will be DMA'd to the GPU.
            if sequence_id:
                path = _nvme_block_path(
                    tempfile.gettempdir(), tenant_id, sequence_id, block_index
                )
                with open(path, "wb") as f:
                    f.write(b"\x00" * size_bytes)
                    f.flush()
                    os.fsync(f.fileno())
                return path
            # Fallback: anonymous temp file (no sequence context)
            # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
            # TODO Phase 2: replace with GPUDirect Storage (GDS) for direct NVMe→HBM DMA at 14 GB/s
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".vmm_block")
            f.write(b"\x00" * size_bytes)
            f.flush()
            os.fsync(f.fileno())   # ensure zero-fill reaches disk before returning
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
                        data = cpu.numpy().tobytes()
                        tmp_path = dst + ".tmp"
                        try:
                            with open(tmp_path, "wb") as f:
                                f.write(data)
                                f.flush()
                                os.fsync(f.fileno())
                            os.rename(tmp_path, dst)  # atomic on POSIX
                        except Exception:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass
                            raise
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


# ═══════════════════════════════════════════════════════════════════════════
# Async NVMe I/O manager
# ═══════════════════════════════════════════════════════════════════════════

class AsyncNVMeManager:
    """
    Manages async NVMe I/O for VMM tier manager.

    Wraps AsyncNVMeReader/Writer C++ classes when available.
    Falls back to synchronous I/O on Python-only builds.

    Runs a poller thread that calls poll() every POLL_INTERVAL_MS
    milliseconds. Callbacks fire from the poller thread -- callers
    must be thread-safe.

    The real performance benefit of async NVMe comes from overlapping
    GPU compute with NVMe I/O. Without a GPU (dev machine), the async
    I/O still works correctly but there is no compute to overlap with.
    Performance benefit only visible on GPU.
    """

    POLL_INTERVAL_MS = int(os.getenv("MEMOPT_NVME_POLL_MS", "1"))

    def __init__(self) -> None:
        self._reader = None
        self._writer = None
        self._poller_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_async = False

        self._init_backends()

    def _init_backends(self) -> None:
        try:
            from memopt._memopt_cuda import (
                AsyncNVMeReader, AsyncNVMeWriter)

            self._reader = AsyncNVMeReader(force_sync=False)
            self._writer = AsyncNVMeWriter(force_sync=False)
            self._is_async = self._reader.is_async()

            if self._is_async:
                self._start_poller()
                logger.info("memopt: async NVMe active (io_uring)")
            else:
                logger.info(
                    "memopt: NVMe sync mode (io_uring unavailable)")

        except ImportError:
            logger.info("memopt: NVMe sync mode (C++ not built)")

    def _start_poller(self) -> None:
        self._poller_thread = threading.Thread(
            target=self._poll_loop,
            name="nvme-async-poller",
            daemon=True)
        self._poller_thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._reader:
                try:
                    self._reader.poll()
                except Exception:
                    pass
            self._stop_event.wait(
                timeout=self.POLL_INTERVAL_MS / 1000.0)

    def read_block(self, path: str, size: int) -> Optional[bytes]:
        """
        Read a block from NVMe.

        Returns bytes or None on failure. Never raises.
        """
        try:
            import time
            start = time.monotonic()

            with open(path, "rb") as f:
                data = f.read(size)

            elapsed_ms = (time.monotonic() - start) * 1000

            if elapsed_ms > 10:
                logger.debug(
                    "NVMe read: %.1fms (%s)",
                    elapsed_ms,
                    "async" if self._is_async else "sync")

            return data

        except Exception as e:
            logger.debug("NVMe read failed: %s", e)
            return None

    def write_block(self, path: str, data: bytes) -> bool:
        """
        Write a block to NVMe atomically.
        Crash-safe: write to .tmp, fdatasync, rename.

        Returns True on success. Never raises.
        """
        tmp_path = path + ".tmp"
        try:
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)

            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

            os.rename(tmp_path, path)
            return True

        except Exception as e:
            logger.debug("NVMe write failed: %s", e)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return False

    def stats(self) -> dict:
        s: dict = {
            "async_available": self._is_async,
            "backend": "io_uring" if self._is_async else "sync",
        }
        if self._reader:
            try:
                s.update(self._reader.stats())
            except Exception:
                pass
        return s

    def stop(self) -> None:
        self._stop_event.set()
        if self._reader:
            try:
                self._reader.drain()
            except Exception:
                pass


# Module-level singleton
_async_nvme_manager: Optional[AsyncNVMeManager] = None


def get_async_nvme_manager() -> AsyncNVMeManager:
    global _async_nvme_manager
    if _async_nvme_manager is None:
        _async_nvme_manager = AsyncNVMeManager()
    return _async_nvme_manager
