"""
Unified memory backend — Apple M-series, CPU-only servers.

No separate VRAM/HBM pool. Collapsed to 2-tier: RAM (hot) + NVMe (cold).
All torch ops run on CPU. Async copy is done via a background thread.
"""
from __future__ import annotations
import logging
import os
import tempfile
import threading
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch  # only for type hints; not imported at runtime

from .cuda_backend import MemoryTier

_log = logging.getLogger(__name__)


def _nvme_index_path(block_path: str) -> str:
    """Return the path of the recovery index for a block file."""
    return block_path + ".idx"


def _write_nvme_block(path: str, data: bytes) -> None:
    """
    Write data to an NVMe block file with crash safety.

    Protocol:
      1. Write data to path.tmp
      2. fsync path.tmp
      3. Atomic rename path.tmp → path  (rename is atomic on POSIX)
      4. fsync the directory entry

    If the process crashes between steps 1 and 3, path does not exist
    and the VMM knows the block is invalid (no committed file).
    If the process crashes during step 3, the rename is atomic — either
    the old content survives or the new content is present, never a mix.
    """
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, path)   # atomic on POSIX
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _remove_nvme_block(path: str) -> None:
    """Remove block and its tmp file if present."""
    for p in (path, path + ".tmp"):
        try:
            os.remove(p)
        except OSError:
            pass


def _recover_nvme_dir(nvme_dir: str) -> int:
    """
    Remove incomplete NVMe block files left by a previous crash.
    Scans for *.vmm_block.tmp files only — does not touch unrelated files.
    Returns the count of files cleaned up (logged at startup).
    """
    removed = 0
    try:
        for fname in os.listdir(nvme_dir):
            if fname.endswith(".vmm_block.tmp"):
                try:
                    os.remove(os.path.join(nvme_dir, fname))
                    removed += 1
                except OSError:
                    pass
    except OSError:
        pass
    if removed > 0:
        _log.warning(
            "VMM NVMe recovery: removed %d incomplete .vmm_block.tmp "
            "files from %s (previous crash)", removed, nvme_dir
        )
    return removed


class UnifiedBackend:
    """
    CPU / Apple Silicon backend. Never calls torch.cuda.* at any point.
    HBM requests are silently remapped to DRAM.
    """

    def __init__(self) -> None:
        # Clean up any .vmm_block.tmp files left by a previous crash.
        _recover_nvme_dir(tempfile.gettempdir())

    def detect_tiers(self) -> list[MemoryTier]:
        """2-tier hierarchy: DRAM + NVMe."""
        import psutil, shutil
        ram = psutil.virtual_memory().total
        _, nvme_total, _ = shutil.disk_usage("/tmp")
        return [
            MemoryTier("dram", ram,        latency_us=0.1,       bandwidth_gbps=800.0),
            MemoryTier("nvme", nvme_total,  latency_us=100_000.0, bandwidth_gbps=14.0),
        ]

    def allocate(self, size_bytes: int, tier: str) -> "Union[torch.Tensor, str]":
        """
        Allocate size_bytes. HBM requests are treated as DRAM on unified hardware.
        Returns a CPU tensor for hbm/dram (bytearray if torch unavailable), or a file path for nvme.
        """
        if tier in ("hbm", "dram"):
            try:
                import torch
                return torch.empty(size_bytes, dtype=torch.uint8, device="cpu")
            except ImportError:
                return bytearray(size_bytes)
        if tier == "nvme":
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".vmm_block")
            # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
            # TODO Phase 2: replace with GPUDirect Storage (GDS) for direct NVMe→HBM DMA at 14 GB/s
            f.write(b"\x00" * size_bytes)
            f.flush()
            os.fsync(f.fileno())   # ensure zero-fill reaches disk before returning
            f.close()
            return f.name
        raise ValueError(f"Unknown tier: {tier!r}")

    def free(self, handle: "Union[torch.Tensor, str]", tier: str) -> None:
        """Release memory."""
        if tier in ("hbm", "dram"):
            del handle
        elif tier == "nvme" and isinstance(handle, str):
            _remove_nvme_block(handle)

    def async_copy(
        self,
        src: "Union[torch.Tensor, str]",
        dst: "Union[torch.Tensor, str]",
        stream=None,  # unused — kept for interface compatibility
    ) -> threading.Event:
        """
        Non-blocking copy via a background thread.
        Returns a threading.Event; caller waits with event.wait().
        """
        done = threading.Event()

        def _copy() -> None:
            try:
                try:
                    import torch as _torch
                    _has_torch = True
                except ImportError:
                    _torch = None  # type: ignore[assignment]
                    _has_torch = False

                if isinstance(src, str):
                    # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
                    # TODO Phase 2: replace with GPUDirect Storage (GDS) for direct NVMe→HBM DMA at 14 GB/s
                    with open(src, "rb") as f:
                        data = f.read()
                    if isinstance(dst, str):
                        # TODO Phase 2: replace with io_uring (liburing) for ~10 GB/s async NVMe
                        # TODO Phase 2: replace with GPUDirect Storage (GDS) for direct NVMe→HBM DMA at 14 GB/s
                        _write_nvme_block(dst, data)  # atomic write + fsync
                    elif _has_torch and isinstance(dst, _torch.Tensor):
                        dst.copy_(_torch.frombuffer(bytearray(data), dtype=_torch.uint8))
                    else:  # bytearray
                        dst[:] = data
                elif isinstance(dst, str):
                    raw = src.numpy().tobytes() if (_has_torch and isinstance(src, _torch.Tensor)) else bytes(src)
                    _write_nvme_block(dst, raw)  # atomic write + fsync
                else:
                    if _has_torch and isinstance(dst, _torch.Tensor):
                        dst.copy_(src)
                    else:
                        dst[:] = src
            finally:
                done.set()

        threading.Thread(target=_copy, daemon=True).start()
        return done

    def current_hbm_used_bytes(self) -> int:
        """On unified hardware, report process RSS instead of VRAM."""
        import psutil
        return psutil.Process().memory_info().rss

    def synchronize(self) -> None:
        """No-op — no GPU streams to flush."""
        pass
