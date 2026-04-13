"""
Intel Gaudi (HPU) backend stub for memopt.

Intel Gaudi 2 / Gaudi 3:
  96 GB HBM2e per card (Gaudi 2), 128 GB HBM2e (Gaudi 3)
  8 cards per server; scale-out ports at 300 Gb/s
  Accessed via Intel Habana SynapseAI SDK — Habana-enabled PyTorch
  exposes `torch.hpu` (not `torch.cuda`) and the `"hpu"` device tag.

Key API differences from CUDA:
  - No CUDA streams; HPU has its own stream abstraction
  - No cuBLAS; matmul routed through SynapseAI operators
  - Device placement: `tensor.to("hpu")` instead of `.to("cuda")`

STATUS: STUB ONLY. This file defines the public contract so the HAL
and test suite can be authored now. A real implementation requires:
  - Intel Gaudi hardware (Gaudi 2 or newer)
  - SynapseAI SDK installed
  - Habana-enabled PyTorch build (habana_frameworks.torch)

When hardware becomes available, replace:
  _detect()         → real device enumeration via habana_frameworks
  allocate_hbm()    → HPU tensor allocation
  copy_to_device()  → tensor.to("hpu") / explicit hpu stream copy
  synchronize()     → habana_frameworks.torch.hpu.synchronize()
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Optional

logger = logging.getLogger(__name__)


class GaudiBackend:
    """
    Intel Gaudi HPU backend stub.

    All memory operations return safe "not implemented" defaults.
    Detection is best-effort — presence of the SynapseAI package,
    `/dev/accel/accel0`, or `hl-smi` on PATH is enough to mark the
    backend as `available`. Thread-safe; never raises from public
    methods.

    STUB — DO NOT USE IN PRODUCTION.
    """

    # Theoretical HBM bandwidth — configurable via env so operators
    # can calibrate once real measurements exist.
    # REQUIRES HARDWARE to validate.
    _HBM_BANDWIDTH_GBPS = float(os.getenv(
        "MEMOPT_GAUDI_HBM_BW_GBPS",
        "2450.0"))   # Gaudi 2 datasheet

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._available = self._detect()
        if self._available:
            logger.info(
                "Gaudi device %d detected (stub)", device_id)
        else:
            logger.debug("Gaudi: not available")

    def _detect(self) -> bool:
        """
        Detect Gaudi hardware. Returns True on any positive signal.

        Signals:
          1. `habana_frameworks.torch` import succeeds
          2. `/dev/accel/accel0` character device exists
          3. `hl-smi` is executable on PATH
        """
        try:
            import habana_frameworks.torch  # noqa: F401
            return True
        except ImportError:
            pass

        if os.path.exists("/dev/accel/accel0"):
            return True

        if shutil.which("hl-smi"):
            return True

        return False

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Memory ops (stub) ────────────────────────────────────────────

    def allocate_hbm(self, size_bytes: int) -> Optional[bytes]:
        """STUB — always returns None."""
        logger.warning(
            "Gaudi allocate_hbm: stub — not implemented")
        return None

    def free_hbm(self, ptr: bytes) -> None:
        """STUB — no-op."""
        return

    def allocate_host(self, size_bytes: int) -> Optional[bytes]:
        """STUB — always returns None."""
        return None

    def free_host(self, ptr: bytes) -> None:
        """STUB — no-op."""
        return

    def copy_to_device(
        self, dst: bytes, src: bytes, size_bytes: int,
    ) -> bool:
        """STUB — always returns False."""
        return False

    def copy_from_device(
        self, dst: bytes, src: bytes, size_bytes: int,
    ) -> bool:
        """STUB — always returns False."""
        return False

    def synchronize(self) -> None:
        """STUB — no-op."""
        return

    # ── Observability ────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "backend":     "intel_gaudi",
            "device_id":   self._device_id,
            "available":   self._available,
            "implemented": False,
            "note": (
                "Stub only. Requires Intel Gaudi hardware "
                "+ SynapseAI SDK."),
            "hbm_bw_gbps": self._HBM_BANDWIDTH_GBPS,
        }


__all__ = ["GaudiBackend"]
