"""
Google TPU backend stub for memopt.

TPU v4 / v5e / v5p:
  v4: 32 GB HBM per chip
  v5e: 16 GB per chip
  v5p: 95 GB per chip
  Accessed via Google Cloud TPU VMs; programmed with JAX or
  torch_xla (no CUDA-style imperative memory API).

Key architectural differences from GPU backends:
  - No explicit memory management — XLA compiler handles placement
  - No `cudaMalloc` equivalent; tensors live as JAX arrays or
    `torch_xla` XLATensors
  - Computation = compiled XLA program, not imperative kernel launches
  - No local NVMe tier — TPU VMs have no direct-attached storage;
    use GCS bucket or attached Persistent Disk via host

STATUS: STUB ONLY. Meaningful memopt integration requires JAX / XLA
support at the serving engine level, which is out of scope until a
TPU customer exists. Defining this stub now keeps the HAL/backend
contract symmetric across vendors.

When TPU support lands, replace:
  _detect()         → jax.devices("tpu") / torch_xla device check
  allocate_hbm()    → jax.numpy.zeros / jax.device_put to TPU
  copy_to_device()  → jax.device_put(array, device)
  synchronize()     → jax.block_until_ready()
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class TPUBackend:
    """
    Google TPU backend stub.

    STUB — DO NOT USE IN PRODUCTION. Actual implementation requires
    JAX or torch_xla and a TPU VM. Never raises from public methods.
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._available = self._detect()
        if self._available:
            logger.info(
                "TPU device %d detected (stub)", device_id)
        else:
            logger.debug("TPU: not available")

    def _detect(self) -> bool:
        """
        Best-effort TPU detection via JAX or torch_xla. Either library
        reports TPU presence when running on a TPU VM.
        """
        try:
            import jax
            try:
                devices = jax.devices("tpu")
                return len(devices) > 0
            except Exception:
                pass
        except Exception:
            pass

        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            return "TPU" in str(device)
        except Exception:
            pass

        return False

    @property
    def is_available(self) -> bool:
        return self._available

    # ── Memory ops (stub) ────────────────────────────────────────────

    def allocate_hbm(self, size_bytes: int) -> Optional[bytes]:
        logger.warning(
            "TPU allocate_hbm: stub — not implemented")
        return None

    def free_hbm(self, ptr: bytes) -> None:
        return

    def allocate_host(self, size_bytes: int) -> Optional[bytes]:
        return None

    def free_host(self, ptr: bytes) -> None:
        return

    def copy_to_device(
        self, dst: bytes, src: bytes, size_bytes: int,
    ) -> bool:
        return False

    def copy_from_device(
        self, dst: bytes, src: bytes, size_bytes: int,
    ) -> bool:
        return False

    def synchronize(self) -> None:
        return

    # ── Observability ────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "backend":     "google_tpu",
            "device_id":   self._device_id,
            "available":   self._available,
            "implemented": False,
            "note": (
                "Stub only. Requires JAX or torch_xla on a TPU VM."),
        }


__all__ = ["TPUBackend"]
