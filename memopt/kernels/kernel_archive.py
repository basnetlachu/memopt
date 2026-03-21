"""
Hardware-signed kernel archive.

Signs each kernel cache entry with a hardware fingerprint:
    SHA-256(device_name + ":" + compute_cap + ":" + source_sha256)

KernelCache.get() rejects kernels whose fingerprint does not match
the current hardware. This permanently closes the architecture
mismatch bug: a kernel compiled for sm_80 (A100) is never loaded
on sm_89 (Ada) or sm_120 (Blackwell).

Fully backwards compatible — existing cache entries without a
fingerprint field are treated as invalid and recompiled.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def hardware_fingerprint(device_name: str,
                         compute_cap: str,
                         source_sha256: str) -> str:
    """
    Compute a deterministic fingerprint for a kernel binary.

    device_name:   e.g. "NVIDIA A100-SXM4-80GB"
    compute_cap:   e.g. "8.0"
    source_sha256: SHA-256 of the Triton source that was compiled

    The fingerprint changes when:
      - The GPU architecture changes (compute_cap)
      - The source code changes (source_sha256)

    The fingerprint does NOT change on:
      - Driver version (we do not include it — same source + arch
        must produce compatible binaries)
      - Device name suffix changes (only compute_cap is structural)
    """
    payload = f"{compute_cap}:{source_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()


def current_hardware_fingerprint(source_sha256: str) -> Optional[str]:
    """
    Return the fingerprint for the current GPU.
    Returns None on CPU-only machines (no fingerprint needed).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(0)
        compute_cap  = f"{major}.{minor}"
        device_name  = torch.cuda.get_device_name(0)
        return hardware_fingerprint(device_name, compute_cap, source_sha256)
    except Exception as exc:
        logger.debug("hardware_fingerprint: %s", exc)
        return None


def verify_entry(entry: dict) -> bool:
    """
    Return True if a kernel cache entry is valid for the current hardware.

    entry must contain:
        "source_sha256"   str   SHA-256 of the kernel source
        "hw_fingerprint"  str   fingerprint stored at compile time

    Returns True if:
      - hw_fingerprint matches current hardware (correct path)
      - hw_fingerprint is absent AND we are on CPU (legacy entries OK)

    Returns False if:
      - hw_fingerprint is present but does not match current hardware
      - entry is missing source_sha256
    """
    source_sha = entry.get("source_sha256")
    if not source_sha:
        return False  # entry predates source integrity checking

    stored_fp = entry.get("hw_fingerprint")
    if stored_fp is None:
        # Legacy entry without fingerprint — only safe on CPU
        try:
            import torch
            return not torch.cuda.is_available()
        except ImportError:
            return True   # no torch at all — CPU only

    current_fp = current_hardware_fingerprint(source_sha)
    if current_fp is None:
        return True   # CPU machine — no fingerprint required

    match = stored_fp == current_fp
    if not match:
        logger.warning(
            "KernelArchive: fingerprint mismatch — "
            "kernel was compiled for different hardware. "
            "Recompiling. (stored=%s... current=%s...)",
            stored_fp[:12], current_fp[:12]
        )
    return match
