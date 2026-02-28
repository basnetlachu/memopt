"""
memopt.utils.hardware_detector
==============================

One-call GPU capability detection.  Called once at MemoptAgent.__init__ so the
capability profile is available for dynamic candidate selection without re-querying
the driver on every round.

Returned dataclass is immutable — create once, share everywhere.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HardwareProfile:
    """Frozen snapshot of the active GPU's relevant capabilities."""

    # Raw identity
    device_name:       str
    compute_capability: tuple  # e.g. (8, 0) for A100, (9, 0) for H100

    # Derived architecture tags
    arch: str   # "hopper" | "ampere" | "ada_lovelace" | "turing" | "volta" | "other"

    # Memory bandwidth in GB/s (from device properties; 0 if unavailable)
    memory_bandwidth_gbps: float

    # Roofline ridge point in FLOPS/byte (FP16 compute / HBM bandwidth)
    ridge_point: float

    # Capability flags — used by dynamic candidate selection
    supports_flash_attn2:   bool   # CC >= 8.0 (Ampere+)
    supports_flash_attn3:   bool   # CC >= 9.0 (Hopper+) — requires FA3 package
    supports_fp8:           bool   # CC >= 9.0 — Hopper-only via Transformer Engine
    supports_int8:          bool   # Always True on modern CUDA; False on CPU-only
    supports_bf16:          bool   # CC >= 8.0
    supports_channels_last: bool   # Always True for CUDA

    # Human-readable summary
    def summary(self) -> str:
        cc = f"{self.compute_capability[0]}.{self.compute_capability[1]}"
        caps = []
        if self.supports_flash_attn3:  caps.append("FA3")
        elif self.supports_flash_attn2: caps.append("FA2")
        if self.supports_fp8:          caps.append("FP8")
        if self.supports_int8:         caps.append("INT8")
        if self.supports_bf16:         caps.append("BF16")
        cap_str = "+".join(caps) if caps else "basic"
        return (
            f"{self.device_name} (CC {cc}, arch={self.arch}, "
            f"ridge={self.ridge_point:.0f} FLOPS/byte, caps={cap_str})"
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_hardware(device: int = 0) -> HardwareProfile:
    """
    Inspect the active CUDA device and return a HardwareProfile.

    Falls back gracefully to a CPU-only profile when CUDA is not available.
    Always succeeds — never raises.

    Args:
        device: CUDA device ordinal (default 0).

    Returns:
        HardwareProfile with all fields populated.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return _cpu_profile()
        return _detect_cuda_profile(device)
    except Exception:
        return _cpu_profile()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_cuda_profile(device: int) -> HardwareProfile:
    import torch
    props = torch.cuda.get_device_properties(device)
    cc    = (props.major, props.minor)

    arch                 = _detect_architecture(cc)
    memory_bandwidth_gbps = _estimate_bandwidth(props)
    ridge_point          = _compute_ridge_point(cc, memory_bandwidth_gbps)

    return HardwareProfile(
        device_name            = props.name,
        compute_capability     = cc,
        arch                   = arch,
        memory_bandwidth_gbps  = memory_bandwidth_gbps,
        ridge_point            = ridge_point,
        supports_flash_attn2   = cc >= (8, 0),
        supports_flash_attn3   = cc >= (9, 0) and _fa3_available(),
        supports_fp8           = cc >= (9, 0),
        supports_int8          = True,
        supports_bf16          = cc >= (8, 0),
        supports_channels_last = True,
    )


def _detect_architecture(cc: tuple) -> str:
    major, minor = cc
    if major == 9:
        return "hopper"
    if major == 8:
        if minor == 9:
            return "ada_lovelace"  # RTX 40-series (CC 8.9)
        return "ampere"            # A100, A6000 (CC 8.0/8.6)
    if major == 7:
        if minor >= 5:
            return "turing"        # T4, RTX 20-series (CC 7.5)
        return "volta"             # V100 (CC 7.0)
    return "other"


def _estimate_bandwidth(props) -> float:
    """
    Estimate peak HBM bandwidth in GB/s from device properties.

    PyTorch device properties don't expose memory_bus_width directly on all
    versions.  Use known values for common GPUs, then fall back to a formula.
    """
    known_bw = {
        # Hopper
        "NVIDIA H100 SXM5 80GB":   3350.0,
        "NVIDIA H100 PCIe":        2000.0,
        "NVIDIA H200":             4800.0,
        # Ampere
        "NVIDIA A100-SXM4-80GB":   2000.0,
        "NVIDIA A100-SXM4-40GB":   1555.0,
        "NVIDIA A100 PCIe":        1935.0,
        "NVIDIA A100 80GB PCIe":   1935.0,
        "NVIDIA A100-SXM":         2000.0,
        "NVIDIA A6000":             768.0,
        "NVIDIA A5000":             448.0,
        "NVIDIA A4000":             448.0,
        # Ada Lovelace (RTX 40-series)
        "NVIDIA GeForce RTX 4090": 1008.0,
        "NVIDIA GeForce RTX 4080": 716.8,
        "NVIDIA GeForce RTX 4070": 504.2,
        # Turing
        "NVIDIA Tesla T4":          320.0,
        "NVIDIA GeForce RTX 2080 Ti": 616.0,
        # Volta
        "Tesla V100-SXM2-16GB":    900.0,
        "Tesla V100-SXM2-32GB":    900.0,
        "Tesla V100-PCIE-16GB":    900.0,
        "Tesla V100-PCIE-32GB":    900.0,
    }
    # Exact match first
    for known_name, bw in known_bw.items():
        if known_name in props.name:
            return bw

    # Fallback: memory_bus_width × memory_clock_rate formula
    # memory_clock_rate is in kHz; memory_bus_width in bits
    try:
        clock_khz = props.memory_clock_rate   # kHz
        bus_bits  = props.memory_bus_width     # bits
        # GB/s = (clock_kHz * 1000 Hz) * (bus_bits / 8 bytes) * 2 (DDR) / 1e9
        bw = (clock_khz * 1000 * bus_bits / 8 * 2) / 1e9
        return bw
    except AttributeError:
        return 1000.0   # safe fallback (A100-ish class)


def _compute_ridge_point(cc: tuple, bw_gbps: float) -> float:
    """
    Roofline ridge point = peak_fp16_tflops / peak_hbm_bandwidth.

    Uses known peak FP16 (tensor core) throughput per GPU class.
    """
    known_tflops = {
        (9, 0): 1979.0,   # H100 SXM5 (FP16 tensor)
        (8, 0): 312.0,    # A100 SXM4 (FP16 tensor)
        (8, 9): 165.2,    # RTX 4090 (FP16 tensor — shader ops; TC higher but vary)
        (8, 6): 125.0,    # A40/A6000 class
        (7, 5): 65.0,     # T4
        (7, 0): 125.0,    # V100 (FP16 tensor)
    }
    tflops = known_tflops.get(cc, 100.0)  # safe fallback: 100 TFLOPS

    if bw_gbps <= 0:
        return 153.0  # A100 default

    # ridge = TFLOPS / (TB/s) = (tflops * 1e12) / (bw_gbps * 1e9)
    #       = tflops * 1000 / bw_gbps
    return tflops * 1000.0 / bw_gbps


def _fa3_available() -> bool:
    """Check if Flash Attention 3 package is importable."""
    try:
        import flash_attn_interface  # FA3 ships as flash_attn_interface
        return True
    except ImportError:
        pass
    try:
        import flash_attn
        # FA3 builds include flash_attn.flash_attn_3_cuda attribute
        import flash_attn.flash_attn_3_cuda  # type: ignore
        return True
    except (ImportError, AttributeError):
        return False


def _cpu_profile() -> HardwareProfile:
    """Return a safe CPU-only profile when CUDA is unavailable."""
    return HardwareProfile(
        device_name            = "cpu",
        compute_capability     = (0, 0),
        arch                   = "other",
        memory_bandwidth_gbps  = 0.0,
        ridge_point            = 0.0,
        supports_flash_attn2   = False,
        supports_flash_attn3   = False,
        supports_fp8           = False,
        supports_int8          = False,
        supports_bf16          = False,
        supports_channels_last = False,
    )
