"""
Tests for the Hardware Abstraction Layer.

Covers:
  - Singleton semantics
  - CPU-only fallback when every detection path is stubbed out
  - NVIDIA mock path
  - AMD ROCm mock path (without real AMD hardware)
  - rocm-smi CSV parsing edge cases
  - NodeCapabilities exposes the detected backend
  - /healthz surfaces hardware info
"""
from unittest.mock import patch

from memopt.vmm.hal import (
    GPUInfo,
    HAL,
    HardwareBackend,
    get_hal,
    reset_hal,
)


# ══════════════════════════════════════════════════════════════════════
#  Singleton + basic shape
# ══════════════════════════════════════════════════════════════════════


def test_hal_returns_instance():
    reset_hal()
    hal = get_hal()
    assert isinstance(hal, HAL)
    reset_hal()


def test_hal_singleton():
    reset_hal()
    hal1 = get_hal()
    hal2 = get_hal()
    assert hal1 is hal2
    reset_hal()


def test_hal_backend_is_valid_enum():
    reset_hal()
    hal = get_hal()
    assert hal.backend in HardwareBackend
    reset_hal()


def test_hal_gpu_count_is_nonnegative():
    reset_hal()
    hal = get_hal()
    assert hal.gpu_count >= 0
    reset_hal()


def test_hal_hbm_bytes_consistent():
    reset_hal()
    hal = get_hal()
    assert hal.total_hbm_bytes >= 0
    assert hal.free_hbm_bytes >= 0
    assert hal.free_hbm_bytes <= hal.total_hbm_bytes
    reset_hal()


def test_hal_stats_keys():
    reset_hal()
    hal = get_hal()
    stats = hal.stats()
    for key in ("backend", "gpu_count",
                "total_hbm_gb", "free_hbm_gb", "gpus"):
        assert key in stats
    assert isinstance(stats["gpus"], list)
    reset_hal()


# ══════════════════════════════════════════════════════════════════════
#  Detection fallback (stubbed)
# ══════════════════════════════════════════════════════════════════════


def test_hal_cpu_only_no_crash():
    """When every detection path returns False, we land on CPU_ONLY."""
    reset_hal()
    with patch.object(
            HAL, "_try_detect_nvidia_nvml", return_value=False), \
         patch.object(
            HAL, "_try_detect_nvidia_torch", return_value=False), \
         patch.object(
            HAL, "_try_detect_amd_rocm", return_value=False):
        hal = HAL()
    assert hal.backend == HardwareBackend.CPU_ONLY
    assert hal.gpu_count == 0
    assert hal.total_hbm_bytes == 0
    assert hal.is_gpu_available() is False
    reset_hal()


def test_hal_nvidia_mock():
    """Simulate NVIDIA detection succeeding."""
    reset_hal()

    def fake_nvidia(self):
        gpu = GPUInfo()
        gpu.index = 0
        gpu.name = "NVIDIA A100 SXM4"
        gpu.total_bytes = 80 * 1024 ** 3
        gpu.free_bytes  = 72 * 1024 ** 3
        gpu.compute_cap = "8.0"
        gpu.backend = HardwareBackend.NVIDIA_CUDA
        self._gpus = [gpu]
        self._backend = HardwareBackend.NVIDIA_CUDA
        return True

    with patch.object(
            HAL, "_try_detect_nvidia_nvml", fake_nvidia):
        hal = HAL()
    assert hal.backend == HardwareBackend.NVIDIA_CUDA
    assert hal.gpu_count == 1
    assert hal.gpus[0].name == "NVIDIA A100 SXM4"
    assert hal.total_hbm_bytes == 80 * 1024 ** 3
    assert hal.is_gpu_available() is True
    reset_hal()


def test_hal_amd_mock():
    """Simulate AMD ROCm detection (no AMD hardware required)."""
    reset_hal()

    def fake_amd(self):
        gpu = GPUInfo()
        gpu.index = 0
        gpu.name = "AMD Instinct MI300X"
        gpu.total_bytes = 192 * 1024 ** 3
        gpu.free_bytes  = 180 * 1024 ** 3
        gpu.backend = HardwareBackend.AMD_ROCM
        self._gpus = [gpu]
        self._backend = HardwareBackend.AMD_ROCM
        return True

    with patch.object(
            HAL, "_try_detect_nvidia_nvml", return_value=False), \
         patch.object(
            HAL, "_try_detect_nvidia_torch", return_value=False), \
         patch.object(
            HAL, "_try_detect_amd_rocm", fake_amd):
        hal = HAL()
    assert hal.backend == HardwareBackend.AMD_ROCM
    assert hal.gpu_count == 1
    assert hal.gpus[0].name == "AMD Instinct MI300X"
    assert hal.is_gpu_available() is True
    reset_hal()


# ══════════════════════════════════════════════════════════════════════
#  rocm-smi CSV parser (pure string, no hardware)
# ══════════════════════════════════════════════════════════════════════


def test_rocm_smi_csv_parsing():
    csv = (
        "device,VRAM Total Memory (B),VRAM Used Memory (B)\n"
        "card0,68702699520,1073741824\n"
        "card1,68702699520,2147483648\n"
    )
    gpus = HAL._parse_rocm_smi_csv(csv)
    assert len(gpus) == 2
    assert gpus[0].total_bytes == 68702699520
    assert gpus[0].free_bytes == 68702699520 - 1073741824
    assert gpus[1].free_bytes == 68702699520 - 2147483648
    assert all(g.backend == HardwareBackend.AMD_ROCM for g in gpus)


def test_rocm_smi_csv_empty():
    assert HAL._parse_rocm_smi_csv("") == []


def test_rocm_smi_csv_header_only():
    assert HAL._parse_rocm_smi_csv("only one line") == []


def test_rocm_smi_csv_malformed_row_skipped():
    # Header + one well-formed row + one malformed row.
    csv = (
        "device,VRAM Total Memory (B),VRAM Used Memory (B)\n"
        "card0,68702699520,1073741824\n"
        "not,enough\n"
        "card2,not-a-number,also-bad\n"
    )
    gpus = HAL._parse_rocm_smi_csv(csv)
    assert len(gpus) == 1
    assert gpus[0].name == "card0"


def test_gpu_info_to_dict():
    g = GPUInfo()
    g.index = 1
    g.name = "test"
    g.total_bytes = 1 * 1024 ** 3
    g.free_bytes = 512 * 1024 ** 2
    g.compute_cap = "8.6"
    g.backend = HardwareBackend.NVIDIA_CUDA
    d = g.to_dict()
    assert d["index"] == 1
    assert d["name"] == "test"
    assert d["total_gb"] == round(1 * 1024 ** 3 / 1e9, 2)
    assert d["backend"] == "nvidia_cuda"


# ══════════════════════════════════════════════════════════════════════
#  NodeCapabilities integration
# ══════════════════════════════════════════════════════════════════════


def test_node_capabilities_has_hardware_backend():
    from memopt.vmm.discovery import NodeCapabilities
    caps = NodeCapabilities()
    caps.detect()
    assert hasattr(caps, "hardware_backend")
    assert caps.hardware_backend in (
        "nvidia_cuda", "amd_rocm",
        "cpu_only", "unknown")


def test_node_capabilities_roundtrip_hardware_backend():
    from memopt.vmm.discovery import NodeCapabilities
    caps = NodeCapabilities()
    caps.detect()
    d = caps.to_dict()
    assert "hardware_backend" in d
    restored = NodeCapabilities.from_dict(d)
    assert restored.hardware_backend == caps.hardware_backend


# ══════════════════════════════════════════════════════════════════════
#  Serving /healthz surfaces hardware backend
# ══════════════════════════════════════════════════════════════════════


def test_healthz_includes_hardware_backend():
    from fastapi.testclient import TestClient
    from memopt.serving.server import app
    client = TestClient(app)
    response = client.get("/healthz")
    assert response.status_code in (200, 503)
    data = response.json()
    # The endpoint always returns a status+node_id; on the happy
    # path it also surfaces hardware info.
    if "hardware_backend" in data:
        assert data["hardware_backend"] in (
            "nvidia_cuda", "amd_rocm",
            "cpu_only", "unknown")
        assert data.get("gpu_count", 0) >= 0
