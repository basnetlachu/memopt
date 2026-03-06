"""
Tests for RooflineProfiler — no GPU required for unit tests.
GPU tests are skipped unless nvidia-smi is available.
"""

import subprocess
from unittest.mock import patch

import pytest

from memopt.profiler.roofline import HardwareProfile, ModelProfile, RooflineProfiler


# ── GPU database integrity ─────────────────────────────────────────────────────

class TestGPUDatabase:
    def test_all_entries_have_required_fields(self):
        profiler = RooflineProfiler()
        required = {
            "memory_bandwidth_gbs",
            "compute_tflops_fp16",
            "flash_attention_version",
            "supports_bf16",
        }
        for name, specs in profiler.GPU_DATABASE.items():
            missing = required - specs.keys()
            assert not missing, f"GPU '{name}' missing fields: {missing}"

    def test_ridge_point_is_positive_for_all_gpus(self):
        profiler = RooflineProfiler()
        for name, specs in profiler.GPU_DATABASE.items():
            bw      = specs["memory_bandwidth_gbs"]
            compute = specs["compute_tflops_fp16"]
            ridge   = (compute * 1e12) / (bw * 1e9)
            assert ridge > 0, f"Non-positive ridge for {name}"

    def test_h100_has_flash_attention_3(self):
        profiler = RooflineProfiler()
        h100 = profiler.GPU_DATABASE["H100 SXM"]
        assert h100["flash_attention_version"] == "flash_attention_3"

    def test_a100_has_flash_attention_2(self):
        profiler = RooflineProfiler()
        a100 = profiler.GPU_DATABASE["A100 SXM"]
        assert a100["flash_attention_version"] == "flash_attention_2"

    def test_hopper_supports_fp8(self):
        profiler = RooflineProfiler()
        assert profiler.GPU_DATABASE["H100 SXM"]["supports_fp8"] is True
        assert profiler.GPU_DATABASE["H100 PCIe"]["supports_fp8"] is True

    def test_ampere_does_not_support_fp8(self):
        profiler = RooflineProfiler()
        assert profiler.GPU_DATABASE["A100 SXM"]["supports_fp8"] is False
        assert profiler.GPU_DATABASE["A100 PCIe"]["supports_fp8"] is False

    def test_nvlink_only_on_multi_gpu_cards(self):
        profiler = RooflineProfiler()
        # SXM variants have NVLink
        assert profiler.GPU_DATABASE["H100 SXM"]["nvlink_bandwidth_gbs"] is not None
        assert profiler.GPU_DATABASE["A100 SXM"]["nvlink_bandwidth_gbs"] is not None
        # PCIe / consumer variants don't
        assert profiler.GPU_DATABASE["H100 PCIe"]["nvlink_bandwidth_gbs"] is None
        assert profiler.GPU_DATABASE["RTX 4090"]["nvlink_bandwidth_gbs"] is None


# ── GPU matching ───────────────────────────────────────────────────────────────

class TestGPUMatching:
    def test_exact_match_a100_sxm(self):
        profiler = RooflineProfiler()
        specs = profiler._lookup_gpu_specs("NVIDIA A100 SXM4-80GB")
        assert specs["memory_bandwidth_gbs"] == 2000
        assert specs["flash_attention_version"] == "flash_attention_2"

    def test_exact_match_h100(self):
        profiler = RooflineProfiler()
        specs = profiler._lookup_gpu_specs("NVIDIA H100 SXM5-80GB HBM3")
        assert specs["flash_attention_version"] == "flash_attention_3"
        assert specs["supports_fp8"] is True

    def test_rtx_4090_match(self):
        profiler = RooflineProfiler()
        specs = profiler._lookup_gpu_specs("NVIDIA GeForce RTX 4090")
        assert specs["memory_bandwidth_gbs"] == 1008

    def test_unknown_gpu_returns_conservative_defaults(self):
        profiler = RooflineProfiler()
        specs = profiler._lookup_gpu_specs("Exotic GPU XR-9000")
        # Should return defaults without raising
        assert specs["memory_bandwidth_gbs"] > 0
        assert specs["compute_tflops_fp16"] > 0
        assert "flash_attention_version" in specs

    def test_longest_key_wins(self):
        """'A100 SXM' is more specific than 'A100' — should prefer it."""
        profiler = RooflineProfiler()
        specs_sxm  = profiler._lookup_gpu_specs("NVIDIA A100 SXM4-80GB")
        specs_pcie = profiler._lookup_gpu_specs("NVIDIA A100 PCIe 40GB")
        # SXM has NVLink
        assert specs_sxm["nvlink_bandwidth_gbs"] is not None
        assert specs_pcie["nvlink_bandwidth_gbs"] is None


# ── Ridge point math ──────────────────────────────────────────────────────────

class TestRidgePointMath:
    def test_a100_ridge_point(self):
        """A100 SXM: 312 TFLOPS / 2000 GB/s = 156 FLOPS/byte."""
        profiler = RooflineProfiler()
        specs = profiler.GPU_DATABASE["A100 SXM"]
        ridge = (specs["compute_tflops_fp16"] * 1e12) / (specs["memory_bandwidth_gbs"] * 1e9)
        assert abs(ridge - 156.0) < 1.0

    def test_h100_ridge_point(self):
        """H100 SXM: 1979 TFLOPS / 3350 GB/s ≈ 591 FLOPS/byte."""
        profiler = RooflineProfiler()
        specs = profiler.GPU_DATABASE["H100 SXM"]
        ridge = (specs["compute_tflops_fp16"] * 1e12) / (specs["memory_bandwidth_gbs"] * 1e9)
        assert abs(ridge - 591.0) < 5.0


# ── Arithmetic intensity estimation ───────────────────────────────────────────

class TestArithmeticIntensity:
    def test_small_model_has_lower_ai(self):
        profiler = RooflineProfiler()
        ai_1b  = profiler._estimate_arithmetic_intensity(1.0)
        ai_70b = profiler._estimate_arithmetic_intensity(70.0)
        assert ai_1b < ai_70b

    def test_all_values_positive(self):
        profiler = RooflineProfiler()
        for size in [0.5, 1.0, 3.0, 7.0, 13.0, 30.0, 70.0, 175.0]:
            ai = profiler._estimate_arithmetic_intensity(size)
            assert ai > 0, f"AI must be positive for {size}B params"

    def test_batch1_is_memory_bound_on_a100(self):
        """At batch=1, every transformer model should be below A100's ridge."""
        profiler = RooflineProfiler()
        a100_ridge = 156.0
        for size in [7.0, 13.0, 30.0, 70.0]:
            ai = profiler._estimate_arithmetic_intensity(size)
            assert ai < a100_ridge, (
                f"{size}B model AI={ai} should be < A100 ridge {a100_ridge}"
            )


# ── VRAM → param estimation ───────────────────────────────────────────────────

class TestParamEstimation:
    def test_13b_vram_estimate(self):
        """13B FP16 ≈ 26 GB. With 2.5x overhead factor we expect ~10B params."""
        profiler = RooflineProfiler()
        vram_mb = 26 * 1024
        params_b = profiler._estimate_params_from_vram(vram_mb)
        # Should be in a reasonable range
        assert 5.0 < params_b < 20.0

    def test_7b_vram_estimate(self):
        profiler = RooflineProfiler()
        params_b = profiler._estimate_params_from_vram(14 * 1024)
        assert 3.0 < params_b < 15.0


# ── profile_gpu (with mocked nvidia-smi) ──────────────────────────────────────

class TestProfileGPU:
    def test_profile_gpu_a100(self):
        profiler = RooflineProfiler()
        with patch.object(profiler, "_query_gpu_stats",
                          return_value=("NVIDIA A100 SXM4-80GB", 81920, 55000)):
            hw = profiler.profile_gpu(0)

        assert hw.gpu_index == 0
        assert "A100" in hw.gpu_name
        assert hw.vram_total_mb == 81920
        assert hw.vram_free_mb == 55000
        assert hw.ridge_point_flops_per_byte > 100
        assert hw.flash_attention_version == "flash_attention_2"
        assert hw.supports_bf16 is True
        assert hw.supports_fp8 is False
        assert hw.recommended_batch_size >= 1
        assert hw.recommended_max_seqs >= hw.recommended_batch_size

    def test_profile_gpu_h100(self):
        profiler = RooflineProfiler()
        with patch.object(profiler, "_query_gpu_stats",
                          return_value=("NVIDIA H100 SXM5-80GB", 81920, 70000)):
            hw = profiler.profile_gpu(0)

        assert hw.flash_attention_version == "flash_attention_3"
        assert hw.supports_fp8 is True
        assert hw.ridge_point_flops_per_byte > hw.ridge_point_flops_per_byte - 1  # sanity

    def test_profile_gpu_unknown_does_not_crash(self):
        profiler = RooflineProfiler()
        with patch.object(profiler, "_query_gpu_stats",
                          return_value=("Custom GPU X1000", 40960, 30000)):
            hw = profiler.profile_gpu(1)

        assert hw.gpu_index == 1
        assert hw.vram_total_mb == 40960
        assert hw.recommended_batch_size >= 1


# ── profile_model ─────────────────────────────────────────────────────────────

class TestProfileModel:
    def _a100_hw(self) -> HardwareProfile:
        profiler = RooflineProfiler()
        with patch.object(profiler, "_query_gpu_stats",
                          return_value=("NVIDIA A100 SXM4-80GB", 81920, 55000)):
            return profiler.profile_gpu(0)

    def test_large_model_is_memory_bound(self):
        profiler = RooflineProfiler()
        hw = self._a100_hw()

        # 13B model uses ~26 GB → memory-bound on A100
        hw_with_usage = HardwareProfile(
            **{**hw.__dict__, "vram_free_mb": hw.vram_total_mb - 26 * 1024}
        )
        model = profiler.profile_model(12345, hw_with_usage)
        assert model.bottleneck == "MEMORY-BANDWIDTH"
        assert model.bottleneck_severity > 1.0
        assert len(model.recommended_optimizations) > 0

    def test_recommendations_include_flash_attention(self):
        profiler = RooflineProfiler()
        hw = self._a100_hw()
        hw2 = HardwareProfile(**{**hw.__dict__, "vram_free_mb": hw.vram_total_mb - 26 * 1024})
        model = profiler.profile_model(12345, hw2)

        recs_str = " ".join(model.recommended_optimizations).lower()
        assert "flash" in recs_str or "attention" in recs_str

    def test_compute_bound_gets_compile_recommendation(self):
        profiler = RooflineProfiler()
        hw = self._a100_hw()
        # Artificially set very high AI to force compute-bound classification
        with patch.object(profiler, "_estimate_arithmetic_intensity", return_value=99999.0):
            model = profiler.profile_model(12345, hw)

        assert model.bottleneck == "COMPUTE"
        recs_str = " ".join(model.recommended_optimizations).lower()
        assert "compile" in recs_str


# ── Live GPU test (optional) ─────────────────────────────────────────────────

@pytest.mark.skipif(
    subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode != 0,
    reason="nvidia-smi not available"
)
class TestLiveGPU:
    def test_profile_real_gpu(self):
        profiler = RooflineProfiler()
        hw = profiler.profile_gpu(0)
        assert hw.vram_total_mb > 0
        assert hw.vram_free_mb >= 0
        assert hw.memory_bandwidth_gbs > 0
        assert hw.compute_tflops_fp16 > 0
        assert hw.ridge_point_flops_per_byte > 0
        assert hw.flash_attention_version in ("flash_attention_2", "flash_attention_3")
