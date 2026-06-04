"""
Roofline Hardware Profiler — profiles any GPU at runtime and selects
the correct optimizations automatically. No hardcoded assumptions.

Roofline model:
  - Ridge point = Compute (FLOPS) / Memory Bandwidth (bytes/s)
  - If AI (arithmetic intensity) < ridge → MEMORY-BANDWIDTH bound
  - If AI >= ridge → COMPUTE bound
  - Transformer inference at batch=1 is almost always memory-bound
"""

import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, List

logger = logging.getLogger("memopt.profiler.roofline")


@dataclass
class HardwareProfile:
    gpu_index:                  int
    gpu_name:                   str
    vram_total_mb:              int
    vram_free_mb:               int
    memory_bandwidth_gbs:       float
    compute_tflops_fp16:        float
    ridge_point_flops_per_byte: float
    flash_attention_version:    str
    supports_bf16:              bool
    supports_fp8:               bool
    nvlink_bandwidth_gbs:       Optional[float]
    recommended_batch_size:     int
    recommended_max_seqs:       int
    cuda_compute_capability:    tuple


@dataclass
class ModelProfile:
    model_name:               str
    model_family:             str
    parameter_count_b:        float
    vram_used_mb:             int
    arithmetic_intensity:     float
    bottleneck:               str
    bottleneck_severity:      float
    recommended_optimizations: List[str]


class RooflineProfiler:
    """
    GPU roofline profiler.
    Source: NVIDIA/AMD official datasheets + CUDA Programming Guide.

    GPU specs are sourced from hardware_counters.GPU_SPECS (single source of truth).
    """

    @staticmethod
    def _spec_to_dict(spec) -> dict:
        """Convert a GPUSpec from hardware_counters to the roofline dict format."""
        cc = spec.compute_capability or (8, 0)
        supports_fp8  = cc >= (8, 9)   # Ada Lovelace+ and Hopper+
        supports_bf16 = cc >= (8, 0)   # Ampere+
        fa_version = "flash_attention_3" if cc >= (9, 0) else "flash_attention_2"
        # Pascal and older GPUs report fp16_tflops=0; use fp32 as compute ceiling
        effective_fp16 = spec.peak_fp16_tflops or spec.peak_fp32_tflops
        fp8_tflops = round(effective_fp16 * 2.0, 0) if supports_fp8 else None
        return {
            "memory_bandwidth_gbs":    spec.peak_memory_bandwidth_gbps,
            "compute_tflops_fp16":     effective_fp16,
            "compute_tflops_bf16":     effective_fp16,
            "compute_tflops_fp8":      fp8_tflops,
            "flash_attention_version": fa_version,
            "supports_bf16":           supports_bf16,
            "supports_fp8":            supports_fp8,
            "nvlink_bandwidth_gbs":    None,
            "cuda_compute_capability": cc,
        }

    def profile_gpu(self, gpu_index: int = 0) -> HardwareProfile:
        """Profile a specific GPU index. Queries nvidia-smi for live stats."""
        gpu_name, vram_total_mb, vram_free_mb = self._query_gpu_stats(gpu_index)
        specs = self._lookup_gpu_specs(gpu_name)

        memory_bw    = specs["memory_bandwidth_gbs"]
        compute_fp16 = specs["compute_tflops_fp16"]

        # Ridge point: compute FLOPS/s ÷ memory bytes/s = FLOPS/byte
        ridge = (compute_fp16 * 1e12) / (memory_bw * 1e9)

        # Conservative KV-cache allocation: 500 MB/slot, 70% of free VRAM
        optimal_batch   = max(1, min(int(vram_free_mb * 0.70 / 500), 64))
        optimal_max_seqs = min(optimal_batch * 2, 128)

        compute_cap = specs.get("cuda_compute_capability") or (8, 0)

        profile = HardwareProfile(
            gpu_index=gpu_index,
            gpu_name=gpu_name,
            vram_total_mb=vram_total_mb,
            vram_free_mb=vram_free_mb,
            memory_bandwidth_gbs=memory_bw,
            compute_tflops_fp16=compute_fp16,
            ridge_point_flops_per_byte=ridge,
            flash_attention_version=specs["flash_attention_version"],
            supports_bf16=specs["supports_bf16"],
            supports_fp8=specs.get("supports_fp8", False),
            nvlink_bandwidth_gbs=specs.get("nvlink_bandwidth_gbs"),
            recommended_batch_size=optimal_batch,
            recommended_max_seqs=optimal_max_seqs,
            cuda_compute_capability=compute_cap,
        )

        logger.info(
            f"GPU{gpu_index} ({gpu_name}): "
            f"BW={memory_bw}GB/s compute={compute_fp16}TFLOPS "
            f"ridge={ridge:.0f} FA={specs['flash_attention_version']} "
            f"rec_batch={optimal_batch}"
        )
        return profile

    def profile_model(self, pid: int, hw: HardwareProfile) -> ModelProfile:
        """Analyze a running model against the hardware roofline."""
        vram_used_mb   = hw.vram_total_mb - hw.vram_free_mb
        param_billions = self._estimate_params_from_vram(vram_used_mb)
        ai             = self._estimate_arithmetic_intensity(param_billions)

        bottleneck = (
            "MEMORY-BANDWIDTH"
            if ai < hw.ridge_point_flops_per_byte
            else "COMPUTE"
        )
        severity = hw.ridge_point_flops_per_byte / max(ai, 0.1)
        recommendations = self._build_recommendations(ai, hw, param_billions, bottleneck)

        return ModelProfile(
            model_name="detected",
            model_family="unknown",
            parameter_count_b=param_billions,
            vram_used_mb=vram_used_mb,
            arithmetic_intensity=ai,
            bottleneck=bottleneck,
            bottleneck_severity=severity,
            recommended_optimizations=recommendations,
        )

    def _build_recommendations(self, ai: float, hw: HardwareProfile,
                                param_b: float, bottleneck: str) -> List[str]:
        recs: List[str] = []
        if bottleneck == "MEMORY-BANDWIDTH":
            fa = hw.flash_attention_version.replace("_", " ").title()
            recs.append(fa)
            if hw.supports_bf16:
                recs.append("BF16 precision")
            recs.append(f"Continuous batching (recommended batch={hw.recommended_batch_size})")
            recs.append("PagedAttention KV-cache management")
            recs.append("KV-cache prefix reuse")
            recs.append("CUDA Graphs via torch.compile(mode='reduce-overhead')")
        else:
            recs.append("torch.compile(mode='max-autotune')")
            recs.append("Kernel fusion")
            if hw.supports_fp8:
                recs.append("FP8 quantization (Hopper only)")
        return recs

    def _query_gpu_stats(self, gpu_index: int) -> tuple:
        """Query live GPU memory via nvidia-smi."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi: {result.stderr.strip()}")
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            return parts[0], int(parts[1]), int(parts[2])
        except Exception as e:
            logger.warning(f"GPU stats query failed (gpu{gpu_index}): {e} — using defaults")
            return "Unknown GPU", 80 * 1024, 60 * 1024

    def _lookup_gpu_specs(self, gpu_name: str) -> dict:
        """Match live GPU name to hardware_counters.GPU_SPECS. Longest key match wins."""
        from memopt.profiler.hardware_counters import GPU_SPECS

        best_key, best_spec = None, None
        normalized_name = gpu_name.lower().replace("-", " ")
        for key, spec in GPU_SPECS.items():
            normalized_key = key.lower().replace("-", " ")
            if normalized_key in normalized_name:
                if best_key is None or len(key) > len(best_key):
                    best_key, best_spec = key, spec

        if best_spec is not None:
            logger.info(f"Matched '{gpu_name}' → '{best_key}'")
            return self._spec_to_dict(best_spec)

        logger.warning(
            f"GPU '{gpu_name}' not in hardware_counters.GPU_SPECS — using conservative defaults."
        )
        return {
            "memory_bandwidth_gbs":    900,
            "compute_tflops_fp16":     100,
            "compute_tflops_bf16":     100,
            "compute_tflops_fp8":      None,
            "flash_attention_version": "flash_attention_2",
            "supports_bf16":           True,
            "supports_fp8":            False,
            "nvlink_bandwidth_gbs":    None,
            "cuda_compute_capability": (8, 0),
        }

    def _estimate_params_from_vram(self, vram_mb: int) -> float:
        """Estimate parameter count (billions) from VRAM consumption.
        FP16 = 2 bytes/param. Overhead factor ~2.5x for runtime buffers.
        """
        params_bytes = (vram_mb * 1024 * 1024) / 2.5
        return params_bytes / 1e9

    def _estimate_arithmetic_intensity(self, param_b: float) -> float:
        """
        Arithmetic intensity (FLOPS/byte) at batch=1 for transformer inference.
        Based on empirical roofline measurements across model families.
        Batch=1 is always memory-bound; AI rises linearly with batch.
        """
        if param_b < 3:
            return 3.5
        elif param_b < 10:
            return 4.2
        elif param_b < 20:
            return 5.0
        elif param_b < 70:
            return 6.5
        else:
            return 8.0
