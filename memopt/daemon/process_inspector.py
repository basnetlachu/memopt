"""
Process inspector for memopt daemon.
Profiles a GPU process to determine bottleneck type and recommendations.
Uses pynvml for utilization sampling + hardware_detector for roofline analysis.
Never crashes — all errors return safe defaults.
"""
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Arithmetic intensity estimates per model family (FLOPS/byte, FP16 inference)
_MODEL_FAMILY_AI = {
    "llama":   4.0,
    "llama2":  4.0,
    "llama3":  4.0,
    "mistral": 4.2,
    "mixtral": 2.8,   # MoE — lower effective AI
    "falcon":  3.5,
    "gemma":   4.1,
    "qwen":    4.0,
    "phi3":    5.2,
    "bert":    6.5,   # encoder-only — higher AI
    "gpt2":    3.8,
    "gptj":    3.8,
    "gptneox": 3.8,
    "resnet":  15.0,  # conv-heavy — compute-bound
    "vit":     8.0,
    "whisper": 5.5,
    "diffusion": 12.0,
    "clip":    7.0,
}

# Expected speedup range per optimization key: (min, max)
# Only techniques with zero quality degradation — no quantization.
_OPT_SPEEDUP: Dict[str, Tuple[float, float]] = {
    "flash_attention":   (1.5, 2.5),
    "bf16":              (1.2, 1.8),
    "torch_compile":     (1.1, 1.5),
    "kv_cache":          (1.3, 2.0),
    "continuous_batch":  (2.0, 5.0),
    "channels_last":     (1.1, 1.3),
    "tensor_parallel":   (1.5, 3.0),
}

# Optimization labels — float16 / zero-quality-loss techniques only
OPT_LABELS = {
    "flash_attention":   "Flash Attention 2",
    "bf16":              "BF16 precision",
    "torch_compile":     "torch.compile()",
    "kv_cache":          "KV-cache reuse",
    "channels_last":     "channels-last layout",
    "tensor_parallel":   "Tensor parallelism",
    "continuous_batch":  "Continuous batching",
}


@dataclass
class ProcessProfile:
    """Full bottleneck profile for one running GPU process."""
    pid: int
    gpu_ids: List[int]
    gpu_memory_mb: int
    avg_utilization_pct: float
    bottleneck: str                      # "memory_bandwidth" | "compute" | "unknown"
    arithmetic_intensity: float          # estimated FLOPS/byte
    ridge_point: float                   # hardware ridge point FLOPS/byte
    # Legacy field — used by report.py / ScanReporter
    recommendations: List[str] = field(default_factory=list)
    # Fields used by zero_touch.py
    recommended_optimizations: List[str] = field(default_factory=list)
    expected_speedup_min: float = 1.0
    expected_speedup_max: float = 1.0
    utilization_efficiency: float = 0.0  # avg_util / 100
    reasoning: str = ""
    # Hardware info
    model_family: str = ""
    mode: str = ""
    hw_arch: str = ""
    hw_name: str = ""
    supports_flash_attn2: bool = False
    supports_bf16: bool = False


class ProcessInspector:
    """
    Samples a running GPU process for ~N seconds and produces a ProcessProfile.

    Usage:
        inspector = ProcessInspector()
        profile = inspector.profile(
            pid=12345,
            gpu_ids=[0],
            gpu_memory_mb=14000,
            model_family="llama3",
            mode="inference",
            sample_seconds=5,
        )
    """

    def profile(
        self,
        pid: int,
        gpu_ids: List[int],
        gpu_memory_mb: int,
        model_family: str,
        mode: str,
        sample_seconds: int = 5,
    ) -> ProcessProfile:
        """
        Profile a process. Never raises.
        Returns a ProcessProfile with safe defaults on any error.
        """
        try:
            from memopt.utils.hardware_detector import detect_hardware
            hw = detect_hardware(device=gpu_ids[0] if gpu_ids else 0)
        except Exception as e:
            log.warning(f"hardware_detector failed for pid {pid}: {e}")
            hw = None

        ridge_point   = hw.ridge_point         if hw else 153.0  # A100 default
        hw_arch       = hw.arch                if hw else "unknown"
        hw_name       = hw.device_name         if hw else "unknown"
        supports_fa2  = hw.supports_flash_attn2 if hw else False
        supports_bf16 = hw.supports_bf16        if hw else False

        # Sample utilization
        avg_util = self._sample_utilization(gpu_ids, sample_seconds)

        # Estimate arithmetic intensity
        ai = _MODEL_FAMILY_AI.get(model_family, 4.0)

        # Diagnose bottleneck
        bottleneck = self._diagnose_bottleneck(
            avg_util, ai, ridge_point, model_family, mode
        )

        # Build recommendations
        recs = self._build_recommendations(
            bottleneck=bottleneck,
            model_family=model_family,
            mode=mode,
            gpu_memory_mb=gpu_memory_mb,
            hw_arch=hw_arch,
            supports_fa2=supports_fa2,
            supports_bf16=supports_bf16,
        )

        # Estimate speedup from top recommendation
        top_key = recs[0] if recs else None
        sp_min, sp_max = _OPT_SPEEDUP.get(top_key, (1.0, 1.0)) if top_key else (1.0, 1.0)

        # Build reasoning string
        reasoning = (
            f"{bottleneck.replace('_', '-')} bottleneck | "
            f"AI={ai:.1f} ridge={ridge_point:.0f} FLOPS/byte | "
            f"util={avg_util:.0f}%"
        )

        return ProcessProfile(
            pid=pid,
            gpu_ids=gpu_ids,
            model_family=model_family,
            mode=mode,
            gpu_memory_mb=gpu_memory_mb,
            avg_utilization_pct=avg_util,
            bottleneck=bottleneck,
            arithmetic_intensity=ai,
            ridge_point=ridge_point,
            recommendations=recs,
            recommended_optimizations=recs,   # mirror for zero_touch.py
            expected_speedup_min=sp_min,
            expected_speedup_max=sp_max,
            utilization_efficiency=avg_util / 100.0,
            reasoning=reasoning,
            hw_arch=hw_arch,
            hw_name=hw_name,
            supports_flash_attn2=supports_fa2,
            supports_bf16=supports_bf16,
        )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _sample_utilization(self, gpu_ids: List[int], sample_seconds: int) -> float:
        """Sample GPU utilization over sample_seconds. Returns average pct."""
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                samples = []
                end_time = time.time() + sample_seconds
                while time.time() < end_time:
                    for gpu_id in gpu_ids:
                        try:
                            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                            samples.append(util.gpu)
                        except pynvml.NVMLError:
                            pass
                    time.sleep(0.5)
                return sum(samples) / len(samples) if samples else 0.0
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Utilization sampling failed: {e}")
            return 0.0

    def _diagnose_bottleneck(
        self,
        avg_util: float,
        ai: float,
        ridge_point: float,
        model_family: str,
        mode: str,
    ) -> str:
        if ridge_point <= 0:
            if avg_util > 85:
                return "compute"
            if avg_util < 40:
                return "memory_bandwidth"
            return "unknown"

        on_compute_side = ai >= ridge_point
        high_util = avg_util > 85
        low_util  = avg_util < 40

        if high_util and on_compute_side:
            return "compute"
        if low_util or not on_compute_side:
            return "memory_bandwidth"
        return "unknown"

    def _build_recommendations(
        self,
        bottleneck: str,
        model_family: str,
        mode: str,
        gpu_memory_mb: int,
        hw_arch: str,
        supports_fa2: bool,
        supports_bf16: bool,
    ) -> List[str]:
        """
        Build recommendations — float16 / zero quality-loss techniques only.
        No quantization (INT8, FP8, GPTQ, AWQ) is ever recommended.
        """
        recs = []
        is_transformer = model_family not in ("resnet", "vit", "diffusion", "clip", "unknown")

        if bottleneck == "memory_bandwidth":
            if is_transformer and supports_fa2:
                recs.append("flash_attention")
            if supports_bf16:
                recs.append("bf16")
            if mode == "inference" and is_transformer:
                recs.append("kv_cache")
            if mode == "inference" and is_transformer:
                recs.append("continuous_batch")
            recs.append("torch_compile")

        elif bottleneck == "compute":
            recs.append("torch_compile")
            if supports_bf16:
                recs.append("bf16")
            if is_transformer and supports_fa2:
                recs.append("flash_attention")
            if model_family in ("resnet", "vit"):
                recs.append("channels_last")

        else:
            if is_transformer and supports_fa2:
                recs.append("flash_attention")
            if supports_bf16:
                recs.append("bf16")
            recs.append("torch_compile")

        # Deduplicate preserving order
        seen: set = set()
        result = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                result.append(r)
        return result
