"""
Report formatter for memopt scan output.
Produces human-readable terminal reports and JSON output.
"""
import json
import logging
from typing import List, Tuple

from .process_inspector import ProcessProfile, OPT_LABELS

log = logging.getLogger(__name__)

# ANSI color codes
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_DIM    = "\033[2m"

BOTTLENECK_COLOR = {
    "memory_bandwidth": _YELLOW,
    "compute":          _RED,
    "unknown":          _CYAN,
}

BOTTLENECK_LABEL = {
    "memory_bandwidth": "memory-bandwidth",
    "compute":          "compute",
    "unknown":          "unknown",
}


class ScanReporter:
    """
    Formats scan results for terminal or JSON output.

    Usage:
        reporter = ScanReporter()
        reporter.print_report(profiles)
        json_str = reporter.to_json(profiles)
    """

    def print_report(
        self,
        profiles: List[ProcessProfile],
        color: bool = True,
    ) -> None:
        """Print a human-readable scan report to stdout."""
        if not profiles:
            print(_fmt("No GPU processes found.", _DIM, color))
            return

        print()
        print(_fmt("=" * 72, _BOLD, color))
        print(_fmt(f"  memopt scan  —  {len(profiles)} GPU process(es) found", _BOLD, color))
        print(_fmt("=" * 72, _BOLD, color))

        for i, prof in enumerate(profiles):
            self._print_process(prof, i + 1, len(profiles), color)

        print()

    def _calculate_batch_opportunity(
        self,
        gpu_ids: List[int],
        model_vram_mb: int,
    ) -> Tuple[int, int, int, float]:
        """
        Query GPU free/total VRAM via pynvml and estimate optimal batch size.

        Returns (total_mb, free_mb, optimal_batch, batch_speedup).
        Returns (0, 0, 1, 1.0) on any error.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            try:
                total_mb = 0
                free_mb = 0
                for gpu_id in gpu_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    total_mb += mem.total // (1024 * 1024)
                    free_mb += mem.free // (1024 * 1024)
            finally:
                try:
                    pynvml.nvmlShutdown()
                except Exception:
                    pass
            # KV-cache per additional batch slot ≈ 15% of model VRAM
            kv_per_slot_mb = max(model_vram_mb * 0.15, 512)
            optimal_batch = min(int(free_mb / kv_per_slot_mb), 32)
            optimal_batch = max(optimal_batch, 1)
            batch_speedup = min(optimal_batch * 0.85, 6.0)
            return total_mb, free_mb, optimal_batch, batch_speedup
        except Exception:
            return 0, 0, 1, 1.0

    def _print_process(
        self,
        prof: ProcessProfile,
        index: int,
        total: int,
        color: bool,
    ) -> None:
        bneck_color = BOTTLENECK_COLOR.get(prof.bottleneck, _CYAN)
        bneck_label = BOTTLENECK_LABEL.get(prof.bottleneck, prof.bottleneck)

        vram_used_gb = prof.gpu_memory_mb / 1024
        total_mb, free_mb, optimal_batch, batch_speedup = self._calculate_batch_opportunity(
            prof.gpu_ids, prof.gpu_memory_mb
        )
        vram_total_gb = total_mb / 1024 if total_mb else 0.0
        vram_free_gb  = free_mb  / 1024 if free_mb  else 0.0

        vram_str = f"{vram_used_gb:.1f} GB used"
        if vram_total_gb > 0:
            vram_str += f"  /  {vram_total_gb:.0f} GB total  ({vram_free_gb:.1f} GB free)"

        print()
        print(_fmt(f"  [{index}/{total}] PID {prof.pid}", _BOLD, color))
        print(f"      GPU(s)   : {', '.join(str(g) for g in prof.gpu_ids)}"
              f"  ({prof.hw_name})")
        print(f"      VRAM     : {vram_str}")
        print(f"      Model    : {prof.model_family}  [{prof.mode}]")
        print(f"      GPU util : {prof.avg_utilization_pct:.0f}%")
        print(
            f"      Bottleneck: "
            + _fmt(bneck_label.upper(), bneck_color, color)
            + f"  (AI={prof.arithmetic_intensity:.1f}  ridge={prof.ridge_point:.0f} FLOPS/byte)"
        )

        if prof.recommendations:
            print(f"      Recommendations:")
            for j, key in enumerate(prof.recommendations, 1):
                label = OPT_LABELS.get(key, key)
                marker = "  →" if j == 1 else "   "
                line = f"        {marker} {j}. {label}"
                if j == 1:
                    line = _fmt(line, _GREEN, color)
                print(line)
            print()
            top_key = prof.recommendations[0]
            top_label = OPT_LABELS.get(top_key, top_key)
            print(
                _fmt(
                    f"      Run: memopt apply --pid {prof.pid}",
                    _BOLD, color,
                )
                + _fmt(
                    f"   # applies {top_label}",
                    _DIM, color,
                )
            )
        else:
            print(f"      No recommendations available.")

        # Show throughput opportunities for inference-mode transformer models
        is_transformer = prof.model_family not in ("resnet", "vit", "diffusion", "clip", "unknown")
        if prof.mode == "inference" and is_transformer:
            self._print_throughput_opportunities(prof, optimal_batch, batch_speedup, color)

        print(_fmt("  " + "-" * 70, _DIM, color))

    def _print_throughput_opportunities(
        self,
        prof: ProcessProfile,
        optimal_batch: int,
        batch_speedup: float,
        color: bool,
    ) -> None:
        """Print the THROUGHPUT OPPORTUNITIES box below recommendations."""
        sep = _fmt("  " + "─" * 66, _CYAN, color)
        header = _fmt(
            "  THROUGHPUT OPPORTUNITIES  (zero quality loss · float16 only)",
            _BOLD, color,
        )
        print()
        print(sep)
        print(header)
        print(sep)

        # Option 1: optimal batching
        batch_line = (
            f"  [1] Optimal batching      batch={optimal_batch}"
            f"  (~{batch_speedup:.1f}x throughput)"
        )
        print(_fmt(batch_line, _GREEN, color))
        print(_fmt(f"      memopt apply --pid {prof.pid} --mode batch", _DIM, color))

        # Option 2: vLLM continuous batching
        vllm_speedup = min(optimal_batch * 0.90, 7.0)
        vllm_line = f"  [2] vLLM continuous batching   (~{vllm_speedup:.1f}x throughput)"
        print(_fmt(vllm_line, _GREEN, color))
        print(_fmt(f"      memopt apply --pid {prof.pid} --mode vllm", _DIM, color))

        # Option 3: speculative decoding
        spec_line = "  [3] Speculative decoding       (~1.8x latency · TinyLlama draft)"
        print(_fmt(spec_line, _GREEN, color))
        print(_fmt(f"      memopt apply --pid {prof.pid} --mode speculative", _DIM, color))

        # Option 4: vLLM + speculative combined
        combined = min(vllm_speedup * 1.8 * 0.75, 8.0)
        combo_line = f"  [4] vLLM + speculative         (~{combined:.1f}x  recommended)"
        print(_fmt(combo_line, _GREEN, color))
        print(_fmt(f"      memopt apply --pid {prof.pid} --mode vllm+spec", _DIM, color))

        # Arithmetic intensity context
        ai = prof.arithmetic_intensity
        ridge = prof.ridge_point
        print()
        print(_fmt(
            f"  Memory bottleneck: AI={ai:.1f}  ({ridge/ai:.0f}x below ridge={ridge:.0f} FLOPS/byte)",
            _DIM, color,
        ))
        print(_fmt(
            f"  At batch={optimal_batch}: projected AI≈{ai*optimal_batch:.0f}"
            f"  ({int(100*(1-1/optimal_batch))}% of bottleneck eliminated)",
            _DIM, color,
        ))

        print(sep)

    def to_json(self, profiles: List[ProcessProfile], indent: int = 2) -> str:
        """Serialize profiles to JSON string."""
        data = []
        for prof in profiles:
            data.append({
                "pid":                  prof.pid,
                "gpu_ids":              prof.gpu_ids,
                "hw_name":              prof.hw_name,
                "hw_arch":              prof.hw_arch,
                "model_family":         prof.model_family,
                "mode":                 prof.mode,
                "gpu_memory_mb":        prof.gpu_memory_mb,
                "avg_utilization_pct":  prof.avg_utilization_pct,
                "bottleneck":           prof.bottleneck,
                "arithmetic_intensity": prof.arithmetic_intensity,
                "ridge_point":          prof.ridge_point,
                "recommendations": [
                    {"key": k, "label": OPT_LABELS.get(k, k)}
                    for k in prof.recommendations
                ],
                "supports_flash_attn2": prof.supports_flash_attn2,
                "supports_bf16":        prof.supports_bf16,
            })
        return json.dumps({"processes": data}, indent=indent)


def _fmt(text: str, code: str, enabled: bool) -> str:
    """Apply ANSI code if color is enabled."""
    if not enabled:
        return text
    return f"{code}{text}{_RESET}"
