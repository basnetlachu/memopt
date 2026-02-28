"""
Report formatter for memopt scan output.
Produces human-readable terminal reports and JSON output.
"""
import json
import logging
from typing import List, Optional

from .scanner import GPUProcess as ScannedProcess
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

    def _print_process(
        self,
        prof: ProcessProfile,
        index: int,
        total: int,
        color: bool,
    ) -> None:
        bneck_color = BOTTLENECK_COLOR.get(prof.bottleneck, _CYAN)
        bneck_label = BOTTLENECK_LABEL.get(prof.bottleneck, prof.bottleneck)

        vram_gb = prof.gpu_memory_mb / 1024

        print()
        print(_fmt(f"  [{index}/{total}] PID {prof.pid}", _BOLD, color))
        print(f"      GPU(s)   : {', '.join(str(g) for g in prof.gpu_ids)}"
              f"  ({prof.hw_name})")
        print(f"      VRAM     : {vram_gb:.1f} GB")
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

        print(_fmt("  " + "-" * 70, _DIM, color))

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
                "supports_fp8":         prof.supports_fp8,
                "supports_bf16":        prof.supports_bf16,
            })
        return json.dumps({"processes": data}, indent=indent)


def _fmt(text: str, code: str, enabled: bool) -> str:
    """Apply ANSI code if color is enabled."""
    if not enabled:
        return text
    return f"{code}{text}{_RESET}"
