"""
GPU Power Sampler
=================

Samples GPU power via NVML every 100ms in a background thread.
Non-blocking — never slows down the model being measured.

Usage:
    with PowerSampler() as sampler:
        model(inputs)
    report = sampler.report(duration_ms=elapsed_ms)
    print(report.summary())

If pynvml is not available, PowerReport.unavailable() is returned — never crashes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

log = logging.getLogger("memopt.power")


# =============================================================================
# PowerReport — immutable result object
# =============================================================================

@dataclass
class PowerReport:
    avg_watts:        float
    peak_watts:       float
    idle_watts:       float
    active_watts:     float          # avg_watts - idle_watts (clamped >=0)
    joules:           float          # avg_watts × duration_s
    joules_per_token: Optional[float]
    tokens_per_watt:  Optional[float]
    sample_count:     int
    duration_ms:      float
    available:        bool = True

    @classmethod
    def unavailable(cls) -> "PowerReport":
        """Return when NVML is not accessible — never crash the caller."""
        return cls(
            avg_watts=0.0,
            peak_watts=0.0,
            idle_watts=0.0,
            active_watts=0.0,
            joules=0.0,
            joules_per_token=None,
            tokens_per_watt=None,
            sample_count=0,
            duration_ms=0.0,
            available=False,
        )

    def summary(self) -> str:
        if not self.available:
            return "Power data unavailable (NVML not accessible)"
        s = (
            f"avg={self.avg_watts:.1f}W "
            f"peak={self.peak_watts:.1f}W "
            f"active={self.active_watts:.1f}W "
            f"energy={self.joules:.2f}J"
        )
        if self.tokens_per_watt is not None:
            s += f" efficiency={self.tokens_per_watt:.2f}tok/W"
        return s


# =============================================================================
# PowerSampler — background-thread NVML poller
# =============================================================================

class PowerSampler:
    """
    Samples GPU power via pynvml every interval_ms milliseconds.

    Measures idle power for ~300ms at construction time so active_watts
    (avg - idle) is a meaningful signal.

    All exceptions are caught internally — never raises to caller.
    If NVML is unavailable, report() returns PowerReport.unavailable().
    """

    def __init__(self, device_index: int = 0, interval_ms: int = 100):
        self.device_index = device_index
        self.interval_ms  = interval_ms
        self._samples: List[float] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._idle_watts: Optional[float] = self._measure_idle()

    # ── Idle baseline (measured before any workload) ──────────────────────────

    def _measure_idle(self) -> Optional[float]:
        """
        Sample GPU power for ~300ms at construction time to get the idle baseline.
        Returns None if NVML is not available.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            readings: List[float] = []
            deadline = time.time() + 0.30   # 300ms
            while time.time() < deadline:
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    readings.append(mw / 1000.0)
                except Exception:
                    pass
                time.sleep(0.05)
            if readings:
                idle = sum(readings) / len(readings)
                log.debug("PowerSampler: idle baseline = %.1f W (%d samples)", idle, len(readings))
                return idle
        except Exception as exc:
            log.debug("PowerSampler: idle measurement unavailable (%s)", exc)
        return None

    # ── Start / stop (for long-running server processes) ────────────────────

    def start(self) -> None:
        """Start background sampling. Safe to call multiple times."""
        if self._running:
            return
        self._running = True
        self._samples = []
        self._thread = threading.Thread(
            target=self._sample_loop,
            daemon=True,
            name="memopt-power-sampler",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop background sampling."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "PowerSampler":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()

    # ── Background sampling loop ──────────────────────────────────────────────

    def _sample_loop(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            while self._running:
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)   # milliwatts
                    self._samples.append(mw / 1000.0)             # → watts
                except Exception as exc:
                    log.warning("Power sample failed: %s", exc)
                time.sleep(self.interval_ms / 1000.0)
        except Exception as exc:
            log.warning("PowerSampler thread failed: %s — no power data", exc)

    def current_avg_watts(self) -> float:
        """Return current average power in watts. 0.0 if no samples."""
        if not self._samples:
            return 0.0
        # Use last 10 samples for a recent average
        recent = self._samples[-10:] if len(self._samples) > 10 \
            else self._samples
        return sum(recent) / len(recent) if recent else 0.0

    @property
    def available(self) -> bool:
        """True if NVML is working and samples are being collected."""
        return self._running and len(self._samples) > 0

    # ── Report builder ────────────────────────────────────────────────────────

    def report(self, duration_ms: float, token_count: int = 0) -> PowerReport:
        """
        Build a PowerReport from collected samples.

        Args:
            duration_ms:  total measurement window in milliseconds
            token_count:  total tokens processed (for efficiency metrics)
        """
        if not self._samples:
            return PowerReport.unavailable()

        # Discard first 2 samples — GPU power stabilises after ~200ms
        samples = self._samples[2:] if len(self._samples) > 2 else self._samples

        avg_watts  = sum(samples) / len(samples)
        peak_watts = max(samples)
        idle_watts = self._idle_watts if self._idle_watts is not None else 60.0
        active_watts = max(0.0, avg_watts - idle_watts)

        duration_s = duration_ms / 1000.0
        joules = avg_watts * duration_s

        return PowerReport(
            avg_watts=avg_watts,
            peak_watts=peak_watts,
            idle_watts=idle_watts,
            active_watts=active_watts,
            joules=joules,
            joules_per_token=(joules / token_count) if token_count > 0 else None,
            tokens_per_watt=(token_count / avg_watts) if token_count > 0 and avg_watts > 0 else None,
            sample_count=len(samples),
            duration_ms=duration_ms,
            available=True,
        )
