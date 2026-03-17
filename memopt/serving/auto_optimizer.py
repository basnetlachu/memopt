"""
Auto-optimizer — background service that synthesises fused kernels
during live inference and registers them in the kernel cache.

Lifecycle:
  1. Server starts → AutoOptimizer.start() called once
  2. Each request calls kernel_hooks.apply_rope() etc.
  3. Hooks call optimizer.notify() with the op name and args
  4. Optimizer tracks call frequency and estimated stall rate
  5. After WARM_UP_CALLS calls on the same op, fires synthesis
  6. Synthesis runs in a daemon thread — never blocks inference
  7. On success, kernel is in cache — next request uses fused path

The warm-up period (default 50 calls) exists so synthesis happens
on shapes that are actually used in production, not cold-start shapes
that may differ from steady-state traffic.

Thread safety: all internal state is protected by threading.RLock.
"""
from __future__ import annotations
import logging
import os
import threading
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

WARM_UP_CALLS    = 50      # calls before triggering synthesis
SYNTHESIS_GAP_S  = 300.0   # seconds between re-synthesis attempts per op
STALL_RATE_PROXY = float(os.environ.get("MEMOPT_STALL_RATE_PROXY", "0.65"))


class AutoOptimizer:
    """
    Wires BottleneckDetector + JITGenerator into the serving layer.

    Usage (called once at server startup):
        cache     = KernelCache()
        portl     = PortabilityLayer()
        gen       = JITGenerator(cache=cache, portability=portl)
        optimizer = AutoOptimizer(generator=gen, cache=cache)
        optimizer.start()
        kernel_hooks.init_hooks(cache=cache, optimizer=optimizer)
    """

    def __init__(self, generator, cache):
        self._gen            = generator
        self._cache          = cache
        self._lock           = threading.RLock()
        self._call_counts:   dict = defaultdict(int)
        self._last_synthesis: dict = {}
        self._running        = False
        self._total_notified = 0
        self._total_fired    = 0

    def start(self) -> None:
        """Start the optimizer. Safe to call multiple times."""
        with self._lock:
            if self._running:
                return
            self._running = True
        logger.info("AutoOptimizer started")

    def stop(self) -> None:
        with self._lock:
            self._running = False

    def notify(self, op_name: str, args: tuple) -> None:
        """
        Called by each kernel hook on every request.
        Increments call count. After WARM_UP_CALLS, fires synthesis
        once if no cached kernel exists for this op + shapes.
        """
        if not self._running:
            return

        with self._lock:
            self._total_notified += 1
            self._call_counts[op_name] += 1
            count = self._call_counts[op_name]

        if count < WARM_UP_CALLS:
            return

        # Check if synthesis already done or in progress
        from memopt.kernels.kernel_cache import cache_key
        hw  = self._get_hardware()
        key = cache_key(op_name, self._extract_shapes(args), hw)

        if self._cache.get(key) is not None:
            return   # already cached

        with self._lock:
            last = self._last_synthesis.get(op_name, 0)
            if time.monotonic() - last < SYNTHESIS_GAP_S:
                return
            self._last_synthesis[op_name] = time.monotonic()
            self._total_fired += 1

        # Fire synthesis in background
        threading.Thread(
            target=self._fire_synthesis,
            args=(op_name, args, hw),
            daemon=True,
        ).start()

    def stats(self) -> dict:
        with self._lock:
            return {
                "running":         self._running,
                "total_notified":  self._total_notified,
                "total_fired":     self._total_fired,
                "tracked_ops":     dict(self._call_counts),
                "generator_stats": self._gen.stats(),
            }

    def _fire_synthesis(self, op_name: str, args: tuple, hw: str) -> None:
        from memopt.kernels.bottleneck_detector import BottleneckEvent
        shapes = self._extract_shapes(args)
        dtype  = self._extract_dtype(args)

        # Measure real stall rate if CUDA events are available
        stall_rate = self._measure_stall_rate(args)

        event = BottleneckEvent(
            op_name=op_name,
            input_shapes=shapes,
            dtype=dtype,
            access_pattern="sequential",
            stall_rate=stall_rate,
            hardware=hw,
        )

        original_build = self._gen._build_prompt
        self._gen._build_prompt = lambda ev: self._build_prompt(ev)
        try:
            self._gen.handle(event)
        finally:
            self._gen._build_prompt = original_build

        logger.info(
            f"AutoOptimizer: synthesis fired for {op_name} "
            f"stall_rate={stall_rate:.2f} shapes={shapes}"
        )

    def _measure_stall_rate(self, args: tuple) -> float:
        """
        Measure real HBM stall rate using CUDA event timing.
        Returns STALL_RATE_PROXY if CUDA is unavailable or measurement fails.

        Method: compare actual elapsed time to the theoretical minimum
        time based on bytes transferred and peak HBM bandwidth.
        Gap between ideal and actual is attributed to memory stalls.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return STALL_RATE_PROXY

            tensors = [a for a in args if isinstance(a, torch.Tensor)]
            if not tensors:
                return STALL_RATE_PROXY

            total_bytes = sum(
                t.numel() * t.element_size() for t in tensors
            )
            if total_bytes < 1024:
                return STALL_RATE_PROXY

            props    = torch.cuda.get_device_properties(0)
            peak_bw  = props.memory_clock_rate * 1e3 * props.memory_bus_width / 8

            # Time a no-op tensor copy as a bandwidth probe
            probe = tensors[0].clone()
            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = probe + probe   # force HBM read + write
            end.record()
            torch.cuda.synchronize()
            elapsed_s = start.elapsed_time(end) / 1000.0

            if elapsed_s <= 0:
                return STALL_RATE_PROXY

            probe_bytes = probe.numel() * probe.element_size() * 3
            ideal_s     = probe_bytes / peak_bw
            stall       = max(0.0, min(0.95, 1.0 - ideal_s / elapsed_s))
            logger.debug(
                f"Measured stall rate: {stall:.2f} "
                f"(elapsed={elapsed_s*1000:.3f}ms ideal={ideal_s*1000:.3f}ms)"
            )
            return stall

        except Exception as e:
            logger.debug(f"Stall rate measurement failed: {e} — using proxy")
            return STALL_RATE_PROXY

    def _build_prompt(self, event) -> str:
        """Route to op-specific prompt builders with shape validation."""
        from memopt.kernels.tests.test_pillar3_fused import (
            _make_rope_prompt,
            _make_ln_residual_prompt,
            _make_softmax_scale_prompt,
        )
        import math  # noqa: F401

        op = event.op_name
        s  = event.input_shapes
        hw = event.hardware

        if op == "memopt.rope_fused":
            if len(s) >= 2 and len(s[0]) == 3:
                seq, n_heads, head_dim = s[0]
                return _make_rope_prompt(seq, n_heads, head_dim, event.dtype, hw)
            logger.warning(
                f"AutoOptimizer: rope_fused expected 3D shapes, got {s} "
                f"— falling back to generic prompt"
            )

        elif op == "memopt.ln_residual_fused":
            if len(s) >= 1 and len(s[0]) == 2:
                seq, hidden = s[0]
                return _make_ln_residual_prompt(seq, hidden, event.dtype, hw)
            logger.warning(
                f"AutoOptimizer: ln_residual_fused expected 2D shapes, got {s} "
                f"— falling back to generic prompt"
            )

        elif op == "memopt.scaled_softmax_fused":
            if len(s) >= 1 and len(s[0]) == 4:
                b, nh, sq, sk = s[0]
                return _make_softmax_scale_prompt(b, nh, sq, event.dtype, hw)
            logger.warning(
                f"AutoOptimizer: scaled_softmax_fused expected 4D shapes, got {s} "
                f"— falling back to generic prompt"
            )

        # Generic fallback — delegates to JITGenerator's own prompt builder
        # Uses direct instance method call, not fragile __class__ lookup
        return self._gen._build_prompt(event)

    def _extract_shapes(self, args: tuple) -> list:
        try:
            import torch
            return [list(a.shape) for a in args if isinstance(a, torch.Tensor)]
        except Exception:
            return []

    def _extract_dtype(self, args: tuple) -> str:
        try:
            import torch
            for a in args:
                if isinstance(a, torch.Tensor):
                    return str(a.dtype).replace("torch.", "")
        except Exception:
            pass
        return "float16"

    def _get_hardware(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return f"cuda:{torch.cuda.get_device_name(0)}"
        except Exception:
            pass
        return "cpu"
