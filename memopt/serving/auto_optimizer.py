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
import threading
import time
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

WARM_UP_CALLS    = 50      # calls before triggering synthesis
SYNTHESIS_GAP_S  = 300.0   # seconds between re-synthesis attempts per op
STALL_RATE_PROXY = 0.65    # assumed stall rate for hooks (not profiled inline)


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

        event = BottleneckEvent(
            op_name=op_name,
            input_shapes=shapes,
            dtype=dtype,
            access_pattern="sequential",
            stall_rate=STALL_RATE_PROXY,
            hardware=hw,
        )

        # Override prompt for known ops
        original_build = self._gen._build_prompt
        self._gen._build_prompt = lambda ev: self._build_prompt(ev)
        try:
            self._gen.handle(event)
        finally:
            self._gen._build_prompt = original_build

        logger.info(f"AutoOptimizer: synthesis fired for {op_name}")

    def _build_prompt(self, event) -> str:
        """Route to op-specific prompt builders from the validated test suite."""
        op  = event.op_name
        s   = event.input_shapes
        hw  = event.hardware

        # Import prompt builders from test suite — these are the exact prompts
        # that produced the validated kernels (RoPE 5.18x, LN 1.30x, SM 1.32x).
        # Wrapped in try/except because test modules have pytest guards at top level.
        try:
            from memopt.kernels.tests.test_pillar3_fused import (
                _make_rope_prompt,
                _make_ln_residual_prompt,
                _make_softmax_scale_prompt,
            )

            if op == "memopt.rope_fused" and len(s) >= 2 and len(s[0]) == 3:
                seq, n_heads, head_dim = s[0]
                return _make_rope_prompt(seq, n_heads, head_dim, event.dtype, hw)

            if op == "memopt.ln_residual_fused" and len(s) >= 1 and len(s[0]) == 2:
                seq, hidden = s[0]
                return _make_ln_residual_prompt(seq, hidden, event.dtype, hw)

            if op == "memopt.scaled_softmax_fused" and len(s) >= 1 and len(s[0]) == 4:
                import math
                b, nh, sq, _ = s[0]
                return _make_softmax_scale_prompt(b, nh, sq, event.dtype, hw)

        except Exception as e:
            logger.debug(f"AutoOptimizer: prompt builder import failed: {e}")

        # Fallback to generic JITGenerator prompt
        return self._gen.__class__._build_prompt(self._gen, event)

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
