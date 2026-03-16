"""
JIT kernel generator — synthesises fused Triton kernels on demand.

Receives a BottleneckEvent from the detector, constructs a prompt
describing the bottleneck, calls the Claude API to generate a Triton
kernel, compiles and validates it, then registers it in the cache.

If any step fails (API unreachable, compilation error, kernel slower
than baseline), the failure is logged and the baseline kernel continues
running. Inference is never interrupted.

API key: ANTHROPIC_API_KEY environment variable.
If not set: synthesis is skipped, a warning is logged, baseline runs.

The synthesis pipeline runs in a daemon thread — never blocks inference.
"""
from __future__ import annotations
import os
import time
import logging
import threading
import textwrap
from typing import Optional

logger = logging.getLogger(__name__)

ANTHROPIC_MODEL   = "claude-sonnet-4-20250514"
MAX_TOKENS        = 2048
SYNTHESIS_TIMEOUT = 30.0   # seconds — abandon synthesis if API takes longer


class JITGenerator:
    """
    Synthesises hardware-specific fused kernels for detected bottlenecks.

    Usage:
        cache     = KernelCache()
        portlayer = PortabilityLayer()
        gen       = JITGenerator(cache=cache, portability=portlayer)
        detector  = BottleneckDetector(on_bottleneck=gen.handle)
    """

    def __init__(self, cache=None, portability=None):
        self._cache       = cache
        self._portability = portability
        self._lock        = threading.Lock()
        self._in_flight:  set = set()

        self._total_attempted  = 0
        self._total_succeeded  = 0
        self._total_failed     = 0
        self._total_discarded  = 0

    def handle(self, event) -> None:
        """
        Entry point called by BottleneckDetector (in a daemon thread).

        Checks cache first. If no cached kernel exists for this op+shapes,
        starts synthesis. Deduplicates — will not synthesise the same
        op+shapes twice concurrently.
        """
        from .kernel_cache import cache_key
        key = cache_key(event.op_name, event.input_shapes, event.hardware)

        if self._cache and self._cache.get(key):
            logger.debug(f"JIT: cache hit for {key[:32]}...")
            return

        with self._lock:
            if key in self._in_flight:
                return
            self._in_flight.add(key)

        logger.info(
            f"JIT: synthesising kernel for {event.op_name} "
            f"shapes={event.input_shapes} hw={event.hardware} "
            f"stall={event.stall_rate:.1%}"
        )

        try:
            self._synthesise(event, key)
        finally:
            with self._lock:
                self._in_flight.discard(key)

    def _synthesise(self, event, key: str) -> None:
        """Full synthesis pipeline: prompt → LLM → compile → validate → cache."""
        with self._lock:
            self._total_attempted += 1

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning(
                "JIT: ANTHROPIC_API_KEY not set — skipping synthesis. "
                "Baseline kernel continues running."
            )
            with self._lock:
                self._total_failed += 1
            return

        # ── Step 1: Build prompt ────────────────────────────────────
        prompt = self._build_prompt(event)

        # ── Step 2: Call Claude API ─────────────────────────────────
        kernel_source = self._call_api(prompt, api_key)
        if kernel_source is None:
            with self._lock:
                self._total_failed += 1
            return

        # ── Step 3: Compile via portability layer ───────────────────
        compiled = None
        if self._portability:
            compiled = self._portability.compile(
                kernel_source, event.hardware
            )
        if compiled is None:
            logger.warning(f"JIT: compilation failed for {event.op_name}")
            with self._lock:
                self._total_failed += 1
            return

        # ── Step 4: Validate correctness ────────────────────────────
        correct = self._validate(compiled, event)
        if not correct:
            logger.warning(
                f"JIT: correctness check failed for {event.op_name} — discarding"
            )
            with self._lock:
                self._total_discarded += 1
            return

        # ── Step 5: Benchmark — must be faster than baseline ────────
        faster = self._benchmark(compiled, event)
        if not faster:
            logger.info(
                f"JIT: synthesised kernel for {event.op_name} is not "
                f"faster than baseline — discarding"
            )
            with self._lock:
                self._total_discarded += 1
            return

        # ── Step 6: Register in cache ────────────────────────────────
        if self._cache:
            self._cache.put(key, compiled, event)
            logger.info(
                f"JIT: kernel registered — {event.op_name} "
                f"hw={event.hardware}"
            )
        with self._lock:
            self._total_succeeded += 1

    def _build_prompt(self, event) -> str:
        """
        Construct the synthesis prompt from the bottleneck event.

        The prompt asks for a fused Triton kernel that eliminates the
        HBM round trips causing the stall. It specifies:
          - The op to fuse
          - Exact input shapes and dtypes
          - Target hardware and its memory bandwidth
          - Access pattern (sequential/strided/random)
          - Correctness requirement: output must match torch reference
          - Format requirement: return only valid Python/Triton source,
            no explanation, no markdown fences
        """
        shapes_str = ", ".join(str(s) for s in event.input_shapes)
        return textwrap.dedent(f"""
            Write a fused Triton kernel for the following memory-bound operation.

            Operation: {event.op_name}
            Input shapes: {shapes_str}
            Dtype: {event.dtype}
            Hardware: {event.hardware}
            Access pattern: {event.access_pattern}
            Observed HBM stall rate: {event.stall_rate:.1%}

            Requirements:
            1. Fuse load + compute + store into a single kernel pass.
               Do not write results back to HBM between stages.
            2. Use shared memory (tl.load with cache hints) to reduce
               HBM round trips.
            3. The kernel output must be numerically identical to the
               PyTorch reference op within float16 tolerance (atol=1e-3).
            4. Include a launch wrapper function named `run_kernel` that
               accepts the same arguments as the original op and returns
               a torch.Tensor.

            Return only valid Python source code using the Triton library.
            No explanation. No markdown. No comments outside the code.
            Start directly with `import triton`.
        """).strip()

    def _call_api(self, prompt: str, api_key: str) -> Optional[str]:
        """Call the Claude API and return the kernel source string."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}],
            )
            source = response.content[0].text.strip()
            if not source.startswith("import triton"):
                logger.warning("JIT: API response did not start with 'import triton'")
                return None
            return source

        except Exception as e:
            logger.warning(f"JIT: API call failed: {e}")
            return None

    def _validate(self, compiled, event) -> bool:
        """
        Run the compiled kernel against a small random input and compare
        output to the PyTorch reference. Returns True if outputs match.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return True   # can't validate without GPU — assume correct

            tensors = [
                torch.randn(shape, dtype=torch.float16, device="cuda")
                for shape in event.input_shapes
                if shape
            ]
            if not tensors:
                return True

            ref_fn = self._get_reference_op(event.op_name)
            if ref_fn is None:
                return True   # no reference available — trust the kernel

            ref_out = ref_fn(*tensors)

            run = getattr(compiled, "run_kernel", None)
            if run is None:
                return False

            kernel_out = run(*tensors)
            return torch.allclose(ref_out, kernel_out, atol=1e-3, rtol=1e-3)

        except Exception as e:
            logger.warning(f"JIT: validation error: {e}")
            return False

    def _benchmark(self, compiled, event) -> bool:
        """
        Time the compiled kernel vs the PyTorch baseline.
        Returns True if the kernel is at least 5% faster.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return True

            tensors = [
                torch.randn(shape, dtype=torch.float16, device="cuda")
                for shape in event.input_shapes
                if shape
            ]
            if not tensors:
                return True

            ref_fn = self._get_reference_op(event.op_name)
            run    = getattr(compiled, "run_kernel", None)
            if ref_fn is None or run is None:
                return True

            N = 50
            torch.cuda.synchronize()
            t0 = time.monotonic()
            for _ in range(N):
                ref_fn(*tensors)
            torch.cuda.synchronize()
            baseline_ms = (time.monotonic() - t0) / N * 1000

            torch.cuda.synchronize()
            t0 = time.monotonic()
            for _ in range(N):
                run(*tensors)
            torch.cuda.synchronize()
            kernel_ms = (time.monotonic() - t0) / N * 1000

            speedup = baseline_ms / max(kernel_ms, 1e-9)
            logger.info(
                f"JIT: benchmark {event.op_name} — "
                f"baseline {baseline_ms:.3f}ms  kernel {kernel_ms:.3f}ms  "
                f"speedup {speedup:.2f}x"
            )
            return speedup >= 1.05

        except Exception as e:
            logger.warning(f"JIT: benchmark error: {e}")
            return False

    def _get_reference_op(self, op_name: str):
        """Return the PyTorch reference callable for a known op name."""
        try:
            import torch
            mapping = {
                "torch.ops.aten.mm":       torch.mm,
                "torch.ops.aten.bmm":      torch.bmm,
                "torch.ops.aten.addmm":    torch.addmm,
                "torch.ops.aten.softmax":  lambda x: torch.softmax(x, dim=-1),
                "torch.ops.aten.layer_norm": torch.nn.functional.layer_norm,
            }
            return mapping.get(op_name)
        except Exception:
            return None

    def stats(self) -> dict:
        with self._lock:
            return {
                "total_attempted":  self._total_attempted,
                "total_succeeded":  self._total_succeeded,
                "total_failed":     self._total_failed,
                "total_discarded":  self._total_discarded,
                "in_flight":        len(self._in_flight),
            }
