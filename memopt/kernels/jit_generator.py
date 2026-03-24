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

ANTHROPIC_MODEL   = os.environ.get("MEMOPT_LLM_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS        = 2048
SYNTHESIS_TIMEOUT = 30.0   # seconds — abandon synthesis if API takes longer

# Retry / circuit breaker constants
_MAX_RETRIES        = 3
_RETRY_BASE_S       = 1.0    # first retry after 1s, then 2s, then 4s
_CIRCUIT_OPEN_S     = 300.0  # stop retrying for 5 min after 3 consecutive failures

# Circuit breaker state — module level so all JITGenerator instances share it
_circuit_failures    = 0
_circuit_opened_at   = 0.0
_circuit_lock        = threading.Lock()

_DTYPE_TOLERANCES = {
    "float16":  (1e-2, 1e-2),   # FP16 has ~3 decimal digits of precision
    "bfloat16": (1e-2, 1e-2),   # BF16 same range as FP16
    "float32":  (1e-5, 1e-5),   # FP32 has ~7 decimal digits
    "float64":  (1e-8, 1e-8),   # FP64 has ~15 decimal digits
}


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

    def generate(self, op_name: str, hardware_profile=None,
                 source_context: str = "",
                 previous_attempt: Optional[dict] = None):
        """
        Generate a Triton kernel for op_name with feedback loop.

        previous_attempt: metadata dict from KernelCache.get_metadata().
        If provided, injects performance history into the prompt so each
        synthesis attempt can improve on the last.

        Returns the compiled module, or None on failure.
        """
        # Auto-load previous attempt from cache if not provided
        if previous_attempt is None:
            try:
                from memopt.kernels.kernel_cache import KernelCache
                cache = KernelCache()
                previous_attempt = cache.get_metadata(op_name)
                if previous_attempt:
                    logger.info(
                        "JITGenerator: found previous attempt for %s "
                        "(speedup=%.2fx, memory_bound=%s)",
                        op_name,
                        previous_attempt.get('speedup', 0),
                        previous_attempt.get('is_memory_bound', '?'),
                    )
            except Exception:
                pass

        # Build previous attempt context for the prompt
        prev_context = ""
        if previous_attempt:
            speedup      = previous_attempt.get('speedup')
            is_mem_bound = previous_attempt.get('is_memory_bound')
            tiling       = previous_attempt.get('tiling_config')
            stall_rate   = previous_attempt.get('stall_rate')

            prev_context = "\n\nPREVIOUS SYNTHESIS ATTEMPT:\n"

            if speedup is not None:
                prev_context += f"- Achieved speedup: {speedup:.2f}x\n"

            if is_mem_bound is not None:
                bound_type = (
                    "memory-bandwidth bound" if is_mem_bound
                    else "compute bound"
                )
                prev_context += f"- Bottleneck: {bound_type}\n"

            if tiling:
                prev_context += f"- Tiling used: {tiling}\n"

            if stall_rate is not None:
                prev_context += f"- HBM stall rate: {stall_rate:.1%}\n"

            prev_context += (
                "\nIMPROVEMENT DIRECTIVE:\n"
                "Based on the above, synthesise a better kernel.\n"
            )

            if is_mem_bound:
                prev_context += (
                    "The previous kernel was memory-bandwidth bound. "
                    "Try: larger tiles to increase arithmetic intensity, "
                    "vectorised loads (float4), software prefetch, "
                    "or shared memory tiling to reduce global memory traffic.\n"
                )
            else:
                prev_context += (
                    "The previous kernel was compute bound. "
                    "Try: improved register reuse, "
                    "reduced synchronisation barriers, "
                    "or pipeline parallelism between memory and compute.\n"
                )

        # Build architecture context
        arch_name = "unknown"
        if hardware_profile is not None:
            arch_name = getattr(hardware_profile, 'arch_name', 'unknown')

        prompt = textwrap.dedent(f"""\
            Synthesise a fused Triton kernel for: {op_name}
            Architecture: {arch_name}
            Context: {source_context}
            {prev_context}
        """).strip()

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        kernel_source = self._call_api(prompt, api_key)
        if kernel_source is None:
            return None

        # Compile via portability layer if available
        if self._portability:
            compiled = self._portability.compile(kernel_source, "")
            return compiled

        return None

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
        Construct an architecture-aware synthesis prompt from the bottleneck event.

        Calls self._portability.profile() (if available) to inject the correct
        BLOCK_SIZE, safe tile dimensions, and architecture name so the LLM
        generates a kernel that compiles and runs correctly on the target hardware.

        For CPU / Apple M (use_triton=False), the prompt requests plain PyTorch
        ops instead of Triton so TorchCompileStrategy can torch.compile the output.
        """
        shapes_str = ", ".join(str(s) for s in event.input_shapes)

        # ── Gather architecture context ──────────────────────────────
        hw_profile = None
        if self._portability is not None:
            try:
                hw_profile = self._portability.profile()
            except Exception:
                pass

        if hw_profile is not None:
            arch_name      = hw_profile.arch_name
            compute_cap    = hw_profile.compute_cap
            max_block_size = hw_profile.max_block_size
            safe_tile_m    = hw_profile.safe_tile_m
            safe_tile_n    = hw_profile.safe_tile_n
            safe_tile_k    = hw_profile.safe_tile_k
            use_triton     = hw_profile.use_triton
        else:
            arch_name      = "unknown"
            compute_cap    = ""
            max_block_size = 64
            safe_tile_m    = 64
            safe_tile_n    = 64
            safe_tile_k    = 32
            use_triton     = True   # optimistic default

        cap_line = f"Compute capability: sm{compute_cap.replace('.', '')}" if compute_cap else ""

        if use_triton:
            lang_instructions = textwrap.dedent(f"""
                Write a fused Triton kernel.

                Architecture constraints (MUST follow to avoid compilation errors):
                  - Architecture: {arch_name}  {cap_line}
                  - Maximum safe BLOCK_SIZE: {max_block_size}  (use this or smaller, power-of-2)
                  - Safe GEMM tile dimensions: M={safe_tile_m}, N={safe_tile_n}, K={safe_tile_k}
                  - tl.arange arguments MUST be compile-time integer literals — NEVER variables.
                  - DO NOT use tl.cat, tl.join, or tensor slicing like x[:64].
                    Instead use SEPARATE tl.load calls for each half of the tensor.
                  - Embed all tile sizes as integer literals in the source, not as parameters.

                Requirements:
                1. Fuse load + compute + store into a single kernel pass.
                   Do not write results back to HBM between stages.
                2. The kernel output must match the PyTorch reference within
                   float16 tolerance (atol=1e-3).
                3. Include a launch wrapper named `run_kernel` that accepts the
                   same arguments as the original op and returns a torch.Tensor.

                Return only valid Python source code using the Triton library.
                No explanation. No markdown fences. No comments outside the code.
                Start directly with `import triton`.
            """).strip()
        else:
            lang_instructions = textwrap.dedent("""
                Write a fused PyTorch implementation (no Triton — target is CPU or Apple MPS).

                Requirements:
                1. Use only torch / torch.nn.functional — no triton imports.
                2. Fuse operations to minimise Python overhead (single expression if possible).
                3. The output must match the reference within float32 tolerance (atol=1e-5).
                4. Include a function named `run_kernel` that accepts the same arguments
                   as the original op and returns a torch.Tensor.

                Return only valid Python source. No markdown fences. No explanation.
                Start directly with `import torch`.
            """).strip()

        return textwrap.dedent(f"""
            {lang_instructions}

            Operation: {event.op_name}
            Input shapes: {shapes_str}
            Dtype: {event.dtype}
            Hardware: {event.hardware}
            Architecture: {arch_name}
            Access pattern: {event.access_pattern}
            Observed HBM stall rate: {event.stall_rate:.1%}
        """).strip()

    def _call_api(self, prompt: str, api_key: str) -> Optional[str]:
        """
        Call the Claude API with exponential backoff and circuit breaker.

        Circuit breaker opens after _MAX_RETRIES consecutive failures and
        stays open for _CIRCUIT_OPEN_S seconds. While open, all synthesis
        attempts fail immediately without hitting the API — this prevents
        a cascade of blocked threads during an outage.

        Retryable: rate limit (429), server error (500, 502, 503).
        Non-retryable: auth error (401), bad request (400).
        """
        global _circuit_failures, _circuit_opened_at

        # Check circuit breaker
        with _circuit_lock:
            if _circuit_failures >= _MAX_RETRIES:
                elapsed = time.monotonic() - _circuit_opened_at
                if elapsed < _CIRCUIT_OPEN_S:
                    logger.warning(
                        f"JIT circuit breaker OPEN — API synthesis suspended "
                        f"for {_CIRCUIT_OPEN_S - elapsed:.0f}s more. "
                        f"Inference continues on unfused path."
                    )
                    return None
                else:
                    # Half-open: allow one probe attempt
                    _circuit_failures = 0
                    logger.info("JIT circuit breaker half-open — probing API")

        last_error = None
        for attempt in range(_MAX_RETRIES):
            try:
                import anthropic
                client   = anthropic.Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=MAX_TOKENS,
                    messages=[{"role": "user", "content": prompt}],
                )
                source = response.content[0].text.strip()

                # Strip markdown fences (exact match only)
                _FENCE_OPENINGS = {"```", "```python", "```triton", "```cuda"}
                _FENCE_CLOSINGS = {"```"}
                lines = source.splitlines()
                if lines and lines[0].strip() in _FENCE_OPENINGS:
                    lines = lines[1:]
                    if lines and lines[-1].strip() in _FENCE_CLOSINGS:
                        lines = lines[:-1]
                    source = "\n".join(lines).strip()

                if not source.startswith(("import triton", "import torch")):
                    logger.warning(
                        "JIT: API response did not start with import — discarding"
                    )
                    # Not a retryable error — the model returned something weird
                    with _circuit_lock:
                        _circuit_failures = 0
                    return None

                # Success — reset circuit breaker
                with _circuit_lock:
                    _circuit_failures = 0
                return source

            except Exception as e:
                last_error = e
                err_str    = str(e).lower()

                # Non-retryable errors
                if any(x in err_str for x in
                       ("401", "unauthorized", "invalid api key",
                        "400", "bad request")):
                    logger.warning(f"JIT: non-retryable API error: {e}")
                    with _circuit_lock:
                        _circuit_failures = 0
                    return None

                # Retryable — apply backoff
                wait = _RETRY_BASE_S * (2 ** attempt)
                logger.warning(
                    f"JIT: API attempt {attempt + 1}/{_MAX_RETRIES} failed: {e}. "
                    f"Retrying in {wait:.1f}s"
                )
                time.sleep(wait)

        # All retries exhausted — open circuit breaker
        with _circuit_lock:
            _circuit_failures += 1
            if _circuit_failures >= _MAX_RETRIES:
                _circuit_opened_at = time.monotonic()
                logger.error(
                    f"JIT circuit breaker OPENED after {_MAX_RETRIES} failures. "
                    f"Last error: {last_error}. "
                    f"Synthesis suspended for {_CIRCUIT_OPEN_S}s. "
                    f"Inference continues on unfused path."
                )
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
            atol, rtol = _DTYPE_TOLERANCES.get(event.dtype, (1e-3, 1e-3))
            return torch.allclose(
                ref_out.float(),        # compare in FP32 to avoid dtype mismatch
                kernel_out.float(),
                atol=atol,
                rtol=rtol,
            )

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


def circuit_breaker_status() -> dict:
    """Return current circuit breaker state for the /metrics endpoint."""
    with _circuit_lock:
        open_s = (_CIRCUIT_OPEN_S - (time.monotonic() - _circuit_opened_at)
                  if _circuit_failures >= _MAX_RETRIES else 0.0)
        return {
            "state":       "open" if open_s > 0 else "closed",
            "failures":    _circuit_failures,
            "reopen_in_s": max(0.0, open_s),
        }
