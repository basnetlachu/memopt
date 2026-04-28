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
MAX_TOKENS        = 6000
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


# ═══════════════════════════════════════════════════════════════════════════
# Flash attention synthesis prompt
# ═══════════════════════════════════════════════════════════════════════════

FLASH_ATTENTION_PROMPT = """\
Write a fused multi-head attention kernel in Triton.

Hardware: {hardware}
Query shape: {q_shape}  (batch, heads, seq, dim)
Key shape:   {k_shape}
Value shape: {v_shape}
dtype:       {dtype}
causal mask: {causal}

Requirements:
1. Fused QK^T matmul + scale + softmax + V matmul
   in a single Triton kernel.
2. Use tiling to keep SRAM usage bounded.
   Tile size should fit in shared memory.
3. The kernel must be numerically stable.
   Use the online softmax algorithm
   (track running max and sum).
4. Handle causal masking inside the kernel
   when causal=True.
5. Output shape must match input Q shape.

The function signature must be:
  def run_kernel(q, k, v, scale=None):
    # q, k, v: torch.Tensor
    # returns: torch.Tensor same shape as q

Do not import torch inside the kernel.
Use triton.language as tl.
Use @triton.jit decorator.

Return only the Python/Triton code.
No explanation. No markdown.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Custom op synthesis prompt and spec
# ═══════════════════════════════════════════════════════════════════════════

CUSTOM_OP_PROMPT = """\
Write a fused CUDA kernel in Triton for the following operation:

{description}

Hardware: {hardware}
Input shapes: {input_shapes}
Input dtypes: {input_dtypes}
Output shape: {output_shape}

Requirements:
1. Implement the exact mathematical operation described above.
2. Use tiling appropriate for the input size.
3. The kernel must be numerically correct within float16 tolerances.
4. Use @triton.jit and triton.language as tl.

The function signature must be:
  def run_kernel(*inputs):
    # inputs: list of torch.Tensor
    # returns: torch.Tensor

Return only the Python/Triton code.
No explanation. No markdown.
"""


class CustomOpSpec:
    """
    Specification for a custom op to synthesize.

    Callers describe their op and we attempt to synthesize a faster
    Triton kernel. No guarantee of success or speedup. The synthesized
    kernel is validated and benchmarked. Only a correct + faster kernel
    is kept.
    """

    def __init__(
        self,
        name: str,
        description: str,
        input_shapes: list,
        input_dtypes: list,
        output_shape: tuple,
        reference_fn=None,
        atol: float = 1e-2,
        rtol: float = 1e-2,
    ):
        if not name:
            raise ValueError("name required")
        if not description:
            raise ValueError("description required")

        self.name = name
        self.description = description
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.output_shape = output_shape
        self.reference_fn = reference_fn
        self.atol = atol
        self.rtol = rtol


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

        # ── Step 7: Save to cross-deployment KernelLibrary ───────────
        # The library is the network-effect surface: every validated
        # kernel becomes available to the next deployment with matching
        # op + hardware. Save AFTER correctness + benchmark gates passed.
        # Library failure must never break inference — swallow + log.
        try:
            from memopt.kernels.kernel_cache import get_kernel_library
            library = get_kernel_library()
            library.save(
                op_name=event.op_name,
                hardware_hash=event.hardware,
                source=kernel_source,
                metadata={
                    "op_name":      event.op_name,
                    "input_shapes": str(event.input_shapes),
                    "hardware":     event.hardware,
                    "validated":    True,
                },
            )
        except Exception as exc:
            logger.debug(f"JIT: KernelLibrary.save failed: {exc}")

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

                valid, cleaned = self._validate_kernel_source(source)
                if not valid:
                    logger.warning(
                        "JIT: API response failed validation (%s) — discarding",
                        cleaned,
                    )
                    with _circuit_lock:
                        _circuit_failures = 0
                    return None

                # Success — reset circuit breaker
                with _circuit_lock:
                    _circuit_failures = 0
                return cleaned

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

    def _validate_kernel_source(self, source: str) -> tuple:
        """
        Static validation of a synthesized kernel source string.

        Returns (is_valid, cleaned_source_or_reason).
          - On success: (True, cleaned_source)
          - On failure: (False, reason_string)

        Strips markdown fences, verifies the source starts with a triton
        or torch import, contains @triton.jit, and is parseable Python.
        Pure string + AST checks — no instance state required, so it can
        be unit-tested without going through __init__.
        """
        import ast

        for fence in ("```python", "```triton", "```cuda", "```"):
            if source.strip().startswith(fence):
                source = source.strip()[len(fence):]
                break
        for fence in ("```python", "```triton", "```cuda", "```"):
            if source.rstrip().endswith(fence):
                source = source.rstrip()[: -len(fence)]
                break
        source = source.strip()

        if not source.startswith(("import triton", "import torch")):
            return False, "missing import"

        if "@triton.jit" not in source:
            return False, "missing @triton.jit"

        try:
            ast.parse(source)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        return True, source

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

    # ── Flash attention synthesis ─────────────────────────────────────

    def synthesize_flash_attention(
        self,
        q_shape: tuple,
        k_shape: tuple,
        v_shape: tuple,
        dtype: str = "float16",
        causal: bool = False,
        hardware_profile=None,
    ) -> Optional[str]:
        """
        Synthesize a fused attention kernel.

        Returns Triton source code or None.

        HONEST NOTE: Whether the resulting kernel compiles, is correct,
        or is faster is unknown until GPU validation. Do not claim
        speedup before measuring. Never raises.
        """
        try:
            hw = hardware_profile or self._get_hardware_description()

            prompt = FLASH_ATTENTION_PROMPT.format(
                hardware=hw,
                q_shape=q_shape,
                k_shape=k_shape,
                v_shape=v_shape,
                dtype=dtype,
                causal=causal,
            )

            source = self._call_api_simple(prompt)
            if source:
                logger.info("Flash attention synthesis: source received from API")
            return source

        except Exception as e:
            logger.debug("Flash attention synthesis failed: %s", e)
            return None

    def validate_attention(
        self,
        module,
        q_shape: tuple,
        dtype_str: str = "float16",
    ) -> bool:
        """
        Validate synthesized attention kernel against PyTorch reference
        (F.scaled_dot_product_attention).

        Tolerances are looser than other ops because online softmax
        accumulates floating point error differently.

        Returns True if output is close enough. Returns False on CPU.
        Never raises.
        """
        try:
            import torch
            import torch.nn.functional as F

            if not torch.cuda.is_available():
                logger.debug("Attention validation: skipped (no GPU)")
                return False

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(dtype_str, torch.float16)

            b, h, s, d = q_shape
            q = torch.randn(b, h, s, d, dtype=dtype, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            scale = d ** -0.5

            with torch.no_grad():
                ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

            with torch.no_grad():
                out = module.run_kernel(q, k, v, scale)

            # Looser tolerances for attention (online softmax has more fp error)
            atol = 1e-1 if dtype != torch.float32 else 1e-3
            rtol = 1e-1 if dtype != torch.float32 else 1e-3

            ok = torch.allclose(ref.float(), out.float(), atol=atol, rtol=rtol)

            if not ok:
                max_err = (ref.float() - out.float()).abs().max()
                logger.warning("Attention validation failed: max_err=%.4f", max_err)

            return ok

        except Exception as e:
            logger.debug("Attention validation error: %s", e)
            return False

    # ── Custom op synthesis ────────────────────────────────────────────

    def synthesize_custom_op(self, spec: "CustomOpSpec") -> Optional[object]:
        """
        Synthesize a kernel for a custom op.

        Returns compiled module or None.

        Steps: check cache -> build prompt -> API call -> compile ->
        validate -> benchmark -> cache if correct AND faster.

        Never raises. Returns None if API unavailable, compilation fails,
        validation fails, or not faster than reference.
        """
        cache_key_str = self._cache.make_key(
            spec.name, spec.input_shapes,
            self._get_hardware_description()
        ) if hasattr(self._cache, 'make_key') else f"{spec.name}_custom"

        # Cache hit
        cached = self._cache.get(cache_key_str) if self._cache else None
        if cached is not None:
            logger.debug("Custom op cache hit: %s", spec.name)
            return cached

        try:
            hw = self._get_hardware_description()

            prompt = CUSTOM_OP_PROMPT.format(
                description=spec.description,
                hardware=hw,
                input_shapes=spec.input_shapes,
                input_dtypes=spec.input_dtypes,
                output_shape=spec.output_shape,
            )

            source = self._call_api_simple(prompt)
            if source is None:
                return None

            module = self._portability.compile(source, hw) if self._portability else None
            if module is None:
                logger.debug("Custom op compile failed: %s", spec.name)
                return None

            valid = self._validate_custom_op(module, spec)
            if not valid:
                logger.warning(
                    "Custom op validation failed: %s. Discarded.", spec.name)
                return None

            speedup = self._benchmark_custom_op(module, spec)
            if speedup < 1.05:
                logger.info(
                    "Custom op not faster: %s speedup=%.3fx. Discarded.",
                    spec.name, speedup)
                return None

            # Keep it
            metadata = {
                "op_name": spec.name,
                "speedup": speedup,
                "description": spec.description,
                "synthesized_at": time.time(),
            }
            if self._cache:
                self._cache.put(
                    cache_key=cache_key_str,
                    kernel_obj=module,
                    metadata=metadata)

            logger.info(
                "Custom op synthesized: %s speedup=%.3fx", spec.name, speedup)
            return module

        except Exception as e:
            logger.debug("Custom op synthesis error: %s", e)
            return None

    def _validate_custom_op(self, module, spec: "CustomOpSpec") -> bool:
        """
        Validate synthesized custom op.
        With reference_fn: compare output. Without: shape/dtype check only.
        Returns False on CPU (cannot run Triton). Never raises.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return False

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }

            inputs = [
                torch.randn(
                    *shape,
                    dtype=dtype_map.get(dtype, torch.float16),
                    device="cuda")
                for shape, dtype in zip(spec.input_shapes, spec.input_dtypes)
            ]

            with torch.no_grad():
                out = module.run_kernel(*inputs)

            if tuple(out.shape) != tuple(spec.output_shape):
                logger.warning(
                    "Shape mismatch: got %s, expected %s",
                    tuple(out.shape), spec.output_shape)
                return False

            if spec.reference_fn is not None:
                with torch.no_grad():
                    ref = spec.reference_fn(*inputs)
                ok = torch.allclose(
                    ref.float(), out.float(),
                    atol=spec.atol, rtol=spec.rtol)
                if not ok:
                    max_err = (ref.float() - out.float()).abs().max().item()
                    logger.warning(
                        "Custom op validation: max_err=%.6f atol=%.6f",
                        max_err, spec.atol)
                return ok

            return True

        except Exception as e:
            logger.debug("Custom op validation error: %s", e)
            return False

    def _benchmark_custom_op(self, module, spec: "CustomOpSpec") -> float:
        """
        Measure speedup vs reference_fn. Returns speedup ratio.
        Returns 0.0 if cannot measure. Uses CUDA events on GPU.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return 0.0
            if spec.reference_fn is None:
                return 0.0

            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
            }

            inputs = [
                torch.randn(
                    *shape,
                    dtype=dtype_map.get(dtype, torch.float16),
                    device="cuda")
                for shape, dtype in zip(spec.input_shapes, spec.input_dtypes)
            ]

            N_WARMUP = 5
            N_ITERS = 20

            for _ in range(N_WARMUP):
                with torch.no_grad():
                    module.run_kernel(*inputs)
                    spec.reference_fn(*inputs)

            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            for _ in range(N_ITERS):
                with torch.no_grad():
                    module.run_kernel(*inputs)
            end.record()
            torch.cuda.synchronize()
            synth_ms = start.elapsed_time(end) / N_ITERS

            start.record()
            for _ in range(N_ITERS):
                with torch.no_grad():
                    spec.reference_fn(*inputs)
            end.record()
            torch.cuda.synchronize()
            ref_ms = start.elapsed_time(end) / N_ITERS

            if synth_ms <= 0:
                return 0.0

            speedup = ref_ms / synth_ms
            logger.debug(
                "Benchmark %s: ref=%.3fms synth=%.3fms speedup=%.3fx",
                spec.name, ref_ms, synth_ms, speedup)
            return speedup

        except Exception as e:
            logger.debug("Benchmark error: %s", e)
            return 0.0

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_hardware_description(self) -> str:
        """Return human-readable hardware description for prompts. Never raises."""
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                return (
                    f"{props.name} "
                    f"({props.total_memory // 1_000_000_000}GB, "
                    f"SM{props.major}{props.minor}, "
                    f"shared_memory="
                    f"{props.max_shared_memory_per_block // 1024}KB)")
        except Exception:
            pass
        return "CPU (no GPU available)"

    def _call_api_simple(self, prompt: str) -> Optional[str]:
        """
        Call Claude API with a raw prompt. Returns source string or None.
        Strips markdown fences. Reuses existing circuit breaker. Never raises.
        """
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        return self._call_api(prompt, api_key)

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
