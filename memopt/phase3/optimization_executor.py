"""
Phase 3: Optimization Executor

Applies a single optimization with test-measure-commit loop:
1. Baseline measurement
2. Apply optimization
3. Validate correctness
4. Measure performance
5. Compare and decide (commit or rollback)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any, Union, Tuple
import time


def _estimate_param_bytes(model: Any) -> int:
    """
    Estimate total parameter storage in bytes.

    Uses p.numel() * p.element_size() — works for any dtype (fp32, fp16, bf16).
    Does NOT count optimizer states or activations — weights only.
    """
    try:
        return sum(p.numel() * p.element_size() for p in model.parameters())
    except Exception:
        return 0


def _get_available_cpu_ram_bytes() -> int:
    """
    Return available CPU RAM in bytes.

    Priority:
        1. psutil.virtual_memory().available  (most accurate)
        2. /proc/meminfo MemAvailable line    (Linux fallback)
        3. 8 GB hardcoded minimum            (conservative last resort)
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb * 1024
    except Exception:
        pass

    return 8 * 1024 ** 3  # 8 GB


# Fraction of available RAM we are willing to use for the CPU weight copy.
# Set below 1.0 to leave headroom for Python allocator overhead, OS buffers,
# and any other in-flight tensors.
SNAPSHOT_RAM_SAFETY_MARGIN: float = 0.75


def safe_copy(model: Any) -> Tuple[Optional[dict], str]:
    """
    Choose and execute a rollback strategy based on available CPU RAM.

    Strategies (returned as second element):
        "state_dict" — all parameters copied to CPU (full rollback available).
                       Used when: available_ram × MARGIN ≥ 2 × param_bytes.
                       (Factor of 2 because load_state_dict allocates a second
                       copy during the copy-in pass.)
        "in_place"   — weights stay on device; no CPU backup.
                       Used when: MARGIN × ram ≥ param_bytes but < 2× param_bytes.
                       Rollback not available (changes cannot be undone).
        "failed"     — available RAM is below even a single copy.
                       Rollback not available; optimization proceeds at user risk.

    Returns:
        (snapshot_dict_or_None, strategy_str)
    """
    param_bytes = _estimate_param_bytes(model)
    avail = _get_available_cpu_ram_bytes()
    budget = int(avail * SNAPSHOT_RAM_SAFETY_MARGIN)

    if param_bytes == 0:
        # Model has no parameters (e.g. wrapper); try state_dict anyway
        try:
            snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return snap, "state_dict"
        except Exception:
            return None, "failed"

    if budget >= param_bytes * 2:
        # Enough RAM for a full CPU copy (including load_state_dict overhead)
        try:
            snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            logger.debug(
                "safe_copy: state_dict strategy "
                "(param=%.1fGB, avail=%.1fGB, budget=%.1fGB)",
                param_bytes / 1e9, avail / 1e9, budget / 1e9,
            )
            return snap, "state_dict"
        except Exception as e:
            logger.warning("safe_copy: state_dict failed (%s), falling through", e)

    if budget >= param_bytes:
        logger.info(
            "safe_copy: in_place strategy — insufficient RAM for CPU backup "
            "(param=%.1fGB, budget=%.1fGB). Rollback unavailable.",
            param_bytes / 1e9, budget / 1e9,
        )
        return None, "in_place"

    logger.warning(
        "safe_copy: failed — available RAM (%.1fGB) < model size (%.1fGB). "
        "Rollback unavailable.",
        budget / 1e9, param_bytes / 1e9,
    )
    return None, "failed"


def _state_dict_snapshot(model: Any) -> Optional[dict]:
    """
    Lightweight weight snapshot using state_dict().

    Avoids copy.deepcopy() which segfaults on transformers 5.x due to
    internal reference cycles combined with Cython metadata in .so files.

    Prefer safe_copy() for new call-sites — it gates on available RAM.
    """
    snap, _ = safe_copy(model)
    return snap


def _restore_snapshot(model: Any, snapshot: Optional[dict]) -> None:
    """Restore model weights from a _state_dict_snapshot() / safe_copy() result."""
    if snapshot is None:
        return
    try:
        model.load_state_dict(snapshot, strict=False)
    except Exception as exc:
        logger.warning("Snapshot restore failed: %s", exc)

logger = logging.getLogger("memopt.phase3")

# Minimum speedup required to commit INT8 — higher bar than the default
# tolerance_pct (5%) because INT8 adds quantization complexity and should
# only be kept when it delivers meaningful bandwidth savings.
INT8_COMMIT_THRESHOLD_PCT = 15.0

# Reference ordering for the optimization pipeline.
# Callers that build OptimizationCandidate lists should respect this order:
#   channels_last  → layout (must precede compile)
#   sdpa           → attention kernel selection
#   int8_quantization → halves weight bandwidth (memory-bound only)
#   torch_compile  → fuses all previous transformations
TRANSFORMATION_ORDER = [
    "channels_last",
    "sdpa",
    "int8_quantization",
    "torch_compile",
]

# Try to import torch, but don't fail if not available
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None


@dataclass
class OptimizationResult:
    """Result of applying a single optimization"""
    success: bool
    baseline_time_ms: float
    optimized_time_ms: float
    speedup_pct: float
    regression_detected: bool
    error_message: Optional[str]
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    correctness_validated: bool = True
    optimization_type: str = ""
    # Optimized model/operation returned when success=True so sequencer
    # can chain them (avoids applying every optimization to the original).
    optimized_model: Any = None
    optimized_op: Optional[Callable] = None
    # Human-readable reason from regime gate if INT8 was skipped
    runtime_warning: Optional[str] = None
    # Whether a rollback snapshot was available (False for very large models)
    rollback_available: bool = True

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return (
            f"OptimizationResult({status}): "
            f"{self.speedup_pct:+.1f}% speedup "
            f"({self.baseline_time_ms:.2f}ms -> {self.optimized_time_ms:.2f}ms)"
        )


class OptimizationExecutor:
    """
    Applies optimizations with safety checks and rollback.

    Implements the test-measure-commit loop:
    1. Measure baseline performance
    2. Apply the optimization transformation
    3. Validate correctness (outputs match within tolerance)
    4. Measure optimized performance
    5. Compare and decide: commit if improvement, rollback if regression
    """

    def __init__(
        self,
        tolerance_pct: float = 5.0,
        correctness_rtol: float = 1e-3,
        correctness_atol: float = 1e-5,
        num_warmup: int = 5,
        num_iterations: int = 10,
        model_name: str = "unknown",
        issue_certificates: bool = True,
    ):
        """
        Args:
            tolerance_pct: Allow up to this % regression (measurement noise)
            correctness_rtol: Relative tolerance for correctness check
            correctness_atol: Absolute tolerance for correctness check
            num_warmup: Number of warmup iterations
            num_iterations: Number of measurement iterations
            model_name: Label for issued certificates
            issue_certificates: If True, persist a certificate for each COMMIT
        """
        self.tolerance_pct = tolerance_pct
        self.correctness_rtol = correctness_rtol
        self.correctness_atol = correctness_atol
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.model_name = model_name
        self.issue_certificates = issue_certificates

        # Lazy certificate store — created on first COMMIT
        self._cert_store = None

        # Lazy import transformation engine
        self._transformation_engine = None

    @property
    def transformation_engine(self):
        """Lazy load transformation engine."""
        if self._transformation_engine is None:
            from .transformations import TransformationEngine
            self._transformation_engine = TransformationEngine()
        return self._transformation_engine

    def apply_optimization(
        self,
        model: Any,  # torch.nn.Module
        operation: Callable,
        candidate: Any,  # OptimizationCandidate from Phase 2
        inputs: Dict[str, Any],
        dry_run: bool = False
    ) -> OptimizationResult:
        """
        Apply optimization with test-measure-commit loop.

        Process:
        1. Baseline measurement
        2. Apply optimization
        3. Validate correctness
        4. Measure performance
        5. Compare and decide (commit or rollback)

        Args:
            model: The PyTorch model
            operation: Function that runs the operation
            candidate: OptimizationCandidate from Phase 2
            inputs: Input tensors for the operation
            dry_run: If True, only simulate (don't actually apply)

        Returns:
            OptimizationResult with success/failure and metrics
        """

        if not HAS_TORCH:
            return OptimizationResult(
                success=False,
                baseline_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                regression_detected=False,
                error_message="PyTorch not available",
                optimization_type=str(candidate.optimization_type) if hasattr(candidate, 'optimization_type') else ""
            )

        opt_type = str(candidate.optimization_type.value) if hasattr(candidate.optimization_type, 'value') else str(candidate.optimization_type)

        logger.info(f"Applying optimization: {opt_type}")

        # Step 1: Baseline measurement
        logger.debug("Measuring baseline performance...")
        try:
            baseline_metrics = self._profile_operation(operation, inputs)
            baseline_time = baseline_metrics['gpu_time_ms']
        except Exception as e:
            return OptimizationResult(
                success=False,
                baseline_time_ms=0.0,
                optimized_time_ms=0.0,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Baseline measurement failed: {str(e)}",
                optimization_type=opt_type
            )

        if dry_run:
            # Estimate without actually applying
            expected_impact = getattr(candidate, 'expected_impact_pct', 10.0)
            return OptimizationResult(
                success=True,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time * (1 - expected_impact / 100),
                speedup_pct=expected_impact,
                regression_detected=False,
                error_message=None,
                metrics_before=baseline_metrics,
                metrics_after={},
                correctness_validated=False,
                optimization_type=opt_type
            )

        # Snapshot weights before any in-place transformation.
        # safe_copy() decides the strategy based on available CPU RAM:
        #   "state_dict" — full CPU copy (rollback available)
        #   "in_place"   — no CPU copy, changes cannot be undone
        #   "failed"     — insufficient RAM even for a single copy
        snapshot, _snap_strategy = safe_copy(model)
        _rollback_avail = (_snap_strategy == "state_dict")
        if not _rollback_avail:
            logger.info(
                "apply_optimization: rollback unavailable (strategy=%s) — "
                "proceeding without rollback guard",
                _snap_strategy,
            )

        # Capture pre-transformation output for correctness validation.
        # Must happen BEFORE apply() because transformation_engine.apply()
        # modifies model in-place; calling operation(**inputs) afterwards
        # would run the already-modified model.
        _pre_output: Any = None
        if HAS_TORCH:
            try:
                with torch.no_grad():
                    _pre_output = operation(**inputs)
            except Exception:
                pass  # correctness check handles None reference gracefully

        # Step 2: Apply transformation on the original model (no deepcopy)
        logger.debug(f"Applying transformation: {opt_type}...")
        try:
            optimized_op, optimized_model = self.transformation_engine.apply(
                model, operation, candidate, inputs
            )
        except Exception as e:
            logger.warning(f"Transformation failed: {e}")
            _restore_snapshot(model, snapshot)
            return OptimizationResult(
                success=False,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Transformation failed: {str(e)}",
                metrics_before=baseline_metrics,
                optimization_type=opt_type,
                rollback_available=_rollback_avail,
            )

        # Step 3: Validate correctness against pre-transformation output
        logger.debug("Validating correctness...")
        is_correct, correctness_error = self._validate_correctness(
            operation, optimized_op, inputs, reference_output=_pre_output
        )

        if not is_correct:
            logger.warning(f"Correctness check failed: {correctness_error}")
            _restore_snapshot(model, snapshot)
            return OptimizationResult(
                success=False,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Correctness check failed: {correctness_error}",
                metrics_before=baseline_metrics,
                correctness_validated=False,
                optimization_type=opt_type,
                rollback_available=_rollback_avail,
            )

        # Step 4: Measure optimized performance
        logger.debug("Measuring optimized performance...")
        try:
            optimized_metrics = self._profile_operation(optimized_op, inputs)
            optimized_time = optimized_metrics['gpu_time_ms']
        except Exception as e:
            _restore_snapshot(model, snapshot)
            return OptimizationResult(
                success=False,
                baseline_time_ms=baseline_time,
                optimized_time_ms=baseline_time,
                speedup_pct=0.0,
                regression_detected=False,
                error_message=f"Optimized measurement failed: {str(e)}",
                metrics_before=baseline_metrics,
                optimization_type=opt_type,
                rollback_available=_rollback_avail,
            )

        # Step 5: Compare and decide
        if baseline_time > 0:
            speedup_pct = ((baseline_time - optimized_time) / baseline_time) * 100
        else:
            speedup_pct = 0.0

        regression_detected = speedup_pct < -self.tolerance_pct

        # INT8 carries quantization complexity — require a higher minimum gain.
        commit_threshold = (
            INT8_COMMIT_THRESHOLD_PCT
            if opt_type in ('int8_quantization', 'INT8_QUANTIZATION')
            else self.tolerance_pct
        )

        # Decision logic
        if regression_detected:
            decision = "ROLLBACK"
            success = False
            logger.warning(f"{decision}: {speedup_pct:.1f}% regression detected")
            _restore_snapshot(model, snapshot)
        elif speedup_pct < commit_threshold:
            decision = "NO_IMPROVEMENT"
            success = False
            logger.info(
                f"{decision}: {speedup_pct:.1f}% change "
                f"(below {commit_threshold:.0f}% commit threshold)"
            )
            _restore_snapshot(model, snapshot)
        else:
            decision = "COMMIT"
            success = True
            logger.info(f"{decision}: {speedup_pct:.1f}% speedup achieved")
            # Model stays in optimized state; snapshot not needed.
            # Issue a signed audit certificate for this committed optimization.
            if self.issue_certificates:
                try:
                    from memopt.certificates import CertificateStore
                    if self._cert_store is None:
                        self._cert_store = CertificateStore()
                    # Build a minimal result-like object for issue()
                    class _R:
                        pass
                    _r = _R()
                    _r.success = True
                    _r.optimization_type = opt_type
                    _r.baseline_time_ms = baseline_time
                    _r.optimized_time_ms = optimized_time
                    _r.speedup_pct = speedup_pct
                    _r.correctness_validated = is_correct
                    self._cert_store.issue(_r, model_name=self.model_name)
                except Exception as _cert_exc:
                    logger.debug("Certificate issuance failed (non-fatal): %s", _cert_exc)

        return OptimizationResult(
            success=success,
            baseline_time_ms=baseline_time,
            optimized_time_ms=optimized_time,
            speedup_pct=speedup_pct,
            regression_detected=regression_detected,
            error_message=None if success else f"Insufficient improvement: {speedup_pct:.1f}%",
            metrics_before=baseline_metrics,
            metrics_after=optimized_metrics,
            correctness_validated=True,
            optimization_type=opt_type,
            # Carry the optimized versions so the sequencer can chain them
            optimized_model=optimized_model if success else None,
            optimized_op=optimized_op if success else None,
            rollback_available=_rollback_avail,
        )

    def _profile_operation(
        self,
        operation: Callable,
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Profile operation and return key metrics."""

        if not HAS_TORCH or not torch.cuda.is_available():
            # CPU-only timing
            times = []

            # Warmup
            for _ in range(self.num_warmup):
                operation(**inputs)

            # Measure
            for _ in range(self.num_iterations):
                start = time.perf_counter()
                operation(**inputs)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # Convert to ms

            if HAS_NUMPY:
                median_time = float(np.median(times))
                std_time = float(np.std(times))
            else:
                times.sort()
                median_time = times[len(times) // 2]
                std_time = (sum((t - median_time) ** 2 for t in times) / len(times)) ** 0.5

            return {
                'gpu_time_ms': median_time,
                'gpu_time_std': std_time,
                'device': 'cpu'
            }

        # GPU timing with CUDA events
        times = []

        # Warmup
        for _ in range(self.num_warmup):
            with torch.no_grad():
                operation(**inputs)
        torch.cuda.synchronize()

        # Measure
        for _ in range(self.num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                operation(**inputs)
            end.record()

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        if HAS_NUMPY:
            median_time = float(np.median(times))
            std_time = float(np.std(times))
        else:
            times.sort()
            median_time = times[len(times) // 2]
            std_time = (sum((t - median_time) ** 2 for t in times) / len(times)) ** 0.5

        return {
            'gpu_time_ms': median_time,
            'gpu_time_std': std_time,
            'device': 'cuda'
        }

    def _validate_correctness(
        self,
        original_op: Callable,
        optimized_op: Callable,
        inputs: Dict[str, Any],
        *,
        reference_output: Any = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate optimized operation produces same results as original.

        Args:
            reference_output: Pre-captured output from before transformation.
                When provided, used as the reference instead of calling
                original_op(**inputs) (which would now run the modified model).

        Returns:
            (is_correct, error_message)
        """

        if not HAS_TORCH:
            return True, None  # Can't validate without torch

        try:
            with torch.no_grad():
                # Use pre-captured reference when available so we compare
                # against the truly-unmodified model output.
                original_output = (
                    reference_output
                    if reference_output is not None
                    else original_op(**inputs)
                )
                optimized_output = optimized_op(**inputs)

                # Compare outputs
                if isinstance(original_output, torch.Tensor):
                    if not torch.allclose(
                        original_output, optimized_output,
                        rtol=self.correctness_rtol, atol=self.correctness_atol
                    ):
                        max_diff = (original_output - optimized_output).abs().max().item()
                        return False, f"Output mismatch (max diff: {max_diff:.2e})"

                elif isinstance(original_output, tuple):
                    for i, (orig, opt) in enumerate(zip(original_output, optimized_output)):
                        if isinstance(orig, torch.Tensor):
                            if not torch.allclose(orig, opt, rtol=self.correctness_rtol, atol=self.correctness_atol):
                                max_diff = (orig - opt).abs().max().item()
                                return False, f"Output {i} mismatch (max diff: {max_diff:.2e})"

                elif isinstance(original_output, dict):
                    for key in original_output:
                        orig = original_output[key]
                        opt = optimized_output.get(key)
                        if isinstance(orig, torch.Tensor) and isinstance(opt, torch.Tensor):
                            if not torch.allclose(orig, opt, rtol=self.correctness_rtol, atol=self.correctness_atol):
                                max_diff = (orig - opt).abs().max().item()
                                return False, f"Output '{key}' mismatch (max diff: {max_diff:.2e})"

                return True, None

        except Exception as e:
            return False, f"Correctness validation error: {str(e)}"
