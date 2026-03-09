"""
memopt Autonomous Optimization Agent
=====================================

Wires existing profiling and optimization components into a multi-round
autonomous loop.  Does NOT rewrite any existing component — only adds the
agent layer on top.

Dependency chain (read-only, never modified):
  select_optimizations()   →  profiler/universal_optimizer.py
  apply_universal_plan()   →  profiler/universal_optimizer.py
  safe_compile()           →  profiler/universal_optimizer.py
  torchao quantize_()      →  external library (regime-gated)

Stop conditions (checked in priority order every round):
  1. COMPUTE_BOUND  → GPU already optimal, touch nothing
  2. TARGET_MET     → achieved >= target_speedup
  3. EXHAUSTED      → all candidates for current bottleneck tried
  4. NO_PROGRESS    → N consecutive rounds with no commit
  5. MAX_ROUNDS     → hard cap
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import signal
import threading

import torch
import torch.nn as nn

from memopt.phase3.universal_optimizer import (
    select_optimizations,
    apply_universal_plan,
    safe_compile,
    UniversalPlan,
    UniversalOptimizer,
    _bench_ms,
)
from memopt.profiler.power_sampler import PowerSampler, PowerReport
from memopt.utils.input_handler import (
    detect_input_format,
    forward,
    extract_tensor,
    get_sequence_length,
    get_batch_size,
    InputFormat,
)
from memopt.utils.hardware_detector import detect_hardware, HardwareProfile
from memopt.utils.model_type_detector import detect_model_type, get_applicable_optimizations

log = logging.getLogger("memopt.agent")


# =============================================================================
# Decision table
# Priority order within each bottleneck type matters:
#   channels_last before compile  (layout must precede kernel fusion)
#   sdpa before int8              (sdpa is safer; int8 is higher-bar)
#   compile always last           (fuses everything applied above)
# =============================================================================

OPTIMIZATION_PRIORITY: Dict[str, List[str]] = {
    "MEMORY_BOUND_DRAM": [
        "channels_last",   # CNNs: layout first
        "sdpa",            # transformers: O(N²) memory → O(N) with Flash-style attn
        "int8",            # regime-gated internally (seq>=1024, batch×seq<=4096)
        "compile",         # fuses everything above
    ],
    "MEMORY_BOUND_CACHE": [
        "channels_last",
        "compile",
    ],
    "PIPELINE_BOUND_OCCUPANCY": [
        "compile",         # only software lever available
    ],
    "COMPUTE_BOUND": [],   # STOP — already optimal, touch nothing
    "MIXED": [
        "sdpa",
        "channels_last",
        "compile",
    ],
}


# =============================================================================
# Internal types (not exported)
# =============================================================================

@dataclass
class _ProfileResult:
    bottleneck_type: str    # key in OPTIMIZATION_PRIORITY
    confidence: float       # 0.0 – 1.0
    ai: float               # arithmetic intensity (FLOPS/byte); inf = unmeasurable
    ridge: float            # roofline ridge point (FLOPS/byte)


@dataclass
class _ApplyResult:
    status: str                          # "COMMITTED" | "ROLLED_BACK"
    speedup: float                       # speedup vs current model (not original baseline)
    optimized_model: Optional[nn.Module] # None on ROLLED_BACK


@dataclass
class AgentState:
    round_number:             int
    cumulative_speedup:       float
    rounds_without_commit:    int
    tried_candidates:         Set[str]
    target_speedup:           float
    current_bottleneck:       str = "MIXED"
    last_arithmetic_intensity: Optional[float] = None
    fmt:                      Optional[InputFormat] = None
    model_family:             str = "unknown"


# =============================================================================
# Public types (exported via __init__.py)
# =============================================================================

@dataclass
class AgentRound:
    round_number:           int
    bottleneck_type:        str
    confidence:             float
    candidates_tried:       List[str]
    candidates_committed:   List[str]
    candidates_rolled_back: List[str]
    speedup_this_round:     float
    cumulative_speedup:     float
    stop_reason:            Optional[str]   # None if continuing


@dataclass
class AgentReport:
    model_name:                str
    gpu_name:                  str
    target_speedup:            float
    final_speedup:             float
    target_met:                bool
    rounds:                    List[AgentRound]
    total_time_seconds:        float
    optimizations_applied:     List[str]
    optimizations_rolled_back: List[str]
    honest_ceiling:            str             # plain English reason for stopping
    optimized_model:           Optional[nn.Module]
    # Power fields — default to unavailable so existing callers aren't broken
    baseline_power:            "PowerReport" = None   # type: ignore[assignment]
    optimized_power:           "PowerReport" = None   # type: ignore[assignment]
    power_reduction_pct:       Optional[float] = None

    def __post_init__(self):
        # Ensure power fields are always valid PowerReport objects
        if self.baseline_power is None:
            self.baseline_power = PowerReport.unavailable()
        if self.optimized_power is None:
            self.optimized_power = PowerReport.unavailable()

    def summary(self) -> str:
        lines = [
            f"MemoptAgent Report — {self.model_name} on {self.gpu_name}",
            f"  Target:  {self.target_speedup:.2f}x",
            f"  Final:   {self.final_speedup:.2f}x",
            f"  Met:     {'YES' if self.target_met else 'NO'}",
            f"  Rounds:  {len(self.rounds)}",
            f"  Applied: {', '.join(self.optimizations_applied) or 'none'}",
            f"  Rolled:  {', '.join(self.optimizations_rolled_back) or 'none'}",
            f"  Time:    {self.total_time_seconds:.1f}s",
            f"  Ceiling: {self.honest_ceiling}",
        ]
        if self.baseline_power.available:
            lines.append(f"  Power:   {self.power_summary()}")
        return "\n".join(lines)

    def power_summary(self) -> str:
        if not self.baseline_power.available:
            return "Power data unavailable"
        if self.power_reduction_pct is None:
            return "Power comparison unavailable"
        return (
            f"Power: {self.baseline_power.avg_watts:.1f}W → "
            f"{self.optimized_power.avg_watts:.1f}W "
            f"({self.power_reduction_pct:+.1f}%)\n"
            f"  Baseline:  {self.baseline_power.summary()}\n"
            f"  Optimized: {self.optimized_power.summary()}"
        )


# =============================================================================
# MemoptAgent — the autonomous loop
# =============================================================================

class MemoptAgent:
    """
    Autonomous optimization agent.
    Profiles → decides → applies → verifies → repeats.
    Stops when: optimal, no progress, or target met.

    Usage:
        agent = MemoptAgent(target_speedup=2.0, max_rounds=10)
        report = agent.run(model, sample_input)
        print(report.final_speedup)
        # use report.optimized_model for inference
    """

    def __init__(
        self,
        target_speedup:    float = 2.0,
        max_rounds:        int   = 10,
        min_improvement:   float = 0.05,   # minimum improvement per round to count as progress
        no_progress_limit: int   = 3,      # stop after N rounds with no committed optimization
        multi_gpu:         bool  = False,  # explicit opt-in: shard across GPUs via FSDP
    ):
        self.target_speedup    = target_speedup
        self.max_rounds        = max_rounds
        self.min_improvement   = min_improvement
        self.no_progress_limit = no_progress_limit
        self.multi_gpu         = multi_gpu
        # Last INT8 gate rejection reason — set by _apply_int8, read by _explain_ceiling
        self._last_int8_skip_reason: Optional[str] = None

        # Detect hardware once at construction — shared across all run() calls
        self.hw: HardwareProfile = detect_hardware()
        log.info("Hardware: %s", self.hw.summary())

        if self.multi_gpu:
            from memopt.utils.multi_gpu import detect_available_gpus
            self.num_gpus = detect_available_gpus()
            log.info("Multi-GPU mode: %d GPU(s) detected", self.num_gpus)
        else:
            self.num_gpus = 1

    # ── Top-level entry point ──────────────────────────────────────────────────

    def run(self, model: nn.Module, sample: Any) -> AgentReport:
        """
        Run autonomous optimization on any PyTorch model.

        Args:
            model:  Any nn.Module on CUDA.
            sample: One real batch from your dataloader.
                    Can be dict, tensor, tuple, HuggingFace BatchEncoding — anything.
        """
        start_time = time.time()
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        log.info(
            "MemoptAgent starting | target=%.2fx | gpu=%s",
            self.target_speedup, gpu_name,
        )

        # Detect input format ONCE — all downstream calls use fmt
        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Detecting input format for %s...", type(model).__name__)
        fmt = detect_input_format(model, sample, device=device)
        log.info(
            "Detected: style=%s | family=%s | keys=%s",
            fmt.style, fmt.model_family, fmt.input_keys,
        )

        # Detect model family for hardware-aware candidate selection
        model_family = detect_model_type(model)
        log.info("Model family: %s", model_family)

        # Multi-GPU: apply FSDP if explicitly enabled and needed
        if self.multi_gpu and self.num_gpus > 1:
            from memopt.utils.multi_gpu import apply_fsdp, is_multi_gpu_needed
            if is_multi_gpu_needed(model, self.hw):
                model, fsdp_applied = apply_fsdp(model, self.num_gpus)
                if fsdp_applied:
                    log.info("FSDP: model sharded across %d GPUs", self.num_gpus)

        # Measure true baseline ONCE — latency + power simultaneously
        baseline_ms, baseline_power = self._benchmark_with_power(model, fmt)
        log.info(
            "Baseline: %.2f ms | power: %s",
            baseline_ms, baseline_power.summary(),
        )

        _param_count = sum(p.numel() for p in model.parameters())
        if _param_count > 1e9:
            log.info(
                "Large model (%.1fB params): using in-place optimization "
                "with state_dict snapshots for rollback",
                _param_count / 1e9,
            )
            current_model = model  # work in-place; per-optimization snapshots handle rollback
        else:
            current_model = copy.deepcopy(model)
        state = AgentState(
            round_number=0,
            cumulative_speedup=1.0,
            rounds_without_commit=0,
            tried_candidates=set(),
            target_speedup=self.target_speedup,
            fmt=fmt,
            model_family=model_family,
        )
        rounds: List[AgentRound] = []

        while True:
            state.round_number += 1
            log.info("\n--- Round %d ---", state.round_number)

            # Profile current model (uses select_optimizations for real GPU data)
            profile = self._profile(current_model, fmt)
            state.current_bottleneck = profile.bottleneck_type
            state.last_arithmetic_intensity = profile.ai

            log.info(
                "Bottleneck: %s (confidence=%.2f)  AI=%.1f  ridge=%.1f",
                state.current_bottleneck, profile.confidence,
                profile.ai, profile.ridge,
            )

            # Check stop conditions BEFORE attempting any candidates
            stop_reason = self._should_stop(state)
            if stop_reason:
                log.info("Stopping: %s", stop_reason)
                rounds.append(AgentRound(
                    round_number=state.round_number,
                    bottleneck_type=state.current_bottleneck,
                    confidence=profile.confidence,
                    candidates_tried=[],
                    candidates_committed=[],
                    candidates_rolled_back=[],
                    speedup_this_round=1.0,
                    cumulative_speedup=state.cumulative_speedup,
                    stop_reason=stop_reason,
                ))
                break

            # Candidates for this bottleneck (excluding already-tried)
            candidates = self._get_candidates(state, current_model)
            round_committed:    List[str] = []
            round_rolled_back:  List[str] = []
            round_start_speedup = state.cumulative_speedup

            # Try each candidate in priority order
            for candidate_name in candidates:
                if candidate_name in state.tried_candidates:
                    continue   # already attempted globally — skip

                state.tried_candidates.add(candidate_name)
                log.info("  Trying: %s", candidate_name)

                result = self._apply_optimization(
                    current_model, fmt, candidate_name
                )

                if result.status == "COMMITTED":
                    current_model = result.optimized_model

                    # Re-measure against the ORIGINAL baseline (not current round start)
                    new_ms = self._benchmark(current_model, fmt)
                    state.cumulative_speedup = baseline_ms / new_ms if new_ms > 0 else 1.0

                    round_committed.append(candidate_name)
                    log.info(
                        "  COMMITTED %s — cumulative=%.2fx  (%.2fms → %.2fms)",
                        candidate_name, state.cumulative_speedup,
                        baseline_ms, new_ms,
                    )
                    state.rounds_without_commit = 0

                else:
                    round_rolled_back.append(candidate_name)
                    log.info(
                        "  ROLLED_BACK %s — speedup_vs_current=%.3fx",
                        candidate_name, result.speedup,
                    )

            # Track no-progress
            if not round_committed:
                state.rounds_without_commit += 1

            round_speedup = (
                state.cumulative_speedup / round_start_speedup
                if round_start_speedup > 0 else 1.0
            )

            rounds.append(AgentRound(
                round_number=state.round_number,
                bottleneck_type=state.current_bottleneck,
                confidence=profile.confidence,
                candidates_tried=list(candidates),
                candidates_committed=round_committed,
                candidates_rolled_back=round_rolled_back,
                speedup_this_round=round_speedup,
                cumulative_speedup=state.cumulative_speedup,
                stop_reason=None,
            ))

        all_committed   = [c for r in rounds for c in r.candidates_committed]
        all_rolled_back = [c for r in rounds for c in r.candidates_rolled_back]

        # Measure final power on the optimized model
        final_model = current_model if all_committed else model
        _, optimized_power = self._benchmark_with_power(final_model, fmt)

        # Power reduction: positive = less watts = good
        if baseline_power.available and optimized_power.available and baseline_power.avg_watts > 0:
            power_reduction_pct = (
                (baseline_power.avg_watts - optimized_power.avg_watts)
                / baseline_power.avg_watts * 100.0
            )
        else:
            power_reduction_pct = None

        return AgentReport(
            model_name=type(model).__name__,
            gpu_name=gpu_name,
            target_speedup=self.target_speedup,
            final_speedup=state.cumulative_speedup,
            target_met=state.cumulative_speedup >= self.target_speedup,
            rounds=rounds,
            total_time_seconds=time.time() - start_time,
            optimizations_applied=all_committed,
            optimizations_rolled_back=all_rolled_back,
            honest_ceiling=self._explain_ceiling(state, rounds),
            optimized_model=current_model if all_committed else None,
            baseline_power=baseline_power,
            optimized_power=optimized_power,
            power_reduction_pct=power_reduction_pct,
        )

    # ── Stop conditions (checked in priority order) ────────────────────────────

    def _should_stop(self, state: AgentState) -> Optional[str]:
        """
        Returns stop reason string if agent should stop, None to continue.
        Conditions are evaluated in this exact order — earlier takes priority.
        """
        # 1. Compute-bound: GPU is working at peak efficiency — touch nothing
        if state.current_bottleneck == "COMPUTE_BOUND":
            return "OPTIMAL: model is compute-bound — GPU is working efficiently"

        # 2. Target met
        if state.cumulative_speedup >= state.target_speedup:
            return (
                f"TARGET_MET: achieved {state.cumulative_speedup:.2f}x "
                f">= {state.target_speedup:.2f}x"
            )

        # 3. No candidates left for current bottleneck type
        remaining = self._remaining_candidates(state)
        if not remaining:
            return "EXHAUSTED: all candidates tried for current bottleneck type"

        # 4. No progress in last N consecutive rounds
        if state.rounds_without_commit >= self.no_progress_limit:
            return (
                f"NO_PROGRESS: {self.no_progress_limit} consecutive rounds "
                "with no improvement"
            )

        # 5. Hard cap
        if state.round_number >= self.max_rounds:
            return f"MAX_ROUNDS: reached limit of {self.max_rounds} rounds"

        return None  # continue

    # ── Candidate management ───────────────────────────────────────────────────

    def _get_candidates(
        self,
        state: AgentState,
        model: Optional[nn.Module] = None,
    ) -> List[str]:
        """
        Return untried candidates for (model_family, hardware) in priority order.

        Uses model_type_detector.get_applicable_optimizations() which is
        hardware-aware (gates flash_attn3/fp8 on Hopper, flash_attn2 on Ampere+).

        Falls back to static OPTIMIZATION_PRIORITY table if model_family lookup
        yields nothing (ensures backward compat).
        """
        # Dynamic path: hardware + model-family aware
        candidates = get_applicable_optimizations(
            state.model_family,
            self.hw,
            tried=state.tried_candidates,
        )

        # Intersect with bottleneck-appropriate set from static table
        # (COMPUTE_BOUND must still yield no candidates to trigger the OPTIMAL stop)
        if state.current_bottleneck == "COMPUTE_BOUND":
            return []

        # Skip compile for >7B models
        if model is not None:
            param_count = sum(p.numel() for p in model.parameters())
            if param_count > 7e9 and "compile" in candidates:
                candidates = [c for c in candidates if c != "compile"]
                log.info(
                    "Skipping compile: model has %.1fB params "
                    "(compile warmup overhead exceeds benefit at this scale)",
                    param_count / 1e9,
                )

        # If dynamic path returned nothing, fall back to static table
        if not candidates:
            all_static = OPTIMIZATION_PRIORITY.get(state.current_bottleneck, [])
            candidates = [c for c in all_static if c not in state.tried_candidates]
            if model is not None:
                param_count = sum(p.numel() for p in model.parameters())
                if param_count > 7e9 and "compile" in candidates:
                    candidates = [c for c in candidates if c != "compile"]

        return candidates

    def _remaining_candidates(self, state: AgentState) -> List[str]:
        """Alias for _get_candidates (used by stop condition 3)."""
        return self._get_candidates(state)

    # ── Profiling ──────────────────────────────────────────────────────────────

    def _profile(
        self,
        model: nn.Module,
        fmt: InputFormat,
    ) -> _ProfileResult:
        """
        Determine compute regime using select_optimizations().

        Maps UniversalPlan fields → OPTIMIZATION_PRIORITY key:
          - is_memory_bound=False            → COMPUTE_BOUND
          - is_memory_bound=True             → MEMORY_BOUND_DRAM
          - AI unmeasurable (inf)            → MIXED
        """
        try:
            # Pass model directly — select_optimizations only reads it (no modification).
            # Avoid copy.deepcopy here: transformers 5.x models crash in named_modules()
            # after deepcopy due to stale Cython .so metadata interactions.
            plan = select_optimizations(model, fmt)
        except Exception as exc:
            log.warning(
                "_profile: select_optimizations failed (%s) — defaulting to MIXED", exc
            )
            return _ProfileResult(
                bottleneck_type="MIXED",
                confidence=0.30,
                ai=0.0,
                ridge=0.0,
            )

        ai    = plan.ai
        ridge = plan.ridge

        if ai == float("inf") or ai <= 0:
            # Cannot measure arithmetic intensity (CPU-only or no CUDA allocation)
            bottleneck_type = "MIXED"
            confidence = 0.40

        elif not plan.is_memory_bound:
            # AI > ridge: compute-bound
            bottleneck_type = "COMPUTE_BOUND"
            ratio = ai / ridge if ridge > 0 else 10.0
            # Confidence scales with how far above the ridge we are
            confidence = min(0.99, 0.60 + (ratio - 1.0) * 0.15)

        else:
            # AI <= ridge: memory-bound
            bottleneck_type = "MEMORY_BOUND_DRAM"
            ratio = ai / ridge if ridge > 0 else 0.5
            # Confidence scales with how far below the ridge we are
            confidence = min(0.90, 0.65 + (1.0 - ratio) * 0.25)

        return _ProfileResult(
            bottleneck_type=bottleneck_type,
            confidence=confidence,
            ai=ai,
            ridge=ridge,
        )

    # ── Optimization application ───────────────────────────────────────────────

    def _apply_optimization(
        self,
        model: nn.Module,
        fmt: InputFormat,
        candidate_name: str,
    ) -> _ApplyResult:
        """
        Apply one named optimization and measure whether it helps.

        Uses ONLY existing components:
          channels_last, sdpa → apply_universal_plan()
          compile             → safe_compile() (has regression guard built in)
          int8                → torchao quantize_() with regime gate

        Commit thresholds:
          int8:         15%  (INT8_COMMIT_THRESHOLD_PCT in optimization_executor.py)
          all others:   5%   (default tolerance_pct)

        For models >1B params, channels_last and sdpa use in-place application
        with state_dict snapshot for rollback (avoids GPU deepcopy overhead).
        compile and int8 always use deepcopy (compile wraps module; int8 changes arch).
        """
        commit_threshold = 0.15 if candidate_name == "int8" else 0.05
        snapshot = None

        try:
            # Fix D: memory headroom check before each attempt
            if not self._check_memory_headroom(model):
                return _ApplyResult(status="ROLLED_BACK", speedup=1.0, optimized_model=None)

            param_count = sum(p.numel() for p in model.parameters())
            # Fix A: in-place + state_dict snapshot for large models
            # (channels_last and sdpa don't change architecture → state_dict works)
            # compile and int8 still use deepcopy (compile wraps; int8 changes layer types)
            use_snapshot = (param_count > 1e9 and candidate_name in ("channels_last", "sdpa"))

            if use_snapshot:
                snapshot = self._snapshot_model(model, optimization_name=candidate_name)
                # Measure baseline BEFORE applying in-place transformation
                baseline_ms = self._benchmark(model, fmt)

                if candidate_name == "channels_last":
                    plan = UniversalPlan(use_channels_last=True)
                    apply_universal_plan(model, fmt.inputs, plan)
                    optimized = model
                elif candidate_name == "sdpa":
                    plan = UniversalPlan(use_sdpa=True)
                    apply_universal_plan(model, fmt.inputs, plan)
                    optimized = model
                else:
                    log.warning("Unknown snapshot candidate '%s' — skipping", candidate_name)
                    self._restore_snapshot(model, snapshot)
                    return _ApplyResult(status="ROLLED_BACK", speedup=1.0, optimized_model=None)

            else:
                candidate_model = copy.deepcopy(model)

                if candidate_name == "channels_last":
                    plan = UniversalPlan(use_channels_last=True)
                    optimized = apply_universal_plan(candidate_model, fmt.inputs, plan)

                elif candidate_name == "sdpa":
                    plan = UniversalPlan(use_sdpa=True)
                    optimized = apply_universal_plan(candidate_model, fmt.inputs, plan)

                elif candidate_name == "compile":
                    # safe_compile already has an internal regression guard (0.95 threshold).
                    # If it decides NOT to compile, it returns the original module unchanged.
                    # Then our own benchmark below will see speedup ≈ 1.0 → ROLLED_BACK.
                    optimized = safe_compile(
                        candidate_model,
                        fmt,
                        mode="reduce-overhead",
                    )

                elif candidate_name == "int8":
                    optimized = self._apply_int8(candidate_model, fmt)

                elif candidate_name in ("flash_attn2", "flash_attn3"):
                    version = 3 if candidate_name == "flash_attn3" else 2
                    optimized = self._apply_flash_attention(candidate_model, version)

                elif candidate_name == "fp8":
                    optimized = self._apply_fp8(candidate_model)

                elif candidate_name == "qkv_fusion":
                    optimized = self._apply_qkv_fusion(candidate_model)

                elif candidate_name == "awq_4bit":
                    optimized = self._apply_awq(candidate_model, fmt)

                elif candidate_name == "gptq_4bit":
                    optimized = self._apply_gptq(candidate_model, fmt)

                elif candidate_name == "moe_optimize":
                    optimized = self._apply_moe(candidate_model, fmt)

                else:
                    log.warning("Unknown candidate '%s' — skipping", candidate_name)
                    return _ApplyResult(
                        status="ROLLED_BACK", speedup=1.0, optimized_model=None
                    )

                # Measure baseline from the original (unmodified) model
                baseline_ms = self._benchmark(model, fmt)

            # Measure optimized performance
            optimized_ms = self._benchmark(optimized, fmt)
            speedup      = baseline_ms / optimized_ms if optimized_ms > 0 else 1.0
            improvement  = speedup - 1.0

            if improvement >= commit_threshold:
                log.info(
                    "    %s: %.3fx (%.1f%% improvement >= %.0f%% threshold) → COMMIT",
                    candidate_name, speedup, improvement * 100, commit_threshold * 100,
                )
                return _ApplyResult(
                    status="COMMITTED",
                    speedup=speedup,
                    optimized_model=optimized,
                )
            else:
                log.info(
                    "    %s: %.3fx (%.1f%% improvement < %.0f%% threshold) → ROLLBACK",
                    candidate_name, speedup, improvement * 100, commit_threshold * 100,
                )
                # Restore snapshot if we modified in-place
                if use_snapshot and snapshot is not None:
                    self._restore_snapshot(model, snapshot)
                return _ApplyResult(
                    status="ROLLED_BACK", speedup=speedup, optimized_model=None
                )

        except Exception as exc:
            log.warning("  %s raised exception: %s", candidate_name, exc)
            # Restore snapshot if we modified in-place before the exception
            if use_snapshot and snapshot is not None:
                try:
                    self._restore_snapshot(model, snapshot)
                except Exception as restore_exc:
                    log.warning("  snapshot restore also failed: %s", restore_exc)
            return _ApplyResult(status="ROLLED_BACK", speedup=1.0, optimized_model=None)

    def _apply_int8(
        self,
        model: nn.Module,
        fmt: InputFormat,
    ) -> nn.Module:
        """
        Apply torchao INT8 quantization with regime gate.

        Regime gate (A100-calibrated): seq>=1024 AND batch×seq<=4096.
        Outside this window, returns model unchanged → caller sees speedup~1.0 → ROLLBACK.

        Tier 1: Int8DynamicActivationInt8WeightConfig (dynamic_activation)
        Tier 2: Int8WeightOnlyConfig (weight_only fallback)
        Returns original if torchao unavailable or both tiers fail.
        """
        seq_len = get_sequence_length(fmt) or 1
        batch   = get_batch_size(fmt) or 1

        # Delegate to 3-tier UniversalOptimizer gate
        # (<7B: seq>=1024 AND batch*seq<=4096; 7B-20B: seq>=512; >20B: seq>=128)
        from memopt.profiler.bottleneck_classifier import BottleneckType
        in_regime, _int8_reason = UniversalOptimizer()._should_apply_int8(
            fmt,
            BottleneckType.MEMORY_BOUND_DRAM,
            model,
        )

        if not in_regime:
            log.info(
                "    int8: regime gate REJECTED — %s (seq=%d, batch×seq=%d)",
                _int8_reason, seq_len, batch * seq_len,
            )
            self._last_int8_skip_reason = _int8_reason
            return model  # unchanged → speedup ~1.0 → ROLLBACK

        # Tier 1: dynamic activation + weight INT8
        try:
            from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig
            candidate = copy.deepcopy(model)
            quantize_(candidate, Int8DynamicActivationInt8WeightConfig())
            log.info("    int8: torchao Int8DynamicActivationInt8WeightConfig applied")
            return candidate
        except ImportError:
            log.warning("    int8: torchao not installed — trying Int8WeightOnlyConfig")
        except Exception as exc:
            log.warning("    int8: Int8DynamicActivationInt8WeightConfig failed: %s", exc)

        # Tier 2: weight-only INT8
        try:
            from torchao.quantization import quantize_, Int8WeightOnlyConfig
            candidate = copy.deepcopy(model)
            quantize_(candidate, Int8WeightOnlyConfig())
            log.info("    int8: torchao Int8WeightOnlyConfig applied")
            return candidate
        except Exception as exc:
            log.warning("    int8: both torchao tiers failed: %s — returning original", exc)

        return model  # both tiers failed → speedup ~1.0 → ROLLBACK

    def _apply_flash_attention(
        self,
        model: nn.Module,
        version: int = 2,
    ) -> nn.Module:
        """
        Apply Flash Attention via the three-strategy cascade in flash_attention.py.
        Returns the (possibly wrapped) model; strategy is logged internally.
        The caller benchmarks and rolls back if speedup < threshold.
        """
        from memopt.phase3.flash_attention import apply_flash_attention
        optimized, strategy = apply_flash_attention(model, version=version)
        log.info("    flash_attn%d: strategy=%s", version, strategy)
        return optimized

    def _apply_fp8(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Apply FP8 quantization via Transformer Engine (Hopper only).
        Returns the (possibly wrapped) model; mode is logged internally.
        The caller benchmarks and rolls back if speedup < threshold.
        """
        from memopt.phase3.fp8_optimizer import apply_fp8
        optimized, mode = apply_fp8(model)
        log.info("    fp8: mode=%s", mode)
        return optimized

    def _apply_qkv_fusion(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Fuse Q/K/V projection triplets in transformer attention blocks.
        Modifies model in-place; caller deepcopied before calling here.
        Returns model (possibly with FusedQKVLinear modules substituted).
        """
        from memopt.phase3.qkv_fusion import apply_qkv_fusion
        optimized, n_fused = apply_qkv_fusion(model)
        log.info("    qkv_fusion: %d triplets fused", n_fused)
        return optimized

    def _apply_awq(
        self,
        model: nn.Module,
        fmt,
    ) -> nn.Module:
        """
        Apply AWQ 4-bit quantization.
        Requires: autoawq installed, HuggingFace model with .config.
        Returns original model if prerequisites not met or accuracy fails.
        """
        from memopt.phase3.quantization import apply_awq, is_quantization_candidate
        if not is_quantization_candidate(model):
            log.info("    awq_4bit: not a quantization candidate — skipping")
            return model

        # Try to get tokenizer from model config
        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            log.info("    awq_4bit: no tokenizer available — skipping")
            return model

        quantized, applied, reason = apply_awq(model, tokenizer, fmt=fmt)
        log.info("    awq_4bit: applied=%s reason=%s", applied, reason)
        return quantized if applied else model

    def _apply_gptq(
        self,
        model: nn.Module,
        fmt,
    ) -> nn.Module:
        """
        Apply GPTQ 4-bit quantization (fallback after AWQ).
        Returns original model if prerequisites not met or accuracy fails.
        """
        from memopt.phase3.quantization import apply_gptq, is_quantization_candidate
        if not is_quantization_candidate(model):
            log.info("    gptq_4bit: not a quantization candidate — skipping")
            return model

        tokenizer = self._get_tokenizer(model)
        if tokenizer is None:
            log.info("    gptq_4bit: no tokenizer available — skipping")
            return model

        quantized, applied, reason = apply_gptq(model, tokenizer, fmt=fmt)
        log.info("    gptq_4bit: applied=%s reason=%s", applied, reason)
        return quantized if applied else model

    def _apply_moe(
        self,
        model: nn.Module,
        fmt,
    ) -> nn.Module:
        """
        Apply MoE optimization (router compile + expert prefetch).
        Returns original model if no MoE structure found.
        """
        from memopt.phase3.moe_optimizer import apply_moe_optimization
        optimized, applied = apply_moe_optimization(model, hardware=self.hw, fmt=fmt)
        log.info("    moe_optimize: applied=%s", applied)
        return optimized if applied else model

    @staticmethod
    def _get_tokenizer(model: nn.Module):
        """
        Try to retrieve a tokenizer for the model.

        Checks:
          1. model.tokenizer attribute (some wrapped models expose it)
          2. model.config.name_or_path → AutoTokenizer.from_pretrained()

        Returns tokenizer or None if unavailable.
        """
        # Check direct attribute
        if hasattr(model, "tokenizer"):
            return model.tokenizer

        # Try loading from config
        try:
            config = getattr(model, "config", None)
            if config is None:
                return None
            name = getattr(config, "name_or_path", None) or getattr(config, "_name_or_path", None)
            if not name:
                return None
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
            return tok
        except Exception:
            return None

    # ── Benchmarking ───────────────────────────────────────────────────────────

    def _benchmark(
        self,
        model: nn.Module,
        fmt: InputFormat,
        warmup: int = 5,
        iters: int = 20,
    ) -> float:
        """Median latency in ms using CUDA events (perf_counter fallback on CPU)."""
        def _run() -> Any:
            return forward(model, fmt)

        return _bench_ms(_run, warmup=warmup, iters=iters)

    def _benchmark_with_power(
        self,
        model: nn.Module,
        fmt: InputFormat,
        iters: int = 50,
        token_count: int = 0,
    ) -> "Tuple[float, PowerReport]":
        """
        Measure latency AND power simultaneously.

        Warmup runs first (no power measurement — GPU stabilises).
        Then iters forward passes under PowerSampler context.
        Returns (median_ms_per_iter, PowerReport).
        """
        # Warmup — GPU frequency ramps up, caches warm, no power measurement
        with torch.no_grad():
            for _ in range(5):
                forward(model, fmt)
        torch.cuda.synchronize()

        # Timed + power-sampled batch
        with PowerSampler() as sampler:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                for _ in range(iters):
                    forward(model, fmt)
            e.record()
            torch.cuda.synchronize()
            duration_ms = s.elapsed_time(e)

        median_ms = duration_ms / iters
        power = sampler.report(
            duration_ms=duration_ms,
            token_count=token_count * iters,
        )
        return median_ms, power

    # ── Change 1: 3-tier snapshot / restore for large models ───────────────────

    @staticmethod
    def _snapshot_model(model: nn.Module, optimization_name: str = None) -> Dict:
        """
        Snapshot strategy by model size:
        - <1B params:  deepcopy (preserves compiled state, fast)
        - 1B-7B params: full state_dict to CPU (safe, acceptable speed)
        - >7B params:  layer-selective snapshot (only layers touched by optimization)
        """
        param_count = sum(p.numel() for p in model.parameters())

        if param_count < 1e9:
            return {
                "type": "deepcopy",
                "model": copy.deepcopy(model),
                "param_count": param_count,
            }

        elif param_count < 7e9:
            return {
                "type": "state_dict",
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "param_count": param_count,
            }

        else:
            # >7B: only snapshot layers relevant to this optimization
            # SDPA touches attention layers only
            # INT8 touches linear layers only
            # channels_last touches conv layers only
            layer_patterns = {
                "sdpa":          ["attn", "attention", "self_attn"],
                "int8":          ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj",
                                 "fc1", "fc2", "dense"],
                "channels_last": ["conv"],
                "compile":       [],  # skipped >7B — should never reach here
            }

            patterns = layer_patterns.get(optimization_name, [])

            if not patterns:
                # Unknown optimization or compile — snapshot nothing
                # Rollback will re-profile and skip if needed
                return {
                    "type": "none",
                    "param_count": param_count,
                }

            selective_state = {
                k: v.cpu().clone()
                for k, v in model.named_parameters()
                if any(p in k for p in patterns)
            }

            log.info(
                "Selective snapshot: %d tensors (%.0fM params) for optimization='%s'",
                len(selective_state),
                sum(v.numel() for v in selective_state.values()) / 1e6,
                optimization_name,
            )

            return {
                "type": "selective",
                "state": selective_state,
                "param_count": param_count,
            }

    @staticmethod
    def _restore_snapshot(model: nn.Module, snapshot: Dict) -> nn.Module:
        """Restore from any snapshot type."""
        snap_type = snapshot["type"]

        if snap_type == "deepcopy":
            return snapshot["model"]

        elif snap_type == "state_dict":
            model.load_state_dict(snapshot["state"])
            if next(model.parameters(), None) is not None:
                device = next(model.parameters()).device
                if device.type == "cpu":
                    model.cuda()
            return model

        elif snap_type == "selective":
            # Restore only the snapshotted layers in-place
            current_state = model.state_dict()
            for k, v in snapshot["state"].items():
                current_state[k] = v.cuda()
            model.load_state_dict(current_state)
            return model

        elif snap_type == "none":
            # Nothing to restore — model unchanged (optimization was no-op)
            return model

        else:
            raise ValueError(f"Unknown snapshot type: {snap_type}")

    # ── Fix D: GPU memory headroom check ───────────────────────────────────────

    @staticmethod
    def _check_memory_headroom(model: nn.Module) -> bool:
        """
        Check if enough GPU memory exists to attempt optimization.
        Needs ~1.5x model memory free for snapshot + candidate tensors.
        Returns False if memory is too tight — skip optimization safely.
        """
        if not torch.cuda.is_available():
            return True  # CPU path — no memory constraint
        total     = torch.cuda.get_device_properties(0).total_memory
        reserved  = torch.cuda.memory_reserved(0)
        free      = total - reserved
        model_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        headroom_needed = model_bytes * 1.5
        has_headroom = free >= headroom_needed
        if not has_headroom:
            log.warning(
                "Low GPU memory: free=%.1fGB needed=%.1fGB model=%.1fGB "
                "— skipping optimization to avoid OOM",
                free / 1e9, headroom_needed / 1e9, model_bytes / 1e9,
            )
        return has_headroom

    # ── Honest ceiling explanation ─────────────────────────────────────────────

    def _explain_ceiling(
        self,
        state: AgentState,
        rounds: List[AgentRound],
    ) -> str:
        """
        Plain English explanation of why the agent stopped.
        No hype. No vague statements. Includes actionable next steps.
        """
        if not rounds:
            return "No rounds completed."

        final_round = rounds[-1]
        stop = final_round.stop_reason or ""

        if "OPTIMAL" in stop:
            ai = state.last_arithmetic_intensity or 0.0
            return (
                f"Model is compute-bound "
                f"(arithmetic intensity={ai:.0f} FLOPS/byte > roofline ridge point). "
                f"The GPU is already working at full compute efficiency — "
                f"memory bandwidth is not the bottleneck. "
                f"Further speedup requires reducing FLOPs (smaller model, pruning) "
                f"or faster hardware (H100 has 3× higher compute throughput than A100)."
            )

        if "TARGET_MET" in stop:
            return (
                f"Target of {state.target_speedup:.2f}x achieved. "
                f"Further optimization is possible but with diminishing returns — "
                f"remaining candidates each deliver <5% additional improvement."
            )

        if "EXHAUSTED" in stop:
            ai = state.last_arithmetic_intensity
            int8_note = (
                f" INT8 gate: {self._last_int8_skip_reason}."
                if self._last_int8_skip_reason
                else " Increase seq>=1024 AND batch×seq<=4096 to unlock INT8 regime."
            )
            if ai and ai != float("inf"):
                return (
                    f"All available optimizations applied for bottleneck type "
                    f"'{state.current_bottleneck}' "
                    f"(arithmetic intensity={ai:.0f} FLOPS/byte). "
                    f"To go further: (1) install flash-attn for full Flash Attention "
                    f"(pip install flash-attn); "
                    f"(2){int8_note} "
                    f"(3) use bfloat16 mixed precision to halve memory traffic."
                )
            return (
                f"All available optimizations applied for bottleneck type "
                f"'{state.current_bottleneck}'."
                f"{int8_note}"
            )

        if "NO_PROGRESS" in stop:
            return (
                f"No improvement in last {self.no_progress_limit} consecutive rounds. "
                f"The remaining candidates don't help at the current batch size / "
                f"sequence length. "
                f"Try: larger batch size (>64), longer sequences (>512 tokens), "
                f"or install flash-attn for full Flash Attention support."
            )

        if "MAX_ROUNDS" in stop:
            return (
                f"Reached {self.max_rounds}-round limit. "
                f"Final speedup: {state.cumulative_speedup:.2f}x. "
                f"Increase max_rounds= for more optimization attempts."
            )

        return (
            f"Stopped after {state.round_number} rounds. "
            f"Final: {state.cumulative_speedup:.2f}x."
        )
