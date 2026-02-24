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

import torch
import torch.nn as nn

from memopt.phase3.universal_optimizer import (
    select_optimizations,
    apply_universal_plan,
    safe_compile,
    UniversalPlan,
    _bench_ms,
)
from memopt.profiler.power_sampler import PowerSampler, PowerReport

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
    ):
        self.target_speedup    = target_speedup
        self.max_rounds        = max_rounds
        self.min_improvement   = min_improvement
        self.no_progress_limit = no_progress_limit

    # ── Top-level entry point ──────────────────────────────────────────────────

    def run(self, model: nn.Module, sample_input: Dict[str, Any]) -> AgentReport:
        start_time = time.time()
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        log.info(
            "MemoptAgent starting | target=%.2fx | gpu=%s",
            self.target_speedup, gpu_name,
        )

        # Measure true baseline ONCE — latency + power simultaneously
        baseline_ms, baseline_power = self._benchmark_with_power(model, sample_input)
        log.info(
            "Baseline: %.2f ms | power: %s",
            baseline_ms, baseline_power.summary(),
        )

        current_model = copy.deepcopy(model)
        state = AgentState(
            round_number=0,
            cumulative_speedup=1.0,
            rounds_without_commit=0,
            tried_candidates=set(),
            target_speedup=self.target_speedup,
        )
        rounds: List[AgentRound] = []

        while True:
            state.round_number += 1
            log.info("\n--- Round %d ---", state.round_number)

            # Profile current model (uses select_optimizations for real GPU data)
            profile = self._profile(current_model, sample_input)
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
            candidates = self._get_candidates(state)
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
                    current_model, sample_input, candidate_name
                )

                if result.status == "COMMITTED":
                    current_model = result.optimized_model

                    # Re-measure against the ORIGINAL baseline (not current round start)
                    new_ms = self._benchmark(current_model, sample_input)
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
        _, optimized_power = self._benchmark_with_power(final_model, sample_input)

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

    def _get_candidates(self, state: AgentState) -> List[str]:
        """Return untried candidates for current bottleneck in priority order."""
        all_candidates = OPTIMIZATION_PRIORITY.get(state.current_bottleneck, [])
        return [c for c in all_candidates if c not in state.tried_candidates]

    def _remaining_candidates(self, state: AgentState) -> List[str]:
        """Alias for _get_candidates (used by stop condition 3)."""
        return self._get_candidates(state)

    # ── Profiling ──────────────────────────────────────────────────────────────

    def _profile(
        self,
        model: nn.Module,
        sample_input: Dict[str, Any],
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
            plan = select_optimizations(model, sample_input)
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
        sample_input: Dict[str, Any],
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
        """
        commit_threshold = 0.15 if candidate_name == "int8" else 0.05

        try:
            candidate_model = copy.deepcopy(model)

            if candidate_name == "channels_last":
                plan = UniversalPlan(use_channels_last=True)
                optimized = apply_universal_plan(candidate_model, sample_input, plan)

            elif candidate_name == "sdpa":
                plan = UniversalPlan(use_sdpa=True)
                optimized = apply_universal_plan(candidate_model, sample_input, plan)

            elif candidate_name == "compile":
                # safe_compile already has an internal regression guard (0.95 threshold).
                # If it decides NOT to compile, it returns the original module unchanged.
                # Then our own benchmark below will see speedup ≈ 1.0 → ROLLED_BACK.
                optimized = safe_compile(
                    candidate_model,
                    sample_input,
                    mode="reduce-overhead",
                )

            elif candidate_name == "int8":
                optimized = self._apply_int8(candidate_model, sample_input)

            else:
                log.warning("Unknown candidate '%s' — skipping", candidate_name)
                return _ApplyResult(
                    status="ROLLED_BACK", speedup=1.0, optimized_model=None
                )

            # Measure speedup of this transformation vs the CURRENT model
            baseline_ms  = self._benchmark(model, sample_input)
            optimized_ms = self._benchmark(optimized, sample_input)
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
                return _ApplyResult(
                    status="ROLLED_BACK", speedup=speedup, optimized_model=None
                )

        except Exception as exc:
            log.warning("  %s raised exception: %s", candidate_name, exc)
            return _ApplyResult(status="ROLLED_BACK", speedup=1.0, optimized_model=None)

    def _apply_int8(
        self,
        model: nn.Module,
        sample_input: Dict[str, Any],
    ) -> nn.Module:
        """
        Apply torchao INT8 quantization with regime gate.

        Regime gate (A100-calibrated): seq>=1024 AND batch×seq<=4096.
        Outside this window, returns model unchanged → caller sees speedup~1.0 → ROLLBACK.

        Tier 1: Int8DynamicActivationInt8WeightConfig (dynamic_activation)
        Tier 2: Int8WeightOnlyConfig (weight_only fallback)
        Returns original if torchao unavailable or both tiers fail.
        """
        # Extract batch and sequence length from sample_input
        seq_len = 1
        batch   = 1
        for v in sample_input.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                batch   = v.shape[0]
                seq_len = v.shape[1]
                break

        in_regime = (seq_len >= 1024) and (batch * seq_len <= 4096)
        if not in_regime:
            log.info(
                "    int8: regime gate REJECTED (seq=%d, batch×seq=%d) — "
                "need seq>=1024 and batch×seq<=4096",
                seq_len, batch * seq_len,
            )
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

    # ── Benchmarking ───────────────────────────────────────────────────────────

    def _benchmark(
        self,
        model: nn.Module,
        sample_input: Dict[str, Any],
        warmup: int = 5,
        iters: int = 20,
    ) -> float:
        """Median latency in ms using CUDA events (perf_counter fallback on CPU)."""
        def _run() -> Any:
            return model(**sample_input)

        return _bench_ms(_run, warmup=warmup, iters=iters)

    def _benchmark_with_power(
        self,
        model: nn.Module,
        sample_input: Dict[str, Any],
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
                model(**sample_input)
        torch.cuda.synchronize()

        # Timed + power-sampled batch
        with PowerSampler() as sampler:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                for _ in range(iters):
                    model(**sample_input)
            e.record()
            torch.cuda.synchronize()
            duration_ms = s.elapsed_time(e)

        median_ms = duration_ms / iters
        power = sampler.report(
            duration_ms=duration_ms,
            token_count=token_count * iters,
        )
        return median_ms, power

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
            if ai and ai != float("inf"):
                return (
                    f"All available optimizations applied for bottleneck type "
                    f"'{state.current_bottleneck}' "
                    f"(arithmetic intensity={ai:.0f} FLOPS/byte). "
                    f"To go further: (1) install flash-attn for full Flash Attention "
                    f"(pip install flash-attn); "
                    f"(2) increase batch size or sequence length to seq>=1024 "
                    f"to unlock INT8 regime; "
                    f"(3) use bfloat16 mixed precision to halve memory traffic."
                )
            return (
                f"All available optimizations applied for bottleneck type "
                f"'{state.current_bottleneck}'. No further candidates in priority table."
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
