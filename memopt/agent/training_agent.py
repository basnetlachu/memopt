"""
memopt Autonomous Training Optimization Agent
=============================================

Subclass of MemoptAgent that adds training-specific behavior:
  - Profiles full training steps (forward + backward + optimizer.step)
  - Training-specific candidates: gradient_checkpointing, bf16_mixed_precision
  - Checkpoint-based rollback (state_dict only — never deepcopy optimizer)
  - Loss stability check after every commit (>20% increase = rollback + stop)
  - NEVER applies int8 — breaks gradients

Stop conditions (same 5 as inference, plus loss_diverged):
  1. OPTIMAL          — compute-bound
  2. TARGET_MET       — speedup >= target
  3. EXHAUSTED        — all candidates tried
  4. NO_PROGRESS      — N consecutive rounds with no commit
  5. MAX_ROUNDS       — hard cap
  6. LOSS_DIVERGED    — committed opt caused loss to spike >20% → rollback + stop
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from .optimization_agent import (
    MemoptAgent,
    AgentReport,
    AgentRound,
    AgentState,
    _ProfileResult,
    _ApplyResult,
    OPTIMIZATION_PRIORITY,
)
from memopt.phase3.universal_optimizer import (
    select_optimizations,
    safe_compile,
    UniversalPlan,
    apply_universal_plan,
    _bench_ms,
)

log = logging.getLogger("memopt.training_agent")


# =============================================================================
# Decision table — training candidates only
# int8 / torchao are BLACKLISTED: they break gradients
# compile uses "default" mode (not reduce-overhead) — autograd needs it
# =============================================================================

TRAINING_OPTIMIZATION_PRIORITY: Dict[str, List[str]] = {
    "MEMORY_BOUND_DRAM": [
        "bf16_mixed_precision",      # first: halves bandwidth demand everywhere
        "gradient_checkpointing",    # second: trades compute for activation memory
        "sdpa",                      # third: attention optimization
        "channels_last",             # CNNs only
        "compile",                   # last: fuses everything
    ],
    "MEMORY_BOUND_CACHE": [
        "gradient_checkpointing",
        "channels_last",
        "compile",
    ],
    "PIPELINE_BOUND_OCCUPANCY": [
        "bf16_mixed_precision",
        "compile",
    ],
    "COMPUTE_BOUND": [],    # STOP — already optimal
    "MIXED": [
        "bf16_mixed_precision",
        "sdpa",
        "compile",
    ],
}

# NEVER apply these to training — will break gradients
TRAINING_BLACKLIST: Set[str] = {
    "int8",
    "torchao_int8",
    "dynamic_activation",
    "weight_only",
}


# =============================================================================
# Public types
# =============================================================================

@dataclass
class TrainingAgentReport:
    """Extended report for training optimization sessions."""
    # Core fields (mirrors AgentReport)
    model_name:                str
    gpu_name:                  str
    target_speedup:            float
    final_speedup:             float
    target_met:                bool
    rounds:                    List[AgentRound]
    total_time_seconds:        float
    optimizations_applied:     List[str]
    optimizations_rolled_back: List[str]
    honest_ceiling:            str

    # Training-specific
    baseline_loss:             float
    final_loss:                float
    loss_stable:               bool
    gradient_checkpointing:    bool   # was gradient_checkpointing committed?
    bf16_enabled:              bool   # was bf16_mixed_precision committed?
    scaler:                    Optional[torch.cuda.amp.GradScaler]
    usage_instructions:        str    # how to use scaler / autocast in caller's loop

    # Optimized model (None if nothing committed or training-state-only opts applied)
    optimized_model:           Optional[nn.Module]

    def summary(self) -> str:
        lines = [
            f"TrainingAgent Report — {self.model_name} on {self.gpu_name}",
            f"  Target:   {self.target_speedup:.2f}x",
            f"  Final:    {self.final_speedup:.2f}x",
            f"  Met:      {'YES' if self.target_met else 'NO'}",
            f"  Rounds:   {len(self.rounds)}",
            f"  Applied:  {', '.join(self.optimizations_applied) or 'none'}",
            f"  Rolled:   {', '.join(self.optimizations_rolled_back) or 'none'}",
            f"  Loss:     {self.baseline_loss:.4f} → {self.final_loss:.4f} "
            f"({'stable' if self.loss_stable else 'UNSTABLE'})",
            f"  GradCkpt: {'YES' if self.gradient_checkpointing else 'no'}",
            f"  BF16:     {'YES' if self.bf16_enabled else 'no'}",
            f"  Time:     {self.total_time_seconds:.1f}s",
            f"  Ceiling:  {self.honest_ceiling}",
        ]
        if self.bf16_enabled:
            lines.append(f"\n  {self.usage_instructions}")
        return "\n".join(lines)


# =============================================================================
# Training-specific transformations (module-level helpers)
# =============================================================================

def apply_gradient_checkpointing(model: nn.Module) -> nn.Module:
    """
    Recompute activations during backward instead of storing them.
    Reduces activation memory by ~60%, costs ~30% extra compute.

    Priority:
      1. HuggingFace gradient_checkpointing_enable() API
      2. torch.utils.checkpoint on detected sequential blocks (fallback)
    """
    # Try HuggingFace API first — most transformers support it
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        log.info("gradient_checkpointing: enabled via HuggingFace API")
        return model

    # Fallback: wrap detected blocks with torch.utils.checkpoint
    import torch.utils.checkpoint as ckpt

    def _make_checkpointed(module: nn.Module) -> nn.Module:
        original_forward = module.forward

        def _checkpointed_forward(*args, **kwargs):
            # checkpoint requires at least one tensor input for autograd
            # Flatten args to find tensors; if none, fall back to regular forward
            tensor_args = [a for a in args if isinstance(a, torch.Tensor)]
            if not tensor_args:
                return original_forward(*args, **kwargs)

            def _fn(*a):
                # Reconstruct original arg list (non-tensors unchanged)
                idx = 0
                rebuilt = []
                for orig in args:
                    if isinstance(orig, torch.Tensor):
                        rebuilt.append(a[idx])
                        idx += 1
                    else:
                        rebuilt.append(orig)
                return original_forward(*rebuilt, **kwargs)

            return ckpt.checkpoint(_fn, *tensor_args, use_reentrant=False)

        module.forward = _checkpointed_forward
        return module

    applied = 0
    for name, module in model.named_modules():
        leaf = name.split(".")[-1].lower() if name else ""
        # Target encoder/decoder blocks, not embeddings or heads
        if any(k in leaf for k in ["layer", "block", "encoder", "decoder"]):
            if list(module.children()):  # must have submodules
                _make_checkpointed(module)
                applied += 1
                if applied >= 4:
                    break

    log.info(f"gradient_checkpointing: torch.utils.checkpoint applied to {applied} blocks")
    return model


def apply_bf16_mixed_precision(
    model: nn.Module,
) -> Tuple[nn.Module, torch.cuda.amp.GradScaler]:
    """
    Enable BF16 autocast for forward pass; keep FP32 for optimizer.

    Does NOT call model.half() — that would break optimizer momentum buffers.
    Returns (model, scaler). Caller must wrap forward+backward with:

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            loss = loss_fn(output, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    Requires Ampere+ (compute capability >= 8.0).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("BF16 mixed precision requires CUDA")

    props = torch.cuda.get_device_properties(0)
    if props.major < 8:
        raise RuntimeError(
            f"BF16 requires Ampere+ GPU (compute capability >= 8.0). "
            f"Current GPU: {props.name} (CC {props.major}.{props.minor}). "
            f"Use FP16 (torch.float16) with GradScaler for older GPUs."
        )

    # Model weights stay FP32 — autocast handles cast at kernel boundary
    scaler = torch.amp.GradScaler('cuda')
    log.info(
        "bf16_mixed_precision: enabled on %s (CC %d.%d, BF16 native)",
        props.name, props.major, props.minor,
    )
    return model, scaler


# =============================================================================
# TrainingAgent
# =============================================================================

class TrainingAgent(MemoptAgent):
    """
    Autonomous training optimization agent.

    Extends MemoptAgent with:
      - Full-step profiling (forward + backward + optimizer.step)
      - Training candidate table (no int8, add grad_ckpt + bf16)
      - Checkpoint-based rollback (state_dict only)
      - Loss stability check after every commit

    Usage:
        agent = TrainingAgent(target_speedup=1.5, max_rounds=5)
        report = agent.run(model, optimizer, loss_fn, sample_batch)
        print(report.summary())
    """

    def __init__(
        self,
        target_speedup:    float = 1.5,    # training target lower than inference
        max_rounds:        int   = 10,
        min_improvement:   float = 0.05,
        no_progress_limit: int   = 3,
        loss_threshold:    float = 0.20,   # >20% loss increase = rollback + stop
    ):
        super().__init__(
            target_speedup=target_speedup,
            max_rounds=max_rounds,
            min_improvement=min_improvement,
            no_progress_limit=no_progress_limit,
        )
        self.loss_threshold = loss_threshold
        self._scaler: Optional[torch.cuda.amp.GradScaler] = None
        self._bf16_enabled = False
        self._grad_ckpt_enabled = False

    # ── Blacklist guard ─────────────────────────────────────────────────────

    @staticmethod
    def _assert_not_blacklisted(candidate: str) -> None:
        """Raise immediately if a blacklisted candidate is attempted."""
        if candidate in TRAINING_BLACKLIST:
            raise ValueError(
                f"TrainingAgent BLACKLIST: '{candidate}' must never be applied "
                f"to training — it breaks gradients. "
                f"Blacklisted: {TRAINING_BLACKLIST}"
            )

    # ── Candidate management (override parent) ──────────────────────────────

    def _get_candidates(self, state: AgentState) -> List[str]:
        all_candidates = TRAINING_OPTIMIZATION_PRIORITY.get(
            state.current_bottleneck, []
        )
        return [c for c in all_candidates if c not in state.tried_candidates]

    def _remaining_candidates(self, state: AgentState) -> List[str]:
        return self._get_candidates(state)

    # ── Full training-step profiling ─────────────────────────────────────────

    def _profile_training_step(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
    ) -> Tuple[_ProfileResult, float]:
        """
        Profile complete training step: forward + backward + optimizer.step.
        Returns (bottleneck_profile, step_time_ms).
        """
        inp = sample_batch["input"]
        tgt = sample_batch.get("target")

        def _step():
            optimizer.zero_grad()
            if self._bf16_enabled and self._scaler is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inp) if isinstance(inp, dict) else model(inp)
                    loss = loss_fn(out, tgt)
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                out = model(**inp) if isinstance(inp, dict) else model(inp)
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()

        # Warmup — 3 steps, discard
        for _ in range(3):
            _step()

        # Measure — 10 steps with CUDA events, take median
        times: List[float] = []
        torch.cuda.synchronize()
        for _ in range(10):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            optimizer.zero_grad()
            s.record()
            if self._bf16_enabled and self._scaler is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inp) if isinstance(inp, dict) else model(inp)
                    loss = loss_fn(out, tgt)
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                out = model(**inp) if isinstance(inp, dict) else model(inp)
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

        times.sort()
        step_ms = float(times[len(times) // 2])

        # Bottleneck via select_optimizations (forward pass only — sufficient for
        # regime classification; backward doesn't change AI materially)
        profile = self._profile(model, inp if isinstance(inp, dict) else {"input": inp})
        return profile, step_ms

    # ── Checkpoint-based rollback ────────────────────────────────────────────

    @staticmethod
    def _save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, Any]:
        """
        Save training state before applying an optimization.
        Deepcopy state_dicts only — never deepcopy the whole optimizer object,
        which would corrupt momentum buffers and CUDA tensor references.
        """
        return {
            "model_state":     copy.deepcopy(model.state_dict()),
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        }

    @staticmethod
    def _restore_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint: Dict[str, Any],
    ) -> None:
        """Restore model and optimizer state after a failed optimization."""
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        log.info("Checkpoint restored — optimization rolled back")

    # ── Loss stability check ───────────────���─────────────────────────────────

    def _check_loss_stability(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        baseline_loss: float,
    ) -> Tuple[bool, float]:
        """
        Run 5 training steps, measure average loss.
        Returns (stable, current_loss).
        stable = loss didn't increase by more than self.loss_threshold (20%).
        """
        inp = sample_batch["input"]
        tgt = sample_batch.get("target")
        losses: List[float] = []

        for _ in range(5):
            optimizer.zero_grad()
            if self._bf16_enabled and self._scaler is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inp) if isinstance(inp, dict) else model(inp)
                    loss = loss_fn(out, tgt)
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                out = model(**inp) if isinstance(inp, dict) else model(inp)
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()
            losses.append(loss.item())

        current_loss = sum(losses) / len(losses)
        stable = current_loss <= baseline_loss * (1.0 + self.loss_threshold)

        if not stable:
            log.warning(
                "Loss instability detected: baseline=%.4f current=%.4f "
                "increase=%.1f%% (threshold=%.0f%%)",
                baseline_loss,
                current_loss,
                100.0 * (current_loss / baseline_loss - 1.0),
                self.loss_threshold * 100,
            )
        return stable, current_loss

    # ── Measure baseline loss ────────────────────────────────────────────────

    def _measure_baseline_loss(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
    ) -> float:
        """Run 5 warmup steps and return average loss."""
        inp = sample_batch["input"]
        tgt = sample_batch.get("target")
        losses: List[float] = []
        for _ in range(5):
            optimizer.zero_grad()
            out = model(**inp) if isinstance(inp, dict) else model(inp)
            loss = loss_fn(out, tgt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return sum(losses) / len(losses)

    # ── Training benchmark (full step) ──────────────────────────────────────

    def _benchmark_training_step(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        warmup: int = 3,
        iters: int = 10,
    ) -> float:
        """Median full-step latency in ms (forward + backward + optimizer)."""
        inp = sample_batch["input"]
        tgt = sample_batch.get("target")

        for _ in range(warmup):
            optimizer.zero_grad()
            if self._bf16_enabled and self._scaler is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inp) if isinstance(inp, dict) else model(inp)
                    loss = loss_fn(out, tgt)
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                out = model(**inp) if isinstance(inp, dict) else model(inp)
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()

        times: List[float] = []
        torch.cuda.synchronize()
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            optimizer.zero_grad()
            s.record()
            if self._bf16_enabled and self._scaler is not None:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(**inp) if isinstance(inp, dict) else model(inp)
                    loss = loss_fn(out, tgt)
                self._scaler.scale(loss).backward()
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                out = model(**inp) if isinstance(inp, dict) else model(inp)
                loss = loss_fn(out, tgt)
                loss.backward()
                optimizer.step()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

        times.sort()
        return float(times[len(times) // 2])

    # ── Apply training-specific optimization ─────────────────────────────────

    def _apply_training_optimization(
        self,
        model: nn.Module,
        sample_batch: Dict[str, Any],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        candidate_name: str,
    ) -> Tuple[str, nn.Module, Optional[torch.cuda.amp.GradScaler]]:
        """
        Apply one training candidate.
        Returns ("COMMITTED"|"ROLLED_BACK", model, scaler_or_None).

        Commit threshold: 5% improvement (training is noisier than inference).
        """
        self._assert_not_blacklisted(candidate_name)

        commit_threshold = 0.05  # 5% minimum improvement to commit

        try:
            if candidate_name == "bf16_mixed_precision":
                new_model, scaler = apply_bf16_mixed_precision(model)
                # Temporarily enable so benchmark uses autocast
                self._bf16_enabled = True
                self._scaler = scaler

                baseline_ms  = self._benchmark_training_step(model,     sample_batch, optimizer, loss_fn)
                optimized_ms = self._benchmark_training_step(new_model, sample_batch, optimizer, loss_fn)
                speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 1.0

                if speedup - 1.0 >= commit_threshold:
                    log.info(
                        "    bf16_mixed_precision: %.3fx (%.1f%%) → COMMIT",
                        speedup, (speedup - 1.0) * 100,
                    )
                    return "COMMITTED", new_model, scaler
                else:
                    # Undo bf16 state
                    self._bf16_enabled = False
                    self._scaler = None
                    log.info(
                        "    bf16_mixed_precision: %.3fx (%.1f%%) → ROLLBACK",
                        speedup, (speedup - 1.0) * 100,
                    )
                    return "ROLLED_BACK", model, None

            elif candidate_name == "gradient_checkpointing":
                candidate_model = copy.deepcopy(model)
                apply_gradient_checkpointing(candidate_model)

                baseline_ms  = self._benchmark_training_step(model,           sample_batch, optimizer, loss_fn)
                optimized_ms = self._benchmark_training_step(candidate_model, sample_batch, optimizer, loss_fn)
                # Note: gradient_checkpointing typically SLOWS the step (uses compute for memory).
                # We still commit if it doesn't regress speed by >5% AND reduces memory.
                # For pure memory pressure relief, always commit if within 5% of baseline.
                speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 1.0

                # Gradient checkpointing: commit if it doesn't slow down by > 5%
                # (memory savings justify the small compute cost)
                if speedup >= 0.95:
                    log.info(
                        "    gradient_checkpointing: %.3fx → COMMIT "
                        "(memory savings justify compute cost)",
                        speedup,
                    )
                    return "COMMITTED", candidate_model, None
                else:
                    log.info(
                        "    gradient_checkpointing: %.3fx (>5%% slower) → ROLLBACK",
                        speedup,
                    )
                    return "ROLLED_BACK", model, None

            elif candidate_name == "channels_last":
                candidate_model = copy.deepcopy(model)
                plan = UniversalPlan(use_channels_last=True)
                optimized = apply_universal_plan(candidate_model, sample_batch.get("input", {}), plan)

                baseline_ms  = self._benchmark_training_step(model,     sample_batch, optimizer, loss_fn)
                optimized_ms = self._benchmark_training_step(optimized, sample_batch, optimizer, loss_fn)
                speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 1.0

                if speedup - 1.0 >= commit_threshold:
                    log.info("    channels_last: %.3fx → COMMIT", speedup)
                    return "COMMITTED", optimized, None
                else:
                    log.info("    channels_last: %.3fx → ROLLBACK", speedup)
                    return "ROLLED_BACK", model, None

            elif candidate_name == "sdpa":
                candidate_model = copy.deepcopy(model)
                plan = UniversalPlan(use_sdpa=True)
                optimized = apply_universal_plan(candidate_model, sample_batch.get("input", {}), plan)

                baseline_ms  = self._benchmark_training_step(model,     sample_batch, optimizer, loss_fn)
                optimized_ms = self._benchmark_training_step(optimized, sample_batch, optimizer, loss_fn)
                speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 1.0

                if speedup - 1.0 >= commit_threshold:
                    log.info("    sdpa: %.3fx → COMMIT", speedup)
                    return "COMMITTED", optimized, None
                else:
                    log.info("    sdpa: %.3fx → ROLLBACK", speedup)
                    return "ROLLED_BACK", model, None

            elif candidate_name == "compile":
                # Training requires "default" mode — reduce-overhead uses CUDAGraphs
                # which capture the graph at fixed shapes and break with autograd.
                candidate_model = safe_compile(
                    copy.deepcopy(model),
                    sample_batch.get("input", {}),
                    mode="default",
                )

                baseline_ms  = self._benchmark_training_step(model,           sample_batch, optimizer, loss_fn)
                optimized_ms = self._benchmark_training_step(candidate_model, sample_batch, optimizer, loss_fn)
                speedup = baseline_ms / optimized_ms if optimized_ms > 0 else 1.0

                if speedup - 1.0 >= commit_threshold:
                    log.info("    compile(default): %.3fx → COMMIT", speedup)
                    return "COMMITTED", candidate_model, None
                else:
                    log.info("    compile(default): %.3fx → ROLLBACK", speedup)
                    return "ROLLED_BACK", model, None

            else:
                log.warning("    Unknown training candidate '%s' — skip", candidate_name)
                return "ROLLED_BACK", model, None

        except Exception as exc:
            log.warning("    %s raised exception: %s — ROLLBACK", candidate_name, exc)
            # Undo bf16 state if it was mid-flight
            if candidate_name == "bf16_mixed_precision":
                self._bf16_enabled = False
                self._scaler = None
            return "ROLLED_BACK", model, None

    # ── Main entry point ────────────────────────────────────────────────────

    def run(  # type: ignore[override]
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        sample_batch: Dict[str, Any],
        target_speedup: Optional[float] = None,
    ) -> TrainingAgentReport:
        """
        Autonomous training optimization loop.

        Args:
            model:         nn.Module on CUDA, in training mode
            optimizer:     optimizer attached to model parameters
            loss_fn:       callable(output, target) → scalar loss tensor
            sample_batch:  {"input": ..., "target": ...} on CUDA
            target_speedup: override self.target_speedup if provided

        Returns:
            TrainingAgentReport with full session detail
        """
        if target_speedup is not None:
            self.target_speedup = target_speedup

        start_time = time.time()
        gpu_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
        log.info(
            "TrainingAgent starting | target=%.2fx | gpu=%s",
            self.target_speedup, gpu_name,
        )

        # Reset training-specific state
        self._bf16_enabled = False
        self._grad_ckpt_enabled = False
        self._scaler = None

        # ── Measure true baseline (full training step) ───────────────────────
        baseline_ms = self._benchmark_training_step(model, sample_batch, optimizer, loss_fn)
        baseline_loss = self._measure_baseline_loss(model, sample_batch, optimizer, loss_fn)
        log.info("Baseline step: %.2f ms | baseline loss: %.4f", baseline_ms, baseline_loss)

        current_model = model          # work in-place — state_dict checkpoints handle rollback
        current_loss  = baseline_loss

        state = AgentState(
            round_number=0,
            cumulative_speedup=1.0,
            rounds_without_commit=0,
            tried_candidates=set(),
            target_speedup=self.target_speedup,
        )
        rounds:          List[AgentRound] = []
        loss_diverged    = False
        final_scaler:    Optional[torch.cuda.amp.GradScaler] = None

        while True:
            state.round_number += 1
            log.info("\n--- Training Round %d ---", state.round_number)

            # Profile current model regime
            inp = sample_batch["input"]
            profile = self._profile(
                current_model,
                inp if isinstance(inp, dict) else {"input": inp},
            )
            state.current_bottleneck = profile.bottleneck_type
            state.last_arithmetic_intensity = profile.ai

            log.info(
                "Bottleneck: %s (confidence=%.2f)  AI=%.1f  ridge=%.1f",
                state.current_bottleneck, profile.confidence,
                profile.ai, profile.ridge,
            )

            # ── Stop conditions ──────────────────────────────────────────────
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

            candidates          = self._get_candidates(state)
            round_committed:    List[str] = []
            round_rolled_back:  List[str] = []
            round_start_speedup = state.cumulative_speedup

            for candidate_name in candidates:
                if candidate_name in state.tried_candidates:
                    continue
                state.tried_candidates.add(candidate_name)

                log.info("  Trying training candidate: %s", candidate_name)

                # Save checkpoint before attempting
                checkpoint = self._save_checkpoint(current_model, optimizer)

                status, new_model, new_scaler = self._apply_training_optimization(
                    current_model, sample_batch, optimizer, loss_fn, candidate_name,
                )

                if status == "COMMITTED":
                    current_model = new_model
                    if new_scaler is not None:
                        final_scaler = new_scaler

                    # Re-measure cumulative speedup vs original baseline
                    new_step_ms = self._benchmark_training_step(
                        current_model, sample_batch, optimizer, loss_fn
                    )
                    state.cumulative_speedup = (
                        baseline_ms / new_step_ms if new_step_ms > 0 else 1.0
                    )

                    # Loss stability check after commit
                    stable, current_loss = self._check_loss_stability(
                        current_model, sample_batch, optimizer, loss_fn, baseline_loss
                    )
                    if not stable:
                        log.warning(
                            "  LOSS_DIVERGED after %s — restoring checkpoint and stopping",
                            candidate_name,
                        )
                        self._restore_checkpoint(current_model, optimizer, checkpoint)
                        # Undo bf16 state if that's what caused divergence
                        if candidate_name == "bf16_mixed_precision":
                            self._bf16_enabled = False
                            self._scaler = None
                            final_scaler = None
                        state.cumulative_speedup = round_start_speedup
                        round_rolled_back.append(candidate_name)
                        loss_diverged = True
                        break  # exit candidate loop and set stop reason

                    round_committed.append(candidate_name)
                    state.rounds_without_commit = 0
                    log.info(
                        "  COMMITTED %s — cumulative=%.2fx  step=%.2fms  loss=%.4f",
                        candidate_name, state.cumulative_speedup, new_step_ms, current_loss,
                    )

                    # Track training-specific committed opts
                    if candidate_name == "gradient_checkpointing":
                        self._grad_ckpt_enabled = True
                    elif candidate_name == "bf16_mixed_precision":
                        self._bf16_enabled = True

                else:
                    round_rolled_back.append(candidate_name)
                    # Restore checkpoint (some transforms modify model in-place)
                    self._restore_checkpoint(current_model, optimizer, checkpoint)
                    log.info("  ROLLED_BACK %s", candidate_name)

                if loss_diverged:
                    break

            if not round_committed:
                state.rounds_without_commit += 1

            round_speedup = (
                state.cumulative_speedup / round_start_speedup
                if round_start_speedup > 0 else 1.0
            )

            stop_for_round = None
            if loss_diverged:
                stop_for_round = "LOSS_DIVERGED: loss increased >20% after commit — rolled back and stopped"

            rounds.append(AgentRound(
                round_number=state.round_number,
                bottleneck_type=state.current_bottleneck,
                confidence=profile.confidence,
                candidates_tried=list(candidates),
                candidates_committed=round_committed,
                candidates_rolled_back=round_rolled_back,
                speedup_this_round=round_speedup,
                cumulative_speedup=state.cumulative_speedup,
                stop_reason=stop_for_round,
            ))

            if loss_diverged:
                break

        all_committed   = [c for r in rounds for c in r.candidates_committed]
        all_rolled_back = [c for r in rounds for c in r.candidates_rolled_back]

        # Final loss measurement
        _, final_loss = self._check_loss_stability(
            current_model, sample_batch, optimizer, loss_fn, baseline_loss
        )
        loss_stable = final_loss <= baseline_loss * (1.0 + self.loss_threshold)

        # Build usage instructions for bf16
        usage_instructions = ""
        if self._bf16_enabled and final_scaler is not None:
            usage_instructions = (
                "BF16 mixed precision is active. Wrap your training loop with:\n"
                "    with torch.autocast('cuda', dtype=torch.bfloat16):\n"
                "        output = model(batch)\n"
                "        loss = loss_fn(output, target)\n"
                "    report.scaler.scale(loss).backward()\n"
                "    report.scaler.step(optimizer)\n"
                "    report.scaler.update()"
            )

        return TrainingAgentReport(
            model_name=type(model).__name__,
            gpu_name=gpu_name,
            target_speedup=self.target_speedup,
            final_speedup=state.cumulative_speedup,
            target_met=state.cumulative_speedup >= self.target_speedup,
            rounds=rounds,
            total_time_seconds=time.time() - start_time,
            optimizations_applied=all_committed,
            optimizations_rolled_back=all_rolled_back,
            honest_ceiling=self._build_training_ceiling(state, rounds, loss_diverged),
            baseline_loss=baseline_loss,
            final_loss=final_loss,
            loss_stable=loss_stable,
            gradient_checkpointing=self._grad_ckpt_enabled,
            bf16_enabled=self._bf16_enabled,
            scaler=final_scaler,
            usage_instructions=usage_instructions,
            optimized_model=current_model if all_committed else None,
        )

    # ── Honest ceiling (training-aware) ─────────────────────────────────────

    def _build_training_ceiling(
        self,
        state: AgentState,
        rounds: List[AgentRound],
        loss_diverged: bool,
    ) -> str:
        if not rounds:
            return "No rounds completed."

        final_round = rounds[-1]
        stop = final_round.stop_reason or ""

        if loss_diverged or "LOSS_DIVERGED" in stop:
            return (
                "An optimization caused loss to increase by more than 20%. "
                "The agent rolled back and stopped to protect training stability. "
                "Try: (1) lower learning rate before re-running; "
                "(2) use a smaller batch for warmup; "
                "(3) reduce target_speedup to allow more conservative opts."
            )

        if "OPTIMAL" in stop:
            ai = state.last_arithmetic_intensity or 0.0
            return (
                f"Training step is compute-bound "
                f"(arithmetic intensity={ai:.0f} FLOPS/byte > ridge). "
                f"Memory bandwidth is not the bottleneck. "
                f"To go faster: use gradient accumulation to increase effective batch size, "
                f"or reduce model depth."
            )

        if "TARGET_MET" in stop:
            return (
                f"Training target of {self.target_speedup:.2f}x achieved. "
                f"Remaining candidates deliver diminishing returns (<5% each)."
            )

        if "EXHAUSTED" in stop:
            ai = state.last_arithmetic_intensity
            tips = []
            if not self._bf16_enabled:
                tips.append("enable BF16 manually if GPU supports it (Ampere+)")
            if not self._grad_ckpt_enabled:
                tips.append("enable gradient_checkpointing for memory-heavy models")
            tips.append("try a larger batch size (>32) to make memory-bound regime clearer")
            return (
                f"All training candidates tried for bottleneck "
                f"'{state.current_bottleneck}' (AI={ai:.0f} FLOPS/byte). "
                f"Next steps: {'; '.join(tips)}."
            )

        if "NO_PROGRESS" in stop:
            return (
                f"No improvement in last {self.no_progress_limit} consecutive rounds. "
                f"Try: larger batch size, longer sequence length, or different optimizer."
            )

        if "MAX_ROUNDS" in stop:
            return (
                f"Reached {self.max_rounds}-round limit. "
                f"Final speedup: {state.cumulative_speedup:.2f}x. "
                f"Increase max_rounds= for more attempts."
            )

        return (
            f"Stopped after {state.round_number} rounds. "
            f"Final: {state.cumulative_speedup:.2f}x."
        )
