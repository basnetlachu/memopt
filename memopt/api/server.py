"""
memopt FastAPI server — 4 endpoints:

  POST /optimize       submit optimization job; returns job_id immediately
  GET  /status/{id}    poll job status
  GET  /metrics        Prometheus text exposition (prometheus_client)
  GET  /health         {"status": "ok", "gpu": <name>}  always <100 ms

Start:
  uvicorn memopt.api.server:app --host 0.0.0.0 --port 8080

GPU work is synchronous (torch CUDA calls block the calling thread), so
each job runs in a ThreadPoolExecutor thread rather than an asyncio task.
max_workers=2 enforces the ≤2 concurrent jobs constraint; additional
submissions queue transparently inside the executor.

Security: model weights are never accepted over HTTP — file paths only.
Job store is in-memory; no persistence across server restarts.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel, field_validator

log = logging.getLogger("memopt.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


# ── Job model ─────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    QUEUED      = "queued"
    RUNNING     = "running"
    COMPLETE    = "complete"
    FAILED      = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class OptimizationJob:
    job_id:         str
    status:         JobStatus
    model_path:     str
    input_shape:    list
    created_at:     float
    started_at:     Optional[float] = None
    completed_at:   Optional[float] = None
    speedup:        Optional[float] = None
    tier_used:      Optional[str]   = None
    error:          Optional[str]   = None
    result_path:    Optional[str]   = None
    # Agent-specific fields (None for regular /optimize jobs)
    target_speedup: Optional[float] = None
    max_rounds:     Optional[int]   = None
    rounds_summary: Optional[str]   = None   # JSON-serialisable round log


# ── Global state ──────────────────────────────────────────────────────────────

job_store: Dict[str, OptimizationJob] = {}

# max_workers=2: never run more than 2 GPU jobs concurrently (VRAM budget).
# Extra submissions queue automatically inside the executor.
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="memopt-gpu")

RESULTS_DIR = Path("/tmp/memopt_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Prometheus metrics ────────────────────────────────────────────────────────

jobs_total = Counter(
    "memopt_jobs_total",
    "Total optimization jobs submitted",
    ["status"],          # labels: complete, failed, rolled_back
)

optimizations_applied = Counter(
    "memopt_optimizations_applied_total",
    "Optimizations committed by type",
    ["type"],            # labels: torchao_int8, sdpa, channels_last, compile
)

rollbacks_total = Counter(
    "memopt_rollbacks_total",
    "Optimizations rolled back due to regression or accuracy failure",
    ["reason"],          # labels: regression, accuracy_failure, exception
)

speedup_histogram = Histogram(
    "memopt_speedup_ratio",
    "Speedup ratio achieved (optimized/baseline)",
    buckets=[0.9, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0],
)

optimization_duration = Histogram(
    "memopt_optimization_duration_seconds",
    "Time taken to complete optimization job",
    buckets=[30, 60, 120, 300, 600, 1200],
)

active_jobs = Gauge(
    "memopt_active_jobs",
    "Currently running optimization jobs",
)

gpu_utilization = Gauge(
    "memopt_gpu_utilization_percent",
    "GPU compute utilization during optimization",
    ["gpu_id"],
)

gpu_memory_used = Gauge(
    "memopt_gpu_memory_used_bytes",
    "GPU memory currently allocated in bytes",
    ["gpu_id"],
)

gpu_power_watts = Gauge(
    "memopt_gpu_power_watts",
    "Current GPU power draw in watts",
    ["gpu_id"],
)

baseline_power_watts = Histogram(
    "memopt_baseline_power_watts",
    "GPU power during baseline measurement",
    buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
)

optimized_power_watts = Histogram(
    "memopt_optimized_power_watts",
    "GPU power during optimized model measurement",
    buckets=[50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
)

power_reduction_pct = Histogram(
    "memopt_power_reduction_percent",
    "Power reduction achieved by optimization",
    buckets=[-10, 0, 5, 10, 15, 20, 25, 30, 35, 40],
)

joules_per_token = Histogram(
    "memopt_joules_per_token",
    "Energy efficiency: joules consumed per token",
    buckets=[0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)


# ── GPU metrics background collector ─────────────────────────────────────────

def _collect_gpu_metrics() -> None:
    """Update GPU gauges every 15 seconds. Runs as a daemon thread."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as exc:
        log.warning("GPU metrics collector: pynvml unavailable (%s) — skipping", exc)
        return

    while True:
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_utilization.labels(gpu_id=str(i)).set(util.gpu)
                gpu_memory_used.labels(gpu_id=str(i)).set(mem.used)
                try:
                    power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    gpu_power_watts.labels(gpu_id=str(i)).set(power_mw / 1000.0)
                except pynvml.NVMLError:
                    pass  # power reading not supported on all GPUs
        except Exception as exc:
            log.warning("GPU metrics collection failed: %s", exc)
        time.sleep(15)


threading.Thread(target=_collect_gpu_metrics, daemon=True, name="memopt-gpu-metrics").start()


# ── Request schema ────────────────────────────────────────────────────────────

class OptimizeRequest(BaseModel):
    model_path:  str
    input_shape: List[int]
    dtype:       str = "fp32"
    target:      str = "inference"

    @field_validator("input_shape")
    @classmethod
    def _shape_nonempty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("input_shape must be a non-empty list of integers")
        if any(d <= 0 for d in v):
            raise ValueError("all input_shape dimensions must be positive integers")
        return v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bench_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    """
    Measure median forward-pass latency in milliseconds.

    Uses CUDA events for GPU (sub-microsecond accuracy) and perf_counter
    for CPU. warmup=10 + iters=50 is enough to absorb JIT compilation
    overhead from torch.compile and torchao quantization.
    """
    import torch

    with torch.no_grad():
        for _ in range(warmup):
            fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        times: List[float] = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.no_grad():
                fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                fn()
            times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    return times[len(times) // 2]


def _build_sample_input(
    model: Any,
    input_shape: list,
    device: Any,
) -> Dict[str, Any]:
    """
    Construct a sample input dict from input_shape.

    Tries three strategies in order:
      1. 2-D [batch, seq]  → transformer token IDs + attention_mask
         Tries BERT (30522), GPT-2 (50257), and LLaMA (32000) vocab sizes.
      2. Any shape → single float randn tensor as keyword arg.
         Tries common kwarg names: x, input, inputs, hidden_states.
      3. Positional fallback: returns {"_positional": tensor}.
         run_optimization() unwraps this and calls model(tensor) directly.

    The probe forward passes are intentionally swallowed so that
    type/shape errors fall through to the next strategy quietly.
    """
    import torch

    shape = tuple(input_shape)

    # ── Strategy 1: transformer input_ids ──────────────────────────────────
    if len(shape) == 2:
        for vocab in (30522, 50257, 32000):
            try:
                ids  = torch.randint(0, vocab, shape, device=device)
                mask = torch.ones(shape, dtype=torch.long, device=device)
                sample: Dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
                with torch.no_grad():
                    model(**sample)
                return sample
            except Exception:
                pass

    # ── Strategy 2: float tensor with common kwarg names ───────────────────
    t = torch.randn(*shape, device=device)
    for kw in ("x", "input", "inputs", "hidden_states"):
        try:
            sample = {kw: t}
            with torch.no_grad():
                model(**sample)
            return sample
        except Exception:
            pass

    # ── Strategy 3: positional fallback ────────────────────────────────────
    log.warning(
        "_build_sample_input: no keyword matched shape=%s; "
        "using positional fallback — model(tensor)",
        shape,
    )
    return {"_positional": t}


def _plan_tier(plan: Any) -> str:
    """Derive a readable tier string from a UniversalPlan."""
    parts: List[str] = []
    if getattr(plan, "use_sdpa", False):
        parts.append("sdpa")
    if getattr(plan, "use_channels_last", False):
        parts.append("channels_last")
    if getattr(plan, "compile_mode", None):
        parts.append(f"compile({plan.compile_mode})")
    return "+".join(parts) if parts else "none"


def load_model_safe(model_path: str, device: Any) -> Any:
    """
    Load a model from disk with actionable error messages on failure.

    Detects three common mistakes and raises ValueError with fix instructions
    instead of letting raw pickle/type errors reach the client:
      1. state_dict saved instead of full model → explains torch.save(model, path)
      2. Non-Module object → explains expected type
      3. AttributeError (custom class not importable) → explains both fix options
    """
    import torch

    try:
        obj = torch.load(model_path, map_location=device, weights_only=False)

        # Detect state_dict: an OrderedDict/dict whose values are all tensors.
        if isinstance(obj, dict) and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            raise ValueError(
                "Loaded file contains a state_dict, not a full model. "
                "memopt requires a full model. "
                "Fix: torch.save(model, path) not torch.save(model.state_dict(), path). "
                "If using a custom architecture, ensure the class is installed "
                "in the server environment."
            )

        if not isinstance(obj, torch.nn.Module):
            raise ValueError(
                f"Expected nn.Module, got {type(obj).__name__}. "
                "Save with: torch.save(model, path)"
            )

        return obj

    except AttributeError as exc:
        # Pickle cannot find the class definition in the server process.
        # AttributeError message format: "Can't get attribute 'Foo' on <module ...>"
        # split("'") → ["Can", "t get attribute ", "Foo", " on ..."]  — class is at [2].
        msg = str(exc)
        parts = msg.split("'")
        class_name = parts[2] if len(parts) >= 3 else "unknown"
        raise ValueError(
            f"Cannot load model: class '{class_name}' not found in server environment.\n"
            f"Option 1 (recommended): Use a standard architecture "
            f"(BERT, GPT-2, ResNet) importable by the server.\n"
            f"Option 2: Install your custom class in the server environment "
            f"and rebuild the Docker image.\n"
            f"Original error: {exc}"
        ) from exc

    except ValueError:
        raise   # re-raise our own messages above unchanged

    except Exception as exc:
        raise ValueError(f"Failed to load model from {model_path}: {exc}") from exc


# ── Background worker ─────────────────────────────────────────────────────────

def run_optimization(job_id: str) -> None:
    """
    Execute one optimization job in a ThreadPoolExecutor thread.

    Lifecycle:
      QUEUED → RUNNING → COMPLETE    (speedup ≥ 1.0)
                       → ROLLED_BACK (speedup < 1.0 — optimization made things worse)
                       → FAILED      (exception before speedup is known)

    All exceptions are caught so the server never crashes.
    """
    import torch
    from memopt.phase3 import select_optimizations, apply_universal_plan
    from memopt.profiler.power_sampler import PowerSampler

    job = job_store[job_id]
    job.status     = JobStatus.RUNNING
    job.started_at = time.time()
    active_jobs.inc()
    log.info(
        "Job %s RUNNING — model=%s shape=%s",
        job_id, job.model_path, job.input_shape,
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Load model ──────────────────────────────────────────────────────
        model = load_model_safe(job.model_path, device)
        model.eval()

        # ── Build sample input ───────────────────────────────────────────────
        raw_sample = _build_sample_input(model, job.input_shape, device)
        positional_tensor = raw_sample.pop("_positional", None)
        sample_input = raw_sample   # may be empty if positional path taken

        # Wrap model calls so positional fallback is transparent everywhere.
        if positional_tensor is not None:
            def _call(m: torch.nn.Module) -> Any:
                return m(positional_tensor)
        else:
            def _call(m: torch.nn.Module) -> Any:
                return m(**sample_input)

        # ── Baseline measurement (with power) ────────────────────────────────
        with PowerSampler() as _bps:
            baseline_ms = _bench_ms(lambda: _call(model))
        _bp = _bps.report(duration_ms=baseline_ms * 50)
        if _bp.available:
            baseline_power_watts.observe(_bp.avg_watts)
        log.info("Job %s baseline: %.2f ms | power: %s", job_id, baseline_ms, _bp.summary())

        # ── Optimization pipeline ────────────────────────────────────────────
        # select_optimizations profiles the model and returns a UniversalPlan.
        # apply_universal_plan executes: channels_last → SDPA → safe_compile.
        # Both operate on deepcopy so the original model is never mutated.
        plan      = select_optimizations(copy.deepcopy(model), sample_input)
        optimized = apply_universal_plan(copy.deepcopy(model), sample_input, plan)
        optimized.eval()

        tier = _plan_tier(plan)
        log.info("Job %s plan: %s | tier=%s", job_id, plan, tier)

        # ── Optimized measurement (with power) ───────────────────────────────
        with PowerSampler() as _ops:
            optimized_ms = _bench_ms(lambda: _call(optimized))
        _op = _ops.report(duration_ms=optimized_ms * 50)
        if _op.available:
            optimized_power_watts.observe(_op.avg_watts)
            if _bp.available and _bp.avg_watts > 0:
                _pct = (_bp.avg_watts - _op.avg_watts) / _bp.avg_watts * 100.0
                power_reduction_pct.observe(_pct)
            if _op.joules_per_token is not None:
                joules_per_token.observe(_op.joules_per_token)
        speedup = baseline_ms / optimized_ms
        log.info(
            "Job %s speedup: %.3fx baseline=%.2f ms optimized=%.2f ms | power: %s",
            job_id, speedup, baseline_ms, optimized_ms, _op.summary(),
        )

        # ── Save result ──────────────────────────────────────────────────────
        result_path = str(RESULTS_DIR / f"{job_id}.pt")
        torch.save(optimized, result_path)

        job.speedup     = round(speedup, 4)
        job.tier_used   = tier
        job.result_path = result_path

        # ── Commit or rollback ───────────────────────────────────────────────
        if speedup < 1.0:
            job.status = JobStatus.ROLLED_BACK
            rollbacks_total.labels(reason="regression").inc()
            jobs_total.labels(status="rolled_back").inc()
            log.warning(
                "Job %s ROLLED_BACK — speedup %.3fx < 1.0 "
                "(optimization regressed; original model recommended)",
                job_id, speedup,
            )
        else:
            job.status = JobStatus.COMPLETE
            jobs_total.labels(status="complete").inc()
            log.info("Job %s COMPLETE — speedup %.3fx", job_id, speedup)

        speedup_histogram.observe(speedup)

        # Increment once per optimization type present in the plan.
        if getattr(plan, "use_sdpa", False):
            optimizations_applied.labels(type="sdpa").inc()
        if getattr(plan, "use_channels_last", False):
            optimizations_applied.labels(type="channels_last").inc()
        if getattr(plan, "compile_mode", None):
            optimizations_applied.labels(type="compile").inc()

    except Exception as exc:
        log.error("Job %s FAILED: %s", job_id, exc, exc_info=True)
        jobs_total.labels(status="failed").inc()
        job.status = JobStatus.FAILED
        job.error  = str(exc)

    finally:
        job.completed_at = time.time()
        optimization_duration.observe(job.completed_at - job.started_at)
        active_jobs.dec()


# ── Agent request schema ──────────────────────────────────────────────────────

class AgentRequest(BaseModel):
    model_path:     str
    input_shape:    List[int]
    target_speedup: float = 2.0
    max_rounds:     int   = 10
    dtype:          str   = "fp32"

    @field_validator("input_shape")
    @classmethod
    def _shape_nonempty(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("input_shape must be a non-empty list of integers")
        if any(d <= 0 for d in v):
            raise ValueError("all input_shape dimensions must be positive integers")
        return v


# ── Agent background worker ───────────────────────────────────────────────────

def run_agent_job(job_id: str) -> None:
    """
    Execute autonomous multi-round optimization in a ThreadPoolExecutor thread.

    Lifecycle: QUEUED → RUNNING → COMPLETE | ROLLED_BACK | FAILED
    Same polling semantics as run_optimization(); use GET /status/{job_id}.
    """
    import json as _json
    import torch
    from memopt.agent import MemoptAgent

    job = job_store[job_id]
    job.status     = JobStatus.RUNNING
    job.started_at = time.time()
    active_jobs.inc()
    log.info(
        "AgentJob %s RUNNING — model=%s shape=%s target=%.2fx rounds=%d",
        job_id, job.model_path, job.input_shape,
        job.target_speedup or 2.0, job.max_rounds or 10,
    )

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = load_model_safe(job.model_path, device)
        model.eval()

        raw_sample        = _build_sample_input(model, job.input_shape, device)
        positional_tensor = raw_sample.pop("_positional", None)
        sample_input      = raw_sample

        # Wrap positional models so agent's model(**sample_input) works correctly
        if positional_tensor is not None:
            _pt = positional_tensor

            class _PosWrap(torch.nn.Module):
                def __init__(self, m: torch.nn.Module) -> None:
                    super().__init__()
                    self._m = m

                def forward(self, **_: Any) -> Any:
                    return self._m(_pt)

            model        = _PosWrap(model)
            sample_input = {}

        agent = MemoptAgent(
            target_speedup=job.target_speedup or 2.0,
            max_rounds=job.max_rounds or 10,
        )

        report = agent.run(model, sample_input)

        speedup = report.final_speedup
        job.speedup = round(speedup, 4)

        # Compact round log for /status response
        round_rows = []
        for r in report.rounds:
            row = {
                "round":     r.round_number,
                "bottleneck": r.bottleneck_type,
                "committed":  r.candidates_committed,
                "rolled":     r.candidates_rolled_back,
                "cumulative": round(r.cumulative_speedup, 4),
            }
            if r.stop_reason:
                row["stop"] = r.stop_reason
            round_rows.append(row)
        job.rounds_summary = _json.dumps(round_rows)

        # Save optimized model if anything was committed
        if report.optimized_model is not None:
            result_path = str(RESULTS_DIR / f"{job_id}.pt")
            torch.save(report.optimized_model, result_path)
            job.result_path = result_path
            job.tier_used   = "+".join(report.optimizations_applied) or "none"

        # Commit or rollback decision
        if speedup < 1.0:
            job.status = JobStatus.ROLLED_BACK
            rollbacks_total.labels(reason="regression").inc()
            jobs_total.labels(status="rolled_back").inc()
            log.warning("AgentJob %s ROLLED_BACK — speedup %.3fx", job_id, speedup)
        else:
            job.status = JobStatus.COMPLETE
            jobs_total.labels(status="complete").inc()
            log.info(
                "AgentJob %s COMPLETE — speedup=%.3fx ceiling=%s",
                job_id, speedup, report.honest_ceiling,
            )

        speedup_histogram.observe(speedup)

        # Prometheus per-type counters
        for opt in report.optimizations_applied:
            optimizations_applied.labels(type=opt).inc()

        # Power metrics from AgentReport
        if report.baseline_power.available:
            baseline_power_watts.observe(report.baseline_power.avg_watts)
        if report.optimized_power.available:
            optimized_power_watts.observe(report.optimized_power.avg_watts)
        if report.power_reduction_pct is not None:
            power_reduction_pct.observe(report.power_reduction_pct)

    except Exception as exc:
        log.error("AgentJob %s FAILED: %s", job_id, exc, exc_info=True)
        jobs_total.labels(status="failed").inc()
        job.status = JobStatus.FAILED
        job.error  = str(exc)

    finally:
        job.completed_at = time.time()
        optimization_duration.observe(job.completed_at - job.started_at)
        active_jobs.dec()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="memopt",
    description="GPU model optimization as a service",
    version="1.1.0",
)


@app.post("/optimize")
async def optimize(request: OptimizeRequest):
    """
    Submit a model optimization job.

    Returns immediately with a job_id.  Poll GET /status/{job_id} until
    status is one of: complete | failed | rolled_back.

    Constraints
    -----------
    - Model must already exist on the server filesystem (no upload endpoint).
    - At most 2 jobs run concurrently; additional submissions queue behind them.
    - Only target=inference is supported.
    """
    if not Path(request.model_path).exists():
        raise HTTPException(404, f"Model not found: {request.model_path}")

    job_id = str(uuid.uuid4())
    job = OptimizationJob(
        job_id      = job_id,
        status      = JobStatus.QUEUED,
        model_path  = request.model_path,
        input_shape = request.input_shape,
        created_at  = time.time(),
    )
    job_store[job_id] = job
    executor.submit(run_optimization, job_id)

    log.info(
        "Job %s QUEUED — model=%s shape=%s",
        job_id, request.model_path, request.input_shape,
    )
    return {"job_id": job_id, "status": "queued", "estimated_minutes": 5}


@app.get("/status/{job_id}")
def status(job_id: str):
    """
    Return the current status of an optimization job.

    Fields
    ------
    status:           queued | running | complete | failed | rolled_back
    speedup:          float ≥ 0 (None while running)
    tier_used:        e.g. "sdpa+compile(reduce-overhead)" (None while running)
    result_path:      server-side path to the saved optimized model (None until complete)
    error:            error message string (None unless failed)
    duration_seconds: wall-clock time from RUNNING → terminal state (None while running)
    """
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    duration = (
        round(job.completed_at - job.started_at, 2)
        if job.completed_at is not None and job.started_at is not None
        else None
    )
    return {
        "job_id":           job_id,
        "status":           job.status.value,
        "speedup":          job.speedup,
        "tier_used":        job.tier_used,
        "result_path":      job.result_path,
        "error":            job.error,
        "duration_seconds": duration,
    }


@app.post("/agent")
async def agent_optimize(request: AgentRequest):
    """
    Submit an autonomous multi-round optimization job.

    Same async job pattern as POST /optimize — returns immediately with a
    job_id.  Poll GET /status/{job_id} until status is complete | failed |
    rolled_back.

    Additional fields in GET /status response for agent jobs:
      rounds_summary: JSON array of per-round results (bottleneck, committed,
                      rolled, cumulative speedup, stop reason if terminal).
    """
    if not Path(request.model_path).exists():
        raise HTTPException(404, f"Model not found: {request.model_path}")

    job_id = str(uuid.uuid4())
    job = OptimizationJob(
        job_id         = job_id,
        status         = JobStatus.QUEUED,
        model_path     = request.model_path,
        input_shape    = request.input_shape,
        created_at     = time.time(),
        target_speedup = request.target_speedup,
        max_rounds     = request.max_rounds,
    )
    job_store[job_id] = job
    executor.submit(run_agent_job, job_id)

    log.info(
        "AgentJob %s QUEUED — model=%s shape=%s target=%.2fx rounds=%d",
        job_id, request.model_path, request.input_shape,
        request.target_speedup, request.max_rounds,
    )
    return {
        "job_id":          job_id,
        "status":          "queued",
        "target_speedup":  request.target_speedup,
        "max_rounds":      request.max_rounds,
        "estimated_minutes": max(5, request.max_rounds * 2),
    }


# /metrics is served by the prometheus_client ASGI app (mounted below).
# It returns text/plain Prometheus exposition format, not JSON.
# Scrape with: curl http://localhost:8080/metrics


@app.get("/health")
def health():
    """
    Liveness probe — always responds in <100 ms.

    Does NOT load models or run GPU code.  GPU name is queried from the
    already-cached CUDA context (negligible overhead).
    """
    try:
        import torch
        gpu = (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        )
    except Exception:
        gpu = "unknown"
    return {
        "status": "ok",
        "gpu": gpu,
        "model_format": "torch.save(model, path) required — not state_dict()",
        "supported_architectures": "any nn.Module importable in server environment",
    }


# ── Mount Prometheus /metrics ─────────────────────────────────────────────────
# make_asgi_app() returns a standard ASGI app that serves the Prometheus text
# exposition at its root path.  Mounting it at "/metrics" means
# GET /metrics returns Content-Type: text/plain; version=0.0.4.
app.mount("/metrics", make_asgi_app())
