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

from fastapi import Depends, FastAPI, Header, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, field_validator
from starlette.responses import Response
from memopt.auth.api_key import (
    get_or_create_key, verify_key, mask_key,
    authenticate, is_admin,
    create_tenant, revoke_tenant, list_tenants,
)

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

    Uses detect_input_format() for auto-detection first.
    Falls back to shape-based heuristics if the model cannot run a probe.

    Returns a plain dict that can be used with model(**sample) or
    {"_positional": tensor} as a last resort.
    """
    import torch
    from memopt.utils.input_handler import detect_input_format

    shape = tuple(input_shape)

    # ── Auto-detection via detect_input_format ──────────────────────────────
    # Build a probe tensor from shape and let detect_input_format try everything
    if len(shape) == 2 and model is not None:
        # Transformer probe: try integer tensor (token IDs)
        try:
            ids = torch.randint(0, 32000, shape, device=device)
            fmt = detect_input_format(model, ids, device=str(device))
            log.info("_build_sample_input: auto-detected style=%s keys=%s", fmt.style, fmt.input_keys)
            return fmt.inputs
        except Exception:
            pass

    if model is not None:
        # General probe: float tensor
        try:
            t = torch.randn(*shape, device=device)
            fmt = detect_input_format(model, t, device=str(device))
            log.info("_build_sample_input: auto-detected style=%s keys=%s", fmt.style, fmt.input_keys)
            # For positional style, wrap in a key the caller understands
            if fmt.style == "positional":
                return {"_positional": fmt.inputs["__tensor__"]}
            return fmt.inputs
        except Exception:
            pass

    # ── Fallback: shape-based heuristics (no model probe available) ─────────
    t = torch.randn(*shape, device=device)
    log.warning(
        "_build_sample_input: auto-detection failed for shape=%s; "
        "returning positional fallback",
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


import os as _os

_ALLOW_UNSAFE_LOAD: bool = _os.environ.get("MEMOPT_ALLOW_UNSAFE_LOAD") == "1"

_MODEL_ROOT_RAW: Optional[str] = _os.environ.get("MEMOPT_MODEL_ROOT")
_MODEL_ROOT: Optional[Path] = (
    Path(_MODEL_ROOT_RAW).resolve() if _MODEL_ROOT_RAW else None
)

try:
    from memopt.phase3 import (
        select_optimizations as _phase3_select,
        apply_universal_plan as _phase3_apply,
    )
    _PHASE3_AVAILABLE = True
except ImportError as _phase3_err:
    _PHASE3_AVAILABLE = False
    _PHASE3_IMPORT_ERROR = str(_phase3_err)


def _validate_model_path(raw: str) -> Path:
    """
    Canonicalise and authorise a client-supplied model path.

    Raises HTTPException(400/404) on any violation so error messages do not
    leak filesystem layout. If MEMOPT_MODEL_ROOT is set, the resolved path
    must live under that directory (symlinks resolved); otherwise any
    resolvable regular file on disk is accepted.
    """
    if not raw or "\x00" in raw:
        raise HTTPException(status_code=400, detail="Invalid model_path")
    try:
        resolved = Path(raw).resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError):
        raise HTTPException(status_code=404, detail="Model not found")
    if not resolved.is_file():
        raise HTTPException(status_code=400, detail="model_path is not a regular file")
    if _MODEL_ROOT is not None:
        try:
            resolved.relative_to(_MODEL_ROOT)
        except ValueError:
            raise HTTPException(status_code=404, detail="Model not found")
    return resolved

# Telemetry collector
try:
    from memopt.observability import MetricsCollector
    _p4_collector = MetricsCollector()
    _p4_collector.start()
except Exception as _p4_err:
    import logging as _log
    _log.getLogger(__name__).warning(
        f"Observability startup failed: {_p4_err}"
    )
    _p4_collector = None


def load_model_safe(model_path: str, device: Any) -> Any:
    """
    Load a model from disk with actionable error messages on failure.

    Uses weights_only=True by default to prevent arbitrary code execution from
    pickle deserialization.  To load models with custom architectures that
    require weights_only=False, set MEMOPT_ALLOW_UNSAFE_LOAD=1 in the
    environment — a CRITICAL warning is logged at server startup if that flag
    is set.

    Detects three common mistakes and raises ValueError with fix instructions
    instead of letting raw pickle/type errors reach the client:
      1. state_dict saved instead of full model → explains torch.save(model, path)
      2. Non-Module object → explains expected type
      3. AttributeError (custom class not importable) → explains both fix options
    """
    import torch

    def _load(weights_only: bool) -> Any:
        return torch.load(model_path, map_location=device, weights_only=weights_only)

    try:
        try:
            obj = _load(weights_only=True)
        except Exception as _exc:
            # weights_only=True raises TypeError/pickle.UnpicklingError for custom
            # class pickles or anything that needs arbitrary Python execution.
            if _ALLOW_UNSAFE_LOAD:
                log.warning(
                    "weights_only=True failed for %s (%s). "
                    "Retrying with weights_only=False because "
                    "MEMOPT_ALLOW_UNSAFE_LOAD=1 is set — "
                    "only load files you trust.",
                    model_path, _exc,
                )
                obj = _load(weights_only=False)
            else:
                raise ValueError(
                    f"Model file requires weights_only=False (custom architecture): {_exc}\n"
                    "To allow unsafe loading set MEMOPT_ALLOW_UNSAFE_LOAD=1 in the "
                    "server environment.  Only do this for files you trust."
                ) from _exc

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
    select_optimizations = _phase3_select
    apply_universal_plan = _phase3_apply
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
        from memopt.utils.input_handler import detect_input_format, forward as _ih_forward
        raw_sample = _build_sample_input(model, job.input_shape, device)
        positional_tensor = raw_sample.pop("_positional", None)
        probe = positional_tensor if positional_tensor is not None else raw_sample
        with torch.no_grad():
            _fmt = detect_input_format(model, probe, device=str(device))
        sample_input = _fmt.inputs   # normalized dict for select_optimizations

        def _call(m: torch.nn.Module) -> Any:
            return _ih_forward(m, _fmt)

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


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="memopt",
    description="GPU model optimization as a service",
    version="1.1.0",
)

# ── GKD store — optional, registered by caller at startup ────────────────────
# Set by register_gkd(); None means GKD is not configured.
# /health returns 503 when _gkd_store.stats()["backend_degraded"] is True.
_gkd_store: object = None


def register_gkd(gkd) -> None:
    """
    Wire a GKDStore into the server so /health can probe its Redis backend.

    Call once at startup:
        from memopt.cluster.gkd_store import GKDStore
        register_gkd(GKDStore(redis_url=os.environ["REDIS_URL"]))
    """
    global _gkd_store
    _gkd_store = gkd


# ── API key authentication ────────────────────────────────────────────────────
# Key is loaded/created once at import time; the closure used by verify_api_key
# captures the module-level _API_KEY string so it is always current.
_API_KEY: str = get_or_create_key()
_api_key_header = APIKeyHeader(name="X-Memopt-API-Key", auto_error=False)
log.info("memopt API server: key loaded (%s)", mask_key(_API_KEY))
if _ALLOW_UNSAFE_LOAD:
    log.critical(
        "MEMOPT_ALLOW_UNSAFE_LOAD=1 is set — torch.load will deserialize arbitrary "
        "pickle payloads.  Only load model files you trust."
    )
if _MODEL_ROOT is None:
    log.warning(
        "MEMOPT_MODEL_ROOT is not set — any resolvable file on disk may be loaded via "
        "POST /optimize. Set MEMOPT_MODEL_ROOT=/path/to/models to restrict."
    )
else:
    log.info("MEMOPT_MODEL_ROOT=%s (model_path must resolve under this root)", _MODEL_ROOT)
if not _PHASE3_AVAILABLE:
    log.warning(
        "memopt.phase3 unavailable (%s) — POST /optimize will return 501.",
        _PHASE3_IMPORT_ERROR,
    )


def verify_api_key(key: str = Security(_api_key_header)) -> str:
    """FastAPI dependency — rejects requests without a valid API key."""
    if not verify_key(key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


def get_tenant(x_memopt_api_key: Optional[str] = Header(default=None)) -> str:
    """
    FastAPI dependency — validates key and returns the tenant_id.
    Supports both the new multi-tenant key format and the legacy single-key mode.
    """
    tenant_id = authenticate(x_memopt_api_key)
    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return tenant_id


def require_admin(tenant_id: str = Depends(get_tenant)) -> str:
    """FastAPI dependency — requires admin privileges."""
    if not is_admin(tenant_id):
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return tenant_id


@app.post("/optimize")
async def optimize(request: OptimizeRequest, _: str = Security(verify_api_key)):
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
    if not _PHASE3_AVAILABLE:
        raise HTTPException(
            status_code=501,
            detail="Optimization pipeline unavailable: memopt.phase3 is not installed on this server.",
        )
    safe_path = _validate_model_path(request.model_path)

    job_id = str(uuid.uuid4())
    job = OptimizationJob(
        job_id      = job_id,
        status      = JobStatus.QUEUED,
        model_path  = str(safe_path),
        input_shape = request.input_shape,
        created_at  = time.time(),
    )
    job_store[job_id] = job
    executor.submit(run_optimization, job_id)

    log.info(
        "Job %s QUEUED — model=%s shape=%s",
        job_id, safe_path, request.input_shape,
    )
    return {"job_id": job_id, "status": "queued", "estimated_minutes": 5}


@app.get("/status/{job_id}")
def status(job_id: str, _: str = Security(verify_api_key)):
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
async def agent_optimize(request: AgentRequest, _: str = Security(verify_api_key)):
    """
    Submit an autonomous multi-round optimization job.

    Same async job pattern as POST /optimize — returns immediately with a
    job_id.  Poll GET /status/{job_id} until status is complete | failed |
    rolled_back.

    Additional fields in GET /status response for agent jobs:
      rounds_summary: JSON array of per-round results (bottleneck, committed,
                      rolled, cumulative speedup, stop reason if terminal).
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=410,
        content={
            "error": "removed",
            "message": "The /agent endpoint has been removed. "
                       "Use /optimize or /profile instead.",
        },
    )


# /metrics is served by the prometheus_client ASGI app (mounted below).
# It returns text/plain Prometheus exposition format, not JSON.
# Scrape with: curl http://localhost:8080/metrics


# ── Compliance ledger endpoints — pillar 4 ─────────────────────────────────
# Pillar 4 (compliance ledger) is wired through memopt.observability.

def _get_ledger():
    """Return a process-singleton OptimizationLedger for the API server."""
    from memopt.observability.ledger import OptimizationLedger
    global _LEDGER_SINGLETON
    try:
        return _LEDGER_SINGLETON
    except NameError:
        pass
    _LEDGER_SINGLETON = OptimizationLedger()
    return _LEDGER_SINGLETON


@app.get("/ledger")
async def ledger_entries(n: int = 100, tenant_id: str = Depends(get_tenant)):
    ledger = _get_ledger()
    entries = ledger.entries(
        tenant_id=tenant_id if tenant_id != "_default" else None,
        limit=max(1, min(int(n), 1000)),
    )
    return {"tenant_id": tenant_id, "entries": entries}


@app.get("/ledger/verify")
async def ledger_verify(
    tenant_id: Optional[str] = None,
    tenant: str = Depends(get_tenant),
):
    ledger = _get_ledger()
    return ledger.verify(
        tenant_id=tenant_id if tenant_id else (tenant if tenant != "_default" else None)
    )


@app.get("/ledger/export/csv")
async def export_ledger_csv(
    period_hours: int = 24,
    tenant_id: str = Depends(get_tenant),
):
    from fastapi.responses import PlainTextResponse
    from memopt.observability.report_exporter import generate_report
    ledger = _get_ledger()
    report = generate_report(ledger, tenant_id=tenant_id, period_hours=period_hours)
    return PlainTextResponse(content=report.to_csv_summary(), media_type="text/csv")


@app.get("/ledger/export/report.html")
async def export_report_html(
    period_hours: int = 24,
    tenant_id: str = Depends(get_tenant),
):
    from fastapi.responses import HTMLResponse
    from memopt.observability.report_exporter import generate_report
    ledger = _get_ledger()
    report = generate_report(ledger, tenant_id=tenant_id, period_hours=period_hours)
    return HTMLResponse(content=report.to_html(), status_code=200)


@app.get("/ledger/export/report.pdf")
async def export_report_pdf(
    period_hours: int = 24,
    tenant_id: str = Depends(get_tenant),
):
    from fastapi.responses import Response
    from memopt.observability.report_exporter import generate_report
    ledger = _get_ledger()
    report = generate_report(ledger, tenant_id=tenant_id, period_hours=period_hours)
    pdf_bytes = report.to_pdf()
    if pdf_bytes is None:
        raise HTTPException(
            status_code=501,
            detail="PDF export requires reportlab; install via `pip install reportlab`.",
        )
    return Response(content=pdf_bytes, media_type="application/pdf")


@app.get("/ledger/export/carbon")
async def export_carbon(
    period_hours: int = 24 * 30,
    tenant_id: str = Depends(get_tenant),
):
    from memopt.observability.ledger import CarbonCalculator
    ledger = _get_ledger()
    totals = ledger.totals(tenant_id=tenant_id if tenant_id != "_default" else None)
    energy_kwh_saved = float(totals.get("energy_kwh_saved", 0.0))
    days = max(1.0, period_hours / 24.0)
    daily_kwh = energy_kwh_saved / days

    calc = CarbonCalculator()
    annual = calc.annual_projection(daily_kwh_saved=daily_kwh)
    return {
        "tenant_id": tenant_id,
        "period_hours": period_hours,
        "totals": totals,
        "annual_projection": annual,
        "disclaimer": annual.get(
            "disclaimer",
            "Estimate only — not a certified carbon accounting figure.",
        ),
    }


# ── Tenant management (admin only) ───────────────────────────────────────────

@app.post("/tenants/{new_tenant_id}", status_code=201)
async def create_tenant_endpoint(
    new_tenant_id: str,
    _admin: str = Depends(require_admin),
):
    """Create a new tenant and return its API key. Admin only."""
    try:
        key = create_tenant(new_tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"tenant_id": new_tenant_id, "api_key": key}


@app.delete("/tenants/{target_tenant_id}")
async def revoke_tenant_endpoint(
    target_tenant_id: str,
    _admin: str = Depends(require_admin),
):
    """Revoke a tenant's API key. Admin only."""
    try:
        removed = revoke_tenant(target_tenant_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if not removed:
        raise HTTPException(status_code=404, detail=f"Tenant '{target_tenant_id}' not found")
    return {"tenant_id": target_tenant_id, "revoked": True}


@app.get("/tenants")
async def list_tenants_endpoint(_admin: str = Depends(require_admin)):
    """List all active tenants. Admin only."""
    return {"tenants": list_tenants()}


@app.get("/hardware/specs")
async def hardware_specs(tenant_id: str = Depends(get_tenant)):
    """
    GPU hardware specifications with honest audit trail.

    measured=False means the number comes from the vendor datasheet
    and has not been validated by memopt on real hardware.
    """
    from memopt.profiler.hardware_counters import (
        GPU_SPECS, get_spec_with_audit)

    result = {}
    for name in GPU_SPECS:
        result[name] = get_spec_with_audit(name)

    return {
        "specs": result,
        "total": len(result),
        "measured_count": sum(
            1 for v in result.values() if v.get("measured")),
        "note": (
            "measured=False means the bandwidth number is from "
            "the vendor datasheet. memopt updates this after "
            "running certification on that hardware."),
    }


@app.get("/health")
async def health():
    """
    Liveness probe — always responds in <100 ms.

    Does NOT load models or run GPU code.  GPU name is queried from the
    already-cached CUDA context (negligible overhead).

    Returns HTTP 503 (with status="degraded") when any registered
    component is degraded (e.g. Redis backend unreachable).
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

    degraded = False
    if _gkd_store is not None:
        try:
            gkd_stats = _gkd_store.stats()
            if gkd_stats.get("backend_degraded"):
                degraded = True
        except Exception:
            degraded = True

    body = {
        "status": "degraded" if degraded else "ok",
        "gpu": gpu,
        "model_format": "torch.save(model, path) required — not state_dict()",
        "supported_architectures": "any nn.Module importable in server environment",
    }

    from fastapi.responses import JSONResponse
    return JSONResponse(
        content=body,
        status_code=503 if degraded else 200,
    )


# ── Prometheus /metrics — authenticated ──────────────────────────────────────

@app.get("/metrics")
async def metrics(_admin: str = Depends(require_admin)):
    """
    Prometheus text exposition — admin key required.
    Scrape with: curl -H 'X-Memopt-Api-Key: <admin_key>' http://localhost:8080/metrics
    """
    body = generate_latest()
    # Append Pillar 4 collector metrics (VMM, GKD, kernel hooks, etc.)
    if _p4_collector is not None:
        try:
            body += _p4_collector.registry.prometheus_text().encode()
        except Exception:
            pass
    return Response(body, media_type=CONTENT_TYPE_LATEST)
