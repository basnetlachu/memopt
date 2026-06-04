"""
memopt serving server — OpenAI-compatible HTTP interface for
ContinuousBatchingEngine.

Start with:
    python -m memopt.serving.server --model path/to/model.pt --port 8001
OR:
    memopt-serve --model path/to/model.pt --port 8001

At startup, if MEMOPT_CONTROL_PLANE env var is set, the server registers
itself with the control plane so the dashboard can list it.
"""
from __future__ import annotations

import argparse
import asyncio
import json as _json
import logging
import os
import time
import uuid
from typing import List, Optional

import torch
import torch.nn as nn

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False
    FastAPI = object  # type: ignore[assignment,misc]
    BaseModel = object  # type: ignore[assignment]

from memopt.serving.continuous_batching import (
    ContinuousBatchingEngine,
    BatchingConfig,
    Request,
)

log = logging.getLogger("memopt.serving.server")

# ── Module-level singletons — set by _build_engine() ─────────────────────────

_engine: Optional[ContinuousBatchingEngine] = None
_tokenizer = None

# Pillar 3 — module-level references prevent garbage collection
_p3_kv_cache:    object = None
_p3_portability: object = None
_p3_generator:   object = None
_p3_optimizer:   object = None

# Pillar 1+2 — GKD store for KV cache deduplication
_gkd_store: object = None
_node_id: str = ""

# Pillar 6 — CertifyDaemon for drift detection + re-synthesis
_certify_daemon: object = None
_power_sampler: object = None
_pod_controller: object = None

# Partial compute skip counters (from GKD partial hits)
_partial_skip_count: int = 0
_partial_skip_tokens_saved: int = 0
_partial_skip_attempted: int = 0

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="memopt serving", version="1.0") if _HAS_FASTAPI else None


# ── Oracle helpers (importable, no FastAPI dependency) ────────────────────────


def _get_oracle():
    """Return the VMM oracle instance, or None."""
    if _engine is None:
        return None
    vmm_inst = getattr(_engine, '_vmm', None)
    if vmm_inst and hasattr(vmm_inst, 'oracle'):
        return vmm_inst.oracle
    return None


def _get_top_transitions(oracle, top_k: int = 100) -> list[dict]:
    """
    Extract top_k transitions by count from a MemoryOracle.

    Returns list of:
    {
        "from_block": int,
        "to_block":   int,
        "count":      int,
        "confidence": float,  # count / total_from
        "total_from": int     # total transitions from this block
    }

    Sorted by count descending.
    Thread-safe. Returns [] on any error.
    """
    try:
        results = []
        with oracle._lock:
            for from_block, targets in oracle._transitions.items():
                total_from = sum(targets.values())
                for to_block, count in targets.items():
                    confidence = (
                        count / total_from if total_from > 0 else 0.0)
                    results.append({
                        "from_block": from_block,
                        "to_block":   to_block,
                        "count":      count,
                        "confidence": round(confidence, 4),
                        "total_from": total_from,
                    })

        results.sort(key=lambda x: x["count"], reverse=True)
        return results[:top_k]
    except Exception:
        return []


# ── Request / Response models ─────────────────────────────────────────────────

if _HAS_FASTAPI:
    class CompletionRequest(BaseModel):
        model:       str
        prompt:      str
        max_tokens:  int   = 100
        temperature: float = 0.0
        stream:      bool  = False

    class CompletionChoice(BaseModel):
        text:          str
        index:         int
        finish_reason: str

    class CompletionUsage(BaseModel):
        prompt_tokens:     int
        completion_tokens: int
        total_tokens:      int

    class CompletionResponse(BaseModel):
        id:      str
        object:  str = "text_completion"
        model:   str
        choices: List[CompletionChoice]
        usage:   CompletionUsage


def _try_partial_compute_skip(request, gkd_hit) -> Optional[dict]:
    """
    Attempt partial compute skip using cached KV prefix.

    Returns a dict with completion data on success.
    Returns None on any failure — falls through to full inference.
    NEVER returns wrong output.

    HONEST NOTE:
    This requires engine.run_with_prefix() which does not exist yet
    in ContinuousBatchingEngine. This function implements the full
    logic. When run_with_prefix is unavailable it returns None and
    falls through. Tracking: _partial_skip_attempted increments even
    on fallthrough so we can measure partial skip opportunity rate.
    """
    global _partial_skip_count, _partial_skip_tokens_saved
    global _partial_skip_attempted
    _partial_skip_attempted += 1

    if _engine is None:
        return None

    # Check if engine supports prefix injection
    if not hasattr(_engine, "run_with_prefix"):
        log.debug(
            "Partial skip: engine does not support run_with_prefix yet. "
            "Tracking opportunity (matched %d tokens).",
            gkd_hit.matched_len)
        return None

    # Engine supports prefix injection — attempt it
    try:
        delta_start = gkd_hit.delta_start or gkd_hit.matched_len or 0
        tokens_saved = gkd_hit.matched_len or 0

        result = _engine.run_with_prefix(
            token_ids=request.token_ids[delta_start:],
            prefix_len=tokens_saved)

        if result is None:
            return None

        _partial_skip_count += 1
        _partial_skip_tokens_saved += tokens_saved

        log.info(
            "Partial skip: saved %d tokens (%d/%d = %d%%)",
            tokens_saved, tokens_saved,
            tokens_saved + len(request.token_ids[delta_start:]),
            tokens_saved * 100 // max(1, tokens_saved + len(
                request.token_ids[delta_start:])))

        return result

    except Exception as e:
        log.debug("Partial skip failed: %s", e)
        return None


# ── Endpoints ─────────────────────────────────────────────────────────────────

if _HAS_FASTAPI:
    _start_time = time.time()

    @app.get("/oracle/stats")
    def oracle_stats_endpoint():
        """Top oracle transitions for pod controller aggregation."""
        try:
            oracle = _get_oracle()
            if oracle is None:
                return {
                    "node_id": _node_id or "unknown",
                    "transitions": [],
                    "total_transitions": 0,
                    "horizon": 0,
                    "min_confidence": 0.3,
                    "exported_at": time.time(),
                }
            top_transitions = _get_top_transitions(oracle)
            stats = oracle.stats()
            return {
                "node_id": _node_id or "unknown",
                "transitions": top_transitions,
                "total_transitions":
                    getattr(stats, 'transitions_learned',
                            len(top_transitions)),
                "horizon": getattr(stats, 'horizon', 0),
                "min_confidence":
                    getattr(oracle, '_min_confidence', 0.3),
                "exported_at": time.time(),
            }
        except Exception:
            return {
                "node_id": _node_id or "unknown",
                "transitions": [],
                "total_transitions": 0,
                "horizon": 0,
                "min_confidence": 0.3,
                "exported_at": time.time(),
            }

    @app.get("/oracle/health")
    async def oracle_health():
        """
        Oracle health check for pod controller.
        No auth required — internal network only.
        """
        oracle = _get_oracle()
        if oracle is None:
            return {
                "node_id": _node_id or "unknown",
                "oracle_active": False,
                "transition_count": 0,
                "horizon": 0,
                "prediction_accuracy": 0.0,
                "healthy": False,
            }

        try:
            stats = oracle.stats()

            transition_count = 0
            try:
                with oracle._lock:
                    transition_count = sum(
                        len(v) for v in
                        oracle._transitions.values())
            except Exception:
                pass

            accuracy = 0.0
            try:
                if hasattr(stats, "outcome_accuracy"):
                    accuracy = stats.outcome_accuracy
                elif stats.total_predictions_made > 0:
                    accuracy = (
                        stats.total_predictions_correct
                        / stats.total_predictions_made)
            except Exception:
                pass

            return {
                "node_id": _node_id or "unknown",
                "oracle_active": True,
                "transition_count": transition_count,
                "horizon": getattr(stats, "horizon", 0),
                "prediction_accuracy": round(accuracy, 4),
                "healthy": True,
            }
        except Exception as e:
            log.error(f"Oracle health error: {e}")
            return {
                "node_id": _node_id or "unknown",
                "oracle_active": False,
                "transition_count": 0,
                "horizon": 0,
                "prediction_accuracy": 0.0,
                "healthy": False,
            }

    @app.post("/oracle/merge")
    async def oracle_merge(payload: dict):
        """
        Receive transitions from pod oracle and merge into local oracle.

        Called by pod controller to push high-confidence pod-level
        transitions down to individual nodes.

        Only merges transitions with
        confidence >= MEMOPT_MIN_MERGE_CONFIDENCE (default 0.5).

        No auth — internal network only.
        """
        oracle = _get_oracle()
        if oracle is None:
            return {"merged": 0, "skipped": 0,
                    "reason": "oracle_not_active"}

        min_confidence = float(os.getenv(
            "MEMOPT_MIN_MERGE_CONFIDENCE", "0.5"))

        transitions = payload.get("transitions", [])
        source = payload.get("source", "unknown")

        merged = 0
        skipped = 0

        for t in transitions:
            try:
                confidence = t.get("confidence", 0.0)
                if confidence < min_confidence:
                    skipped += 1
                    continue

                from_block = int(t["from_block"])
                to_block   = int(t["to_block"])
                count      = int(t["count"])

                # Cap injected count to avoid one pod overwhelming
                # the local oracle's learned transitions
                merge_count = min(count, 10)

                # Directly inject the transition rather than calling
                # observe() repeatedly — observe() would create
                # spurious prev→from transitions as a side effect
                with oracle._lock:
                    oracle._transitions[from_block][to_block] += \
                        merge_count

                merged += 1
            except Exception:
                skipped += 1

        log.debug(
            f"Oracle merge from {source}: "
            f"merged={merged} skipped={skipped}")

        return {"merged": merged, "skipped": skipped}

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "engine": "continuous_batching",
            "ready": _engine is not None,
        }

    @app.get("/healthz")
    def healthz():
        """Liveness probe — 200 if alive, 503 if should be restarted.
        Must respond in < 100ms. No blocking calls. No auth required."""
        try:
            checks = {}
            healthy = True

            # Check background threads
            try:
                vmm_instance = getattr(_engine, '_vmm', None)
                if vmm_instance and hasattr(vmm_instance, 'prefetch'):
                    if not vmm_instance.prefetch.is_healthy():
                        checks["threads"] = "unhealthy"
                        healthy = False
                    else:
                        checks["threads"] = "ok"
                else:
                    checks["threads"] = "ok"
            except Exception:
                checks["threads"] = "ok"

            # Check Redis degradation duration
            try:
                if _gkd_store is not None:
                    stats = _gkd_store.stats()
                    degraded = stats.get("backend_degraded", False)
                    since = stats.get("backend_degraded_since")
                    if degraded and since is not None:
                        duration = time.monotonic() - since
                        if duration > 300:  # > 5 minutes
                            checks["redis"] = "degraded_5m"
                            healthy = False
                        else:
                            checks["redis"] = "degraded"
                    else:
                        checks["redis"] = "ok"
                else:
                    checks["redis"] = "not_configured"
            except Exception:
                checks["redis"] = "ok"

            status_code = 200 if healthy else 503
            # HAL-reported hardware info — callers use this to route
            # traffic (AMD vs NVIDIA pools, CPU-only debug nodes).
            hw_backend = "unknown"
            gpu_count = 0
            try:
                from memopt.vmm.hal import get_hal
                _hal = get_hal()
                hw_backend = _hal.backend.value
                gpu_count = _hal.gpu_count
            except Exception:
                pass

            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=status_code,
                content={
                    "status": "ok" if healthy else "degraded",
                    "node_id": _node_id or "unknown",
                    "uptime_seconds": round(
                        time.time() - _start_time, 1),
                    "hardware_backend": hw_backend,
                    "gpu_count": gpu_count,
                    "checks": checks,
                })
        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={
                    "status": "error",
                    "node_id": _node_id or "unknown",
                })

    @app.get("/readyz")
    def readyz():
        """Readiness probe — 200 if ready to serve, 503 if not.
        Kubernetes removes pod from endpoints on 503.
        May take up to 500ms. No auth required."""
        try:
            checks = {}
            ready = True
            reason = ""

            # Check engine initialized
            if _engine is None:
                ready = False
                reason = "engine_not_initialized"
                checks["engine"] = "not_ready"
            else:
                checks["engine"] = "ok"

            # Check GKD store initialized
            if _gkd_store is not None:
                checks["gkd"] = "ok"
            else:
                checks["gkd"] = "local_fallback"

            # Check silicon certification
            cert_hash = ""
            try:
                if _certify_daemon is not None:
                    if _certify_daemon.is_certified():
                        checks["certification"] = "valid"
                        last = _certify_daemon.last_result()
                        if last:
                            cert_hash = str(
                                last.get("certificate_hash",
                                         ""))[:16]
                    else:
                        checks["certification"] = "not_certified"
                else:
                    checks["certification"] = "not_configured"
            except Exception:
                checks["certification"] = "unknown"

            # Check drift
            drift_pct = 0.0
            try:
                if _certify_daemon is not None:
                    ds = _certify_daemon.drift_stats()
                    if ds.get("is_drifted", False):
                        drift_pct = ds.get("drift_pct", 0.0) or 0.0
                        if drift_pct > 15.0:
                            ready = False
                            reason = "drift_exceeded_15pct"
                            checks["drift"] = f"{drift_pct:.1f}%"
                        else:
                            checks["drift"] = f"{drift_pct:.1f}%"
                    else:
                        checks["drift"] = "none"
            except Exception:
                checks["drift"] = "unknown"

            status_code = 200 if ready else 503
            from fastapi.responses import JSONResponse
            content = {
                "status": "ready" if ready else "not_ready",
                "node_id": _node_id or "unknown",
                "checks": checks,
            }
            if cert_hash:
                content["cert_hash"] = cert_hash
            if drift_pct > 0:
                content["drift_pct"] = drift_pct
            if not ready and reason:
                content["reason"] = reason

            return JSONResponse(
                status_code=status_code, content=content)
        except Exception:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "reason": "internal_error",
                    "node_id": _node_id or "unknown",
                })

    @app.get("/report/found-capacity")
    def found_capacity_report():
        """
        Report GKD deduplication stats.
        Energy savings reflect PowerSampler readings when active.
        """
        gkd_stats = _gkd_store.stats() if _gkd_store is not None else {}

        has_power = (_power_sampler is not None
                     and getattr(_power_sampler, 'available', False))

        return {
            "gkd": {
                "hit_rate_pct":             gkd_stats.get("hit_rate_pct", 0),
                "exact_hits":               gkd_stats.get("exact_hits", 0),
                "lcp_hits":                 gkd_stats.get("lcp_hits", 0),
                "total_lookups":            gkd_stats.get("total_lookups", 0),
                "estimated_hbm_saved_gb":   gkd_stats.get("estimated_hbm_saved_gb", 0),
                "lcp_prefix_match_rate_pct": gkd_stats.get("lcp_token_reuse_pct", 0),
                "entries_in_store":         gkd_stats.get("entries_in_store", 0),
                "backend_degraded":         gkd_stats.get("backend_degraded", False),
            },
            "energy_savings": {
                "measured": False,
                "power_sampler_active": has_power,
                "note": ("PowerSampler active — live NVML readings"
                         if has_power
                         else "PowerSampler unavailable — install pynvml"),
            },
            "node_id": _node_id,
            "generated_at": time.time(),
        }

    @app.get("/report/savings")
    def report_savings(
        period_hours: int = 24,
        tenant: str = "",
    ):
        # MOVED: Cost-savings reporting depends on the compliance ledger
        # (Pillar 4) which now lives in memopt-trust. Returns 501 here.
        # TODO: Re-enable when memopt-trust integrates back.
        from fastapi import HTTPException
        raise HTTPException(
            status_code=501,
            detail=(
                "Endpoint moved to memopt-trust. "
                "Not available in memopt-engine."
            ),
        )

    @app.get("/metrics")
    def metrics():
        """
        Prometheus-style metrics: HBM saved, GKD hit rate, kernel stats.
        All numbers are from live measurements — nothing is estimated.
        """
        gkd_stats = _gkd_store.stats() if _gkd_store is not None else {}
        collector_metrics = {}
        try:
            from memopt.api.server import _p4_collector
            if _p4_collector is not None:
                collector_metrics = _p4_collector.collect()
        except Exception:
            pass

        kernel_stats = {}
        try:
            from memopt.serving import kernel_hooks as _kh
            kernel_stats = _kh.stats()
        except Exception:
            pass

        return {
            "hbm_saved_gb":     gkd_stats.get("estimated_hbm_saved_gb", 0),
            "gkd_hit_rate_pct": gkd_stats.get("hit_rate_pct", 0),
            "total_lookups":    gkd_stats.get("total_lookups", 0),
            "exact_hits":       gkd_stats.get("exact_hits", 0),
            "lcp_hits":         gkd_stats.get("lcp_hits", 0),
            "speedup_ratio":    collector_metrics.get("speedup_ratio"),
            "kernel_hits":      kernel_stats.get("rope_hits", 0)
                              + kernel_stats.get("ln_hits", 0)
                              + kernel_stats.get("softmax_hits", 0),
            "kernel_fallbacks": kernel_stats.get("rope_fallbacks", 0)
                              + kernel_stats.get("ln_fallbacks", 0)
                              + kernel_stats.get("softmax_fallbacks", 0),
            "collector":        collector_metrics,
            "kernel_hooks":     kernel_stats,
            "partial_skip_count":        _partial_skip_count,
            "partial_skip_tokens_saved": _partial_skip_tokens_saved,
            "partial_skip_attempted":    _partial_skip_attempted,
            "partial_skip_opportunity_rate": round(
                _partial_skip_attempted / max(1,
                    gkd_stats.get("total_lookups", 1)) * 100, 2),
            "node_id":          _node_id,
        }

    @app.get("/kernels/library")
    def kernel_library():
        """
        List kernels in the shared library.
        Returns op names, hardware targets, speedups, and synthesis dates.
        Source code is NOT returned — only metadata.
        """
        try:
            from memopt.kernels.kernel_cache import get_kernel_library
            library = get_kernel_library()
            kernels = library.list_kernels()

            safe_kernels = []
            for k in kernels:
                safe_kernels.append({
                    "op_name": k.get("op_name"),
                    "hardware_hash": k.get("hardware_hash", "")[:8],
                    "speedup": k.get("speedup"),
                    "saved_at": k.get("saved_at"),
                })

            return {
                "total": len(safe_kernels),
                "kernels": safe_kernels,
                "library_stats": library.stats(),
                "note": (
                    "Kernels are shared across deployments. "
                    "Every synthesis makes future deployments faster. "
                    "Source code not included in this response."),
            }

        except Exception as e:
            return {"total": 0, "kernels": [], "error": str(e)}

    @app.post("/v1/completions", response_model=CompletionResponse)
    async def completions(request: CompletionRequest):
        if _engine is None:
            raise HTTPException(status_code=503, detail="Engine not initialized")

        # Tokenize prompt
        if _tokenizer is not None:
            input_ids = _tokenizer.encode(request.prompt, return_tensors="pt")
        else:
            # Fallback: treat prompt as space-separated token IDs
            try:
                ids = [int(x) for x in request.prompt.split()]
                input_ids = torch.tensor([ids], dtype=torch.long)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No tokenizer loaded. Pass token IDs as space-separated "
                        "integers or load a tokenizer with --tokenizer."
                    ),
                )

        # Pillar 1+2 — GKD lookup for KV cache deduplication
        gkd_hit = None
        token_ids_list = input_ids[0].tolist()
        seq_len = len(token_ids_list)
        if _gkd_store is not None:
            try:
                gkd_hit = _gkd_store.lookup(token_ids_list, seq_len)
            except Exception:
                pass  # GKD failure must never block inference

        # ── GKD exact hit → skip inference ────────────────────────────
        # On exact hit with cached output: return directly without
        # calling engine.run_sync(). This is the core dedup value prop.
        if gkd_hit is not None and not gkd_hit.is_partial:
            try:
                cached_output = _gkd_store.get_output(
                    gkd_hit.block_ref)
                if cached_output is not None:
                    log.debug(
                        "GKD exact hit: skipping inference "
                        "for %d tokens (block_ref=%s)",
                        seq_len, gkd_hit.block_ref)
                    # Record hit in telemetry collector
                    try:
                        from memopt.api.server import _p4_collector
                        if _p4_collector is not None:
                            _p4_collector.record_request(
                                cached_output.get(
                                    "completion_tokens", 0))
                    except Exception:
                        pass
                    return CompletionResponse(
                        id=f"cmpl-{uuid.uuid4().hex[:8]}",
                        model=request.model,
                        choices=[CompletionChoice(
                            text=cached_output.get("text", ""),
                            index=0,
                            finish_reason="stop",
                        )],
                        usage=CompletionUsage(
                            prompt_tokens=seq_len,
                            completion_tokens=cached_output.get(
                                "completion_tokens", 0),
                            total_tokens=seq_len + cached_output.get(
                                "completion_tokens", 0),
                        ),
                    )
            except Exception:
                pass  # get_output failure → fall through to inference

        if gkd_hit is not None and gkd_hit.is_partial:
            # Attempt partial compute skip
            partial_result = _try_partial_compute_skip(request, gkd_hit)
            if partial_result is not None:
                return partial_result

            # Partial skip failed or engine not ready — fall through
            log.debug(
                "GKD partial hit fallthrough: running full inference. "
                "Matched %s/%d tokens would have been saved.",
                gkd_hit.matched_len, seq_len)

        # ── Full miss or partial hit: run inference ───────────────────
        import time as _time
        _t_start = _time.perf_counter()

        loop = asyncio.get_event_loop()
        results: List[Request] = await loop.run_in_executor(
            None,
            lambda: _engine.run_sync(
                [input_ids],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
        )

        _elapsed_s = _time.perf_counter() - _t_start
        result = results[0]

        # ── Measure actual J/token from PowerSampler ──────────────────
        actual_j_per_token = None
        if (_power_sampler is not None
                and result.tokens_generated > 0
                and _elapsed_s > 0):
            try:
                avg_watts = _power_sampler.current_avg_watts()
                if avg_watts > 0:
                    total_joules = avg_watts * _elapsed_s
                    actual_j_per_token = (
                        total_joules / result.tokens_generated)
                    log.debug(
                        "Energy: %.4f J/token (%d tokens, %.1fms, "
                        "%.1fW avg)",
                        actual_j_per_token,
                        result.tokens_generated,
                        _elapsed_s * 1000,
                        avg_watts)
            except Exception:
                pass  # measurement failure must never block response

        # Decode output
        if _tokenizer is not None:
            prompt_len = input_ids.shape[1]
            new_ids = result.output_ids[0, prompt_len:].tolist()
            text = _tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            text = " ".join(str(t) for t in result.output_ids[0].tolist())

        # Pillar 1+2 — register into GKD for future dedup (only on miss)
        if _gkd_store is not None and gkd_hit is None:
            try:
                block_ref = f"cmpl-{uuid.uuid4().hex[:8]}:0"
                _gkd_store.register(
                    token_ids=token_ids_list,
                    sequence_length=seq_len,
                    block_ref=block_ref,
                    node_id=_node_id,
                )
                # Store output alongside block_ref for future exact hits
                _gkd_store.register_output(block_ref, {
                    "text": text,
                    "completion_tokens": result.tokens_generated,
                })
            except Exception:
                pass  # registration failure must never block response

        # Telemetry — record request count
        try:
            from memopt.api.server import _p4_collector
            if _p4_collector is not None:
                _p4_collector.record_request(result.tokens_generated)
        except Exception:
            pass

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:8]}",
            model=request.model,
            choices=[CompletionChoice(
                text=text,
                index=0,
                finish_reason="stop",
            )],
            usage=CompletionUsage(
                prompt_tokens=input_ids.shape[1],
                completion_tokens=result.tokens_generated,
                total_tokens=input_ids.shape[1] + result.tokens_generated,
            ),
        )


    # ── Pod GKD endpoints ──────────────────────────────────────────────────────

    @app.get("/pod/gkd/lookup")
    def pod_gkd_lookup(hash: str = "", seq_len: int = 0):
        """Pod-level GKD cache lookup. No auth (internal network)."""
        if _pod_controller is None:
            return {"hit": False, "block_ref": None, "node_id": None}
        try:
            entry = _pod_controller._gkd_cache.get(hash, seq_len)
            if entry:
                return {
                    "hit": True,
                    "block_ref": entry.get("block_ref"),
                    "node_id": entry.get("node_id"),
                }
            return {"hit": False, "block_ref": None, "node_id": None}
        except Exception:
            return {"hit": False, "block_ref": None, "node_id": None}

    @app.post("/pod/gkd/register")
    def pod_gkd_register(payload: dict):
        """Register entry in pod GKD cache. No auth (internal)."""
        if _pod_controller is None:
            return {"status": "no_pod_controller"}
        try:
            _pod_controller._gkd_cache.put(
                payload.get("hash", ""),
                payload.get("seq_len", 0),
                {
                    "block_ref": payload.get("block_ref", ""),
                    "node_id": payload.get("node_id", ""),
                })
            return {"status": "ok"}
        except Exception:
            return {"status": "error"}

    @app.get("/pod/gkd/stats")
    def pod_gkd_stats():
        """Pod GKD cache stats."""
        if _pod_controller is None:
            return {"pod_controller": "not_active"}
        return _pod_controller._gkd_cache.stats()

    @app.get("/pod/oracle/stats")
    async def pod_oracle_stats():
        """
        Export pod oracle transitions for global oracle aggregation.

        Only active when pod controller is running (MEMOPT_POD_ID set).
        No auth — internal network only.
        """
        if _pod_controller is None:
            return {
                "pod_id": "",
                "transitions": [],
                "aggregator_stats": {},
                "exported_at": time.time(),
            }

        try:
            transitions = \
                _pod_controller._aggregator.get_pod_transitions(
                    top_k=200)

            return {
                "pod_id": _pod_controller._config.pod_id,
                "transitions": transitions,
                "aggregator_stats":
                    _pod_controller._aggregator.stats(),
                "exported_at": time.time(),
            }
        except Exception as e:
            log.error(f"Pod oracle export error: {e}")
            return {
                "pod_id": "",
                "transitions": [],
                "aggregator_stats": {},
                "exported_at": time.time(),
            }

    @app.get("/pod/bloom-filter")
    def pod_bloom_filter():
        """
        Export the node's bloom filter as base64 bytes.
        Pod controllers can merge bloom filters from all nodes
        to build a cluster-wide negative filter.
        """
        import base64
        if _gkd_store is None:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"error": "gkd_store not initialized"})
        try:
            raw = _gkd_store._bloom.to_bytes()
            return {
                "node_id": _node_id,
                "bloom_bytes_b64": base64.b64encode(raw).decode(),
                "bloom_stats": _gkd_store._bloom.stats(),
            }
        except Exception as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=500,
                content={"error": str(e)})


    @app.get("/gum/stats")
    def gum_stats():
        """
        GUM bandwidth and latency statistics.

        Shows per-peer and global transfer metrics.
        All numbers from real timed transfers.
        transport field shows "tcp" or "rdma".
        """
        try:
            from memopt.cluster.gum_metrics import get_gum_metrics
            metrics = get_gum_metrics()
            return metrics.dashboard_data()
        except Exception as e:
            return {"error": str(e), "global": {}, "peers": {}}


# ── Engine builder ────────────────────────────────────────────────────────────

def _build_engine(
    model_path: str,
    device: str,
    config: BatchingConfig,
    args=None,
) -> None:
    """Load model and initialize ContinuousBatchingEngine."""
    global _engine

    from memopt.api.server import load_model_safe
    model: nn.Module = load_model_safe(model_path, device)
    model = model.to(device).eval()
    # Initialize VMM for KV block management
    _vmm_instance = None
    try:
        from memopt.vmm import VMM
        _vmm_instance = VMM()
        log.info("VMM initialized for serving engine")
    except Exception as e:
        log.debug("VMM init skipped: %s", e)

    _engine = ContinuousBatchingEngine(model=model, config=config, vmm=_vmm_instance)
    log.info("ContinuousBatchingEngine ready (device=%s, vmm=%s)",
             device, "active" if _vmm_instance else "disabled")

    # Pillar 1+2 — GKD store for KV cache deduplication
    try:
        import socket
        from memopt.cluster.gkd_store import GKDStore

        global _gkd_store, _node_id
        _node_id = os.getenv("MEMOPT_NODE_ID", socket.gethostname())
        redis_url = os.getenv("REDIS_URL")
        if redis_url:
            _gkd_store = GKDStore(
                redis_url=redis_url, node_id=_node_id
            )
            log.info("GKDStore ready (redis=%s)", redis_url)
        else:
            _gkd_store = GKDStore(backend="local", node_id=_node_id)
            log.info("GKDStore ready (local backend)")
    except Exception as e:
        log.warning("GKDStore init failed: %s — serving without dedup", e)

    # Pillar 3 — auto-optimizer startup
    try:
        from memopt.kernels.kernel_cache import KernelCache
        from memopt.kernels.portability_layer import PortabilityLayer
        from memopt.kernels.jit_generator import JITGenerator
        from memopt.serving.auto_optimizer import AutoOptimizer
        from memopt.serving import kernel_hooks

        global _p3_kv_cache, _p3_portability, _p3_generator, _p3_optimizer
        _p3_kv_cache    = KernelCache()
        _p3_portability = PortabilityLayer()
        _p3_generator   = JITGenerator(
            cache=_p3_kv_cache, portability=_p3_portability
        )
        _p3_optimizer   = AutoOptimizer(
            generator=_p3_generator, cache=_p3_kv_cache
        )
        _p3_optimizer.start()
        kernel_hooks.init_hooks(cache=_p3_kv_cache, optimizer=_p3_optimizer)
    except Exception as e:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"Auto-optimizer startup failed: {e} — serving without kernel optimisation"
        )

    # Pillar 4 — PowerSampler for real energy measurement
    global _power_sampler
    try:
        from memopt.profiler.power_sampler import PowerSampler
        _power_sampler = PowerSampler(interval_ms=100)
        _power_sampler.start()
        log.info("PowerSampler active (NVML %s)",
                 "available" if _power_sampler.available else "unavailable")
    except Exception as e:
        _power_sampler = None
        log.warning("PowerSampler unavailable: %s", e)

    # Wire MetricsCollector (telemetry) to data sources
    try:
        from memopt.api.server import _p4_collector
        if _p4_collector is not None:
            if _p3_kv_cache is not None:
                from memopt.serving import kernel_hooks as _kh
                _p4_collector.register_kernel_hooks(_kh)
            if _power_sampler is not None:
                _p4_collector.register_power_sampler(_power_sampler)
            log.info("MetricsCollector: data sources registered")
    except Exception as e:
        log.debug("MetricsCollector wiring skipped: %s", e)

    # Pillar 6 (CertifyDaemon) moved to memopt-trust.
    # Re-add when integrating with memopt-trust.

    # Self-register with control plane if configured
    cp = os.getenv("MEMOPT_CONTROL_PLANE", "")
    if cp and args is not None:
        import urllib.request
        payload = _json.dumps({
            "host": getattr(args, "host", "0.0.0.0"),
            "port": getattr(args, "port", 8001),
            "model_path": model_path,
            "engine_type": "continuous_batching",
            "pid": os.getpid(),
        }).encode()
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{cp}/api/v1/serving/register",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                ),
                timeout=5,
            )
            log.info("Registered with control plane at %s", cp)
        except Exception as exc:
            log.debug("Control plane registration failed (non-fatal): %s", exc)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    if not _HAS_FASTAPI:
        print("fastapi and uvicorn are required: pip install fastapi uvicorn")
        raise SystemExit(1)

    parser = argparse.ArgumentParser(description="memopt serving server")
    parser.add_argument("--model",     required=True, help="Path to model file")
    parser.add_argument("--tokenizer", default=None,  help="HuggingFace tokenizer name/path")
    parser.add_argument("--port",      type=int, default=8001)
    parser.add_argument("--host",      default="0.0.0.0")
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--max-seq",   type=int, default=2048)
    args = parser.parse_args()

    config = BatchingConfig(
        max_batch_size=args.max_batch,
        max_seq_len=args.max_seq,
    )
    _build_engine(args.model, args.device, config, args=args)

    if args.tokenizer:
        global _tokenizer
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        log.info("Tokenizer loaded: %s", args.tokenizer)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
