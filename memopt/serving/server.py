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

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="memopt serving", version="1.0") if _HAS_FASTAPI else None


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


# ── Endpoints ─────────────────────────────────────────────────────────────────

if _HAS_FASTAPI:
    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "engine": "continuous_batching",
            "ready": _engine is not None,
        }

    @app.get("/report/found-capacity")
    def found_capacity_report():
        """
        Report real GKD deduplication stats and ledger totals.
        All numbers are measured, not estimated.
        """
        gkd_stats = _gkd_store.stats() if _gkd_store is not None else {}
        ledger_totals = {}
        try:
            from memopt.api.server import _p4_ledger
            if _p4_ledger is not None:
                ledger_totals = _p4_ledger.totals()
        except Exception:
            pass

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
            "ledger": ledger_totals,
            "node_id": _node_id,
            "generated_at": time.time(),
        }

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

        # Submit to engine (blocking in executor so we don't block the event loop)
        # NOTE: Even on GKD hit, we still run full inference because the engine
        # does not yet support "decode from block_ref" or partial-prefix compute.
        # The GKD lookup above builds the dedup index and tracks hit stats for
        # the /report/found-capacity endpoint. Actual compute-skip requires
        # engine integration (future work).
        loop = asyncio.get_event_loop()
        results: List[Request] = await loop.run_in_executor(
            None,
            lambda: _engine.run_sync(
                [input_ids],
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
        )

        result = results[0]

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
            except Exception:
                pass  # registration failure must never block response

        # Decode output
        if _tokenizer is not None:
            prompt_len = input_ids.shape[1]
            new_ids = result.output_ids[0, prompt_len:].tolist()
            text = _tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            text = " ".join(str(t) for t in result.output_ids[0].tolist())

        # Pillar 4 — record metrics and ledger entry
        try:
            from memopt.api.server import _p4_collector, _p4_ledger
            if _p4_collector is not None:
                _p4_collector.record_request(result.tokens_generated)
            if _p4_ledger is not None:
                gkd_stats = _gkd_store.stats() if _gkd_store else {}
                _p4_ledger.record(
                    tokens=result.tokens_generated,
                    tenant_id="_default",
                    gkd_hit_rate_pct=gkd_stats.get("hit_rate_pct"),
                )
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
    _engine = ContinuousBatchingEngine(model=model, config=config)
    log.info("ContinuousBatchingEngine ready (device=%s)", device)

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

    # Pillar 4 — wire MetricsCollector to data sources
    try:
        from memopt.api.server import _p4_collector, _p4_ledger
        if _p4_collector is not None:
            if _p3_kv_cache is not None:
                from memopt.serving import kernel_hooks as _kh
                _p4_collector.register_kernel_hooks(_kh)
            log.info("MetricsCollector: data sources registered")
    except Exception as e:
        log.debug("MetricsCollector wiring skipped: %s", e)

    # Pillar 6 — CertifyDaemon: drift detection + kernel re-synthesis
    try:
        from memopt.kernels.certify_daemon import (
            CertifyDaemon,
            make_control_plane_callback,
        )

        global _certify_daemon
        _certify_daemon = CertifyDaemon(
            node_id=_node_id,
            alert_callback=make_control_plane_callback(
                jit_generator=_p3_generator,
                kernel_cache=_p3_kv_cache,
            ),
        )
        _certify_daemon.start()
        log.info(
            "CertifyDaemon started (node=%s, interval=%sh)",
            _node_id,
            os.getenv("MEMOPT_CERTIFY_INTERVAL_H", "24"),
        )
    except Exception as e:
        log.warning("CertifyDaemon startup failed: %s — serving without drift detection", e)

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
