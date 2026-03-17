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

        # Submit to engine (blocking in executor so we don't block the event loop)
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

        # Decode output
        if _tokenizer is not None:
            prompt_len = input_ids.shape[1]
            new_ids = result.output_ids[0, prompt_len:].tolist()
            text = _tokenizer.decode(new_ids, skip_special_tokens=True)
        else:
            text = " ".join(str(t) for t in result.output_ids[0].tolist())

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
