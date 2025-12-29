"""
MemOpt Worker Service
Secure GPU inference worker - accepts requests only from control plane
"""
import os
import time
from typing import Optional
import torch
from fastapi import FastAPI, HTTPException, Header, status
from pydantic import BaseModel, Field
import uvicorn

from memopt import OptimizedLLM

# Configuration from environment
WORKER_TOKEN = os.getenv("WORKER_TOKEN")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-2-7b-hf")
HF_TOKEN = os.getenv("HF_TOKEN")
WORKER_PORT = int(os.getenv("WORKER_PORT", "8001"))
OPTIMIZATION_LEVEL = int(os.getenv("OPTIMIZATION_LEVEL", "7"))

# Validate required config
if not WORKER_TOKEN:
    raise RuntimeError("WORKER_TOKEN environment variable is required")
if len(WORKER_TOKEN) < 32:
    raise RuntimeError("WORKER_TOKEN must be at least 32 characters")

app = FastAPI(title="MemOpt Worker Service", version="1.0.0")

# Global model cache
_model: Optional[OptimizedLLM] = None
_model_loaded = False
_model_load_error: Optional[str] = None


def load_model():
    """Load model at startup"""
    global _model, _model_loaded, _model_load_error

    try:
        print(f"Loading model: {MODEL_NAME} with optimization level {OPTIMIZATION_LEVEL}")
        _model = OptimizedLLM(
            model=MODEL_NAME,
            optimization_level=OPTIMIZATION_LEVEL,
            enable_profiling=True,
        )
        _model_loaded = True
        print(f"Model loaded successfully: {MODEL_NAME}")
    except Exception as e:
        _model_load_error = str(e)
        print(f"ERROR: Failed to load model: {e}")
        raise


# Load model on startup
@app.on_event("startup")
async def startup_event():
    load_model()


class GenerateRequest(BaseModel):
    """Request for text generation"""
    request_id: str = Field(..., description="Request ID for tracing")
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(..., description="Model name (must match MODEL_NAME)")
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    optimization_level: int = Field(default=7, ge=1, le=10)


class GenerateResponse(BaseModel):
    """Response from text generation"""
    request_id: str
    model: str
    generated_text: str
    usage: dict  # {prompt_tokens, completion_tokens, total_tokens}
    latency_ms: float


def verify_worker_token(x_worker_token: Optional[str] = Header(None)):
    """Verify worker token from header"""
    if not x_worker_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-Worker-Token header"
        )

    if x_worker_token != WORKER_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid worker token"
        )


@app.get("/health")
async def health():
    """Health check endpoint - no auth required"""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    gpu_name = torch.cuda.get_device_name(0) if gpu_available else None

    return {
        "status": "healthy" if _model_loaded else "unhealthy",
        "model_loaded": _model_loaded,
        "model_name": MODEL_NAME if _model_loaded else None,
        "model_load_error": _model_load_error,
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "optimization_level": OPTIMIZATION_LEVEL
    }


@app.post("/generate", response_model=GenerateResponse, dependencies=[verify_worker_token])
async def generate(
    request: GenerateRequest,
    x_worker_token: str = Header(..., alias="X-Worker-Token")
):
    """
    Generate text using loaded model
    Requires X-Worker-Token header for authentication
    """
    start_time = time.time()

    # Verify model is loaded
    if not _model_loaded or _model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {_model_load_error}"
        )

    # Verify model name matches
    if request.model != MODEL_NAME:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model mismatch: requested '{request.model}', loaded '{MODEL_NAME}'"
        )

    try:
        # Generate text
        generated_text = _model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=(request.temperature > 0),
        )

        # Calculate token usage (rough estimate)
        prompt_tokens = len(request.prompt.split())
        completion_tokens = len(generated_text.split())
        total_tokens = prompt_tokens + completion_tokens

        latency_ms = (time.time() - start_time) * 1000

        print(f"[{request.request_id}] Generated {total_tokens} tokens in {latency_ms:.2f}ms")

        return GenerateResponse(
            request_id=request.request_id,
            model=MODEL_NAME,
            generated_text=generated_text,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            },
            latency_ms=latency_ms
        )

    except Exception as e:
        print(f"[{request.request_id}] Generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=WORKER_PORT,
        log_level="info"
    )
