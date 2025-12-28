"""
Memopt FastAPI Server
Production-ready API for optimized LLM inference
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import uvicorn

from Memopt import OptimizedLLM

# Initialize FastAPI
app = FastAPI(
    title="Memopt API",
    description="Memory-optimized LLM inference API - 6x faster, 82% cheaper",
    version="1.0.0"
)

# Global model cache
_model_cache = {}


class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str = Field(..., description="Input text prompt")
    max_tokens: int = Field(default=256, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    model: str = Field(default="gpt2-xl", description="Model name")
    optimization_level: str = Field(
        default="high",
        pattern="^(conservative|balanced|high|aggressive)$",
        description="Optimization level"
    )


class GenerateResponse(BaseModel):
    """Response model for text generation"""
    generated_text: str
    tokens_generated: int
    throughput_tok_s: Optional[float] = None
    latency_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    models_loaded: List[str]


def get_model(model_name: str, optimization_level: str) -> OptimizedLLM:
    """Get or create model instance (cached)"""
    cache_key = f"{model_name}_{optimization_level}"
    
    if cache_key not in _model_cache:
        print(f"Loading model: {model_name} with {optimization_level} optimization...")
        _model_cache[cache_key] = OptimizedLLM(
            model=model_name,
            optimization_level=optimization_level,
            enable_profiling=True
        )
    
    return _model_cache[cache_key]


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "name": "Memopt API",
        "version": "1.0.0",
        "description": "Memory-optimized LLM inference - 6x faster, 82% cheaper",
        "endpoints": {
            "health": "/health",
            "generate": "/generate (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    cuda_available = torch.cuda.is_available()
    gpu_name = None
    
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except:
            pass
    
    return HealthResponse(
        status="healthy",
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        models_loaded=list(_model_cache.keys())
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text using Memopt optimized inference
    
    Example:
    ```
    curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Explain quantum computing",
            "max_tokens": 256,
            "model": "gpt2-xl",
            "optimization_level": "high"
        }'
    ```
    """
    try:
        # Get model
        model = get_model(request.model, request.optimization_level)
        
        # Generate
        output = model.generate(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=(request.temperature > 0)
        )
        
        # Get stats
        stats = model.get_profiling_stats()
        
        # Count tokens (approximate)
        tokens_generated = len(output.split())  # Simple approximation
        
        return GenerateResponse(
            generated_text=output,
            tokens_generated=tokens_generated,
            throughput_tok_s=stats.tokens_per_second if stats else None,
            latency_ms=stats.latency_per_token_ms if stats else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_model=dict)
async def list_models():
    """List loaded models"""
    return {
        "loaded_models": list(_model_cache.keys()),
        "count": len(_model_cache)
    }


@app.post("/unload/{model_key}")
async def unload_model(model_key: str):
    """Unload a model from cache"""
    if model_key in _model_cache:
        del _model_cache[model_key]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"status": "unloaded", "model": model_key}
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.get("/stats", response_model=dict)
async def get_stats():
    """Get GPU memory statistics"""
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    
    return {
        "cuda_available": True,
        "gpu_name": torch.cuda.get_device_name(0),
        "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
    }


if __name__ == "__main__":
    print("="*70)
    print("Memopt API Server")
    print("="*70)
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )