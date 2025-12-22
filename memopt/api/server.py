"""
FastAPI server for MemOpt

PURPOSE: Production REST API for LLM inference
WHY: Standard interface for datacenter integration

Usage:
    python -m memopt.api.server --model gpt2-xl --port 8000
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import time
import json
from typing import Optional

from memopt.utils.license_client import validate_license, LicenseError

logger = get_logger(__name__)

# ADD THIS: Validate license on module load (before anything else)
try:
    logger.info("Validating MemOpt license...")
    validate_license()
    logger.info("✓ License validation successful")
except LicenseError as e:
    logger.error(f"❌ License validation failed: {e}")
    logger.error("MemOpt cannot start without valid license")
    sys.exit(1)
except Exception as e:
    logger.error(f"❌ Unexpected error during license validation: {e}")
    sys.exit(1)

from memopt.core.model import OptimizedLLM
from memopt.monitoring.logger import get_logger
from memopt.utils.errors import MemOptError
from .models import (
    GenerateRequest,
    GenerateResponse,
    BatchGenerateRequest,
    BatchGenerateResponse,
    ModelInfo,
    HealthResponse
)

logger = get_logger(__name__)


class MemOptAPI:
    """
    MemOpt API Server
    
    Provides REST API for optimized LLM inference
    """
    
    def __init__(
        self,
        model_name: str = "gpt2-xl",
        optimization_level: str = "high",
        enable_cors: bool = True,
        max_batch_size: int = 32
    ):
        self.model_name = model_name
        self.optimization_level = optimization_level
        self.max_batch_size = max_batch_size
        
        # Create FastAPI app
        self.app = FastAPI(
            title="MemOpt API",
            description="High-performance LLM inference API",
            version="1.1.0"
        )
        
        # Enable CORS
        if enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = OptimizedLLM(
            model=model_name,
            optimization_level=optimization_level,
            enable_profiling=True
        )
        logger.info("✓ Model loaded successfully")
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self):
        """Register API routes"""
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "service": "MemOpt API",
                "version": "1.1.0",
                "model": self.model_name,
                "optimization": self.optimization_level
            }
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint"""
            stats = self.model.get_memory_stats()
            
            return HealthResponse(
                status="healthy",
                model_loaded=True,
                gpu_available=stats['total_gb'] > 0,
                memory_free_gb=stats['free_gb']
            )
        
        @self.app.get("/model/info", response_model=ModelInfo)
        async def model_info():
            """Get model information"""
            stats = self.model.get_memory_stats()
            
            return ModelInfo(
                model_name=self.model.model_name,
                optimization_level=self.model.optimization_level,
                num_layers=self.model.num_layers,
                num_heads=self.model.num_heads,
                hidden_size=self.model.hidden_size,
                device=self.model.device,
                memory_allocated_gb=stats['allocated_gb'],
                kv_cache_enabled=self.model.use_paging,
                quantization_enabled=self.model.use_quantization,
                flash_attention_enabled=self.model.use_flash_attention
            )
        
        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Generate text from prompt"""
            try:
                start_time = time.time()
                
                # Check if streaming
                if request.stream:
                    raise HTTPException(
                        status_code=400,
                        detail="Use /generate/stream for streaming responses"
                    )
                
                # Generate
                output = self.model.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    do_sample=request.do_sample
                )
                
                # Calculate metrics
                generation_time = time.time() - start_time
                
                # Get token counts (approximate)
                input_tokens = len(self.model.tokenizer.encode(request.prompt))
                output_tokens = len(self.model.tokenizer.encode(output)) - input_tokens
                tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0
                
                return GenerateResponse(
                    output=output,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    generation_time=generation_time,
                    tokens_per_second=tokens_per_second,
                    model=self.model_name,
                    optimization_level=self.optimization_level
                )
                
            except MemOptError as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Internal server error")
        
        @self.app.post("/generate/stream")
        async def generate_stream(request: GenerateRequest):
            """Stream text generation (token by token)"""
            
            async def stream_tokens():
                try:
                    # For now, generate full text then stream
                    # TODO: Implement true token-by-token streaming
                    output = self.model.generate(
                        prompt=request.prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=request.do_sample
                    )
                    
                    # Stream word by word (simulated)
                    words = output.split()
                    for word in words:
                        yield f"data: {json.dumps({'token': word})}\n\n"
                        await asyncio.sleep(0.01)  # Small delay
                    
                    yield f"data: {json.dumps({'done': True})}\n\n"
                    
                except Exception as e:
                    logger.error(f"Streaming failed: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            return StreamingResponse(
                stream_tokens(),
                media_type="text/event-stream"
            )
        
        @self.app.post("/generate/batch", response_model=BatchGenerateResponse)
        async def generate_batch(request: BatchGenerateRequest):
            """Batch text generation"""
            
            if len(request.prompts) > self.max_batch_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"Batch size {len(request.prompts)} exceeds maximum {self.max_batch_size}"
                )
            
            try:
                start_time = time.time()
                outputs = []
                
                # Process each prompt
                for prompt in request.prompts:
                    output = self.model.generate(
                        prompt=prompt,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        do_sample=request.do_sample
                    )
                    outputs.append(output)
                
                total_time = time.time() - start_time
                
                # Calculate total tokens
                total_tokens = sum(
                    len(self.model.tokenizer.encode(out))
                    for out in outputs
                )
                avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0
                
                return BatchGenerateResponse(
                    outputs=outputs,
                    total_time=total_time,
                    average_tokens_per_second=avg_tokens_per_second,
                    batch_size=len(request.prompts)
                )
                
            except MemOptError as e:
                logger.error(f"Batch generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/metrics")
        async def metrics():
            """Get performance metrics"""
            stats = self.model.get_profiling_stats()
            
            if not stats:
                return {"error": "Profiling not enabled"}
            
            return stats.to_dict()


def create_app(
    model_name: str = "gpt2-xl",
    optimization_level: str = "high",
    **kwargs
) -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        model_name: Model to load
        optimization_level: Optimization level
        **kwargs: Additional arguments
    
    Returns:
        FastAPI application instance
    """
    api = MemOptAPI(
        model_name=model_name,
        optimization_level=optimization_level,
        **kwargs
    )
    return api.app


def run_server(
    model_name: str = "gpt2-xl",
    optimization_level: str = "high",
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    **kwargs
):
    """
    Run the API server
    
    Args:
        model_name: Model to load
        optimization_level: Optimization level
        host: Host to bind to
        port: Port to bind to
        workers: Number of workers
        **kwargs: Additional arguments
    """
    logger.info("="*70)
    logger.info("Starting MemOpt API Server")
    logger.info("="*70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Optimization: {optimization_level}")
    logger.info(f"Host: {host}:{port}")
    logger.info(f"Workers: {workers}")
    
    app = create_app(
        model_name=model_name,
        optimization_level=optimization_level,
        **kwargs
    )
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MemOpt API Server")
    parser.add_argument("--model", default="gpt2-xl", help="Model name")
    parser.add_argument("--optimization", default="high", help="Optimization level")
    parser.add_argument("--host", default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    run_server(
        model_name=args.model,
        optimization_level=args.optimization,
        host=args.host,
        port=args.port,
        workers=args.workers
    )