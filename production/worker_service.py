"""
Production worker service - one process per GPU.

Each worker:
- Runs on exactly one GPU (via CUDA_VISIBLE_DEVICES)
- Exposes HTTP API for inference requests
- Performs dynamic batching
- Reports metrics to registry
- Sends periodic heartbeats
"""
import os
import sys
import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memopt import OptimizedLLM
from production.config import WorkerConfig
from production.registry import WorkerRegistry, WorkerInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """Request format for generation."""
    prompt: str
    max_tokens: int = 256
    temperature: float = 1.0
    do_sample: bool = False
    session_id: Optional[str] = None  # For session affinity


class GenerateResponse(BaseModel):
    """Response format for generation."""
    text: str
    tokens_generated: int
    latency_ms: float
    worker_id: str
    gpu_id: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    worker_id: str
    gpu_id: int
    model_name: str
    queue_depth: int
    tokens_per_second: float


@dataclass
class QueuedRequest:
    """A queued inference request."""
    prompt: str
    max_tokens: int
    temperature: float
    do_sample: bool
    session_id: Optional[str]
    future: asyncio.Future
    enqueue_time: float


class WorkerService:
    """
    Production worker service.

    Handles:
    - HTTP API for inference
    - Dynamic batching of concurrent requests
    - Metrics tracking and reporting
    - Health checks and heartbeats
    """

    def __init__(self, config: WorkerConfig):
        self.config = config

        # Set GPU before importing torch
        os.environ['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)

        self.worker_id = f"worker-{config.worker_id}-gpu-{config.gpu_id}"

        # Request queue for batching
        self.request_queue = asyncio.Queue(maxsize=config.max_queue_size)

        # Metrics tracking
        self.total_requests = 0
        self.total_tokens = 0
        self.latencies = deque(maxlen=1000)  # Keep last 1000 latencies

        # Model (loaded during startup)
        self.model: Optional[OptimizedLLM] = None

        # Registry client
        self.registry = WorkerRegistry(
            host=config.registry_host,
            port=config.registry_port,
            db=config.registry_db
        )

        # FastAPI app
        self.app = FastAPI(title=f"Memopt Worker {self.worker_id}")
        self._setup_routes()

        # Background tasks
        self._batch_processor_task = None
        self._heartbeat_task = None

    def _setup_routes(self):
        """Setup FastAPI routes."""

        @self.app.post("/generate", response_model=GenerateResponse)
        async def generate(request: GenerateRequest):
            """Generate text for a prompt."""
            return await self.generate(request)

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.model is not None else "starting",
                worker_id=self.worker_id,
                gpu_id=self.config.gpu_id,
                model_name=self.config.model_name,
                queue_depth=self.request_queue.qsize(),
                tokens_per_second=self.get_throughput()
            )

        @self.app.get("/metrics")
        async def metrics():
            """Prometheus-compatible metrics."""
            return self.get_metrics()

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Queue a generation request and wait for result.

        This uses dynamic batching - multiple concurrent requests
        will be batched together for efficient inference.
        """
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Check queue capacity
        if self.request_queue.full():
            raise HTTPException(status_code=429, detail="Worker queue full")

        # Create queued request
        future = asyncio.Future()
        queued_request = QueuedRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.do_sample,
            session_id=request.session_id,
            future=future,
            enqueue_time=time.time()
        )

        # Add to queue
        await self.request_queue.put(queued_request)

        # Wait for result
        result = await future

        return result

    async def _batch_processor(self):
        """
        Background task that processes batches of requests.

        Waits up to batch_timeout_ms to accumulate requests,
        then processes them as a batch.
        """
        logger.info(f"Starting batch processor for {self.worker_id}")

        while True:
            try:
                # Collect requests for batch
                batch = []
                deadline = time.time() + (self.config.batch_timeout_ms / 1000.0)

                # Get first request (blocking)
                first_request = await self.request_queue.get()
                batch.append(first_request)

                # Try to get more requests until timeout or batch full
                while len(batch) < self.config.max_batch_size:
                    remaining_time = max(0, deadline - time.time())

                    if remaining_time <= 0:
                        break

                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=remaining_time
                        )
                        batch.append(request)
                    except asyncio.TimeoutError:
                        break

                # Process batch
                await self._process_batch(batch)

            except Exception as e:
                logger.error(f"Error in batch processor: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[QueuedRequest]):
        """Process a batch of requests."""
        if not batch:
            return

        start_time = time.time()

        try:
            # Extract prompts
            prompts = [req.prompt for req in batch]
            max_tokens = batch[0].max_tokens  # Use first request's max_tokens

            # Run inference (in thread pool to avoid blocking event loop)
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                self.model.generate_batch,
                prompts,
                max_tokens,
                False  # do_sample
            )

            # Process results
            for req, output in zip(batch, outputs):
                latency_ms = (time.time() - req.enqueue_time) * 1000
                tokens_generated = len(output.split())

                # Update metrics
                self.total_requests += 1
                self.total_tokens += tokens_generated
                self.latencies.append(latency_ms)

                # Create response
                response = GenerateResponse(
                    text=output,
                    tokens_generated=tokens_generated,
                    latency_ms=latency_ms,
                    worker_id=self.worker_id,
                    gpu_id=self.config.gpu_id
                )

                # Resolve future
                if not req.future.done():
                    req.future.set_result(response)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)

            # Fail all requests in batch
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(
                        HTTPException(status_code=500, detail=str(e))
                    )

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to registry."""
        logger.info(f"Starting heartbeat loop for {self.worker_id}")

        while True:
            try:
                # Report metrics to registry
                self.registry.update_worker_metrics(
                    worker_id=self.worker_id,
                    tokens_per_second=self.get_throughput(),
                    queue_depth=self.request_queue.qsize(),
                    current_batch_size=0,  # TODO: track actual batch size
                    memory_used_gb=self.get_memory_used(),
                    p95_latency_ms=self.get_p95_latency()
                )

                await asyncio.sleep(self.config.heartbeat_interval_seconds)

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self.config.heartbeat_interval_seconds)

    def get_throughput(self) -> float:
        """Calculate tokens per second over recent history."""
        if not self.latencies or self.total_tokens == 0:
            return 0.0

        # Use last 100 requests for throughput calculation
        recent_latencies = list(self.latencies)[-100:]
        if not recent_latencies:
            return 0.0

        total_time_seconds = sum(recent_latencies) / 1000.0
        if total_time_seconds == 0:
            return 0.0

        # Estimate tokens (rough approximation)
        tokens_per_request = 100  # Rough average
        tokens = len(recent_latencies) * tokens_per_request

        return tokens / total_time_seconds

    def get_memory_used(self) -> float:
        """Get GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 3)
        return 0.0

    def get_p95_latency(self) -> float:
        """Get 95th percentile latency."""
        if not self.latencies:
            return 0.0

        return float(np.percentile(list(self.latencies), 95))

    def get_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            "worker_id": self.worker_id,
            "gpu_id": self.config.gpu_id,
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "queue_depth": self.request_queue.qsize(),
            "tokens_per_second": self.get_throughput(),
            "p95_latency_ms": self.get_p95_latency(),
            "memory_used_gb": self.get_memory_used(),
        }

    async def startup(self):
        """Initialize worker on startup."""
        logger.info(f"Starting worker {self.worker_id} on GPU {self.config.gpu_id}")

        # Load model
        logger.info(f"Loading model {self.config.model_name}...")
        model_kwargs = {
            'model': self.config.model_name,
            'optimization_level': self.config.optimization_level,
            'enable_profiling': True,
            'device': 'cuda',
        }

        if self.config.rl_scheduler_path:
            model_kwargs['rl_scheduler_path'] = self.config.rl_scheduler_path
        if self.config.memory_predictor_path:
            model_kwargs['memory_predictor_path'] = self.config.memory_predictor_path

        self.model = OptimizedLLM(**model_kwargs)
        logger.info(f"Model loaded successfully")

        # Register with registry
        worker_info = WorkerInfo(
            worker_id=self.worker_id,
            host=self.config.host if self.config.host != "0.0.0.0" else "localhost",
            port=self.config.port,
            gpu_id=self.config.gpu_id,
            model_name=self.config.model_name,
            last_heartbeat=time.time()
        )
        self.registry.register_worker(worker_info)

        # Start background tasks
        self._batch_processor_task = asyncio.create_task(self._batch_processor())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info(f"Worker {self.worker_id} ready at {self.config.host}:{self.config.port}")

    async def shutdown(self):
        """Cleanup on shutdown."""
        logger.info(f"Shutting down worker {self.worker_id}")

        # Cancel background tasks
        if self._batch_processor_task:
            self._batch_processor_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        # Deregister from registry
        self.registry.deregister_worker(self.worker_id)

        logger.info(f"Worker {self.worker_id} shutdown complete")


def main():
    """Main entry point for worker service."""
    # Load configuration
    config = WorkerConfig.from_env()

    # Create worker service
    worker = WorkerService(config)

    # Setup startup/shutdown hooks
    @worker.app.on_event("startup")
    async def startup_event():
        await worker.startup()

    @worker.app.on_event("shutdown")
    async def shutdown_event():
        await worker.shutdown()

    # Run server
    uvicorn.run(
        worker.app,
        host=config.host,
        port=config.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
