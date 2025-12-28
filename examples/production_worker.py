#!/usr/bin/env python3
"""
Production MemOpt Worker

Complete end-to-end example of a production inference worker using:
- Redis Streams for request queue
- vLLM for GPU inference
- Distributed state coordination

This worker:
1. Connects to Redis for queue and state
2. Initializes vLLM inference engine
3. Consumes requests from queue
4. Executes inference on GPU
5. Handles failures with retry/DLQ
6. Participates in leader election
7. Reports metrics

Run with:
    python -m examples.production_worker --model meta-llama/Llama-2-7b-hf --gpus 1

Environment variables:
    REDIS_HOST - Redis hostname (default: localhost)
    REDIS_PORT - Redis port (default: 6379)
    REDIS_PASSWORD - Redis password (optional)
    MODEL_NAME - Model to load
    TENSOR_PARALLEL_SIZE - Number of GPUs
"""

import os
import sys
import time
import signal
import asyncio
import logging
import argparse
from threading import Event
from typing import Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis

from memopt.backends import (
    create_redis_backend,
    RedisRequestQueue,
    create_vllm_adapter
)
from memopt.leader_election import LeaderElection
from memopt.metrics import MetricsCollector
from memopt.scheduler import InferenceRequest


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionWorker:
    """Production inference worker."""

    def __init__(
        self,
        redis_host: str,
        redis_port: int,
        redis_password: Optional[str],
        model_name: str,
        tensor_parallel_size: int,
        node_id: Optional[str] = None
    ):
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_password = redis_password
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.node_id = node_id or f"worker-{os.getpid()}"

        # Components (initialized in setup)
        self.redis_client: Optional[redis.Redis] = None
        self.state_backend = None
        self.request_queue: Optional[RedisRequestQueue] = None
        self.vllm_adapter = None
        self.leader_election: Optional[LeaderElection] = None
        self.metrics: Optional[MetricsCollector] = None

        # Shutdown event
        self.shutdown_event = Event()

        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()

    async def setup(self):
        """Initialize all components."""
        logger.info(f"Initializing worker {self.node_id}")

        # 1. Connect to Redis
        logger.info(f"Connecting to Redis at {self.redis_host}:{self.redis_port}")
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            health_check_interval=30
        )

        # Test connection
        self.redis_client.ping()
        logger.info("✓ Redis connection established")

        # 2. Create distributed state backend
        self.state_backend = create_redis_backend(
            host=self.redis_host,
            port=self.redis_port,
            password=self.redis_password
        )
        logger.info("✓ Distributed state backend initialized")

        # 3. Create request queue
        self.request_queue = RedisRequestQueue(self.redis_client)
        logger.info("✓ Request queue initialized")

        # 4. Initialize vLLM adapter
        logger.info(f"Loading model: {self.model_name}")
        self.vllm_adapter = create_vllm_adapter(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            max_num_seqs=256,
            gpu_memory_utilization=0.90
        )

        await self.vllm_adapter.initialize()
        logger.info("✓ vLLM inference engine initialized")

        # 5. Initialize metrics
        self.metrics = MetricsCollector(window_size=10000)
        logger.info("✓ Metrics collector initialized")

        # 6. Start leader election
        self.leader_election = LeaderElection(
            node_id=self.node_id,
            backend=self.state_backend,
            lease_duration=10,
            renew_interval=3
        )

        self.leader_election.on_elected(self._on_became_leader)
        self.leader_election.on_lost_leadership(self._on_lost_leadership)
        self.leader_election.start()
        logger.info("✓ Leader election started")

        logger.info("=" * 60)
        logger.info("Worker initialization complete!")
        logger.info(f"Node ID: {self.node_id}")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"GPUs: {self.tensor_parallel_size}")
        logger.info("=" * 60)

    def _on_became_leader(self):
        """Callback when this worker becomes the leader."""
        logger.info("🎉 This worker is now the LEADER")
        # Leader-specific tasks can be added here
        # e.g., global scheduling, coordination, monitoring

    def _on_lost_leadership(self):
        """Callback when this worker loses leadership."""
        logger.info("📉 This worker is no longer the leader")

    async def process_request(self, request: InferenceRequest) -> bool:
        """
        Process a single inference request.

        Args:
            request: Request to process

        Returns:
            True if successful, False otherwise
        """
        start_time = time.time()
        request_id = request.request_id

        try:
            logger.info(f"Processing request {request_id}")

            # Check deadline
            if request.deadline and time.time() > request.deadline:
                logger.warning(f"Request {request_id} already past deadline, skipping")
                self.metrics.counter("requests_skipped_deadline")
                return False

            # Generate response
            final_response = None
            async for response in self.vllm_adapter.generate(request):
                final_response = response

            # Record metrics
            latency = time.time() - start_time
            self.metrics.counter("requests_completed")
            self.metrics.histogram("request_latency_ms", latency * 1000)
            self.metrics.histogram("tokens_generated", final_response.tokens_generated)

            logger.info(
                f"✓ Request {request_id} completed: "
                f"{final_response.tokens_generated} tokens in {latency:.2f}s"
            )

            return True

        except asyncio.CancelledError:
            logger.info(f"Request {request_id} cancelled")
            self.metrics.counter("requests_cancelled")
            return False

        except Exception as e:
            logger.error(f"✗ Request {request_id} failed: {e}", exc_info=True)
            self.metrics.counter("requests_failed")
            return False

    async def run(self):
        """Main worker loop."""
        logger.info("Starting request processing loop...")

        while not self.shutdown_event.is_set():
            try:
                # Dequeue request (blocks for 5s)
                result = self.request_queue.dequeue(block=True)

                if result is None:
                    # Timeout, check shutdown and continue
                    continue

                message_id, request = result

                # Process request
                success = await self.process_request(request)

                # ACK or NACK
                if success:
                    self.request_queue.ack(message_id)
                else:
                    self.request_queue.nack(message_id, request)

                # Log stats periodically
                if int(time.time()) % 60 == 0:  # Every minute
                    self._log_stats()

            except Exception as e:
                logger.error(f"Worker loop error: {e}", exc_info=True)
                await asyncio.sleep(1)

        logger.info("Worker loop stopped")

    def _log_stats(self):
        """Log worker statistics."""
        stats = {
            'queue_depth': self.request_queue.get_queue_depth(),
            'pending_count': self.request_queue.get_pending_count(),
            'dlq_depth': self.request_queue.get_dlq_depth(),
            'is_leader': self.leader_election.is_leader,
            'requests_completed': self.metrics.get_counter('requests_completed'),
            'requests_failed': self.metrics.get_counter('requests_failed')
        }

        latency_stats = self.metrics.get_histogram_stats('request_latency_ms')

        logger.info(
            f"Stats: queue={stats['queue_depth']} pending={stats['pending_count']} "
            f"dlq={stats['dlq_depth']} completed={stats['requests_completed']} "
            f"failed={stats['requests_failed']} p50_latency={latency_stats.get('p50', 0):.1f}ms "
            f"leader={stats['is_leader']}"
        )

    async def shutdown(self):
        """Graceful shutdown."""
        logger.info("Shutting down worker...")

        # Stop leader election
        if self.leader_election:
            self.leader_election.stop()

        # Shutdown vLLM
        if self.vllm_adapter:
            await self.vllm_adapter.shutdown()

        # Close queue
        if self.request_queue:
            self.request_queue.close()

        # Close Redis backend
        if self.state_backend:
            self.state_backend.close()

        # Close Redis client
        if self.redis_client:
            self.redis_client.close()

        logger.info("✓ Worker shutdown complete")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='MemOpt Production Worker')
    parser.add_argument('--redis-host', default=os.getenv('REDIS_HOST', 'localhost'),
                        help='Redis hostname')
    parser.add_argument('--redis-port', type=int, default=int(os.getenv('REDIS_PORT', 6379)),
                        help='Redis port')
    parser.add_argument('--redis-password', default=os.getenv('REDIS_PASSWORD'),
                        help='Redis password')
    parser.add_argument('--model', required=True,
                        help='Model name or path (e.g., meta-llama/Llama-2-7b-hf)')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--node-id', default=None,
                        help='Worker node ID (auto-generated if not provided)')

    args = parser.parse_args()

    # Create worker
    worker = ProductionWorker(
        redis_host=args.redis_host,
        redis_port=args.redis_port,
        redis_password=args.redis_password,
        model_name=args.model,
        tensor_parallel_size=args.gpus,
        node_id=args.node_id
    )

    try:
        # Initialize
        await worker.setup()

        # Run
        await worker.run()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    finally:
        # Cleanup
        await worker.shutdown()

    return 0


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
