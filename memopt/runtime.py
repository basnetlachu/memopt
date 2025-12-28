"""
Runtime Configuration and Backend Selection

Provides factory functions that select appropriate backends based on environment:
- MEMOPT_ENV=dev → In-memory, simulated (for testing)
- MEMOPT_ENV=prod → Redis, vLLM (for production)

In production mode, fails fast if required configuration is missing.
"""

import os
import logging
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class RuntimeMode(Enum):
    """Runtime environment mode."""
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class RuntimeConfig:
    """
    Runtime configuration from environment variables.

    Required for prod:
    - MEMOPT_ENV=prod
    - REDIS_URL=redis://host:port
    - MODEL_NAME=meta-llama/Llama-2-7b-hf (or model path)

    Optional:
    - TENSOR_PARALLEL_SIZE=1 (number of GPUs)
    - MAX_NUM_SEQS=256 (max concurrent sequences)
    - GPU_MEMORY_UTILIZATION=0.90
    """

    def __init__(self):
        # Runtime mode
        env_mode = os.getenv("MEMOPT_ENV", "dev").lower()
        try:
            self.mode = RuntimeMode(env_mode)
        except ValueError:
            raise ValueError(
                f"Invalid MEMOPT_ENV='{env_mode}'. "
                f"Must be one of: {[m.value for m in RuntimeMode]}"
            )

        # Redis configuration (required in prod)
        self.redis_url = os.getenv("REDIS_URL")

        # Model configuration (required in prod)
        self.model_name = os.getenv("MODEL_NAME")
        self.tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        self.max_num_seqs = int(os.getenv("MAX_NUM_SEQS", "256"))
        self.gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))

        # Validate production requirements
        if self.mode == RuntimeMode.PROD:
            self._validate_prod_config()

    def _validate_prod_config(self):
        """Validate that all required production config is present."""
        errors = []

        if not self.redis_url:
            errors.append("REDIS_URL environment variable required in prod mode")

        if not self.model_name:
            errors.append("MODEL_NAME environment variable required in prod mode")

        if errors:
            raise RuntimeError(
                "Production configuration incomplete:\n" +
                "\n".join(f"  - {e}" for e in errors) +
                "\n\nRequired environment variables:\n"
                "  export MEMOPT_ENV=prod\n"
                "  export REDIS_URL=redis://localhost:6379\n"
                "  export MODEL_NAME=meta-llama/Llama-2-7b-hf\n"
            )

    @property
    def is_prod(self) -> bool:
        """Check if running in production mode."""
        return self.mode == RuntimeMode.PROD

    @property
    def is_dev(self) -> bool:
        """Check if running in dev/test mode."""
        return self.mode in (RuntimeMode.DEV, RuntimeMode.TEST)


# Global config instance
_config: Optional[RuntimeConfig] = None


def get_config() -> RuntimeConfig:
    """Get or create runtime configuration."""
    global _config
    if _config is None:
        _config = RuntimeConfig()
    return _config


def create_inference_engine(model_name: Optional[str] = None):
    """
    Create inference engine based on runtime mode.

    - prod: vLLM engine (real GPU inference)
    - dev/test: HuggingFace model (for development)

    Args:
        model_name: Override model name (uses config if None)

    Returns:
        Inference engine instance

    Raises:
        RuntimeError: If prod mode but vLLM not available
    """
    config = get_config()
    model = model_name or config.model_name

    if config.is_prod:
        # Production: Use vLLM
        try:
            from memopt.backends.vllm_adapter import create_vllm_adapter
        except ImportError:
            raise RuntimeError(
                "vLLM not installed. Required for production mode.\n"
                "Install with: pip install vllm>=0.3.0"
            )

        if not model:
            raise RuntimeError("MODEL_NAME required in production mode")

        logger.info(f"Creating vLLM inference engine: {model}")
        adapter = create_vllm_adapter(
            model=model,
            tensor_parallel_size=config.tensor_parallel_size,
            max_num_seqs=config.max_num_seqs,
            gpu_memory_utilization=config.gpu_memory_utilization
        )

        return adapter

    else:
        # Dev/Test: Use HuggingFace model
        from memopt.model import OptimizedLLM

        model = model or "gpt2"  # Default for dev
        logger.info(f"Creating HuggingFace model (dev mode): {model}")

        return OptimizedLLM(model, optimization_level="balanced")


def create_distributed_state_backend():
    """
    Create distributed state backend based on runtime mode.

    - prod: RedisBackend (real distributed coordination)
    - dev/test: InMemoryBackend (local only)

    Returns:
        DistributedStateBackend instance

    Raises:
        RuntimeError: If prod mode but Redis not available
    """
    config = get_config()

    if config.is_prod:
        # Production: Use Redis
        try:
            from memopt.backends.redis_backend import create_redis_backend
        except ImportError:
            raise RuntimeError(
                "redis-py not installed. Required for production mode.\n"
                "Install with: pip install redis>=4.5.0"
            )

        if not config.redis_url:
            raise RuntimeError("REDIS_URL required in production mode")

        # Parse Redis URL
        import urllib.parse
        parsed = urllib.parse.urlparse(config.redis_url)

        logger.info(f"Creating Redis backend: {parsed.hostname}:{parsed.port}")

        backend = create_redis_backend(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password
        )

        # Test connection
        if not backend.health_check():
            raise RuntimeError(
                f"Cannot connect to Redis at {config.redis_url}. "
                "Ensure Redis is running and accessible."
            )

        return backend

    else:
        # Dev/Test: Use in-memory
        from memopt.distributed_state import InMemoryBackend

        logger.info("Creating in-memory backend (dev mode)")
        return InMemoryBackend()


def create_request_queue():
    """
    Create request queue based on runtime mode.

    - prod: RedisRequestQueue (persistent, distributed)
    - dev/test: In-memory queue (local only)

    Returns:
        Request queue instance

    Raises:
        RuntimeError: If prod mode but Redis not available
    """
    config = get_config()

    if config.is_prod:
        # Production: Use Redis Streams
        try:
            import redis
            from memopt.backends.redis_queue import RedisRequestQueue
        except ImportError:
            raise RuntimeError(
                "redis-py not installed. Required for production mode.\n"
                "Install with: pip install redis>=4.5.0"
            )

        if not config.redis_url:
            raise RuntimeError("REDIS_URL required in production mode")

        # Parse Redis URL
        import urllib.parse
        parsed = urllib.parse.urlparse(config.redis_url)

        logger.info(f"Creating Redis request queue: {parsed.hostname}:{parsed.port}")

        # Create Redis client
        redis_client = redis.Redis(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            decode_responses=False,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            health_check_interval=30
        )

        # Test connection
        try:
            redis_client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Redis at {config.redis_url}: {e}\n"
                "Ensure Redis is running and accessible."
            )

        queue = RedisRequestQueue(redis_client)
        return queue

    else:
        # Dev/Test: Use in-memory queue
        # Note: For simplicity, return a simple wrapper
        # In dev mode, the scheduler's built-in queue is sufficient
        logger.info("Using in-memory request queue (dev mode)")

        # Return a simple in-memory queue wrapper
        from collections import deque

        class InMemoryQueue:
            """Simple in-memory queue for dev/test."""
            def __init__(self):
                self._queue = deque(maxlen=10000)

            def enqueue(self, request):
                self._queue.append(request)
                return f"mem-{id(request)}"

            def dequeue(self, block=False):
                if self._queue:
                    return (f"mem-{id(self._queue[0])}", self._queue.popleft())
                return None

            def ack(self, message_id):
                pass

            def nack(self, message_id, request):
                self._queue.appendleft(request)

            def get_queue_depth(self):
                return len(self._queue)

        return InMemoryQueue()


def validate_production_runtime():
    """
    Validate that production runtime is correctly configured.

    Checks that in prod mode:
    - Inference engine is vLLM (not HuggingFace)
    - Request queue is Redis (not in-memory)
    - State backend is Redis (not in-memory)

    Raises:
        RuntimeError: If production validation fails
    """
    config = get_config()

    if not config.is_prod:
        logger.info(f"Running in {config.mode.value} mode - skipping prod validation")
        return

    logger.info("Validating production runtime configuration...")

    errors = []

    # Check inference engine
    try:
        engine = create_inference_engine()
        engine_type = type(engine).__name__

        if "vllm" not in engine_type.lower() and "VLLMAdapter" not in engine_type:
            errors.append(
                f"Inference engine is {engine_type}, expected VLLMAdapter. "
                "Ensure vLLM is installed and MODEL_NAME is set."
            )
    except Exception as e:
        errors.append(f"Failed to create inference engine: {e}")

    # Check state backend
    try:
        backend = create_distributed_state_backend()
        backend_type = type(backend).__name__

        if backend_type == "InMemoryBackend":
            errors.append(
                "State backend is InMemoryBackend, expected RedisBackend. "
                "Ensure REDIS_URL is set and Redis is running."
            )
    except Exception as e:
        errors.append(f"Failed to create state backend: {e}")

    # Check request queue
    try:
        queue = create_request_queue()
        queue_type = type(queue).__name__

        if queue_type == "InMemoryQueue":
            errors.append(
                "Request queue is InMemoryQueue, expected RedisRequestQueue. "
                "Ensure REDIS_URL is set and Redis is running."
            )
    except Exception as e:
        errors.append(f"Failed to create request queue: {e}")

    if errors:
        raise RuntimeError(
            "❌ PRODUCTION VALIDATION FAILED:\n\n" +
            "\n".join(f"  • {e}" for e in errors) +
            "\n\nProduction deployment BLOCKED. Fix configuration before deploying."
        )

    logger.info("✅ Production runtime validation PASSED")
    logger.info(f"  • Inference engine: vLLM ({config.model_name})")
    logger.info(f"  • State backend: Redis ({config.redis_url})")
    logger.info(f"  • Request queue: Redis Streams ({config.redis_url})")
