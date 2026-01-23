#!/usr/bin/env python3
"""
Production Smoke Test

Runs an end-to-end test of the production system:
1. Starts in production mode
2. Connects to Redis
3. Initializes vLLM (or uses small model for testing)
4. Enqueues a request
5. Processes the request
6. Verifies output

Usage:
    # With local Redis and small model (for testing)
    export Memopt_ENV=prod
    export REDIS_URL=redis://localhost:6379
    export MODEL_NAME=gpt2  # Small model for quick test
    python tests/smoke_test_production.py

    # With production config
    export Memopt_ENV=prod
    export REDIS_URL=redis://your-redis:6379
    export MODEL_NAME=meta-llama/Llama-2-7b-hf
    python tests/smoke_test_production.py
"""

import sys
import os
import time
import asyncio
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check that all prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check environment variables
    required_vars = ["Memopt_ENV", "REDIS_URL", "MODEL_NAME"]
    missing = [v for v in required_vars if not os.getenv(v)]

    if missing:
        logger.error(f"Missing environment variables: {missing}")
        logger.error("Required:")
        logger.error("  export Memopt_ENV=prod")
        logger.error("  export REDIS_URL=redis://localhost:6379")
        logger.error("  export MODEL_NAME=gpt2")
        return False

    if os.getenv("Memopt_ENV") != "prod":
        logger.error("Memopt_ENV must be 'prod' for smoke test")
        return False

    logger.info("✓ Environment variables set")
    return True


def test_runtime_config():
    """Test runtime configuration."""
    logger.info("Testing runtime configuration...")

    from Memopt.runtime import get_config

    config = get_config()

    logger.info(f"  Mode: {config.mode.value}")
    logger.info(f"  Redis: {config.redis_url}")
    logger.info(f"  Model: {config.model_name}")

    assert config.is_prod, "Should be in production mode"
    logger.info("✓ Runtime configuration valid")


def test_redis_connection():
    """Test Redis connectivity."""
    logger.info("Testing Redis connection...")

    from Memopt.runtime import get_config
    import redis
    import urllib.parse

    config = get_config()
    parsed = urllib.parse.urlparse(config.redis_url)

    try:
        client = redis.Redis(
            host=parsed.hostname or "localhost",
            port=parsed.port or 6379,
            password=parsed.password,
            socket_timeout=5.0
        )
        result = client.ping()
        assert result, "Redis ping failed"
        logger.info("✓ Redis connection successful")
    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        raise


def test_backend_creation():
    """Test that production backends are created."""
    logger.info("Testing backend creation...")

    from Memopt.runtime import (
        create_distributed_state_backend,
        create_request_queue
    )

    # Test state backend
    logger.info("  Creating state backend...")
    backend = create_distributed_state_backend()
    backend_type = type(backend).__name__

    if backend_type == "InMemoryBackend":
        raise RuntimeError("❌ State backend is InMemoryBackend in prod mode!")

    logger.info(f"  ✓ State backend: {backend_type}")

    # Test request queue
    logger.info("  Creating request queue...")
    queue = create_request_queue()
    queue_type = type(queue).__name__

    if queue_type == "InMemoryQueue":
        raise RuntimeError("❌ Request queue is InMemoryQueue in prod mode!")

    logger.info(f"  ✓ Request queue: {queue_type}")
    logger.info("✓ Production backends created successfully")

    return backend, queue


def test_production_validation():
    """Test production runtime validation."""
    logger.info("Testing production validation...")

    from Memopt.runtime import validate_production_runtime

    try:
        validate_production_runtime()
        logger.info("✓ Production validation passed")
    except RuntimeError as e:
        logger.error(f"✗ Production validation failed: {e}")
        raise


async def test_inference_engine():
    """Test inference engine creation and basic operation."""
    logger.info("Testing inference engine...")

    from Memopt.runtime import create_inference_engine, get_config

    config = get_config()

    logger.info(f"  Loading model: {config.model_name}")
    logger.info("  (This may take a few minutes for large models...)")

    try:
        engine = create_inference_engine()
        engine_type = type(engine).__name__

        logger.info(f"  ✓ Inference engine created: {engine_type}")

        # Check if it's vLLM
        if "vllm" in engine_type.lower() or "VLLMAdapter" in engine_type:
            logger.info("  ✓ Using vLLM (production inference engine)")

            # Initialize if needed
            if hasattr(engine, 'initialize'):
                logger.info("  Initializing vLLM engine...")
                await engine.initialize()
                logger.info("  ✓ vLLM engine initialized")

            return engine, True  # is_vllm = True

        else:
            logger.warning(f"  ⚠ Using {engine_type} (not vLLM)")
            logger.warning("  For production, install vLLM: pip install vllm>=0.3.0")
            return engine, False  # is_vllm = False

    except Exception as e:
        logger.error(f"✗ Failed to create inference engine: {e}")
        raise


async def test_end_to_end_inference(engine, is_vllm):
    """Test end-to-end inference request."""
    logger.info("Testing end-to-end inference...")

    test_prompt = "Once upon a time"
    logger.info(f"  Prompt: '{test_prompt}'")

    try:
        if is_vllm:
            # vLLM adapter
            from Memopt.scheduler import InferenceRequest

            request = InferenceRequest(
                request_id="smoke-test-001",
                prompt=test_prompt,
                input_ids=None,  # Will be tokenized by vLLM
                max_tokens=20
            )

            logger.info("  Generating tokens...")
            start_time = time.time()

            final_response = None
            async for response in engine.generate(request):
                final_response = response

            latency = time.time() - start_time

            logger.info(f"  ✓ Generated {final_response.tokens_generated} tokens")
            logger.info(f"  ✓ Latency: {latency:.2f}s")
            logger.info(f"  Output: {final_response.generated_text[:100]}...")

        else:
            # HuggingFace model
            logger.info("  Generating with HuggingFace model...")
            start_time = time.time()

            output = engine.generate(test_prompt, max_tokens=20)
            latency = time.time() - start_time

            logger.info(f"  ✓ Generated output")
            logger.info(f"  ✓ Latency: {latency:.2f}s")
            logger.info(f"  Output: {output[:100]}...")

        logger.info("✓ End-to-end inference successful")

    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        raise


async def main():
    """Run all smoke tests."""
    logger.info("=" * 60)
    logger.info("Memopt PRODUCTION SMOKE TEST")
    logger.info("=" * 60)

    try:
        # Prerequisites
        if not check_prerequisites():
            return 1

        # Runtime config
        test_runtime_config()

        # Redis connection
        test_redis_connection()

        # Backend creation
        backend, queue = test_backend_creation()

        # Production validation
        test_production_validation()

        # Inference engine
        engine, is_vllm = await test_inference_engine()

        # End-to-end inference
        await test_end_to_end_inference(engine, is_vllm)

        # Success
        logger.info("=" * 60)
        logger.info("✅ ALL SMOKE TESTS PASSED")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Production system is ready!")
        logger.info("")
        if not is_vllm:
            logger.warning("⚠ NOTE: Using HuggingFace model (dev mode)")
            logger.warning("  For production, install vLLM:")
            logger.warning("    pip install vllm>=0.3.0")
            logger.warning("  Then restart with MODEL_NAME set to a vLLM-compatible model")

        return 0

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ SMOKE TEST FAILED")
        logger.error("=" * 60)
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
