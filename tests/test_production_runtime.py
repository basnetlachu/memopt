"""
Test production runtime configuration and backend selection.

Verifies that:
1. Production mode uses vLLM (not HuggingFace)
2. Production mode uses RedisRequestQueue (not in-memory)
3. Production mode uses RedisBackend (not InMemoryBackend)
4. Production validation catches misconfigurations
"""

import os
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment before each test."""
    # Remove all MEMOPT env vars
    for key in list(os.environ.keys()):
        if key.startswith("MEMOPT") or key in ("REDIS_URL", "MODEL_NAME"):
            monkeypatch.delenv(key, raising=False)


@pytest.fixture
def prod_env(monkeypatch):
    """Set production environment variables."""
    monkeypatch.setenv("MEMOPT_ENV", "prod")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
    monkeypatch.setenv("MODEL_NAME", "gpt2")  # Use small model for tests


@pytest.fixture
def dev_env(monkeypatch):
    """Set development environment variables."""
    monkeypatch.setenv("MEMOPT_ENV", "dev")


class TestRuntimeMode:
    """Test runtime mode detection."""

    def test_default_is_dev(self, clean_env):
        """Default mode should be dev."""
        from memopt.runtime import RuntimeConfig

        config = RuntimeConfig()
        assert config.mode.value == "dev"
        assert config.is_dev
        assert not config.is_prod

    def test_prod_mode_detected(self, prod_env):
        """Production mode should be detected."""
        from memopt.runtime import RuntimeConfig

        config = RuntimeConfig()
        assert config.mode.value == "prod"
        assert config.is_prod
        assert not config.is_dev

    def test_invalid_mode_raises(self, monkeypatch):
        """Invalid mode should raise error."""
        monkeypatch.setenv("MEMOPT_ENV", "invalid")

        from memopt.runtime import RuntimeConfig

        with pytest.raises(ValueError, match="Invalid MEMOPT_ENV"):
            RuntimeConfig()


class TestProductionValidation:
    """Test production configuration validation."""

    def test_prod_requires_redis_url(self, monkeypatch):
        """Production mode requires REDIS_URL."""
        monkeypatch.setenv("MEMOPT_ENV", "prod")
        monkeypatch.setenv("MODEL_NAME", "gpt2")
        # No REDIS_URL

        from memopt.runtime import RuntimeConfig

        with pytest.raises(RuntimeError, match="REDIS_URL.*required"):
            RuntimeConfig()

    def test_prod_requires_model_name(self, monkeypatch):
        """Production mode requires MODEL_NAME."""
        monkeypatch.setenv("MEMOPT_ENV", "prod")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        # No MODEL_NAME

        from memopt.runtime import RuntimeConfig

        with pytest.raises(RuntimeError, match="MODEL_NAME.*required"):
            RuntimeConfig()

    def test_prod_with_all_config_succeeds(self, prod_env):
        """Production mode with all config should succeed."""
        from memopt.runtime import RuntimeConfig

        config = RuntimeConfig()
        assert config.is_prod
        assert config.redis_url == "redis://localhost:6379"
        assert config.model_name == "gpt2"


class TestInferenceEngineSelection:
    """Test inference engine backend selection."""

    def test_dev_mode_uses_huggingface(self, dev_env):
        """Dev mode should use HuggingFace OptimizedLLM."""
        from memopt.runtime import create_inference_engine

        engine = create_inference_engine()
        engine_type = type(engine).__name__

        # Should be OptimizedLLM (HuggingFace-based)
        assert engine_type == "OptimizedLLM"

    @patch("memopt.runtime.create_vllm_adapter")
    def test_prod_mode_uses_vllm(self, mock_vllm, prod_env):
        """Production mode should use vLLM adapter."""
        from memopt.runtime import create_inference_engine

        # Mock vLLM adapter
        mock_adapter = MagicMock()
        mock_adapter.__class__.__name__ = "VLLMAdapter"
        mock_vllm.return_value = mock_adapter

        engine = create_inference_engine()

        # Should call vLLM factory
        mock_vllm.assert_called_once()
        assert engine == mock_adapter

    def test_prod_mode_without_vllm_raises(self, prod_env, monkeypatch):
        """Production mode without vLLM should fail fast."""
        from memopt.runtime import create_inference_engine

        # Mock import failure
        import sys
        original_import = __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if "vllm" in name:
                raise ImportError("vLLM not installed")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(__builtins__, "__import__", mock_import)

        with pytest.raises(RuntimeError, match="vLLM not installed"):
            create_inference_engine()


class TestStateBackendSelection:
    """Test distributed state backend selection."""

    def test_dev_mode_uses_inmemory(self, dev_env):
        """Dev mode should use InMemoryBackend."""
        from memopt.runtime import create_distributed_state_backend

        backend = create_distributed_state_backend()
        backend_type = type(backend).__name__

        assert backend_type == "InMemoryBackend"

    @patch("memopt.runtime.create_redis_backend")
    def test_prod_mode_uses_redis(self, mock_redis, prod_env):
        """Production mode should use RedisBackend."""
        from memopt.runtime import create_distributed_state_backend

        # Mock Redis backend
        mock_backend = MagicMock()
        mock_backend.__class__.__name__ = "RedisBackend"
        mock_backend.health_check.return_value = True
        mock_redis.return_value = mock_backend

        backend = create_distributed_state_backend()

        # Should call Redis factory
        mock_redis.assert_called_once()
        assert backend == mock_backend
        mock_backend.health_check.assert_called_once()

    @patch("memopt.runtime.create_redis_backend")
    def test_prod_mode_redis_health_check_fails(self, mock_redis, prod_env):
        """Production mode should fail if Redis unreachable."""
        from memopt.runtime import create_distributed_state_backend

        # Mock unhealthy Redis
        mock_backend = MagicMock()
        mock_backend.health_check.return_value = False
        mock_redis.return_value = mock_backend

        with pytest.raises(RuntimeError, match="Cannot connect to Redis"):
            create_distributed_state_backend()


class TestRequestQueueSelection:
    """Test request queue backend selection."""

    def test_dev_mode_uses_inmemory(self, dev_env):
        """Dev mode should use in-memory queue."""
        from memopt.runtime import create_request_queue

        queue = create_request_queue()
        queue_type = type(queue).__name__

        assert queue_type == "InMemoryQueue"

    @patch("memopt.backends.redis_queue.RedisRequestQueue")
    @patch("redis.Redis")
    def test_prod_mode_uses_redis_streams(self, mock_redis_client, mock_queue, prod_env):
        """Production mode should use RedisRequestQueue."""
        from memopt.runtime import create_request_queue

        # Mock Redis client
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis_client.return_value = mock_client

        # Mock Redis queue
        mock_queue_instance = MagicMock()
        mock_queue_instance.__class__.__name__ = "RedisRequestQueue"
        mock_queue.return_value = mock_queue_instance

        queue = create_request_queue()

        # Should create Redis client
        mock_redis_client.assert_called_once()
        mock_client.ping.assert_called_once()

        # Should create Redis queue
        mock_queue.assert_called_once_with(mock_client)
        assert queue == mock_queue_instance


class TestProductionRuntimeValidation:
    """Test complete production runtime validation."""

    @patch("memopt.runtime.create_inference_engine")
    @patch("memopt.runtime.create_distributed_state_backend")
    @patch("memopt.runtime.create_request_queue")
    def test_validation_passes_with_correct_backends(
        self, mock_queue, mock_backend, mock_engine, prod_env
    ):
        """Validation should pass when all backends are correct."""
        from memopt.runtime import validate_production_runtime

        # Mock correct backends
        mock_engine.return_value = MagicMock(__class__=type("VLLMAdapter", (), {}))
        mock_backend.return_value = MagicMock(__class__=type("RedisBackend", (), {}))
        mock_queue.return_value = MagicMock(__class__=type("RedisRequestQueue", (), {}))

        # Should not raise
        validate_production_runtime()

    @patch("memopt.runtime.create_inference_engine")
    @patch("memopt.runtime.create_distributed_state_backend")
    @patch("memopt.runtime.create_request_queue")
    def test_validation_fails_with_inmemory_backend(
        self, mock_queue, mock_backend, mock_engine, prod_env
    ):
        """Validation should fail if InMemoryBackend used in prod."""
        from memopt.runtime import validate_production_runtime

        # Mock wrong backend
        mock_engine.return_value = MagicMock(__class__=type("VLLMAdapter", (), {}))
        mock_backend.return_value = MagicMock(__class__=type("InMemoryBackend", (), {}))
        mock_queue.return_value = MagicMock(__class__=type("RedisRequestQueue", (), {}))

        with pytest.raises(RuntimeError, match="PRODUCTION VALIDATION FAILED"):
            validate_production_runtime()

    def test_validation_skipped_in_dev(self, dev_env):
        """Validation should be skipped in dev mode."""
        from memopt.runtime import validate_production_runtime

        # Should not raise (skipped in dev)
        validate_production_runtime()


class TestMemoryLeakFix:
    """Test that deployment history memory leak is fixed."""

    def test_deployment_history_is_bounded(self):
        """Deployment history should be bounded."""
        from memopt.deployment import DeploymentController, DeploymentConfig
        from collections import deque

        controller = DeploymentController(DeploymentConfig())

        # Check history is deque with maxlen
        assert isinstance(controller._deployment_history, deque)
        assert controller._deployment_history.maxlen == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
