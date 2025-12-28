"""
Memopt Production Backends

This package contains production-ready implementations of:
- Redis-based distributed state (RedisBackend)
- Redis Streams request queue (RedisRequestQueue)
- vLLM inference engine adapter (VLLMAdapter)

All backends are production-hardened with:
- Connection pooling
- Retry with exponential backoff
- Circuit breakers
- Health checks
- Comprehensive error handling
"""

from Memopt.backends.redis_backend import (
    RedisBackend,
    RedisConfig,
    create_redis_backend
)

from Memopt.backends.redis_queue import (
    RedisRequestQueue,
    RedisQueueConfig,
    consume_requests
)

from Memopt.backends.vllm_adapter import (
    VLLMAdapter,
    VLLMConfig,
    VLLMSyncAdapter,
    create_vllm_adapter
)

__all__ = [
    # Redis Backend
    'RedisBackend',
    'RedisConfig',
    'create_redis_backend',
    # Redis Queue
    'RedisRequestQueue',
    'RedisQueueConfig',
    'consume_requests',
    # vLLM Adapter
    'VLLMAdapter',
    'VLLMConfig',
    'VLLMSyncAdapter',
    'create_vllm_adapter',
]
