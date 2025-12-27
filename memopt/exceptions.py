"""
Custom exceptions for MemOpt.

These exceptions enable graceful degradation and backpressure
without changing inference behavior or performance.
"""


class QueueFullError(Exception):
    """
    Raised when the request queue has reached capacity.

    This indicates the system is overloaded. Callers should:
    - Return HTTP 429 (Too Many Requests)
    - Implement exponential backoff
    - Shed load at the edge

    PERFORMANCE IMPACT: None (raised at admission, not during inference)
    """
    pass


class CacheEvictionError(Exception):
    """
    Raised when KV cache cannot evict blocks to satisfy allocation.

    This indicates all cache blocks are in active use by running requests.
    System should reject new requests until capacity is available.

    PERFORMANCE IMPACT: None (raised at allocation, not during inference)
    """
    pass
