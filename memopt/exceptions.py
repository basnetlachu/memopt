"""
Custom exceptions for Memopt.

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


# Production-grade exceptions for Phase 1 safety features

class MemOptException(Exception):
    """Base exception for all MemOpt production errors."""
    pass


class ResourceExhaustedError(MemOptException):
    """KV cache, queue, or GPU memory exhausted.

    This exception indicates that a hard resource limit has been hit.
    In production, this should trigger request rejection (HTTP 429)
    rather than service crash.
    """
    def __init__(self, resource_type: str):
        self.resource_type = resource_type
        super().__init__(f"Resource exhausted: {resource_type}")


class RequestRejectedError(MemOptException):
    """Request rejected due to safety limits.

    Raised when pre-flight safety checks fail (queue full, request too large, etc.).
    """
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Request rejected: {reason}")


class ShutdownInProgressError(MemOptException):
    """System is shutting down, rejecting new requests.

    Raised when graceful shutdown has been initiated and new requests
    should not be accepted.
    """
    def __init__(self):
        super().__init__("System is shutting down, cannot accept new requests")
