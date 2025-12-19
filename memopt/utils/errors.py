"""
Custom exceptions for MemOpt

PURPOSE: Define all custom exception types used throughout MemOpt
WHY: Allows precise error handling and debugging
WHEN TO USE: Raise these instead of generic Exception

All MemOpt-specific errors inherit from MemOptError for easy catching.
"""


class MemOptError(Exception):
    """
    Base exception for all MemOpt errors
    
    All custom exceptions inherit from this, allowing:
        try:
            ...
        except MemOptError:
            # Catches ALL MemOpt errors
    """
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ModelLoadError(MemOptError):
    """
    Raised when model fails to load
    
    Examples:
    - Model file not found
    - Insufficient memory to load model
    - Corrupted model weights
    """
    pass


class GenerationError(MemOptError):
    """
    Raised when text generation fails
    
    Examples:
    - Generation timeout
    - Invalid model state
    - Unexpected model output
    """
    pass


class OutOfMemoryError(MemOptError):
    """
    Raised when GPU runs out of memory
    
    Examples:
    - Model too large for GPU
    - Batch size too large
    - KV cache full
    """
    pass


class InvalidInputError(MemOptError):
    """
    Raised when input validation fails
    
    Examples:
    - Empty prompt
    - Negative max_tokens
    - Invalid temperature range
    """
    pass


class CacheError(MemOptError):
    """
    Raised when KV cache operations fail
    
    Examples:
    - Cache corruption
    - Invalid cache state
    - Cache allocation failure
    """
    pass


class DistributedError(MemOptError):
    """
    Raised when multi-GPU operations fail
    
    Examples:
    - GPU communication failure
    - Load balancing error
    - GPU synchronization timeout
    """
    pass


class ConfigurationError(MemOptError):
    """
    Raised when configuration is invalid
    
    Examples:
    - Invalid config file format
    - Missing required config
    - Conflicting config values
    """
    pass


class APIError(MemOptError):
    """
    Raised when API operations fail
    
    Examples:
    - Invalid API request
    - Server startup failure
    - Request timeout
    """
    pass


class IntegrationError(MemOptError):
    """
    Raised when third-party integration fails
    
    Examples:
    - vLLM adapter failure
    - TGI integration error
    - External service unavailable
    """
    pass