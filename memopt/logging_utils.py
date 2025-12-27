"""
Phase 2: Structured Logging with Correlation IDs

Provides distributed tracing capabilities for hyperscale deployments.
Every request gets a correlation ID that flows through all components.
"""

import logging
import json
import time
import uuid
from typing import Optional, Dict, Any
from contextvars import ContextVar
from dataclasses import dataclass, asdict
from enum import Enum


# Context variable for correlation ID (thread-safe)
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
request_start_time_var: ContextVar[Optional[float]] = ContextVar('request_start_time', default=None)


class LogLevel(Enum):
    """Log levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    message: str
    correlation_id: Optional[str] = None
    request_id: Optional[str] = None
    component: Optional[str] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = None

    def to_json(self) -> str:
        """Convert to JSON string."""
        data = asdict(self)
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        return json.dumps(data)


class StructuredLogger:
    """
    Phase 2: Structured logger with correlation IDs.

    Enables distributed tracing across microservices and GPU nodes.
    All logs include correlation_id, timestamp, and structured metadata.
    """

    def __init__(
        self,
        name: str,
        level: LogLevel = LogLevel.INFO,
        enable_console: bool = True,
        enable_file: bool = False,
        file_path: Optional[str] = None
    ):
        """
        Args:
            name: Logger name (typically module name)
            level: Minimum log level
            enable_console: Print to console
            enable_file: Write to file
            file_path: Log file path
        """
        self.name = name
        self.level = level
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.file_path = file_path

        # Create underlying Python logger
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self._level_to_logging(level))

        # Configure handlers
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(JsonFormatter())
            self._logger.addHandler(console_handler)

        if enable_file and file_path:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(JsonFormatter())
            self._logger.addHandler(file_handler)

    def _level_to_logging(self, level: LogLevel) -> int:
        """Convert LogLevel to logging level."""
        mapping = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return mapping[level]

    def _create_entry(
        self,
        level: LogLevel,
        message: str,
        **metadata
    ) -> LogEntry:
        """Create structured log entry."""
        correlation_id = correlation_id_var.get()
        start_time = request_start_time_var.get()

        duration_ms = None
        if start_time is not None:
            duration_ms = (time.time() - start_time) * 1000

        return LogEntry(
            timestamp=time.time(),
            level=level.value,
            message=message,
            correlation_id=correlation_id,
            request_id=correlation_id,  # Same as correlation_id
            component=self.name,
            duration_ms=duration_ms,
            metadata=metadata if metadata else None
        )

    def debug(self, message: str, **metadata):
        """Log debug message."""
        entry = self._create_entry(LogLevel.DEBUG, message, **metadata)
        self._logger.debug(entry.to_json())

    def info(self, message: str, **metadata):
        """Log info message."""
        entry = self._create_entry(LogLevel.INFO, message, **metadata)
        self._logger.info(entry.to_json())

    def warning(self, message: str, **metadata):
        """Log warning message."""
        entry = self._create_entry(LogLevel.WARNING, message, **metadata)
        self._logger.warning(entry.to_json())

    def error(self, message: str, **metadata):
        """Log error message."""
        entry = self._create_entry(LogLevel.ERROR, message, **metadata)
        self._logger.error(entry.to_json())

    def critical(self, message: str, **metadata):
        """Log critical message."""
        entry = self._create_entry(LogLevel.CRITICAL, message, **metadata)
        self._logger.critical(entry.to_json())


class JsonFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # If message is already JSON, return as-is
        if record.msg.startswith('{'):
            return record.msg

        # Otherwise, create structured entry
        entry = {
            "timestamp": record.created,
            "level": record.levelname.lower(),
            "message": record.getMessage(),
            "component": record.name
        }

        correlation_id = correlation_id_var.get()
        if correlation_id:
            entry["correlation_id"] = correlation_id
            entry["request_id"] = correlation_id

        return json.dumps(entry)


# ============================================================================
# Phase 2: Request Context Management
# ============================================================================

class RequestContext:
    """
    Phase 2: Request context for distributed tracing.

    Manages correlation ID and timing for a single request.
    Use as context manager to automatically track request lifecycle.
    """

    def __init__(self, correlation_id: Optional[str] = None):
        """
        Args:
            correlation_id: Optional correlation ID (generates if None)
        """
        self.correlation_id = correlation_id or self._generate_correlation_id()
        self.start_time = time.time()
        self._token_corr = None
        self._token_time = None

    @staticmethod
    def _generate_correlation_id() -> str:
        """Generate unique correlation ID."""
        return f"req_{uuid.uuid4().hex[:16]}"

    def __enter__(self):
        """Enter request context."""
        self._token_corr = correlation_id_var.set(self.correlation_id)
        self._token_time = request_start_time_var.set(self.start_time)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit request context."""
        if self._token_corr is not None:
            correlation_id_var.reset(self._token_corr)
        if self._token_time is not None:
            request_start_time_var.reset(self._token_time)
        return False

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.time() - self.start_time) * 1000


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID from context."""
    return correlation_id_var.get()


def get_request_duration_ms() -> Optional[float]:
    """Get current request duration in milliseconds."""
    start_time = request_start_time_var.get()
    if start_time is None:
        return None
    return (time.time() - start_time) * 1000


# ============================================================================
# Phase 2: Global Logger Factory
# ============================================================================

_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    level: LogLevel = LogLevel.INFO
) -> StructuredLogger:
    """
    Get or create structured logger.

    Args:
        name: Logger name (typically __name__)
        level: Minimum log level

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(
            name=name,
            level=level,
            enable_console=True,
            enable_file=False
        )
    return _loggers[name]


# ============================================================================
# Phase 2: Performance Logging Helpers
# ============================================================================

class TimedOperation:
    """
    Context manager for timing operations.

    Usage:
        with TimedOperation("cache_eviction") as op:
            evict_blocks()
        logger.info(f"Eviction took {op.elapsed_ms:.2f}ms")
    """

    def __init__(self, operation_name: str, logger: Optional[StructuredLogger] = None):
        """
        Args:
            operation_name: Operation being timed
            logger: Optional logger for automatic logging
        """
        self.operation_name = operation_name
        self.logger = logger
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log if logger provided."""
        self.end_time = time.time()

        if self.logger:
            if exc_type is None:
                self.logger.debug(
                    f"{self.operation_name} completed",
                    duration_ms=self.elapsed_ms,
                    status="success"
                )
            else:
                self.logger.error(
                    f"{self.operation_name} failed",
                    duration_ms=self.elapsed_ms,
                    status="error",
                    error=str(exc_val)
                )

        return False

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return (end - self.start_time) * 1000


def log_request_lifecycle(logger: StructuredLogger, phase: str, **metadata):
    """
    Log request lifecycle event.

    Args:
        logger: Logger instance
        phase: Lifecycle phase (e.g., "admission", "inference", "completion")
        **metadata: Additional metadata
    """
    correlation_id = get_correlation_id()
    duration_ms = get_request_duration_ms()

    logger.info(
        f"Request {phase}",
        correlation_id=correlation_id,
        phase=phase,
        duration_ms=duration_ms,
        **metadata
    )
