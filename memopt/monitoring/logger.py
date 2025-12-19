"""
Centralized logging for MemOpt

PURPOSE: Provide structured logging throughout the application
WHY: Essential for debugging, monitoring, and troubleshooting
WHEN TO USE: Import in every module: logger = get_logger(__name__)

Features:
- Console output (INFO and above) - what you see in terminal
- File logging (DEBUG and above) - detailed logs in ~/.memopt/logs/
- Error file (ERROR and above) - separate file for errors only

Usage:
    from memopt.monitoring.logger import get_logger
    logger = get_logger(__name__)
    
    logger.debug("Detailed info for debugging")
    logger.info("General information")
    logger.warning("Warning message")
    logger.error("Error occurred", exc_info=True)  # Include stack trace
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


class MemOptLogger:
    """
    Singleton logger manager for MemOpt
    
    WHY SINGLETON: Ensures logging is initialized only once
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """Configure logging system with multiple handlers"""
        # Create logs directory in user's home
        log_dir = Path.home() / ".memopt" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Timestamp for this session's log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Console format (simple, readable)
        console_format = "%(levelname)s - %(name)s - %(message)s"
        console_formatter = logging.Formatter(console_format)
        
        # File format (detailed, includes line numbers)
        file_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "%(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        )
        file_formatter = logging.Formatter(file_format)
        
        # HANDLER 1: Console (INFO and above)
        # Shows in terminal when you run the code
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # HANDLER 2: File (DEBUG and above)
        # Detailed logs for debugging, saved to file
        file_handler = logging.FileHandler(
            log_dir / f"memopt_{timestamp}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        
        # HANDLER 3: Error file (ERROR and above)
        # Separate file for errors only, easier to find issues
        error_handler = logging.FileHandler(
            log_dir / f"memopt_errors_{timestamp}.log",
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Configure root memopt logger
        root_logger = logging.getLogger("memopt")
        root_logger.setLevel(logging.DEBUG)  # Capture everything
        
        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        # Add all handlers
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(error_handler)
        
        # Prevent propagation to avoid duplicate logs
        root_logger.propagate = False
        
        # Log initialization
        root_logger.info("MemOpt logging initialized")
        root_logger.info(f"Log directory: {log_dir}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module
    
    Args:
        name: Logger name (typically __name__ from your module)
    
    Returns:
        Configured logger instance
    
    Example:
        # In your module:
        from memopt.monitoring.logger import get_logger
        logger = get_logger(__name__)
        
        # Use it:
        logger.info("Operation started")
        logger.error("Operation failed", exc_info=True)
    """
    # Ensure logging is initialized
    MemOptLogger()
    
    # Return logger with memopt prefix
    return logging.getLogger(f"memopt.{name}")