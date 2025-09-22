"""Structured logging configuration for AutoDoc v2

This module provides comprehensive logging configuration with structured
logging, correlation IDs, and environment-specific settings.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Dict, List, Optional

import structlog
from structlog.stdlib import LoggerFactory

from .config_loader import get_settings


class AutoDocLogger:
    """AutoDoc logging configuration manager

    Provides structured logging setup with correlation IDs, JSON formatting,
    and environment-specific configuration.
    """

    def __init__(self):
        """Initialize logging configuration"""
        self.settings = get_settings()
        self.log_level = self.settings.log_level.upper()
        self.environment = self.settings.environment
        self.debug = self.settings.debug

        # Configure logging
        self._setup_standard_logging()
        self._setup_structured_logging()

    def _setup_standard_logging(self) -> None:
        """Setup standard Python logging configuration"""

        # Logging configuration
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "detailed": {
                    "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
                "json": {
                    "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                    "format": "%(asctime)s %(name)s %(levelname)s %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": self.log_level,
                    "formatter": "detailed" if self.debug else "standard",
                    "stream": sys.stdout,
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": self.log_level,
                    "formatter": "json" if not self.debug else "detailed",
                    "filename": "logs/autodoc.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8",
                },
                "error_file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "ERROR",
                    "formatter": "json",
                    "filename": "logs/autodoc_errors.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10,
                    "encoding": "utf8",
                },
            },
            "loggers": {
                "": {  # Root logger
                    "handlers": ["console"],
                    "level": self.log_level,
                    "propagate": False,
                },
                "autodoc": {
                    "handlers": ["console", "file"],
                    "level": self.log_level,
                    "propagate": False,
                },
                "src": {
                    "handlers": ["console", "file"],
                    "level": self.log_level,
                    "propagate": False,
                },
                "uvicorn": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.error": {
                    "handlers": ["console", "error_file"],
                    "level": "INFO",
                    "propagate": False,
                },
                "uvicorn.access": {
                    "handlers": ["console"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Apply configuration
        logging.config.dictConfig(logging_config)

    def _setup_structured_logging(self) -> None:
        """Setup structlog for structured logging"""

        # Determine processors based on environment
        if self.debug:
            # Development: Human-readable console output
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.dev.ConsoleRenderer(colors=True),
            ]
        else:
            # Production: JSON output
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer(),
            ]

        # Configure structlog
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )

    def get_logger(self, name: str) -> structlog.stdlib.BoundLogger:
        """Get a structured logger instance

        Args:
            name: Logger name

        Returns:
            Structured logger instance
        """
        return structlog.get_logger(name)

    def add_correlation_id_processor(self) -> None:
        """Add correlation ID processor to structlog"""

        def add_correlation_id(logger, method_name, event_dict):
            """Add correlation ID to log events"""
            # This would extract correlation ID from context
            # For now, it's a placeholder
            return event_dict

        # Add processor to existing configuration
        current_processors = structlog.get_config()["processors"]
        current_processors.insert(-1, add_correlation_id)  # Insert before renderer

        structlog.configure(processors=current_processors)

    def configure_external_loggers(self) -> None:
        """Configure external library loggers"""

        # Configure external library log levels
        external_loggers = {
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "boto3": "WARNING",
            "botocore": "WARNING",
            "pymongo": "WARNING",
            "motor": "WARNING",
            "langchain": "INFO",
            "langgraph": "INFO",
            "openai": "WARNING",
        }

        for logger_name, level in external_loggers.items():
            logging.getLogger(logger_name).setLevel(getattr(logging, level))

    def setup_request_logging(self) -> None:
        """Setup request-specific logging configuration"""

        # Create request logger
        request_logger = logging.getLogger("autodoc.requests")
        request_logger.setLevel(logging.INFO)

        # Create performance logger
        performance_logger = logging.getLogger("autodoc.performance")
        performance_logger.setLevel(logging.INFO)

        # Create security logger
        security_logger = logging.getLogger("autodoc.security")
        security_logger.setLevel(logging.WARNING)

    def get_logging_config(self) -> Dict[str, Any]:
        """Get current logging configuration

        Returns:
            Dictionary with logging configuration
        """
        return {
            "log_level": self.log_level,
            "environment": self.environment,
            "debug": self.debug,
            "structured_logging": True,
            "log_files": {
                "main": "logs/autodoc.log",
                "errors": "logs/autodoc_errors.log",
            },
            "external_loggers_configured": True,
        }


# Performance logging utilities


class PerformanceLogger:
    """Performance logging utilities for monitoring API performance"""

    def __init__(self):
        """Initialize performance logger"""
        self.logger = structlog.get_logger("autodoc.performance")

    async def log_operation_time(
        self,
        operation_name: str,
        duration: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log operation timing

        Args:
            operation_name: Name of the operation
            duration: Duration in seconds
            metadata: Additional metadata
        """
        log_data = {
            "operation": operation_name,
            "duration_seconds": round(duration, 4),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            log_data.update(metadata)

        if duration > 5.0:  # Log slow operations
            self.logger.warning("Slow operation detected", **log_data)
        elif duration > 1.0:  # Log moderately slow operations
            self.logger.info("Operation timing", **log_data)
        else:
            self.logger.debug("Operation timing", **log_data)

    async def log_api_metrics(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        user_id: Optional[str] = None,
    ) -> None:
        """Log API endpoint metrics

        Args:
            endpoint: API endpoint path
            method: HTTP method
            status_code: Response status code
            response_time: Response time in seconds
            user_id: Optional user ID
        """
        self.logger.info(
            "API endpoint metrics",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            response_time=round(response_time, 4),
            user_id=user_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# Security logging utilities


class SecurityLogger:
    """Security logging utilities for monitoring security events"""

    def __init__(self):
        """Initialize security logger"""
        self.logger = structlog.get_logger("autodoc.security")

    async def log_authentication_attempt(
        self,
        username: str,
        success: bool,
        ip_address: str,
        user_agent: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log authentication attempt

        Args:
            username: Username or email
            success: Whether authentication succeeded
            ip_address: Client IP address
            user_agent: Client user agent
            correlation_id: Request correlation ID
        """
        self.logger.info(
            "Authentication attempt",
            username=username,
            success=success,
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def log_authorization_failure(
        self,
        user_id: str,
        endpoint: str,
        required_permissions: List[str],
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log authorization failure

        Args:
            user_id: User ID
            endpoint: Requested endpoint
            required_permissions: Required permissions
            correlation_id: Request correlation ID
        """
        self.logger.warning(
            "Authorization failure",
            user_id=user_id,
            endpoint=endpoint,
            required_permissions=required_permissions,
            correlation_id=correlation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def log_webhook_validation_failure(
        self,
        provider: str,
        repository_url: str,
        event_type: str,
        ip_address: str,
        correlation_id: Optional[str] = None,
    ) -> None:
        """Log webhook signature validation failure

        Args:
            provider: Webhook provider
            repository_url: Repository URL
            event_type: Webhook event type
            ip_address: Client IP address
            correlation_id: Request correlation ID
        """
        self.logger.warning(
            "Webhook validation failure",
            provider=provider,
            repository_url=repository_url,
            event_type=event_type,
            ip_address=ip_address,
            correlation_id=correlation_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


# Global logging instances
autodoc_logger = None
performance_logger = None
security_logger = None


def setup_logging() -> AutoDocLogger:
    """Setup AutoDoc logging configuration

    Returns:
        Configured AutoDoc logger instance
    """
    global autodoc_logger, performance_logger, security_logger

    # Initialize logging
    autodoc_logger = AutoDocLogger()

    # Setup specialized loggers
    performance_logger = PerformanceLogger()
    security_logger = SecurityLogger()

    # Configure external loggers
    autodoc_logger.configure_external_loggers()

    # Setup request logging
    autodoc_logger.setup_request_logging()

    return autodoc_logger


def get_autodoc_logger() -> AutoDocLogger:
    """Get AutoDoc logger instance

    Returns:
        AutoDoc logger instance
    """
    global autodoc_logger

    if autodoc_logger is None:
        autodoc_logger = setup_logging()

    return autodoc_logger


def get_performance_logger() -> PerformanceLogger:
    """Get performance logger instance

    Returns:
        Performance logger instance
    """
    global performance_logger

    if performance_logger is None:
        setup_logging()

    return performance_logger


def get_security_logger() -> SecurityLogger:
    """Get security logger instance

    Returns:
        Security logger instance
    """
    global security_logger

    if security_logger is None:
        setup_logging()

    return security_logger


# Context managers for logging


class LoggingContext:
    """Context manager for adding context to logs"""

    def __init__(self, **context):
        """Initialize logging context

        Args:
            **context: Context variables to add to logs
        """
        self.context = context
        self.logger = structlog.get_logger()

    def __enter__(self):
        """Enter logging context"""
        self.bound_logger = self.logger.bind(**self.context)
        return self.bound_logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context"""
        pass


class PerformanceContext:
    """Context manager for performance logging"""

    def __init__(self, operation_name: str, **metadata):
        """Initialize performance context

        Args:
            operation_name: Name of the operation being measured
            **metadata: Additional metadata
        """
        self.operation_name = operation_name
        self.metadata = metadata
        self.start_time = None
        self.performance_logger = get_performance_logger()

    async def __aenter__(self):
        """Enter performance context"""
        import time

        self.start_time = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit performance context and log timing"""
        if self.start_time is not None:
            import time

            duration = time.time() - self.start_time

            # Add exception info if error occurred
            if exc_type is not None:
                self.metadata["error"] = True
                self.metadata["error_type"] = exc_type.__name__

            await self.performance_logger.log_operation_time(
                self.operation_name, duration, self.metadata
            )


# Utility functions


def log_function_call(
    func_name: str, args: tuple, kwargs: dict, logger: Optional[logging.Logger] = None
) -> None:
    """Log function call details

    Args:
        func_name: Function name
        args: Function arguments
        kwargs: Function keyword arguments
        logger: Optional logger instance
    """
    if logger is None:
        logger = structlog.get_logger()

    # Sanitize arguments (remove sensitive data)
    safe_args = [str(arg)[:100] for arg in args]  # Truncate long arguments
    safe_kwargs = {
        k: (
            str(v)[:100]
            if not k.lower().endswith(("password", "secret", "key", "token"))
            else "***"
        )
        for k, v in kwargs.items()
    }

    logger.debug(
        "Function call", function=func_name, args=safe_args, kwargs=safe_kwargs
    )


def log_error_with_context(
    error: Exception, context: Dict[str, Any], logger: Optional[logging.Logger] = None
) -> None:
    """Log error with additional context

    Args:
        error: Exception that occurred
        context: Additional context information
        logger: Optional logger instance
    """
    if logger is None:
        logger = structlog.get_logger()

    logger.error(
        "Error occurred", error=str(error), error_type=type(error).__name__, **context
    )


# Decorators for automatic logging


def log_async_function_calls(logger_name: Optional[str] = None):
    """Decorator to automatically log async function calls

    Args:
        logger_name: Optional logger name

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            func_logger = structlog.get_logger(logger_name or func.__module__)

            # Log function entry
            log_function_call(func.__name__, args, kwargs, func_logger)

            try:
                # Execute function with performance timing
                async with PerformanceContext(f"{func.__module__}.{func.__name__}"):
                    result = await func(*args, **kwargs)

                # Log successful completion
                func_logger.debug(
                    "Function completed successfully", function=func.__name__
                )

                return result

            except Exception as e:
                # Log error
                log_error_with_context(
                    e,
                    {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                    func_logger,
                )
                raise

        return wrapper

    return decorator


def log_function_calls(logger_name: Optional[str] = None):
    """Decorator to automatically log function calls

    Args:
        logger_name: Optional logger name

    Returns:
        Decorator function
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = structlog.get_logger(logger_name or func.__module__)

            # Log function entry
            log_function_call(func.__name__, args, kwargs, func_logger)

            try:
                # Execute function
                import time

                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log completion
                func_logger.debug(
                    "Function completed successfully",
                    function=func.__name__,
                    duration=round(duration, 4),
                )

                return result

            except Exception as e:
                # Log error
                log_error_with_context(
                    e,
                    {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                    func_logger,
                )
                raise

        return wrapper

    return decorator


# Initialize logging on module import
if autodoc_logger is None:
    try:
        setup_logging()
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
        logging.getLogger(__name__).error(f"Failed to setup advanced logging: {e}")
