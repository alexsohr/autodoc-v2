"""Retry logic utilities with exponential backoff

This module provides retry decorators and utilities using the tenacity library
for robust error handling and recovery in external service calls.

Also includes LangGraph RetryPolicy factories for agent node retry configuration.
"""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from tenacity import (
    after_log,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)
from tenacity.asyncio import AsyncRetrying

logger = logging.getLogger(__name__)


class RetryConfig:
    """Configuration for retry behavior

    Provides predefined retry configurations for different types of operations.
    """

    # Database operations
    DATABASE_RETRY = {
        "stop": stop_after_attempt(3),
        "wait": wait_exponential(multiplier=1, min=1, max=10),
        "retry": retry_if_exception_type((ConnectionError, TimeoutError)),
        "before_sleep": before_sleep_log(logger, logging.WARNING),
        "after": after_log(logger, logging.INFO),
    }

    # External API calls
    API_RETRY = {
        "stop": stop_after_attempt(5),
        "wait": wait_exponential(multiplier=2, min=2, max=60),
        "retry": retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        "before_sleep": before_sleep_log(logger, logging.WARNING),
        "after": after_log(logger, logging.INFO),
    }

    # LLM provider calls
    LLM_RETRY = {
        "stop": stop_after_attempt(3),
        "wait": wait_exponential(multiplier=2, min=5, max=30),
        "retry": retry_if_exception_type((ConnectionError, TimeoutError)),
        "before_sleep": before_sleep_log(logger, logging.WARNING),
        "after": after_log(logger, logging.INFO),
    }

    # File operations
    FILE_RETRY = {
        "stop": stop_after_attempt(3),
        "wait": wait_fixed(1),
        "retry": retry_if_exception_type((OSError, IOError)),
        "before_sleep": before_sleep_log(logger, logging.WARNING),
        "after": after_log(logger, logging.INFO),
    }

    # Git operations
    GIT_RETRY = {
        "stop": stop_after_attempt(3),
        "wait": wait_exponential(multiplier=1, min=2, max=15),
        "retry": retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        "before_sleep": before_sleep_log(logger, logging.WARNING),
        "after": after_log(logger, logging.INFO),
    }

    # Webhook processing
    WEBHOOK_RETRY = {
        "stop": stop_after_attempt(2),
        "wait": wait_fixed(2),
        "retry": retry_if_exception_type((ConnectionError, TimeoutError)),
        "before_sleep": before_sleep_log(logger, logging.WARNING),
        "after": after_log(logger, logging.INFO),
    }


def retry_on_failure(
    retry_config: Optional[Dict[str, Any]] = None,
    custom_exceptions: Optional[List[Type[Exception]]] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
):
    """Decorator for adding retry logic to functions

    Args:
        retry_config: Custom retry configuration (uses API_RETRY if None)
        custom_exceptions: Custom exception types to retry on
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff base

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        # Use provided config or default
        config = retry_config or RetryConfig.API_RETRY.copy()

        # Override with custom parameters if provided
        if custom_exceptions:
            config["retry"] = retry_if_exception_type(tuple(custom_exceptions))
        if max_attempts != 3:
            config["stop"] = stop_after_attempt(max_attempts)
        if base_delay != 1.0 or max_delay != 60.0 or exponential_base != 2.0:
            config["wait"] = wait_exponential(
                multiplier=exponential_base, min=base_delay, max=max_delay
            )

        # Apply retry decorator
        return retry(**config)(func)

    return decorator


def retry_async_on_failure(
    retry_config: Optional[Dict[str, Any]] = None,
    custom_exceptions: Optional[List[Type[Exception]]] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
):
    """Decorator for adding retry logic to async functions

    Args:
        retry_config: Custom retry configuration (uses API_RETRY if None)
        custom_exceptions: Custom exception types to retry on
        max_attempts: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Exponential backoff base

    Returns:
        Decorator function for async functions
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Use provided config or default
            config = retry_config or RetryConfig.API_RETRY.copy()

            # Override with custom parameters if provided
            if custom_exceptions:
                config["retry"] = retry_if_exception_type(tuple(custom_exceptions))
            if max_attempts != 3:
                config["stop"] = stop_after_attempt(max_attempts)
            if base_delay != 1.0 or max_delay != 60.0 or exponential_base != 2.0:
                config["wait"] = wait_exponential(
                    multiplier=exponential_base, min=base_delay, max=max_delay
                )

            # Execute with retry
            async for attempt in AsyncRetrying(**config):
                with attempt:
                    return await func(*args, **kwargs)

        return wrapper

    return decorator


# Predefined retry decorators for common use cases


def retry_database_operation(func: Callable) -> Callable:
    """Retry decorator for database operations

    Args:
        func: Function to decorate

    Returns:
        Decorated function with database retry logic
    """
    return retry(**RetryConfig.DATABASE_RETRY)(func)


def retry_api_call(func: Callable) -> Callable:
    """Retry decorator for external API calls

    Args:
        func: Function to decorate

    Returns:
        Decorated function with API retry logic
    """
    return retry(**RetryConfig.API_RETRY)(func)


def retry_llm_call(func: Callable) -> Callable:
    """Retry decorator for LLM provider calls

    Args:
        func: Function to decorate

    Returns:
        Decorated function with LLM retry logic
    """
    return retry(**RetryConfig.LLM_RETRY)(func)


def retry_file_operation(func: Callable) -> Callable:
    """Retry decorator for file operations

    Args:
        func: Function to decorate

    Returns:
        Decorated function with file operation retry logic
    """
    return retry(**RetryConfig.FILE_RETRY)(func)


def retry_git_operation(func: Callable) -> Callable:
    """Retry decorator for Git operations

    Args:
        func: Function to decorate

    Returns:
        Decorated function with Git operation retry logic
    """
    return retry(**RetryConfig.GIT_RETRY)(func)


# Async versions


def retry_async_database_operation(func: Callable) -> Callable:
    """Async retry decorator for database operations"""
    return retry_async_on_failure(RetryConfig.DATABASE_RETRY)(func)


def retry_async_api_call(func: Callable) -> Callable:
    """Async retry decorator for external API calls"""
    return retry_async_on_failure(RetryConfig.API_RETRY)(func)


def retry_async_llm_call(func: Callable) -> Callable:
    """Async retry decorator for LLM provider calls"""
    return retry_async_on_failure(RetryConfig.LLM_RETRY)(func)


def retry_async_file_operation(func: Callable) -> Callable:
    """Async retry decorator for file operations"""
    return retry_async_on_failure(RetryConfig.FILE_RETRY)(func)


def retry_async_git_operation(func: Callable) -> Callable:
    """Async retry decorator for Git operations"""
    return retry_async_on_failure(RetryConfig.GIT_RETRY)(func)


# Circuit breaker pattern


class CircuitBreaker:
    """Circuit breaker for preventing cascading failures

    Implements the circuit breaker pattern to protect against
    cascading failures in external service calls.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
    ):
        """Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before attempting recovery
            expected_exception: Exception type that triggers circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function through circuit breaker

        Args:
            func: Function to call
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Function result

        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == "open":
            if self._should_attempt_reset():
                self.state = "half_open"
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            # Call function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            # Success - reset failure count
            self._on_success()
            return result

        except self.expected_exception as e:
            # Handle expected failures
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return True

        import time

        return time.time() - self.last_failure_time >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call"""
        self.failure_count = 0
        self.state = "closed"

    def _on_failure(self):
        """Handle failed call"""
        import time

        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )


# Utility functions for common retry patterns


async def retry_with_backoff(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Optional[List[Type[Exception]]] = None,
) -> Any:
    """Retry function with exponential backoff

    Args:
        func: Function to retry
        args: Function arguments
        kwargs: Function keyword arguments
        max_attempts: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exception types to retry on

    Returns:
        Function result

    Raises:
        Exception: If all retry attempts fail
    """
    kwargs = kwargs or {}
    exceptions = exceptions or [Exception]

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=base_delay, max=max_delay),
        retry=retry_if_exception_type(tuple(exceptions)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    ):
        with attempt:
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                return func(*args, **kwargs)


def create_retry_session(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: Optional[List[Type[Exception]]] = None,
) -> AsyncRetrying:
    """Create a reusable retry session

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Exception types to retry on

    Returns:
        AsyncRetrying instance
    """
    exceptions = exceptions or [ConnectionError, TimeoutError]

    return AsyncRetrying(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=2, min=base_delay, max=max_delay),
        retry=retry_if_exception_type(tuple(exceptions)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        after=after_log(logger, logging.INFO),
    )


# Context manager for retry operations


class RetryContext:
    """Context manager for retry operations

    Provides a context manager interface for retry logic
    with automatic resource cleanup and error handling.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exceptions: Optional[List[Type[Exception]]] = None,
    ):
        """Initialize retry context

        Args:
            max_attempts: Maximum retry attempts
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds
            exceptions: Exception types to retry on
        """
        self.retry_session = create_retry_session(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            exceptions=exceptions,
        )
        self.attempt_count = 0
        self.last_exception = None

    async def __aenter__(self):
        """Enter retry context"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit retry context"""
        if exc_type is not None:
            self.last_exception = exc_val
        return False  # Don't suppress exceptions

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic

        Args:
            func: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            Function result
        """
        async for attempt in self.retry_session:
            with attempt:
                self.attempt_count = attempt.retry_state.attempt_number

                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)


# Specialized retry utilities


async def retry_mongodb_operation(func: Callable, *args, **kwargs) -> Any:
    """Retry MongoDB operation with appropriate backoff

    Args:
        func: MongoDB operation function
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        Operation result
    """
    return await retry_with_backoff(
        func=func,
        args=args,
        kwargs=kwargs,
        max_attempts=3,
        base_delay=1.0,
        max_delay=10.0,
        exceptions=[ConnectionError, TimeoutError],
    )


async def retry_llm_request(func: Callable, *args, **kwargs) -> Any:
    """Retry LLM provider request with appropriate backoff

    Args:
        func: LLM request function
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        LLM response
    """
    return await retry_with_backoff(
        func=func,
        args=args,
        kwargs=kwargs,
        max_attempts=3,
        base_delay=5.0,
        max_delay=30.0,
        exceptions=[ConnectionError, TimeoutError, OSError],
    )


async def retry_git_operation(func: Callable, *args, **kwargs) -> Any:
    """Retry Git operation with appropriate backoff

    Args:
        func: Git operation function
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        Git operation result
    """
    return await retry_with_backoff(
        func=func,
        args=args,
        kwargs=kwargs,
        max_attempts=3,
        base_delay=2.0,
        max_delay=15.0,
        exceptions=[ConnectionError, TimeoutError, OSError],
    )


async def retry_webhook_processing(func: Callable, *args, **kwargs) -> Any:
    """Retry webhook processing with minimal backoff

    Args:
        func: Webhook processing function
        args: Function arguments
        kwargs: Function keyword arguments

    Returns:
        Processing result
    """
    return await retry_with_backoff(
        func=func,
        args=args,
        kwargs=kwargs,
        max_attempts=2,
        base_delay=2.0,
        max_delay=5.0,
        exceptions=[ConnectionError, TimeoutError],
    )


# LangGraph RetryPolicy Factories
# These create retry policies for use with LangGraph nodes


def get_langgraph_retry_policy(
    initial_interval: float = 1.0,
    max_interval: float = 10.0,
    max_attempts: int = 3,
    retry_on: Optional[Tuple[Type[Exception], ...]] = None,
) -> Dict[str, Any]:
    """Create a LangGraph-compatible retry policy configuration.

    LangGraph RetryPolicy is used for configuring retry behavior on graph nodes.
    This function creates a dictionary that can be passed to RetryPolicy constructor.

    Args:
        initial_interval: Initial delay between retries in seconds
        max_interval: Maximum delay between retries in seconds
        max_attempts: Maximum number of retry attempts
        retry_on: Tuple of exception types to retry on

    Returns:
        Dictionary with RetryPolicy configuration

    Example:
        >>> from langgraph.types import RetryPolicy
        >>> config = get_langgraph_retry_policy(max_attempts=5)
        >>> policy = RetryPolicy(**config)
    """
    if retry_on is None:
        retry_on = (ValueError, TimeoutError, ConnectionError, Exception)

    return {
        "initial_interval": initial_interval,
        "max_interval": max_interval,
        "max_attempts": max_attempts,
        "retry_on": retry_on,
    }


def get_page_worker_retry_policy() -> Dict[str, Any]:
    """Get retry policy for wiki page generation workers.

    Optimized for LLM-based page generation with:
    - Moderate initial delay to handle rate limits
    - Maximum 3 attempts to prevent infinite loops
    - Retries on common transient errors

    Returns:
        Dictionary with RetryPolicy configuration for page workers

    Example:
        >>> from langgraph.types import RetryPolicy
        >>> config = get_page_worker_retry_policy()
        >>> policy = RetryPolicy(**config)
    """
    return get_langgraph_retry_policy(
        initial_interval=2.0,
        max_interval=30.0,
        max_attempts=3,
        retry_on=(ValueError, TimeoutError, ConnectionError, OSError, Exception),
    )


def get_llm_node_retry_policy() -> Dict[str, Any]:
    """Get retry policy for LLM interaction nodes.

    Optimized for LLM API calls with:
    - Longer initial delay for rate limit recovery
    - Fewer attempts to avoid excessive API costs
    - Focus on transient network errors

    Returns:
        Dictionary with RetryPolicy configuration for LLM nodes
    """
    return get_langgraph_retry_policy(
        initial_interval=5.0,
        max_interval=60.0,
        max_attempts=3,
        retry_on=(TimeoutError, ConnectionError, OSError),
    )


def get_database_node_retry_policy() -> Dict[str, Any]:
    """Get retry policy for database operation nodes.

    Optimized for MongoDB operations with:
    - Short initial delay for quick recovery
    - More attempts for transient connection issues

    Returns:
        Dictionary with RetryPolicy configuration for database nodes
    """
    return get_langgraph_retry_policy(
        initial_interval=1.0,
        max_interval=10.0,
        max_attempts=5,
        retry_on=(ConnectionError, TimeoutError),
    )


def get_file_operation_retry_policy() -> Dict[str, Any]:
    """Get retry policy for file system operation nodes.

    Optimized for file I/O with:
    - Short delays for file locking issues
    - Fewer attempts as file errors are often persistent

    Returns:
        Dictionary with RetryPolicy configuration for file operations
    """
    return get_langgraph_retry_policy(
        initial_interval=0.5,
        max_interval=5.0,
        max_attempts=3,
        retry_on=(OSError, IOError, PermissionError),
    )


# Batch retry utilities


async def retry_batch_operation(
    items: List[Any],
    func: Callable,
    batch_size: int = 10,
    max_attempts: int = 3,
    continue_on_failure: bool = True,
) -> Dict[str, Any]:
    """Retry batch operations with individual item retry

    Args:
        items: List of items to process
        func: Function to apply to each item
        batch_size: Number of items to process in each batch
        max_attempts: Maximum retry attempts per item
        continue_on_failure: Continue processing if individual items fail

    Returns:
        Dictionary with batch processing results
    """
    successful_items = []
    failed_items = []

    # Process items in batches
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]

        # Process each item in the batch
        batch_tasks = []
        for item in batch:

            async def process_item(item_data):
                try:
                    return await retry_with_backoff(
                        func=func, args=(item_data,), max_attempts=max_attempts
                    )
                except Exception as e:
                    if continue_on_failure:
                        logger.warning(f"Batch item processing failed: {e}")
                        return {"error": str(e), "item": item_data}
                    else:
                        raise

            batch_tasks.append(process_item(item))

        # Execute batch
        try:
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            for item, result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    failed_items.append({"item": item, "error": str(result)})
                elif isinstance(result, dict) and "error" in result:
                    failed_items.append(result)
                else:
                    successful_items.append({"item": item, "result": result})

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            if not continue_on_failure:
                raise

    return {
        "total_items": len(items),
        "successful_count": len(successful_items),
        "failed_count": len(failed_items),
        "successful_items": successful_items,
        "failed_items": failed_items,
        "success_rate": len(successful_items) / len(items) if items else 0.0,
    }
