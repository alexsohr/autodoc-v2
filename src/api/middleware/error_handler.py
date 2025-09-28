"""Error handling middleware for FastAPI

This module provides comprehensive error handling middleware with
structured error responses, logging, and monitoring integration.
"""

import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

import structlog
from fastapi import HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from ...utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


class ErrorHandlingMiddleware:
    """Error handling middleware for comprehensive error management

    Provides structured error responses, detailed logging, and
    monitoring integration for all API errors.
    """

    def __init__(self):
        """Initialize error handling middleware"""
        self.settings = get_settings()
        self.include_traceback = self.settings.debug

        # Error type mappings
        self.error_mappings = {
            "ValidationError": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "ValueError": status.HTTP_400_BAD_REQUEST,
            "FileNotFoundError": status.HTTP_404_NOT_FOUND,
            "PermissionError": status.HTTP_403_FORBIDDEN,
            "TimeoutError": status.HTTP_408_REQUEST_TIMEOUT,
            "ConnectionError": status.HTTP_503_SERVICE_UNAVAILABLE,
            "NotImplementedError": status.HTTP_501_NOT_IMPLEMENTED,
        }

    async def __call__(self, request: Request, call_next):
        """Process request through error handling middleware

        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint in chain

        Returns:
            Response from next middleware/endpoint or error response
        """
        try:
            # Process request
            response = await call_next(request)
            return response

        except HTTPException as e:
            # Handle FastAPI HTTP exceptions
            return await self._handle_http_exception(request, e)

        except StarletteHTTPException as e:
            # Handle Starlette HTTP exceptions
            return await self._handle_starlette_exception(request, e)

        except RequestValidationError as e:
            # Handle Pydantic validation errors
            return await self._handle_validation_error(request, e)

        except ValidationError as e:
            # Handle Pydantic model validation errors
            return await self._handle_pydantic_validation_error(request, e)

        except Exception as e:
            # Handle all other exceptions
            return await self._handle_general_exception(request, e)

    async def _handle_http_exception(
        self, request: Request, exc: HTTPException
    ) -> JSONResponse:
        """Handle FastAPI HTTP exceptions

        Args:
            request: FastAPI request object
            exc: HTTP exception

        Returns:
            JSON error response
        """
        try:
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

            # Extract error details
            error_detail = exc.detail
            if isinstance(error_detail, dict):
                error_message = error_detail.get("message", str(exc.detail))
                error_type = error_detail.get("error", "HTTPException")
            else:
                error_message = str(error_detail)
                error_type = "HTTPException"

            # Log error
            logger.warning(
                "HTTP exception occurred",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                status_code=exc.status_code,
                error_type=error_type,
                error_message=error_message,
                user_id=getattr(
                    getattr(request.state, "current_user", None), "id", None
                ),
            )

            # Create error response
            error_response = {
                "error": error_type,
                "message": error_message,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
            }

            # Add details if available
            if isinstance(exc.detail, dict) and "details" in exc.detail:
                error_response["details"] = exc.detail["details"]

            return JSONResponse(
                status_code=exc.status_code,
                content=error_response,
                headers=getattr(exc, "headers", None),
            )

        except Exception as e:
            logger.error(f"Error handling HTTP exception: {e}")
            return self._create_fallback_error_response(
                request, 500, "Error handling failed"
            )

    async def _handle_starlette_exception(
        self, request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """Handle Starlette HTTP exceptions

        Args:
            request: FastAPI request object
            exc: Starlette HTTP exception

        Returns:
            JSON error response
        """
        try:
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

            # Log error
            logger.warning(
                "Starlette HTTP exception occurred",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                status_code=exc.status_code,
                error_message=str(exc.detail),
            )

            # Create error response
            error_response = {
                "error": "HTTPException",
                "message": str(exc.detail),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
            }

            return JSONResponse(status_code=exc.status_code, content=error_response)

        except Exception as e:
            logger.error(f"Error handling Starlette exception: {e}")
            return self._create_fallback_error_response(
                request, 500, "Error handling failed"
            )

    async def _handle_validation_error(
        self, request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic request validation errors

        Args:
            request: FastAPI request object
            exc: Request validation error

        Returns:
            JSON error response
        """
        try:
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

            # Extract validation details
            validation_errors = []
            for error in exc.errors():
                validation_errors.append(
                    {
                        "field": " -> ".join(str(loc) for loc in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"],
                    }
                )

            # Log error
            logger.warning(
                "Request validation error",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                validation_errors=validation_errors,
            )

            # Create error response
            error_response = {
                "error": "ValidationError",
                "message": "Request validation failed",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
                "details": {"validation_errors": validation_errors},
            }

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response
            )

        except Exception as e:
            logger.error(f"Error handling validation error: {e}")
            return self._create_fallback_error_response(
                request, 422, "Validation error handling failed"
            )

    async def _handle_pydantic_validation_error(
        self, request: Request, exc: ValidationError
    ) -> JSONResponse:
        """Handle Pydantic model validation errors

        Args:
            request: FastAPI request object
            exc: Pydantic validation error

        Returns:
            JSON error response
        """
        try:
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

            # Extract validation details
            validation_errors = []
            for error in exc.errors():
                validation_errors.append(
                    {
                        "field": " -> ".join(str(loc) for loc in error["loc"]),
                        "message": error["msg"],
                        "type": error["type"],
                    }
                )

            # Log error
            logger.warning(
                "Pydantic validation error",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                validation_errors=validation_errors,
            )

            # Create error response
            error_response = {
                "error": "ModelValidationError",
                "message": "Data model validation failed",
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
                "details": {"validation_errors": validation_errors},
            }

            return JSONResponse(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=error_response
            )

        except Exception as e:
            logger.error(f"Error handling Pydantic validation error: {e}")
            return self._create_fallback_error_response(
                request, 422, "Model validation error handling failed"
            )

    async def _handle_general_exception(
        self, request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle general exceptions

        Args:
            request: FastAPI request object
            exc: General exception

        Returns:
            JSON error response
        """
        try:
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

            # Determine appropriate status code
            exception_type = type(exc).__name__
            status_code = self.error_mappings.get(
                exception_type, status.HTTP_500_INTERNAL_SERVER_ERROR
            )

            # Log error with full details
            logger.error(
                "Unhandled exception occurred",
                correlation_id=correlation_id,
                method=request.method,
                path=request.url.path,
                error_type=exception_type,
                error_message=str(exc),
                traceback=traceback.format_exc() if self.include_traceback else None,
                user_id=getattr(
                    getattr(request.state, "current_user", None), "id", None
                ),
            )

            # Create error response
            error_response = {
                "error": exception_type,
                "message": str(exc),
                "correlation_id": correlation_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
            }

            # Include traceback in debug mode
            if self.include_traceback:
                error_response["details"] = {"traceback": traceback.format_exc()}

            return JSONResponse(status_code=status_code, content=error_response)

        except Exception as e:
            logger.error(f"Error handling general exception: {e}")
            return self._create_fallback_error_response(
                request, 500, "Exception handling failed"
            )

    def _create_fallback_error_response(
        self, request: Request, status_code: int, message: str
    ) -> JSONResponse:
        """Create fallback error response when error handling itself fails

        Args:
            request: FastAPI request object
            status_code: HTTP status code
            message: Error message

        Returns:
            Basic JSON error response
        """
        try:
            correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

            return JSONResponse(
                status_code=status_code,
                content={
                    "error": "InternalServerError",
                    "message": message,
                    "correlation_id": correlation_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        except Exception:
            # Ultimate fallback
            return JSONResponse(
                status_code=500,
                content={
                    "error": "CriticalError",
                    "message": "Critical error in error handling system",
                },
            )


# Custom exception handlers


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Global HTTP exception handler

    Args:
        request: FastAPI request object
        exc: HTTP exception

    Returns:
        JSON error response
    """
    middleware = ErrorHandlingMiddleware()
    return await middleware._handle_http_exception(request, exc)


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Global validation exception handler

    Args:
        request: FastAPI request object
        exc: Validation error

    Returns:
        JSON error response
    """
    middleware = ErrorHandlingMiddleware()
    return await middleware._handle_validation_error(request, exc)


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global general exception handler

    Args:
        request: FastAPI request object
        exc: General exception

    Returns:
        JSON error response
    """
    middleware = ErrorHandlingMiddleware()
    return await middleware._handle_general_exception(request, exc)


# Middleware instance
error_handling_middleware = ErrorHandlingMiddleware()
