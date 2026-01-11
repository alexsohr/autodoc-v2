"""Request logging middleware for FastAPI

This module provides comprehensive request logging middleware with
structured logging, correlation IDs, and performance monitoring.
"""

import structlog
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.dependencies import get_auth_service
import structlog
from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...models.user import User
from ...services.auth_service import AuthenticationService
from ...utils.config_loader import get_settings

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class RequestLoggingMiddleware:
    """Request logging middleware for comprehensive API monitoring

    Provides structured logging with correlation IDs, performance metrics,
    and detailed request/response tracking.
    """

    def __init__(self):
        """Initialize request logging middleware"""
        self.settings = get_settings()
        self.log_level = self.settings.log_level.upper()

        # Paths to exclude from detailed logging
        self.exclude_paths = {
            "/health",
            "/health/",
            "/health/ready",
            "/health/live",
            "/metrics",
        }

        # Sensitive headers to mask in logs
        self.sensitive_headers = {
            "authorization",
            "x-api-key",
            "x-hub-signature-256",
            "cookie",
        }

    async def __call__(self, request: Request, call_next):
        """Process request through logging middleware

        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint in chain

        Returns:
            Response from next middleware/endpoint
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())

        # Add correlation ID to request state
        request.state.correlation_id = correlation_id

        # Start timing
        start_time = time.time()
        request_timestamp = datetime.now(timezone.utc)

        # Extract request information
        request_info = self._extract_request_info(request, correlation_id)

        # Log request (if not excluded)
        if not self._should_exclude_path(request.url.path):
            logger.info(
                "HTTP request started",
                **request_info,
                timestamp=request_timestamp.isoformat(),
            )

        try:
            # Process request
            response = await call_next(request)

            # Calculate response time
            response_time = time.time() - start_time

            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id

            # Extract response information
            response_info = self._extract_response_info(response, response_time)

            # Log response (if not excluded)
            if not self._should_exclude_path(request.url.path):
                logger.info(
                    "HTTP request completed",
                    **request_info,
                    **response_info,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            # Log performance metrics for slow requests
            if response_time > 1.0:  # Log slow requests (>1 second)
                logger.warning(
                    "Slow request detected",
                    **request_info,
                    **response_info,
                    performance_warning=True,
                )

            return response

        except Exception as e:
            # Calculate error response time
            error_response_time = time.time() - start_time

            # Log error
            logger.error(
                "HTTP request failed",
                **request_info,
                error=str(e),
                error_type=type(e).__name__,
                response_time=error_response_time,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            # Re-raise the exception
            raise

    def _extract_request_info(
        self, request: Request, correlation_id: str
    ) -> Dict[str, Any]:
        """Extract request information for logging

        Args:
            request: FastAPI request object
            correlation_id: Correlation ID

        Returns:
            Dictionary with request information
        """
        try:
            # Get user information if available
            user_info = {}
            if hasattr(request.state, "current_user"):
                user = request.state.current_user
                user_info = {
                    "user_id": str(user.id),
                    "username": user.username,
                    "is_admin": user.is_admin,
                }

            # Mask sensitive headers
            headers = {}
            for key, value in request.headers.items():
                if key.lower() in self.sensitive_headers:
                    headers[key] = "***MASKED***"
                else:
                    headers[key] = value

            return {
                "correlation_id": correlation_id,
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "headers": headers,
                "client_ip": self._get_client_ip(request),
                "user_agent": request.headers.get("user-agent", ""),
                **user_info,
            }

        except Exception as e:
            logger.debug(f"Request info extraction failed: {e}")
            return {
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "error": "Failed to extract request info",
            }

    def _extract_response_info(
        self, response: Response, response_time: float
    ) -> Dict[str, Any]:
        """Extract response information for logging

        Args:
            response: FastAPI response object
            response_time: Response time in seconds

        Returns:
            Dictionary with response information
        """
        try:
            # Get response size
            response_size = 0
            if hasattr(response, "body"):
                response_size = len(response.body) if response.body else 0

            return {
                "status_code": response.status_code,
                "response_time": round(response_time, 4),
                "response_size": response_size,
                "content_type": response.headers.get("content-type", ""),
                "response_headers": dict(response.headers),
            }

        except Exception as e:
            logger.debug(f"Response info extraction failed: {e}")
            return {
                "status_code": getattr(response, "status_code", 0),
                "response_time": round(response_time, 4),
                "error": "Failed to extract response info",
            }

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address from request

        Args:
            request: FastAPI request object

        Returns:
            Client IP address
        """
        try:
            # Check for forwarded headers (for load balancers/proxies)
            forwarded_for = request.headers.get("x-forwarded-for")
            if forwarded_for:
                # Take the first IP in the chain
                return forwarded_for.split(",")[0].strip()

            # Check for real IP header
            real_ip = request.headers.get("x-real-ip")
            if real_ip:
                return real_ip.strip()

            # Fall back to client host
            if request.client:
                return request.client.host

            return "unknown"

        except Exception:
            return "unknown"

    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from detailed logging

        Args:
            path: Request path

        Returns:
            True if path should be excluded
        """
        normalized_path = path.rstrip("/")
        return normalized_path in self.exclude_paths or path in self.exclude_paths


# Dependency functions


async def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request state

    Args:
        request: FastAPI request object

    Returns:
        Correlation ID
    """
    return getattr(request.state, "correlation_id", str(uuid.uuid4()))


async def get_authenticated_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """Get authenticated user with proper error handling

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Authenticated user object

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from auth service
    auth_service = get_auth_service()
    user = await auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive"
        )

    return user


async def get_admin_user(current_user: User = Depends(get_authenticated_user)) -> User:
    """Get authenticated admin user

    Args:
        current_user: Current authenticated user

    Returns:
        Admin user object

    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator privileges required",
        )

    return current_user


def require_permissions(required_scopes: List[str]):
    """Create dependency that requires specific permissions

    Args:
        required_scopes: List of required permission scopes

    Returns:
        Dependency function
    """

    async def check_permissions(
        current_user: User = Depends(get_authenticated_user),
    ) -> User:
        """Check if user has required permissions

        Args:
            current_user: Current authenticated user

        Returns:
            User if authorized

        Raises:
            HTTPException: If user lacks required permissions
        """
        auth_service = get_auth_service()
        if not auth_service.check_permissions(current_user, required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": f"Required permissions: {', '.join(required_scopes)}",
                },
            )

        return current_user

    return check_permissions


# Middleware instance
request_logging_middleware = RequestLoggingMiddleware()
