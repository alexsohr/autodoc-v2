"""Authentication middleware for FastAPI

This module provides authentication middleware for JWT token validation
and user context management across API requests.
"""

import structlog
from typing import List, Optional
from uuid import UUID

from dependencies import get_auth_service
from fastapi import Depends, HTTPException, status
from fastapi.requests import Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ...models.user import TokenData, User
from ...services.auth_service import AuthenticationService
from ...utils.config_loader import get_settings

logger = structlog.get_logger(__name__)

# Security scheme for JWT tokens
security = HTTPBearer(auto_error=False)


class AuthMiddleware:
    """Authentication middleware for FastAPI

    Provides JWT token validation, user context management,
    and permission checking for API endpoints.
    """

    def __init__(self, auth_service: AuthenticationService = None):
        """Initialize authentication middleware with dependency injection.
        
        Args:
            auth_service: AuthenticationService instance. If None, creates a new instance
                         (for backward compatibility).
        """
        self.settings = get_settings()
        self.auth_service = auth_service if auth_service is not None else AuthenticationService()
        self.public_paths = {
            "/",
            "/health",
            "/health/",
            "/health/ready",
            "/health/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/webhooks/github",
            "/webhooks/bitbucket",
        }

    async def __call__(self, request: Request, call_next):
        """Process request through authentication middleware

        Args:
            request: FastAPI request object
            call_next: Next middleware/endpoint in chain

        Returns:
            Response from next middleware/endpoint
        """
        try:
            # Skip authentication for public paths
            if self._is_public_path(request.url.path):
                return await call_next(request)

            # Extract and validate token
            token = self._extract_token_from_request(request)
            if not token:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Validate token and get user
            user = await self.auth_service.get_current_user(token)
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Check if user is active
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User account is inactive",
                )

            # Add user to request state
            request.state.current_user = user
            request.state.token = token

            # Process request
            response = await call_next(request)

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication processing failed",
            )

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public (no authentication required)

        Args:
            path: Request path

        Returns:
            True if path is public
        """
        # Remove trailing slash for comparison
        normalized_path = path.rstrip("/")

        # Check exact matches
        if normalized_path in self.public_paths or path in self.public_paths:
            return True

        # Check path prefixes
        public_prefixes = ["/health", "/docs", "/redoc", "/webhooks"]
        for prefix in public_prefixes:
            if path.startswith(prefix):
                return True

        return False

    def _extract_token_from_request(self, request: Request) -> Optional[str]:
        """Extract JWT token from request

        Args:
            request: FastAPI request object

        Returns:
            JWT token string or None
        """
        try:
            # Check Authorization header
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                return auth_header[7:]  # Remove "Bearer " prefix

            # Check query parameter (for SSE and other special cases)
            token_param = request.query_params.get("token")
            if token_param:
                return token_param

            return None

        except Exception as e:
            logger.debug(f"Token extraction failed: {e}")
            return None


# Dependency functions for FastAPI


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """Get current authenticated user from JWT token

    Args:
        credentials: HTTP authorization credentials

    Returns:
        Current user object

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token and get user
    # Note: This function should be converted to accept auth_service as dependency
    auth_service = get_auth_service()
    user = await auth_service.get_current_user(credentials.credentials)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is inactive"
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Get current active user (alias for get_current_user)

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Current active user
    """
    return current_user


async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current user with admin privileges

    Args:
        current_user: Current user from get_current_user dependency

    Returns:
        Current user if admin

    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin privileges required"
        )

    return current_user


def require_scopes(required_scopes: List[str]):
    """Dependency factory for scope-based authorization

    Args:
        required_scopes: List of required scopes

    Returns:
        Dependency function that checks scopes
    """

    async def check_scopes(current_user: User = Depends(get_current_user)) -> User:
        """Check if user has required scopes

        Args:
            current_user: Current user

        Returns:
            Current user if authorized

        Raises:
            HTTPException: If user lacks required scopes
        """
        auth_service = get_auth_service()
        if not auth_service.check_permissions(current_user, required_scopes):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scopes: {required_scopes}",
            )

        return current_user

    return check_scopes
