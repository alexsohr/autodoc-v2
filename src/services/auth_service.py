"""Authentication service for AutoDoc v2

This module provides authentication and authorization services
including JWT token management, user validation, and security utilities.
"""

import structlog
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from jose import JWTError, jwt
from passlib.context import CryptContext

from ..models.user import Token, TokenData, User, UserCreate, UserLogin, UserUpdate
from ..repository.user_repository import UserRepository
from ..utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


class AuthenticationService:
    """Authentication service for user management and JWT tokens

    Provides comprehensive authentication services including user registration,
    login, JWT token management, and authorization checks.
    """

    def __init__(self, user_repository: UserRepository):
        """Initialize authentication service with dependency injection.
        
        Args:
            user_repository: UserRepository instance (injected via DI).
        """
        self.settings = get_settings()
        self.user_repository = user_repository

        # Password hashing context
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

        # JWT settings
        self.secret_key = self.settings.secret_key
        self.algorithm = self.settings.algorithm
        self.access_token_expire_minutes = self.settings.access_token_expire_minutes

    async def create_user(self, user_data: UserCreate) -> Dict[str, Any]:
        """Create a new user

        Args:
            user_data: User creation data

        Returns:
            Dictionary with creation result
        """
        try:
            # Check if user already exists
            if await self.user_repository.username_exists(user_data.username):
                return {
                    "status": "error",
                    "error": "Username already exists",
                    "error_type": "DuplicateUser",
                }

            # Check if email already exists
            if await self.user_repository.email_exists(user_data.email):
                return {
                    "status": "error",
                    "error": "Email already exists",
                    "error_type": "DuplicateEmail",
                }

            # Hash password
            hashed_password = self.get_password_hash(user_data.password)

            # Create user
            user_doc = await self.user_repository.create_user(
                username=user_data.username,
                email=user_data.email,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
            )

            return {
                "status": "success",
                "user_id": str(user_doc.id),
                "username": user_doc.username,
                "email": user_doc.email,
            }

        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/email and password

        Args:
            username: Username or email
            password: Password

        Returns:
            User object if authentication successful, None otherwise
        """
        try:
            # Get user by username or email
            user_doc = await self.user_repository.get_by_username_or_email(username)
            if not user_doc:
                return None

            # Verify password
            if not self.verify_password(password, user_doc.hashed_password):
                return None

            # Check if user is active
            if not user_doc.is_active:
                return None

            # Update last login
            await self.user_repository.update_last_login(user_doc.id)

            # Convert to User model (without sensitive data)
            return self.user_repository.to_user_model(user_doc)

        except Exception as e:
            logger.error(f"User authentication failed: {e}")
            return None

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username

        Args:
            username: Username

        Returns:
            User model or None
        """
        try:
            user_doc = await self.user_repository.get_by_username(username)
            return self.user_repository.to_user_model(user_doc) if user_doc else None
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email

        Args:
            email: Email address

        Returns:
            User model or None
        """
        try:
            user_doc = await self.user_repository.get_by_email(email)
            return self.user_repository.to_user_model(user_doc) if user_doc else None
        except Exception as e:
            logger.error(f"Failed to get user by email: {e}")
            return None

    async def get_user_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID

        Args:
            user_id: User UUID

        Returns:
            User object or None
        """
        try:
            user_doc = await self.user_repository.get_by_id(user_id)
            return self.user_repository.to_user_model(user_doc) if user_doc else None
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None

    def get_password_hash(self, password: str) -> str:
        """Hash password using bcrypt

        Args:
            password: Plain text password

        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash

        Args:
            plain_password: Plain text password
            hashed_password: Hashed password

        Returns:
            True if password matches, False otherwise
        """
        try:
            return self.pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification failed: {e}")
            return False

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token

        Args:
            data: Token payload data
            expires_delta: Token expiration time (defaults to configured value)

        Returns:
            JWT token string
        """
        try:
            to_encode = data.copy()

            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(
                    minutes=self.access_token_expire_minutes
                )

            to_encode.update({"exp": expire})

            encoded_jwt = jwt.encode(
                to_encode, self.secret_key, algorithm=self.algorithm
            )
            return encoded_jwt

        except Exception as e:
            logger.error(f"Token creation failed: {e}")
            raise ValueError("Failed to create access token")

    def verify_token(self, token: str) -> Optional[TokenData]:
        """Verify and decode JWT token

        Args:
            token: JWT token string

        Returns:
            TokenData object if valid, None otherwise
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])

            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            scopes: List[str] = payload.get("scopes", [])

            if username is None and user_id is None:
                return None

            return TokenData(username=username, user_id=user_id, scopes=scopes)

        except JWTError as e:
            logger.debug(f"Token verification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return None

    async def login_user(self, login_data: UserLogin) -> Dict[str, Any]:
        """Login user and generate access token

        Args:
            login_data: User login credentials

        Returns:
            Dictionary with login result and token
        """
        try:
            # Authenticate user
            user = await self.authenticate_user(
                login_data.username, login_data.password
            )
            if not user:
                return {
                    "status": "error",
                    "error": "Invalid username or password",
                    "error_type": "AuthenticationFailed",
                }

            # Create access token
            token_data = {
                "sub": user.username,
                "user_id": str(user.id),
                "scopes": ["read", "write"] if user.is_admin else ["read"],
            }

            access_token = self.create_access_token(token_data)

            return {
                "status": "success",
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60,
                "user": {
                    "id": str(user.id),
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "is_admin": user.is_admin,
                },
            }

        except Exception as e:
            logger.error(f"User login failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def get_current_user(self, token: str) -> Optional[User]:
        """Get current user from JWT token

        Args:
            token: JWT token string

        Returns:
            User object if token is valid, None otherwise
        """
        try:
            # Verify token
            token_data = self.verify_token(token)
            if not token_data:
                return None

            # Get user by ID or username
            if token_data.user_id:
                user = await self.get_user_by_id(UUID(token_data.user_id))
            elif token_data.username:
                user = await self.get_user_by_username(token_data.username)
            else:
                return None

            return user

        except Exception as e:
            logger.error(f"Get current user failed: {e}")
            return None

    def check_permissions(self, user: User, required_scopes: List[str]) -> bool:
        """Check if user has required permissions

        Args:
            user: User object
            required_scopes: List of required scopes

        Returns:
            True if user has all required scopes
        """
        try:
            # Admin users have all permissions
            if user.is_admin:
                return True

            # Check specific scopes (simplified for now)
            user_scopes = ["read"]  # Default scope for regular users
            if user.is_admin:
                user_scopes.extend(["write", "admin"])

            return all(scope in user_scopes for scope in required_scopes)

        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False

    async def refresh_token(self, token: str) -> Dict[str, Any]:
        """Refresh JWT token

        Args:
            token: Current JWT token

        Returns:
            Dictionary with new token or error
        """
        try:
            # Verify current token
            token_data = self.verify_token(token)
            if not token_data:
                return {
                    "status": "error",
                    "error": "Invalid token",
                    "error_type": "InvalidToken",
                }

            # Get user
            user = await self.get_current_user(token)
            if not user:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound",
                }

            # Create new token
            new_token_data = {
                "sub": user.username,
                "user_id": str(user.id),
                "scopes": ["read", "write"] if user.is_admin else ["read"],
            }

            new_access_token = self.create_access_token(new_token_data)

            return {
                "status": "success",
                "access_token": new_access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60,
            }

        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def logout_user(self, token: str) -> Dict[str, Any]:
        """Logout user (invalidate token)

        Args:
            token: JWT token to invalidate

        Returns:
            Dictionary with logout result
        """
        try:
            # In a production system, you would add the token to a blacklist
            # For now, we'll just return success since JWT tokens are stateless

            token_data = self.verify_token(token)
            if not token_data:
                return {
                    "status": "error",
                    "error": "Invalid token",
                    "error_type": "InvalidToken",
                }

            # TODO: Add token to blacklist in Redis or database
            # For now, just log the logout
            logger.info(f"User logged out: {token_data.username}")

            return {"status": "success", "message": "Successfully logged out"}

        except Exception as e:
            logger.error(f"User logout failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def change_password(
        self, user_id: UUID, current_password: str, new_password: str
    ) -> Dict[str, Any]:
        """Change user password

        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password

        Returns:
            Dictionary with change result
        """
        try:
            # Get user document (with password)
            user_doc = await self.user_repository.get_by_id(user_id)
            if not user_doc:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound",
                }

            # Verify current password
            if not self.verify_password(current_password, user_doc.hashed_password):
                return {
                    "status": "error",
                    "error": "Current password is incorrect",
                    "error_type": "InvalidPassword",
                }

            # Hash new password
            new_hashed_password = self.get_password_hash(new_password)

            # Update password
            success = await self.user_repository.update_password(
                user_id, new_hashed_password
            )

            if success:
                return {"status": "success", "message": "Password changed successfully"}
            else:
                return {
                    "status": "error",
                    "error": "Failed to update password",
                    "error_type": "UpdateFailed",
                }

        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def update_user_profile(
        self, user_id: UUID, updates: UserUpdate
    ) -> Dict[str, Any]:
        """Update user profile

        Args:
            user_id: User ID
            updates: Fields to update

        Returns:
            Dictionary with update result
        """
        try:
            # Convert to dict and remove None values
            update_dict = updates.model_dump(exclude_none=True)

            if not update_dict:
                return {
                    "status": "error",
                    "error": "No valid updates provided",
                    "error_type": "InvalidInput",
                }

            # Check if email already exists (if updating email)
            if "email" in update_dict:
                if await self.user_repository.email_exists(update_dict["email"]):
                    # Check if it's not the same user
                    existing_user = await self.user_repository.get_by_email(
                        update_dict["email"]
                    )
                    if existing_user and existing_user.id != user_id:
                        return {
                            "status": "error",
                            "error": "Email already exists",
                            "error_type": "DuplicateEmail",
                        }

            # Update user
            success = await self.user_repository.update_profile(user_id, update_dict)

            if success:
                return {"status": "success", "message": "Profile updated successfully"}
            else:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound",
                }

        except Exception as e:
            logger.error(f"Profile update failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def list_users(
        self, limit: int = 50, offset: int = 0, admin_only: bool = False
    ) -> Dict[str, Any]:
        """List users (admin only)

        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            admin_only: Only return admin users

        Returns:
            Dictionary with user list
        """
        try:
            # Get users
            user_docs = await self.user_repository.list_users(
                limit=limit, offset=offset, admin_only=admin_only
            )

            # Convert to User models
            users = self.user_repository.to_user_models(user_docs)

            # Get total count
            total_count = await self.user_repository.count_users(admin_only=admin_only)

            return {
                "status": "success",
                "users": [user.model_dump() for user in users],
                "total": total_count,
                "limit": limit,
                "offset": offset,
            }

        except Exception as e:
            logger.error(f"List users failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "users": [],
            }

    async def delete_user(self, user_id: UUID) -> Dict[str, Any]:
        """Delete user (admin only)

        Args:
            user_id: User ID to delete

        Returns:
            Dictionary with deletion result
        """
        try:
            # Check if user exists
            user_doc = await self.user_repository.get_by_id(user_id)
            if not user_doc:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound",
                }

            # Delete user
            success = await self.user_repository.delete_user(user_id)

            if success:
                return {"status": "success", "message": "User deleted successfully"}
            else:
                return {
                    "status": "error",
                    "error": "Failed to delete user",
                    "error_type": "DeletionFailed",
                }

        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on authentication service

        Returns:
            Dictionary with health check results
        """
        try:
            # Test password hashing
            test_hash = self.get_password_hash("test")
            hash_verify = self.verify_password("test", test_hash)

            # Test JWT operations
            test_token = self.create_access_token({"sub": "test", "user_id": "test"})
            token_verify = self.verify_token(test_token)

            # Test repository connection (basic count)
            user_count = await self.user_repository.count_users(active_only=False)

            return {
                "status": "healthy",
                "password_hashing": "working" if hash_verify else "failed",
                "jwt_operations": "working" if token_verify else "failed",
                "repository_connection": "working",
                "user_count": user_count,
                "token_expire_minutes": self.access_token_expire_minutes,
            }

        except Exception as e:
            logger.error(f"Auth service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__,
            }


# Deprecated: Use get_auth_service() from src.dependencies instead
# This singleton is kept for backward compatibility only
# auth_service = AuthenticationService()  # REMOVED - use dependency injection