"""Authentication service for AutoDoc v2

This module provides authentication and authorization services
including JWT token management, user validation, and security utilities.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from ..utils.config_loader import get_settings
from ..utils.mongodb_adapter import get_mongodb_adapter

logger = logging.getLogger(__name__)


class User(BaseModel):
    """User model for authentication"""
    id: UUID = Field(default_factory=uuid4, description="User ID")
    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    full_name: Optional[str] = Field(default=None, description="Full name")
    is_active: bool = Field(default=True, description="User is active")
    is_admin: bool = Field(default=False, description="User has admin privileges")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = Field(default=None, description="Last login timestamp")


class UserCreate(BaseModel):
    """User creation model"""
    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    password: str = Field(description="Password")
    full_name: Optional[str] = Field(default=None, description="Full name")


class UserLogin(BaseModel):
    """User login model"""
    username: str = Field(description="Username or email")
    password: str = Field(description="Password")


class Token(BaseModel):
    """JWT token model"""
    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")


class TokenData(BaseModel):
    """Token payload data"""
    username: Optional[str] = Field(default=None, description="Username")
    user_id: Optional[str] = Field(default=None, description="User ID")
    scopes: List[str] = Field(default_factory=list, description="Token scopes")


class AuthenticationService:
    """Authentication service for user management and JWT tokens
    
    Provides comprehensive authentication services including user registration,
    login, JWT token management, and authorization checks.
    """
    
    def __init__(self):
        """Initialize authentication service"""
        self.settings = get_settings()
        
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
            existing_user = await self.get_user_by_username(user_data.username)
            if existing_user:
                return {
                    "status": "error",
                    "error": "Username already exists",
                    "error_type": "DuplicateUser"
                }
            
            # Check if email already exists
            existing_email = await self.get_user_by_email(user_data.email)
            if existing_email:
                return {
                    "status": "error",
                    "error": "Email already exists",
                    "error_type": "DuplicateEmail"
                }
            
            # Hash password
            hashed_password = self.get_password_hash(user_data.password)
            
            # Create user
            user = User(
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name
            )
            
            # Store user in database
            mongodb = await get_mongodb_adapter()
            user_dict = user.model_dump()
            user_dict["id"] = str(user.id)
            user_dict["hashed_password"] = hashed_password
            
            await mongodb.insert_document("users", user_dict)
            
            return {
                "status": "success",
                "user_id": str(user.id),
                "username": user.username,
                "email": user.email
            }
            
        except Exception as e:
            logger.error(f"User creation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
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
            user_data = await self.get_user_by_username(username)
            if not user_data:
                user_data = await self.get_user_by_email(username)
            
            if not user_data:
                return None
            
            # Verify password
            hashed_password = user_data.get("hashed_password", "")
            if not self.verify_password(password, hashed_password):
                return None
            
            # Check if user is active
            if not user_data.get("is_active", True):
                return None
            
            # Update last login
            await self.update_last_login(user_data["id"])
            
            # Convert to User model
            user_data["id"] = UUID(user_data["id"])
            user_data.pop("hashed_password", None)  # Remove password from response
            
            return User(**user_data)
            
        except Exception as e:
            logger.error(f"User authentication failed: {e}")
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username
        
        Args:
            username: Username
            
        Returns:
            User data dictionary or None
        """
        try:
            mongodb = await get_mongodb_adapter()
            return await mongodb.find_document("users", {"username": username})
        except Exception as e:
            logger.error(f"Failed to get user by username: {e}")
            return None
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email
        
        Args:
            email: Email address
            
        Returns:
            User data dictionary or None
        """
        try:
            mongodb = await get_mongodb_adapter()
            return await mongodb.find_document("users", {"email": email})
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
            mongodb = await get_mongodb_adapter()
            user_data = await mongodb.find_document("users", {"id": str(user_id)})
            
            if user_data:
                user_data["id"] = UUID(user_data["id"])
                user_data.pop("hashed_password", None)  # Remove password
                return User(**user_data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
    
    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp
        
        Args:
            user_id: User ID
        """
        try:
            mongodb = await get_mongodb_adapter()
            await mongodb.update_document(
                "users",
                {"id": user_id},
                {"last_login": datetime.now(timezone.utc)}
            )
        except Exception as e:
            logger.error(f"Failed to update last login: {e}")
    
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
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
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
                expire = datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes)
            
            to_encode.update({"exp": expire})
            
            encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
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
            user = await self.authenticate_user(login_data.username, login_data.password)
            if not user:
                return {
                    "status": "error",
                    "error": "Invalid username or password",
                    "error_type": "AuthenticationFailed"
                }
            
            # Create access token
            token_data = {
                "sub": user.username,
                "user_id": str(user.id),
                "scopes": ["read", "write"] if user.is_admin else ["read"]
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
                    "is_admin": user.is_admin
                }
            }
            
        except Exception as e:
            logger.error(f"User login failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
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
                user_data = await self.get_user_by_username(token_data.username)
                if user_data:
                    user_data["id"] = UUID(user_data["id"])
                    user_data.pop("hashed_password", None)
                    user = User(**user_data)
                else:
                    user = None
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
                user_scopes.append("write")
                user_scopes.append("admin")
            
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
                    "error_type": "InvalidToken"
                }
            
            # Get user
            user = await self.get_current_user(token)
            if not user:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound"
                }
            
            # Create new token
            new_token_data = {
                "sub": user.username,
                "user_id": str(user.id),
                "scopes": ["read", "write"] if user.is_admin else ["read"]
            }
            
            new_access_token = self.create_access_token(new_token_data)
            
            return {
                "status": "success",
                "access_token": new_access_token,
                "token_type": "bearer",
                "expires_in": self.access_token_expire_minutes * 60
            }
            
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
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
                    "error_type": "InvalidToken"
                }
            
            # TODO: Add token to blacklist in Redis or database
            # For now, just log the logout
            logger.info(f"User logged out: {token_data.username}")
            
            return {
                "status": "success",
                "message": "Successfully logged out"
            }
            
        except Exception as e:
            logger.error(f"User logout failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def change_password(self, user_id: UUID, current_password: str, new_password: str) -> Dict[str, Any]:
        """Change user password
        
        Args:
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            Dictionary with change result
        """
        try:
            # Get user
            mongodb = await get_mongodb_adapter()
            user_data = await mongodb.find_document("users", {"id": str(user_id)})
            
            if not user_data:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound"
                }
            
            # Verify current password
            hashed_password = user_data.get("hashed_password", "")
            if not self.verify_password(current_password, hashed_password):
                return {
                    "status": "error",
                    "error": "Current password is incorrect",
                    "error_type": "InvalidPassword"
                }
            
            # Hash new password
            new_hashed_password = self.get_password_hash(new_password)
            
            # Update password
            await mongodb.update_document(
                "users",
                {"id": str(user_id)},
                {"hashed_password": new_hashed_password}
            )
            
            return {
                "status": "success",
                "message": "Password changed successfully"
            }
            
        except Exception as e:
            logger.error(f"Password change failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def update_user_profile(self, user_id: UUID, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile
        
        Args:
            user_id: User ID
            updates: Fields to update
            
        Returns:
            Dictionary with update result
        """
        try:
            # Remove sensitive fields from updates
            allowed_updates = {
                "full_name": updates.get("full_name"),
                "email": updates.get("email")
            }
            
            # Remove None values
            allowed_updates = {k: v for k, v in allowed_updates.items() if v is not None}
            
            if not allowed_updates:
                return {
                    "status": "error",
                    "error": "No valid updates provided",
                    "error_type": "InvalidInput"
                }
            
            # Check if email already exists (if updating email)
            if "email" in allowed_updates:
                existing_email = await self.get_user_by_email(allowed_updates["email"])
                if existing_email and existing_email["id"] != str(user_id):
                    return {
                        "status": "error",
                        "error": "Email already exists",
                        "error_type": "DuplicateEmail"
                    }
            
            # Update user
            mongodb = await get_mongodb_adapter()
            success = await mongodb.update_document(
                "users",
                {"id": str(user_id)},
                allowed_updates
            )
            
            if success:
                return {
                    "status": "success",
                    "message": "Profile updated successfully"
                }
            else:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound"
                }
            
        except Exception as e:
            logger.error(f"Profile update failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def list_users(self, limit: int = 50, offset: int = 0, admin_only: bool = False) -> Dict[str, Any]:
        """List users (admin only)
        
        Args:
            limit: Maximum number of users to return
            offset: Number of users to skip
            admin_only: Only return admin users
            
        Returns:
            Dictionary with user list
        """
        try:
            mongodb = await get_mongodb_adapter()
            
            # Build query
            query = {}
            if admin_only:
                query["is_admin"] = True
            
            # Get users
            users_data = await mongodb.find_documents(
                "users",
                query,
                limit=limit,
                offset=offset,
                sort_field="created_at"
            )
            
            # Convert to User objects and remove passwords
            users = []
            for user_data in users_data:
                user_data["id"] = UUID(user_data["id"])
                user_data.pop("hashed_password", None)
                users.append(User(**user_data).model_dump())
            
            # Get total count
            total_count = await mongodb.count_documents("users", query)
            
            return {
                "status": "success",
                "users": users,
                "total": total_count,
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error(f"List users failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "users": []
            }
    
    async def delete_user(self, user_id: UUID) -> Dict[str, Any]:
        """Delete user (admin only)
        
        Args:
            user_id: User ID to delete
            
        Returns:
            Dictionary with deletion result
        """
        try:
            mongodb = await get_mongodb_adapter()
            
            # Check if user exists
            user_data = await mongodb.find_document("users", {"id": str(user_id)})
            if not user_data:
                return {
                    "status": "error",
                    "error": "User not found",
                    "error_type": "UserNotFound"
                }
            
            # Delete user
            success = await mongodb.delete_document("users", {"id": str(user_id)})
            
            if success:
                return {
                    "status": "success",
                    "message": "User deleted successfully"
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to delete user",
                    "error_type": "DeletionFailed"
                }
            
        except Exception as e:
            logger.error(f"User deletion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on authentication service
        
        Returns:
            Dictionary with health check results
        """
        try:
            # Test database connection
            mongodb = await get_mongodb_adapter()
            health_result = await mongodb.health_check()
            
            # Test password hashing
            test_hash = self.get_password_hash("test")
            hash_verify = self.verify_password("test", test_hash)
            
            # Test JWT operations
            test_token = self.create_access_token({"sub": "test", "user_id": "test"})
            token_verify = self.verify_token(test_token)
            
            return {
                "status": "healthy",
                "database_status": health_result.get("status", "unknown"),
                "password_hashing": "working" if hash_verify else "failed",
                "jwt_operations": "working" if token_verify else "failed",
                "token_expire_minutes": self.access_token_expire_minutes
            }
            
        except Exception as e:
            logger.error(f"Auth service health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "error_type": type(e).__name__
            }


# Global authentication service instance
auth_service = AuthenticationService()
