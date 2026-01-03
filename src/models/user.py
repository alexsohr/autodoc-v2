"""User data models for authentication and authorization

This module defines user-related Pydantic models and Beanie documents
for authentication, authorization, and user management.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import UUID, uuid4

from beanie import Document
from pydantic import BaseModel, Field


class UserDocument(Document):
    """User document for MongoDB storage using Beanie"""

    id: UUID = Field(default_factory=uuid4, description="User ID")
    username: str = Field(description="Username", unique=True)
    email: str = Field(description="Email address", unique=True)
    hashed_password: str = Field(description="Hashed password")
    full_name: Optional[str] = Field(default=None, description="Full name")
    is_active: bool = Field(default=True, description="User is active")
    is_admin: bool = Field(default=False, description="User has admin privileges")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = Field(
        default=None, description="Last login timestamp"
    )

    class Settings:
        name = "users"
        indexes = [
            "username",
            "email",
            "is_active",
            "is_admin",
            "created_at",
        ]


class User(BaseModel):
    """User model for API responses (without sensitive data)"""

    id: UUID = Field(description="User ID")
    username: str = Field(description="Username")
    email: str = Field(description="Email address")
    full_name: Optional[str] = Field(default=None, description="Full name")
    is_active: bool = Field(default=True, description="User is active")
    is_admin: bool = Field(default=False, description="User has admin privileges")
    created_at: Optional[datetime] = Field(default=None, description="Creation timestamp")
    last_login: Optional[datetime] = Field(
        default=None, description="Last login timestamp"
    )

    @classmethod
    def from_document(cls, doc: UserDocument) -> "User":
        """Create User from UserDocument"""
        return cls(
            id=doc.id,
            username=doc.username,
            email=doc.email,
            full_name=doc.full_name,
            is_active=doc.is_active,
            is_admin=doc.is_admin,
            created_at=doc.created_at,
            last_login=doc.last_login,
        )


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


class UserUpdate(BaseModel):
    """User update model"""

    full_name: Optional[str] = Field(default=None, description="Full name")
    email: Optional[str] = Field(default=None, description="Email address")
    is_active: Optional[bool] = Field(default=None, description="User is active")
    is_admin: Optional[bool] = Field(default=None, description="User has admin privileges")


class Token(BaseModel):
    """JWT token model"""

    access_token: str = Field(description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")


class TokenData(BaseModel):
    """Token payload data"""

    username: Optional[str] = Field(default=None, description="Username")
    user_id: Optional[str] = Field(default=None, description="User ID")
    scopes: list[str] = Field(default_factory=list, description="Token scopes")
