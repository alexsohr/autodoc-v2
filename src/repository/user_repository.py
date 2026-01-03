"""User repository for authentication and user management

This module provides data access methods for user-related operations
using the Beanie ODM and repository pattern.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from ..models.user import User, UserDocument
from .base import BaseRepository


class UserRepository(BaseRepository[UserDocument]):
    """Repository for user data access operations"""

    def __init__(self):
        super().__init__(UserDocument)

    async def create_user(
        self,
        username: str,
        email: str,
        hashed_password: str,
        full_name: Optional[str] = None,
        is_admin: bool = False,
    ) -> UserDocument:
        """Create a new user
        
        Args:
            username: Username
            email: Email address
            hashed_password: Hashed password
            full_name: Full name (optional)
            is_admin: Whether user is admin
            
        Returns:
            Created user document
        """
        user_doc = UserDocument(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_admin=is_admin,
        )
        return await self.insert(user_doc)

    async def get_by_username(self, username: str) -> Optional[UserDocument]:
        """Get user by username
        
        Args:
            username: Username
            
        Returns:
            User document or None
        """
        return await self.find_one({"username": username})

    async def get_by_email(self, email: str) -> Optional[UserDocument]:
        """Get user by email
        
        Args:
            email: Email address
            
        Returns:
            User document or None
        """
        return await self.find_one({"email": email})

    async def get_by_id(self, user_id: UUID) -> Optional[UserDocument]:
        """Get user by ID
        
        Args:
            user_id: User UUID
            
        Returns:
            User document or None
        """
        return await self.find_one({"id": user_id})

    async def get_by_username_or_email(self, identifier: str) -> Optional[UserDocument]:
        """Get user by username or email
        
        Args:
            identifier: Username or email
            
        Returns:
            User document or None
        """
        # Try username first
        user = await self.get_by_username(identifier)
        if user:
            return user
        
        # Try email if username not found
        return await self.get_by_email(identifier)

    async def update_last_login(self, user_id: UUID) -> bool:
        """Update user's last login timestamp
        
        Args:
            user_id: User ID
            
        Returns:
            True if updated successfully
        """
        return await self.update_one(
            {"id": user_id}, 
            {"last_login": datetime.now(timezone.utc)}
        )

    async def update_password(self, user_id: UUID, hashed_password: str) -> bool:
        """Update user's password
        
        Args:
            user_id: User ID
            hashed_password: New hashed password
            
        Returns:
            True if updated successfully
        """
        return await self.update_one(
            {"id": user_id}, 
            {"hashed_password": hashed_password}
        )

    async def update_profile(
        self, user_id: UUID, updates: Dict[str, any]
    ) -> bool:
        """Update user profile
        
        Args:
            user_id: User ID
            updates: Fields to update
            
        Returns:
            True if updated successfully
        """
        # Filter allowed updates
        allowed_fields = {"full_name", "email", "is_active", "is_admin"}
        filtered_updates = {
            k: v for k, v in updates.items() 
            if k in allowed_fields and v is not None
        }
        
        if not filtered_updates:
            return False
            
        return await self.update_one({"id": user_id}, filtered_updates)

    async def list_users(
        self, 
        limit: int = 50, 
        offset: int = 0, 
        admin_only: bool = False,
        active_only: bool = True,
    ) -> List[UserDocument]:
        """List users with pagination
        
        Args:
            limit: Maximum number of users
            offset: Number of users to skip
            admin_only: Only return admin users
            active_only: Only return active users
            
        Returns:
            List of user documents
        """
        query = {}
        if admin_only:
            query["is_admin"] = True
        if active_only:
            query["is_active"] = True
            
        return await self.find_many(
            query, 
            limit=limit, 
            offset=offset, 
            sort=[("created_at", -1)]
        )

    async def count_users(
        self, admin_only: bool = False, active_only: bool = True
    ) -> int:
        """Count users
        
        Args:
            admin_only: Only count admin users
            active_only: Only count active users
            
        Returns:
            Number of users
        """
        query = {}
        if admin_only:
            query["is_admin"] = True
        if active_only:
            query["is_active"] = True
            
        return await self.count(query)

    async def delete_user(self, user_id: UUID) -> bool:
        """Delete user
        
        Args:
            user_id: User ID
            
        Returns:
            True if deleted successfully
        """
        return await self.delete_one({"id": user_id})

    async def username_exists(self, username: str) -> bool:
        """Check if username exists
        
        Args:
            username: Username to check
            
        Returns:
            True if username exists
        """
        user = await self.get_by_username(username)
        return user is not None

    async def email_exists(self, email: str) -> bool:
        """Check if email exists
        
        Args:
            email: Email to check
            
        Returns:
            True if email exists
        """
        user = await self.get_by_email(email)
        return user is not None

    def to_user_model(self, user_doc: UserDocument) -> User:
        """Convert UserDocument to User model (without sensitive data)
        
        Args:
            user_doc: User document
            
        Returns:
            User model
        """
        return User.from_document(user_doc)

    def to_user_models(self, user_docs: List[UserDocument]) -> List[User]:
        """Convert list of UserDocuments to User models
        
        Args:
            user_docs: List of user documents
            
        Returns:
            List of user models
        """
        return [self.to_user_model(doc) for doc in user_docs]
