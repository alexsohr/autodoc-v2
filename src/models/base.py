"""
Base model classes and shared functionality for Pydantic models.

This module provides common serializers and utilities that can be inherited
by other model classes to avoid code duplication.
"""

from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, field_serializer


class BaseSerializers(BaseModel):
    """
    Base class providing common field serializers for datetime and UUID fields.
    
    This class can be inherited by Pydantic models to provide consistent
    serialization behavior for common field types without code duplication.
    """

    @field_serializer(
        'timestamp', 'created_at', 'updated_at', 'last_activity', 
        'last_analyzed', 'last_webhook_event', check_fields=False
    )
    def serialize_datetime(self, value: datetime) -> str | None:
        """
        Serialize datetime fields to ISO format string.
        
        Args:
            value: The datetime value to serialize
            
        Returns:
            ISO format datetime string, or None if value is None
        """
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @field_serializer(
        'id', 'question_id', 'session_id', 'repository_id', 
        check_fields=False
    )
    def serialize_uuid(self, value: UUID) -> str | None:
        """
        Serialize UUID fields to string format.
        
        Args:
            value: The UUID value to serialize
            
        Returns:
            String representation of UUID, or None if value is None
        """
        if isinstance(value, UUID):
            return str(value)
        return value
