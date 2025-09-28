"""Shared model base classes for Pydantic and Beanie documents.

Provides serializer mixins so both API schemas and persistence models share
consistent datetime/UUID formatting behaviour.
"""

from datetime import datetime
from typing import Any
from uuid import UUID

from beanie import Document
from pydantic import BaseModel, field_serializer


class _SerializerMixin:
    """Common field serializers reused across Pydantic and Beanie models."""

    @field_serializer(
        "timestamp",
        "created_at",
        "updated_at",
        "last_activity",
        "last_analyzed",
        "last_webhook_event",
        check_fields=False,
    )
    def serialize_datetime(self, value: datetime) -> str | None:
        """Serialize datetime fields to ISO format strings when present."""
        if isinstance(value, datetime):
            return value.isoformat()
        return value

    @field_serializer(
        "id",
        "question_id",
        "session_id",
        "repository_id",
        check_fields=False,
    )
    def serialize_uuid(self, value: UUID) -> str | None:
        """Serialize UUID fields to canonical string representations."""
        if isinstance(value, UUID):
            return str(value)
        return value


class BaseSerializers(_SerializerMixin, BaseModel):
    """Base Pydantic model providing shared serializer behaviour."""


class BaseDocument(_SerializerMixin, Document):
    """Base Beanie document with shared serializers and BSON encoders."""

    class Settings:
        bson_encoders = {UUID: str}

    def __init__(self, **data: Any) -> None:
        # Allow instantiation before init_beanie during unit tests
        if getattr(self.__class__, "_document_settings", None) is None:
            self.__dict__["__initializing__"] = True
            try:
                BaseModel.__init__(self, **data)
            finally:
                self.__dict__.pop("__initializing__", None)
        else:
            super().__init__(**data)
