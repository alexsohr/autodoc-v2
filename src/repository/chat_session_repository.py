"""Repository helpers for chat sessions."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from ..models.chat import ChatSession
from .base import BaseRepository


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Repository helpers for chat sessions."""

    async def touch_session(self, session_id: UUID) -> bool:
        return await self.update_one(
            {"id": session_id},
            {"last_activity": datetime.now(timezone.utc)},
        )
