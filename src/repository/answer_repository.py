"""Repository helpers for stored answers."""

from __future__ import annotations

from ..models.chat import Answer
from .base import BaseRepository


class AnswerRepository(BaseRepository[Answer]):
    """Repository helpers for stored answers."""
