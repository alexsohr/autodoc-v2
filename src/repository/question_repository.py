"""Repository helpers for stored questions."""

from __future__ import annotations

from ..models.chat import Question
from .base import BaseRepository


class QuestionRepository(BaseRepository[Question]):
    """Repository helpers for stored questions."""
