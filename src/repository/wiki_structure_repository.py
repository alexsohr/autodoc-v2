"""Repository helper for wiki structures."""

from __future__ import annotations

from uuid import UUID

from ..models.wiki import WikiStructure
from .base import BaseRepository


class WikiStructureRepository(BaseRepository[WikiStructure]):
    """Repository wrapper for wiki structures."""

    async def upsert(self, repository_id: UUID, wiki: WikiStructure) -> None:
        wiki.repository_id = repository_id
        # Exclude _id from the update - let MongoDB handle _id generation
        # This prevents type mismatch when retrieving (ObjectId vs str)
        data = wiki.model_dump(mode="python")
        data.pop("_id", None)
        await self.collection.update_one(
            {"repository_id": str(repository_id)},
            {"$set": self._prepare_updates(data)},
            upsert=True,
        )
