"""Repository helper for wiki structures."""

from __future__ import annotations

from uuid import UUID

from ..models.wiki import WikiStructure
from .base import BaseRepository


class WikiStructureRepository(BaseRepository[WikiStructure]):
    """Repository wrapper for wiki structures."""

    async def upsert(self, repository_id: UUID, wiki: WikiStructure) -> None:
        wiki.repository_id = repository_id
        await self.collection.update_one(
            {"repository_id": str(repository_id)},
            {"$set": self._prepare_updates(wiki.model_dump(mode="python"))},
            upsert=True,
        )
