"""Repository helper for wiki structures."""

from __future__ import annotations

from uuid import UUID

from ..models.wiki import WikiStructure
from .base import BaseRepository


class WikiStructureRepository(BaseRepository[WikiStructure]):
    """Repository wrapper for wiki structures."""

    async def upsert(self, repository_id: UUID, wiki: WikiStructure) -> None:
        wiki.repository_id = repository_id
        # Get data with by_alias=True so "_id" is used (Beanie/MongoDB standard)
        data = wiki.model_dump(mode="python", by_alias=True)
        
        # Extract _id for $setOnInsert (only set on new document creation)
        wiki_id = data.pop("_id", wiki.id)
        
        await self.collection.update_one(
            {"repository_id": repository_id},  # Use UUID directly, not string
            {
                "$set": self._prepare_updates(data),
                "$setOnInsert": {"_id": wiki_id},  # Set UUID _id only on insert
            },
            upsert=True,
        )
