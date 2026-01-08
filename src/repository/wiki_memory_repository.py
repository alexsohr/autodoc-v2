"""Repository for WikiMemory documents with vector search capabilities.

Provides data access methods for WikiMemory entities including:
- Vector search for semantic similarity queries
- File path-based lookups
- Bulk deletion by repository
"""

from __future__ import annotations

import logging
from typing import List, Optional
from uuid import UUID

from pymongo.errors import OperationFailure

from ..models.wiki_memory import MemoryType, WikiMemory
from .base import BaseRepository

logger = logging.getLogger(__name__)


class WikiMemoryRepository(BaseRepository[WikiMemory]):
    """Repository for WikiMemory documents with vector search support.

    Extends BaseRepository to provide specialized data access methods
    for wiki agent memories, including semantic vector search and
    file path-based queries.
    """

    def __init__(self) -> None:
        """Initialize the repository with WikiMemory document type."""
        super().__init__(WikiMemory)

    async def vector_search(
        self,
        repository_id: UUID,
        query_embedding: List[float],
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        score_threshold: float = 0.7,
    ) -> List[WikiMemory]:
        """Execute MongoDB Atlas Vector Search for semantically similar memories.

        Args:
            repository_id: UUID of the repository to search within.
            query_embedding: Vector embedding to search for (384 dimensions).
            limit: Maximum number of results to return.
            memory_type: Optional filter for specific memory type.
            score_threshold: Minimum similarity score (0.0-1.0) for results.

        Returns:
            List of WikiMemory documents ordered by similarity score.
        """
        pipeline: List[dict] = []

        # Stage 1: $vectorSearch with filter by repository_id
        vector_stage: dict = {
            "$vectorSearch": {
                "index": "wiki_memories_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": limit * 10,
                "limit": limit * 2,  # Get extra to allow for threshold filtering
                "filter": {"repository_id": str(repository_id)},
            }
        }
        pipeline.append(vector_stage)

        # Stage 2: $addFields to capture the vector search score
        pipeline.append(
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}}
        )

        # Stage 3: $match to filter by score threshold
        pipeline.append({"$match": {"score": {"$gte": score_threshold}}})

        # Stage 4: Optional $match for memory_type filter
        if memory_type is not None:
            pipeline.append({"$match": {"memory_type": memory_type.value}})

        # Stage 5: $limit to enforce final result count
        pipeline.append({"$limit": limit})

        results: List[WikiMemory] = []
        try:
            async for raw in self.collection.aggregate(pipeline):
                # Map _id to id for Beanie document construction
                raw_id = raw.get("_id")
                if raw_id is not None:
                    raw["id"] = raw_id

                # Remove computed score field before constructing document
                raw.pop("score", None)

                document = self.document(**raw)
                results.append(document)
        except OperationFailure as e:
            logger.warning(f"Vector search failed for wiki memories: {e}")
            return []

        return results

    async def find_by_file_paths(
        self,
        repository_id: UUID,
        file_paths: List[str],
    ) -> List[WikiMemory]:
        """Find memories associated with any of the given file paths.

        Args:
            repository_id: UUID of the repository to search within.
            file_paths: List of file paths to search for.

        Returns:
            List of WikiMemory documents that reference any of the given paths.
        """
        query = {
            "repository_id": repository_id,
            "file_paths": {"$in": file_paths},
        }
        return await self.find_many(query, limit=1000)

    async def delete_by_repository(self, repository_id: UUID) -> int:
        """Delete all memories for a specific repository.

        Args:
            repository_id: UUID of the repository whose memories to delete.

        Returns:
            Number of documents deleted.
        """
        query = {"repository_id": repository_id}
        return await self.delete_many(query)
