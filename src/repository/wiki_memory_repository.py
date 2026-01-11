"""Repository for WikiMemory documents with vector search capabilities.

Provides data access methods for WikiMemory entities including:
- Vector search for semantic similarity queries (Atlas or ChromaDB fallback)
- Automatic ChromaDB sync on insert for local development
- File path-based lookups
- Bulk deletion by repository (from both MongoDB and ChromaDB)
"""

from __future__ import annotations

import structlog
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import UUID

from pymongo.errors import OperationFailure

from ..models.wiki_memory import MemoryType, WikiMemory
from .base import BaseRepository

if TYPE_CHECKING:
    import chromadb

logger = structlog.get_logger(__name__)

# Global ChromaDB client (lazy initialized)
_chroma_client: Optional["chromadb.ClientAPI"] = None
_chroma_available: Optional[bool] = None


def _get_chroma_client() -> Optional["chromadb.ClientAPI"]:
    """Get or create ChromaDB client for local vector search fallback.

    Returns:
        ChromaDB client or None if ChromaDB is not installed.
    """
    global _chroma_client, _chroma_available

    # Return cached result if we already know ChromaDB isn't available
    if _chroma_available is False:
        return None

    if _chroma_client is None:
        try:
            import chromadb

            # Use persistent storage in .chroma directory
            _chroma_client = chromadb.PersistentClient(path=".chroma")
            _chroma_available = True
            logger.info("Initialized ChromaDB client for local vector search")
        except ImportError:
            _chroma_available = False
            logger.debug("ChromaDB not installed, fallback disabled")
            return None

    return _chroma_client


def _get_collection_name(repository_id: UUID) -> str:
    """Get ChromaDB collection name for a repository."""
    return f"wiki_memories_{str(repository_id).replace('-', '_')}"


class WikiMemoryRepository(BaseRepository[WikiMemory]):
    """Repository for WikiMemory documents with vector search support.

    Extends BaseRepository to provide specialized data access methods
    for wiki agent memories, including semantic vector search and
    file path-based queries. Automatically syncs to ChromaDB for
    local development environments without MongoDB Atlas Vector Search.
    """

    def __init__(self) -> None:
        """Initialize the repository with WikiMemory document type."""
        super().__init__(WikiMemory)

    async def insert(self, document: WikiMemory) -> WikiMemory:
        """Insert a WikiMemory document into MongoDB and sync to ChromaDB.

        Args:
            document: WikiMemory document to insert.

        Returns:
            Inserted WikiMemory document with ID.
        """
        # Insert into MongoDB (primary storage)
        inserted = await super().insert(document)

        # Sync to ChromaDB for fallback vector search
        self._sync_to_chromadb(inserted)

        return inserted

    def _sync_to_chromadb(self, memory: WikiMemory) -> None:
        """Sync a memory document to ChromaDB for local vector search.

        Args:
            memory: WikiMemory document to sync.
        """
        if not memory.embedding or not memory.id:
            return

        client = _get_chroma_client()
        if client is None:
            return

        try:
            collection_name = _get_collection_name(memory.repository_id)
            collection = client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            collection.upsert(
                ids=[str(memory.id)],
                embeddings=[memory.embedding],
                metadatas=[{
                    "memory_type": memory.memory_type.value,
                    "source_agent": memory.source_agent,
                }],
            )
            logger.debug(f"Synced memory {memory.id} to ChromaDB")
        except Exception as e:
            # Don't fail the insert if ChromaDB sync fails
            logger.warning(f"Failed to sync memory to ChromaDB: {e}")

    def _delete_chromadb_collection(self, repository_id: UUID) -> None:
        """Delete ChromaDB collection for a repository.

        Args:
            repository_id: Repository UUID.
        """
        client = _get_chroma_client()
        if client is None:
            return

        try:
            collection_name = _get_collection_name(repository_id)
            client.delete_collection(name=collection_name)
            logger.debug(f"Deleted ChromaDB collection {collection_name}")
        except ValueError:
            # Collection doesn't exist - that's fine
            pass
        except Exception as e:
            logger.warning(f"Failed to delete ChromaDB collection: {e}")

    async def vector_search(
        self,
        repository_id: UUID,
        query_embedding: List[float],
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        score_threshold: float = 0.7,
    ) -> List[WikiMemory]:
        """Execute MongoDB Atlas Vector Search for semantically similar memories.

        Falls back to ChromaDB if Atlas Vector Search is not available.

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
            # Atlas Vector Search not available - use ChromaDB fallback
            logger.info(
                f"Atlas Vector Search unavailable, using ChromaDB fallback: {e}"
            )
            return await self._fallback_vector_search(
                repository_id=repository_id,
                query_embedding=query_embedding,
                limit=limit,
                memory_type=memory_type,
                score_threshold=score_threshold,
            )

        return results

    async def _fallback_vector_search(
        self,
        repository_id: UUID,
        query_embedding: List[float],
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        score_threshold: float = 0.7,
    ) -> List[WikiMemory]:
        """ChromaDB-based vector search fallback for MongoDB Community Edition.

        Uses ChromaDB for efficient local vector search when Atlas Vector Search
        is not available. Data is already synced via insert().

        Args:
            repository_id: UUID of the repository to search within.
            query_embedding: Vector embedding to search for.
            limit: Maximum number of results to return.
            memory_type: Optional filter for specific memory type.
            score_threshold: Minimum similarity score (0.0-1.0) for results.

        Returns:
            List of WikiMemory documents ordered by similarity score.
        """
        client = _get_chroma_client()
        if client is None:
            logger.warning(
                "ChromaDB not available. Install with: pip install chromadb"
            )
            return []

        try:
            # Get collection for this repository
            collection_name = _get_collection_name(repository_id)
            try:
                collection = client.get_collection(
                    name=collection_name,
                )
            except ValueError:
                logger.warning(f"ChromaDB collection {collection_name} not found")
                # Still no data
                return []

            # Build ChromaDB where filter
            where_filter = None
            if memory_type is not None:
                where_filter = {"memory_type": memory_type.value}

            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_filter,
                include=["distances"],
            )

            # Get memory IDs from results
            if not results["ids"] or not results["ids"][0]:
                return []

            distances = results["distances"][0] if results["distances"] else []
            memory_ids_with_scores: List[tuple[str, float]] = []

            for i, mem_id in enumerate(results["ids"][0]):
                # ChromaDB returns distances (lower = more similar for cosine)
                # Convert to similarity score: 1 - distance for cosine
                distance = distances[i] if i < len(distances) else 0
                similarity = 1 - distance

                if similarity >= score_threshold:
                    memory_ids_with_scores.append((mem_id, similarity))

            if not memory_ids_with_scores:
                return []

            # Fetch full documents from MongoDB
            memory_ids = [mid for mid, _ in memory_ids_with_scores]
            memories = await self.find_many(
                {"id": {"$in": memory_ids}},
                limit=limit,
            )

            # Create lookup for ordering
            memory_map = {str(m.id): m for m in memories}

            # Return in similarity order
            ordered_memories = []
            for mem_id, _ in memory_ids_with_scores:
                if mem_id in memory_map:
                    ordered_memories.append(memory_map[mem_id])

            logger.debug(
                f"ChromaDB fallback found {len(ordered_memories)} memories "
                f"for repository {repository_id}"
            )

            return ordered_memories

        except Exception as e:
            logger.error(f"ChromaDB fallback search failed: {e}")
            return []

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

        Deletes from both MongoDB and ChromaDB.

        Args:
            repository_id: UUID of the repository whose memories to delete.

        Returns:
            Number of documents deleted from MongoDB.
        """
        # Delete from ChromaDB first
        self._delete_chromadb_collection(repository_id)

        # Delete from MongoDB
        query = {"repository_id": repository_id}
        return await self.delete_many(query)
