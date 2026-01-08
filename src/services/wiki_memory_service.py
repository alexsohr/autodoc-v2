"""Wiki memory service for managing agent memories during wiki generation.

This module provides the WikiMemoryService class for storing, searching,
and managing memories created by wiki agents during repository documentation
generation. Memories enable cross-referencing, pattern recognition, and
maintaining structural decisions across agent runs.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models.wiki_memory import MemoryType, WikiMemory
from ..repository.wiki_memory_repository import WikiMemoryRepository
from ..tools.embedding_tool import EmbeddingTool
from ..utils.config_loader import get_settings

logger = logging.getLogger(__name__)

# Maximum characters for memory content
MAX_MEMORY_CHARS = 4000


class WikiMemoryService:
    """Service for managing wiki agent memories.

    Provides functionality for storing, searching, and managing memories
    created during wiki generation. Memories are stored with vector embeddings
    for semantic search capabilities.
    """

    def __init__(
        self,
        wiki_memory_repo: WikiMemoryRepository,
        embedding_tool: EmbeddingTool,
    ):
        """Initialize WikiMemoryService with dependency injection.

        Args:
            wiki_memory_repo: WikiMemoryRepository instance for data access.
            embedding_tool: EmbeddingTool instance for generating embeddings.
        """
        self.settings = get_settings()
        self._wiki_memory_repo = wiki_memory_repo
        self._embedding_tool = embedding_tool

    async def store_memory(
        self,
        repository_id: UUID,
        content: str,
        memory_type: MemoryType,
        source_agent: str,
        file_paths: Optional[List[str]] = None,
        related_pages: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Store a new memory with vector embedding.

        Creates a new WikiMemory document with the provided content and metadata.
        Content is truncated to MAX_MEMORY_CHARS if necessary, and a vector
        embedding is generated for semantic search.

        Args:
            repository_id: UUID of the repository this memory belongs to.
            content: Memory content text.
            memory_type: Type of memory (structural_decision, pattern_found, cross_reference).
            source_agent: Identifier of the agent creating the memory.
            file_paths: Optional list of related file paths.
            related_pages: Optional list of related wiki page identifiers.

        Returns:
            Dict with status and memory details on success, or error information.
        """
        try:
            # Truncate content if necessary
            truncated_content = content[:MAX_MEMORY_CHARS] if len(content) > MAX_MEMORY_CHARS else content

            # Generate embedding for the content
            embedding = await self._embedding_tool.get_embedding_for_query(truncated_content)
            if embedding is None:
                logger.warning(
                    f"Failed to generate embedding for memory in repository {repository_id}"
                )
                # Store memory without embedding rather than failing completely
                embedding = []

            # Create WikiMemory instance
            memory = WikiMemory(
                repository_id=repository_id,
                memory_type=memory_type,
                content=truncated_content,
                embedding=embedding,
                source_agent=source_agent,
                file_paths=file_paths or [],
                related_pages=related_pages or [],
                embedding_metadata={
                    "model": "text-embedding-3-small",
                    "dimension": len(embedding) if embedding else 0,
                    "truncated": len(content) > MAX_MEMORY_CHARS,
                    "original_length": len(content),
                },
            )

            # Insert via repository
            inserted_memory = await self._wiki_memory_repo.insert(memory)

            logger.info(
                f"Stored memory {inserted_memory.id} for repository {repository_id}, "
                f"type={memory_type.value}, source={source_agent}"
            )

            return {
                "status": "success",
                "memory_id": str(inserted_memory.id),
                "repository_id": str(repository_id),
            }

        except Exception as e:
            logger.error(f"Failed to store memory for repository {repository_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def search_memories(
        self,
        repository_id: UUID,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> Dict[str, Any]:
        """Search memories using semantic vector search.

        Finds memories similar to the query using vector embeddings.
        Results can be filtered by memory type.

        Args:
            repository_id: UUID of the repository to search within.
            query: Search query text.
            limit: Maximum number of results to return (default 10).
            memory_type: Optional filter for specific memory type.

        Returns:
            Dict with status and list of matching memories, or error information.
        """
        try:
            # Generate embedding for query
            query_embedding = await self._embedding_tool.get_embedding_for_query(query)
            if query_embedding is None:
                logger.warning(f"Failed to generate query embedding for search")
                return {
                    "status": "error",
                    "error": "Failed to generate query embedding",
                    "error_type": "EmbeddingError",
                    "memories": [],
                    "count": 0,
                }

            # Perform vector search
            memories = await self._wiki_memory_repo.vector_search(
                repository_id=repository_id,
                query_embedding=query_embedding,
                limit=limit,
                memory_type=memory_type,
            )

            # Serialize memories
            serialized_memories = self._wiki_memory_repo.serialize_many(memories)

            logger.debug(
                f"Found {len(serialized_memories)} memories for repository {repository_id}, "
                f"query='{query[:50]}...', type={memory_type}"
            )

            return {
                "status": "success",
                "memories": serialized_memories,
                "count": len(serialized_memories),
            }

        except Exception as e:
            logger.error(f"Failed to search memories for repository {repository_id}: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "memories": [],
                "count": 0,
            }

    async def get_memories_by_files(
        self,
        repository_id: UUID,
        file_paths: List[str],
    ) -> Dict[str, Any]:
        """Get memories associated with specific file paths.

        Finds all memories that reference any of the given file paths.

        Args:
            repository_id: UUID of the repository to search within.
            file_paths: List of file paths to search for.

        Returns:
            Dict with status and list of matching memories, or error information.
        """
        try:
            if not file_paths:
                return {
                    "status": "success",
                    "memories": [],
                    "count": 0,
                }

            # Find memories by file paths
            memories = await self._wiki_memory_repo.find_by_file_paths(
                repository_id=repository_id,
                file_paths=file_paths,
            )

            # Serialize memories
            serialized_memories = self._wiki_memory_repo.serialize_many(memories)

            logger.debug(
                f"Found {len(serialized_memories)} memories for {len(file_paths)} file paths "
                f"in repository {repository_id}"
            )

            return {
                "status": "success",
                "memories": serialized_memories,
                "count": len(serialized_memories),
            }

        except Exception as e:
            logger.error(
                f"Failed to get memories by files for repository {repository_id}: {e}"
            )
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "memories": [],
                "count": 0,
            }

    async def purge_memories(self, repository_id: UUID) -> Dict[str, Any]:
        """Delete all memories for a specific repository.

        Removes all WikiMemory documents associated with the given repository.
        Used during repository re-processing or cleanup.

        Args:
            repository_id: UUID of the repository whose memories to delete.

        Returns:
            Dict with status and count of deleted memories, or error information.
        """
        try:
            # Delete all memories for the repository
            deleted_count = await self._wiki_memory_repo.delete_by_repository(
                repository_id=repository_id
            )

            logger.info(
                f"Purged {deleted_count} memories for repository {repository_id}"
            )

            return {
                "status": "success",
                "deleted_count": deleted_count,
                "repository_id": str(repository_id),
            }

        except Exception as e:
            logger.error(
                f"Failed to purge memories for repository {repository_id}: {e}"
            )
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    @classmethod
    async def purge_for_repository(cls, repository_id: UUID) -> Dict[str, Any]:
        """Class method to purge memories - creates its own service instance.

        Convenience method for workflows that need to purge memories without
        having access to an existing service instance. Creates necessary
        dependencies internally.

        Args:
            repository_id: UUID of the repository whose memories to delete.

        Returns:
            Dict with status and count of deleted memories, or error information.
        """
        from ..models.code_document import CodeDocument
        from ..repository.code_document_repository import CodeDocumentRepository
        from ..repository.wiki_memory_repository import WikiMemoryRepository
        from ..tools.embedding_tool import EmbeddingTool

        wiki_memory_repo = WikiMemoryRepository()
        # EmbeddingTool requires CodeDocumentRepository
        code_document_repo = CodeDocumentRepository(CodeDocument)
        embedding_tool = EmbeddingTool(code_document_repo)
        service = cls(wiki_memory_repo, embedding_tool)
        return await service.purge_memories(repository_id)
