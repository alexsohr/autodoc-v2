"""Wiki memory service for managing agent memories during wiki generation.

This module provides the WikiMemoryService class for storing, searching,
and managing memories created by wiki agents during repository documentation
generation. Memories enable cross-referencing, pattern recognition, and
maintaining structural decisions across agent runs.
"""

import structlog
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..models.wiki_memory import MemoryType, WikiMemory
from ..repository.wiki_memory_repository import WikiMemoryRepository
from ..tools.embedding_tool import EmbeddingTool
from ..utils.config_loader import get_settings

logger = structlog.get_logger(__name__)

# Maximum characters for memory content chunk
MAX_MEMORY_CHARS = 4000
# Overlap between chunks for context continuity
CHUNK_OVERLAP = 200


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
        """Store memory, splitting into multiple records if content exceeds limit.

        Creates WikiMemory documents with vector embeddings. If content exceeds
        MAX_MEMORY_CHARS, it is split into overlapping chunks using
        RecursiveCharacterTextSplitter, with each chunk stored as a separate record.

        Args:
            repository_id: UUID of the repository this memory belongs to.
            content: Memory content text.
            memory_type: Type of memory (structural_decision, pattern_found, cross_reference).
            source_agent: Identifier of the agent creating the memory.
            file_paths: Optional list of related file paths.
            related_pages: Optional list of related wiki page identifiers.

        Returns:
            Dict with status, memory_ids list, and chunks_created count on success,
            or error information on failure.
        """
        try:
            # Split content into chunks if needed
            if len(content) > MAX_MEMORY_CHARS:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=MAX_MEMORY_CHARS,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                )
                chunks = text_splitter.split_text(content)
                logger.info(
                    f"Split memory content into {len(chunks)} chunks for repository {repository_id}"
                )
            else:
                chunks = [content]

            memory_ids = []
            for i, chunk in enumerate(chunks):
                # Generate embedding for this chunk
                embedding = await self._embedding_tool.get_embedding_for_query(chunk)
                if embedding is None:
                    logger.warning(
                        f"Failed to generate embedding for chunk {i} in repository {repository_id}"
                    )
                    embedding = []

                # Create WikiMemory for this chunk
                memory = WikiMemory(
                    repository_id=repository_id,
                    memory_type=memory_type,
                    content=chunk,
                    embedding=embedding,
                    source_agent=source_agent,
                    file_paths=file_paths or [],
                    related_pages=related_pages or [],
                    embedding_metadata={
                        "model": "text-embedding-3-small",
                        "dimension": len(embedding) if embedding else 0,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_length": len(content),
                    },
                )

                # Repository handles insert + ChromaDB sync
                inserted = await self._wiki_memory_repo.insert(memory)
                memory_ids.append(str(inserted.id))

            logger.info(
                f"Stored {len(memory_ids)} memory chunks for repository {repository_id}, "
                f"type={memory_type.value}, source={source_agent}"
            )

            return {
                "status": "success",
                "memory_ids": memory_ids,
                "chunks_created": len(chunks),
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

            # Perform vector search (repository handles Atlas/ChromaDB fallback)
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

        Removes all WikiMemory documents associated with the given repository
        from both MongoDB and ChromaDB (handled by repository).
        Used during repository re-processing or cleanup.

        Args:
            repository_id: UUID of the repository whose memories to delete.

        Returns:
            Dict with status and count of deleted memories, or error information.
        """
        try:
            # Repository handles deletion from both MongoDB and ChromaDB
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
