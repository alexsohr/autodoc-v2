"""Wiki memory middleware for agents - provides persistent memory capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, cast
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import logging
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from ...models.wiki_memory import MemoryType

logger = logging.getLogger(__name__)


# System prompt to guide agent on memory usage
WIKI_MEMORY_SYSTEM_PROMPT = """## Wiki Memory System

You have access to a persistent memory system that helps you maintain context across wiki generations.
Use these memory tools to store and recall structural decisions, patterns, and cross-references.

### Memory Tools Available:
- `store_memory`: Store structural decisions, patterns found, or cross-references for future generations
- `recall_memories`: Search for relevant past memories using semantic similarity
- `get_file_memories`: Get memories related to specific source files you're analyzing

### When to Store Memories:
- **Structural decisions**: When you decide on wiki organization, section groupings, or page hierarchies
- **Patterns found**: When you identify recurring patterns in the codebase
- **Cross-references**: When you notice relationships between different parts of the codebase

### Memory Types:
- `structural_decision`: Decisions about wiki structure and organization
- `pattern_found`: Patterns or conventions discovered in the codebase
- `cross_reference`: Relationships between different code areas or wiki sections

Be proactive about storing important decisions so they persist across regenerations."""


class WikiMemoryMiddleware:
    """Middleware that provides persistent memory capabilities to wiki agents.

    Self-contained middleware that:
    - Extracts repository_id from agent state automatically
    - Creates its own WikiMemoryService internally (lazy loading)
    - Injects system prompt guidance for using the memory system

    Usage:
        middleware=[
            WikiMemoryMiddleware("structure_agent"),
            TodoListMiddleware(),
            ...
        ]
    """

    def __init__(
        self,
        source_agent: str,
        system_prompt: str = WIKI_MEMORY_SYSTEM_PROMPT,
    ):
        """Initialize middleware.

        Args:
            source_agent: Agent identifier ("structure_agent" or "page_agent").
            system_prompt: System prompt for memory usage guidance.
        """
        self.source_agent = source_agent
        self.system_prompt = system_prompt
        self._service = None
        self._repository_id: Optional[UUID] = None

        # Create tools with closure over self
        @tool
        async def store_memory(
            content: str,
            memory_type: str,
            file_paths: Optional[List[str]] = None,
            related_pages: Optional[List[str]] = None,
        ) -> str:
            """Store a structural decision, pattern, or cross-reference for future wiki generations.

            Args:
                content: The memory content to store
                memory_type: Type of memory - one of 'structural_decision', 'pattern_found', 'cross_reference'
                file_paths: Optional list of related file paths
                related_pages: Optional list of related wiki page identifiers
            """
            if self._repository_id is None:
                return "Error: repository_id not yet available from agent state"
            service = self._get_service()
            result = await service.store_memory(
                repository_id=self._repository_id,
                content=content,
                memory_type=MemoryType(memory_type),
                source_agent=self.source_agent,
                file_paths=file_paths,
                related_pages=related_pages,
            )
            if result["status"] == "success":
                return f"Memory stored successfully (ID: {result['memory_id']})"
            return f"Failed to store memory: {result.get('error', 'Unknown error')}"

        @tool
        async def recall_memories(
            query: str,
            limit: int = 5,
            memory_type: Optional[str] = None,
        ) -> str:
            """Recall relevant memories based on semantic similarity to your query.

            Args:
                query: Search query to find relevant memories
                limit: Maximum number of memories to return (default 5)
                memory_type: Optional filter by type - 'structural_decision', 'pattern_found', or 'cross_reference'
            """
            if self._repository_id is None:
                return "Error: repository_id not yet available from agent state"
            service = self._get_service()
            result = await service.search_memories(
                repository_id=self._repository_id,
                query=query,
                limit=limit,
                memory_type=MemoryType(memory_type) if memory_type else None,
            )
            if result["status"] == "success":
                memories = result["memories"]
                if not memories:
                    return "No relevant memories found."
                formatted = []
                for m in memories:
                    formatted.append(f"- [{m['memory_type']}] {m['content'][:300]}...")
                return f"Found {len(memories)} relevant memories:\n" + "\n".join(formatted)
            return f"Failed to recall memories: {result.get('error', 'Unknown error')}"

        @tool
        async def get_file_memories(file_paths: List[str]) -> str:
            """Get memories related to specific source files you're analyzing.

            Args:
                file_paths: List of file paths to find related memories for
            """
            if self._repository_id is None:
                return "Error: repository_id not yet available from agent state"
            service = self._get_service()
            result = await service.get_memories_by_files(
                repository_id=self._repository_id,
                file_paths=file_paths,
            )
            if result["status"] == "success":
                memories = result["memories"]
                if not memories:
                    return "No memories found for these files."
                formatted = []
                for m in memories:
                    formatted.append(f"- [{m['memory_type']}] {m['content'][:300]}...")
                return f"Found {len(memories)} file-related memories:\n" + "\n".join(formatted)
            return f"Failed to get file memories: {result.get('error', 'Unknown error')}"

        # Set tools as instance attribute
        self.tools = [store_memory, recall_memories, get_file_memories]

    def _get_service(self):
        """Lazy initialization of WikiMemoryService with its dependencies."""
        if self._service is None:
            from ...models.code_document import CodeDocument
            from ...repository.code_document_repository import CodeDocumentRepository
            from ...repository.wiki_memory_repository import WikiMemoryRepository
            from ...services.wiki_memory_service import WikiMemoryService
            from ...tools.embedding_tool import EmbeddingTool

            wiki_memory_repo = WikiMemoryRepository()
            code_document_repo = CodeDocumentRepository(CodeDocument)
            embedding_tool = EmbeddingTool(code_document_repo)
            self._service = WikiMemoryService(
                wiki_memory_repo=wiki_memory_repo,
                embedding_tool=embedding_tool,
            )
        return self._service

    def set_repository_id(self, repository_id: UUID) -> None:
        """Set the repository ID for the middleware.

        This should be called by the workflow/agent before invoking tools.

        Args:
            repository_id: The repository ID to use for all memory operations
        """
        self._repository_id = repository_id
        logger.debug(f"WikiMemoryMiddleware repository_id set to {repository_id}")
