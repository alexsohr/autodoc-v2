"""Wiki memory middleware for agents - provides persistent memory capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, List, Optional
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from typing_extensions import NotRequired

from ...models.wiki_memory import MemoryType

logger = structlog.get_logger(__name__)


class WikiMemoryState(AgentState):
    """State schema for wiki memory middleware - adds repository_id and generated_content to agent state."""

    repository_id: NotRequired[Optional[str]]
    """Repository ID for memory operations."""

    generated_content: NotRequired[Optional[str]]
    """Extracted wiki page content from agent messages."""


# System prompt to guide agent on memory usage
WIKI_MEMORY_SYSTEM_PROMPT = """## Wiki Memory System - REQUIRED WORKFLOW

You have access to a persistent memory system. Using this system is **MANDATORY** - not optional.

### REQUIRED: Memory Workflow

**STEP 1 - BEFORE starting ANY work:**
Call `recall_memories` with a query describing what you're about to work on.
This retrieves decisions from previous wiki generations that you MUST consider.
Example: `recall_memories(query="wiki structure decisions for repository")`

**STEP 2 - DURING your work:**
When analyzing files, call `get_file_memories` to retrieve past observations about those files.
Example: `get_file_memories(file_paths=["src/main.py", "src/utils.py"])`

**STEP 3 - BEFORE completing your task:**
Store at least ONE memory capturing your key decisions or findings.
Example: `store_memory(content="Organized wiki into 3 sections: Core, API, Utils based on package structure", memory_type="structural_decision")`

### Memory Tools:
- `recall_memories(query, limit=5)` - Search for relevant past memories
- `get_file_memories(file_paths)` - Get memories for specific files
- `store_memory(content, memory_type, file_paths?, related_pages?)` - Store a new memory

### Memory Types (for store_memory):
- `structural_decision` - Wiki organization choices (sections, page hierarchy)
- `pattern_found` - Coding patterns/conventions discovered
- `cross_reference` - Relationships between code areas

**FAILURE TO USE MEMORY TOOLS = INCOMPLETE TASK**
Your work is NOT complete until you have:
1. Recalled relevant memories at the start
2. Stored at least one memory with your decisions"""


class WikiMemoryMiddleware(AgentMiddleware):
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

    state_schema = WikiMemoryState

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
                chunks = result.get("chunks_created", 1)
                memory_ids = result.get("memory_ids", [])
                if chunks > 1:
                    return f"Memory stored in {chunks} chunks (IDs: {', '.join(memory_ids)})"
                return f"Memory stored successfully (ID: {memory_ids[0] if memory_ids else 'unknown'})"
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
                    content_preview = m['content'][:300] + "..." if len(m['content']) > 300 else m['content']
                    formatted.append(f"- [{m['memory_type']}] {content_preview}")
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
                    content_preview = m['content'][:300] + "..." if len(m['content']) > 300 else m['content']
                    formatted.append(f"- [{m['memory_type']}] {content_preview}")
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

    async def awrap_model_call(
        self,
        request: Any,
        handler: Any,
    ) -> Any:
        """Extract repository_id from state and inject memory system prompt.

        This method is called by the agent framework before each model call.
        """
        # Try to extract repository_id from agent state on first model call
        if self._repository_id is None:
            state = getattr(request, 'state', None)
            if state is not None:
                if isinstance(state, dict):
                    repo_id = state.get("repository_id")
                else:
                    repo_id = getattr(state, "repository_id", None)
                if repo_id is not None:
                    self._repository_id = repo_id if isinstance(repo_id, UUID) else UUID(str(repo_id))
                    logger.debug(f"WikiMemoryMiddleware extracted repository_id: {self._repository_id}")

        # Inject memory system prompt into the request
        if hasattr(request, 'system_message') and request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]

        new_system_message = SystemMessage(
            content=new_system_content  # type: ignore[arg-type]
        )

        # Call handler with modified request
        if hasattr(request, 'override'):
            return await handler(request.override(system_message=new_system_message))
        else:
            # Fallback if request doesn't have override method
            return await handler(request)
