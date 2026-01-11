# Plan: Wiki Agent Shared Memory - Revised Architecture

## Goal

Add persistent, repository-scoped memory to wiki agents (structure and page agents) following strict architectural principles:
- **Self-Contained Middleware**: Middleware initializes its own service (like TodoListMiddleware)
- **Service-Only Repository Access**: Tools/middleware call services, not repositories
- **Business Logic Centralization**: All logic in `WikiMemoryService`
- **LangGraph Middleware**: Use `AgentMiddleware` pattern (same as `TodoListMiddleware`)
- **LangGraph Tool Pattern**: Use `@tool` decorator inside middleware closure
- **Constructor Injection**: Services still use constructor injection for their dependencies

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     Wiki React Agents                           │
│  (create_structure_agent / create_page_agent)                   │
│                                                                 │
│  Agent State contains: repository_id, repo_dir, etc.            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │        WikiMemoryMiddleware("structure_agent")              ││
│  │              (Self-Contained Class)                         ││
│  │                                                             ││
│  │  - Extracts repository_id from agent state automatically    ││
│  │  - Initializes own WikiMemoryService (lazy)                 ││
│  │  - tools = [store_memory, recall_memories, get_file_memories]│
│  │  - awrap_model_call() → extracts state + injects prompt     ││
│  └───────────────────────────┬─────────────────────────────────┘│
│                              │                                  │
│                              ▼  (created internally)            │
│            ┌──────────────────────────────┐                     │
│            │     WikiMemoryService        │                     │
│            │     (Business Logic)         │                     │
│            │                              │                     │
│            │  - store_memory()            │                     │
│            │  - search_memories()         │                     │
│            │  - get_memories_by_files()   │                     │
│            │  - purge_memories()          │                     │
│            │  + purge_for_repository()    │ ← class method      │
│            └──────────────┬───────────────┘                     │
│                           ▼                                     │
│            ┌──────────────────────────────┐                     │
│            │   WikiMemoryRepository       │                     │
│            │   (Data Access Only)         │                     │
│            └──────────────┬───────────────┘                     │
│                           ▼                                     │
│            ┌──────────────────────────────┐                     │
│            │      WikiMemory Model        │                     │
│            │      (Beanie Document)       │                     │
│            └──────────────────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘

Workflow calls WikiMemoryService.purge_for_repository() directly for force_regenerate
```

---

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/models/wiki_memory.py` | `WikiMemory` Beanie document, `MemoryType` enum |
| `src/repository/wiki_memory_repository.py` | `WikiMemoryRepository(BaseRepository[WikiMemory])` - data access only |
| `src/services/wiki_memory_service.py` | `WikiMemoryService` - all business logic, embedding generation |
| `src/agents/middleware/__init__.py` | Middleware package init (if doesn't exist) |
| `src/agents/middleware/wiki_memory_middleware.py` | `WikiMemoryMiddleware` using `AgentMiddleware` pattern - provides tools + system prompt |

### Modified Files

| File | Changes |
|------|---------|
| `src/repository/database.py` | Register `WikiMemory` in `init_beanie()` document_models list |
| `src/models/__init__.py` | Export `WikiMemory`, `MemoryType` |
| `src/repository/__init__.py` | Export `WikiMemoryRepository` |
| `src/services/__init__.py` | Export `WikiMemoryService` |
| `src/agents/wiki_react_agents.py` | Add memory middleware to middleware stack (no service param needed) |
| `src/agents/wiki_workflow.py` | Add `force_regenerate` to state, call `WikiMemoryService.purge_for_repository()` |

---

## Implementation Details

### 1. WikiMemory Model (`src/models/wiki_memory.py`)

```python
from enum import Enum
from typing import List, Optional, Dict, Any
from uuid import UUID
from pydantic import Field
from pymongo import IndexModel
from .base import BaseDocument

class MemoryType(str, Enum):
    STRUCTURAL_DECISION = "structural_decision"
    PATTERN_FOUND = "pattern_found"
    CROSS_REFERENCE = "cross_reference"

class WikiMemory(BaseDocument):
    """Persistent memory for wiki generation agents."""

    repository_id: UUID
    memory_type: MemoryType
    content: str  # Max 4000 chars
    embedding: List[float] = Field(default_factory=list)  # 384 dimensions
    source_agent: str  # "structure_agent" | "page_agent"
    file_paths: List[str] = Field(default_factory=list)
    related_pages: List[str] = Field(default_factory=list)
    embedding_metadata: Dict[str, Any] = Field(default_factory=dict)

    class Settings:
        name = "wiki_memories"
        indexes = [
            IndexModel([("repository_id", 1)]),
            IndexModel([("repository_id", 1), ("memory_type", 1)]),
            IndexModel([("repository_id", 1), ("file_paths", 1)]),
        ]
```

### 2. WikiMemoryRepository (`src/repository/wiki_memory_repository.py`)

**Data access only - no business logic:**

```python
from typing import List, Optional
from uuid import UUID
from ..repository.base import BaseRepository
from ..models.wiki_memory import WikiMemory, MemoryType

class WikiMemoryRepository(BaseRepository[WikiMemory]):
    """Repository for wiki memory data access. No business logic."""

    def __init__(self):
        super().__init__(WikiMemory)

    async def vector_search(
        self,
        repository_id: UUID,
        query_embedding: List[float],
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
        score_threshold: float = 0.7,
    ) -> List[WikiMemory]:
        """Execute vector search aggregation pipeline."""
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "wiki_memories_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"repository_id": str(repository_id)}
                }
            },
            {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
            {"$match": {"score": {"$gte": score_threshold}}},
        ]
        if memory_type:
            pipeline.append({"$match": {"memory_type": memory_type.value}})

        return await self.aggregate(pipeline)

    async def find_by_file_paths(
        self, repository_id: UUID, file_paths: List[str]
    ) -> List[WikiMemory]:
        """Find memories related to specific file paths."""
        return await self.find_many({
            "repository_id": str(repository_id),
            "file_paths": {"$in": file_paths}
        })

    async def delete_by_repository(self, repository_id: UUID) -> int:
        """Delete all memories for a repository."""
        result = await self.delete_many({"repository_id": str(repository_id)})
        return result.deleted_count if result else 0
```

### 3. WikiMemoryService (`src/services/wiki_memory_service.py`)

**All business logic centralized here:**

```python
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models.wiki_memory import WikiMemory, MemoryType
from ..repository.wiki_memory_repository import WikiMemoryRepository
from ..tools.embedding_tool import EmbeddingTool
from ..utils.config_loader import get_settings

logger = structlog.get_logger(__name__)

MAX_MEMORY_CHARS = 4000

class WikiMemoryService:
    """Service for wiki memory operations. All business logic lives here."""

    def __init__(
        self,
        wiki_memory_repo: WikiMemoryRepository,
        embedding_tool: EmbeddingTool,
    ):
        """Initialize with dependency injection.

        Args:
            wiki_memory_repo: WikiMemoryRepository instance (injected via DI).
            embedding_tool: EmbeddingTool instance (injected via DI).
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
        """Store a new memory with embedding generation."""
        try:
            # Truncate if needed
            if len(content) > MAX_MEMORY_CHARS:
                content = content[:MAX_MEMORY_CHARS]
                logger.warning(f"Memory content truncated to {MAX_MEMORY_CHARS} chars")

            # Generate embedding
            embedding = await self._embedding_tool.create_embedding(content)

            # Create and store memory
            memory = WikiMemory(
                repository_id=repository_id,
                content=content,
                memory_type=memory_type,
                source_agent=source_agent,
                embedding=embedding,
                file_paths=file_paths or [],
                related_pages=related_pages or [],
            )
            await self._wiki_memory_repo.insert(memory)

            return {
                "status": "success",
                "memory_id": str(memory.id),
                "repository_id": str(repository_id),
            }
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def search_memories(
        self,
        repository_id: UUID,
        query: str,
        limit: int = 10,
        memory_type: Optional[MemoryType] = None,
    ) -> Dict[str, Any]:
        """Search memories using semantic similarity."""
        try:
            # Generate query embedding
            query_embedding = await self._embedding_tool.create_embedding(query)

            # Search
            memories = await self._wiki_memory_repo.vector_search(
                repository_id=repository_id,
                query_embedding=query_embedding,
                limit=limit,
                memory_type=memory_type,
            )

            return {
                "status": "success",
                "memories": [self._wiki_memory_repo.serialize(m) for m in memories],
                "count": len(memories),
            }
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def get_memories_by_files(
        self,
        repository_id: UUID,
        file_paths: List[str],
    ) -> Dict[str, Any]:
        """Get memories related to specific files."""
        try:
            memories = await self._wiki_memory_repo.find_by_file_paths(
                repository_id=repository_id,
                file_paths=file_paths,
            )
            return {
                "status": "success",
                "memories": [self._wiki_memory_repo.serialize(m) for m in memories],
                "count": len(memories),
            }
        except Exception as e:
            logger.error(f"Failed to get file memories: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def purge_memories(self, repository_id: UUID) -> Dict[str, Any]:
        """Purge all memories for a repository (force regenerate)."""
        try:
            deleted_count = await self._wiki_memory_repo.delete_by_repository(
                repository_id=repository_id
            )
            logger.info(f"Purged {deleted_count} memories for repository {repository_id}")
            return {
                "status": "success",
                "deleted_count": deleted_count,
                "repository_id": str(repository_id),
            }
        except Exception as e:
            logger.error(f"Failed to purge memories: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
            }

    @classmethod
    async def purge_for_repository(cls, repository_id: UUID) -> Dict[str, Any]:
        """Class method to purge memories - creates its own service instance.

        This is used by workflow nodes that need to purge memories without
        having a service instance (e.g., force_regenerate in wiki_workflow.py).

        Args:
            repository_id: Repository ID to purge memories for.

        Returns:
            Dict with status and deleted_count.
        """
        from ..repository.wiki_memory_repository import WikiMemoryRepository
        from ..tools.embedding_tool import EmbeddingTool

        wiki_memory_repo = WikiMemoryRepository()
        embedding_tool = EmbeddingTool()
        service = cls(wiki_memory_repo, embedding_tool)
        return await service.purge_memories(repository_id)
```

### 4. WikiMemoryMiddleware (`src/agents/middleware/wiki_memory_middleware.py`)

**Self-contained middleware class that extracts `repository_id` from agent state:**

```python
"""Wiki memory middleware for agents - provides persistent memory capabilities.

This middleware is self-contained:
- Initializes its own WikiMemoryService internally (lazy loading)
- Extracts repository_id from agent state automatically (no need to pass it)
- Only requires source_agent to be specified in constructor
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, cast
from uuid import UUID

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

from ...models.wiki_memory import MemoryType


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
- **Patterns found**: When you identify recurring patterns in the codebase (e.g., "all handlers follow X pattern")
- **Cross-references**: When you notice relationships between different parts of the codebase

### When to Recall Memories:
- Before making structural decisions, recall relevant past memories
- When analyzing files, check for existing memories about those files
- When you need context from previous wiki generations

### Memory Types:
- `structural_decision`: Decisions about wiki structure and organization
- `pattern_found`: Patterns or conventions discovered in the codebase
- `cross_reference`: Relationships between different code areas or wiki sections

Be proactive about storing important decisions so they persist across regenerations."""


class WikiMemoryMiddleware(AgentMiddleware):
    """Middleware that provides persistent memory capabilities to wiki agents.

    Self-contained middleware that:
    - Extracts repository_id from agent state automatically
    - Creates its own WikiMemoryService internally (lazy loading)
    - Injects system prompt guidance for using the memory system

    Usage:
        middleware=[
            WikiMemoryMiddleware("structure_agent"),  # Just pass source_agent
            TodoListMiddleware(),
            ...
        ]
    """

    state_schema = AgentState

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
        super().__init__()
        self.source_agent = source_agent
        self.system_prompt = system_prompt
        self._service = None
        self._repository_id: Optional[UUID] = None

        # Create tools with closure over self
        # Tools access self._repository_id and self._get_service()

        @tool(description="Store a structural decision, pattern, or cross-reference for future wiki generations.")
        async def store_memory(
            content: str,
            memory_type: str,
            file_paths: Optional[List[str]] = None,
            related_pages: Optional[List[str]] = None,
        ) -> str:
            """Store a memory for future reference."""
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

        @tool(description="Recall relevant memories based on semantic similarity to your query.")
        async def recall_memories(
            query: str,
            limit: int = 5,
            memory_type: Optional[str] = None,
        ) -> str:
            """Recall relevant past memories using semantic search."""
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

        @tool(description="Get memories related to specific source files you're analyzing.")
        async def get_file_memories(
            file_paths: List[str],
        ) -> str:
            """Get memories related to specific source files."""
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
            from ...repository.wiki_memory_repository import WikiMemoryRepository
            from ...services.wiki_memory_service import WikiMemoryService
            from ...tools.embedding_tool import EmbeddingTool

            wiki_memory_repo = WikiMemoryRepository()
            embedding_tool = EmbeddingTool()
            self._service = WikiMemoryService(
                wiki_memory_repo=wiki_memory_repo,
                embedding_tool=embedding_tool,
            )
        return self._service

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Extract repository_id from state and inject memory system prompt."""
        # Extract repository_id from agent state on first model call
        if self._repository_id is None:
            state = getattr(request, 'state', None)
            if state is not None:
                if isinstance(state, dict):
                    repo_id = state.get("repository_id")
                else:
                    repo_id = getattr(state, "repository_id", None)
                if repo_id is not None:
                    self._repository_id = repo_id if isinstance(repo_id, UUID) else UUID(str(repo_id))

        # Inject system prompt
        if request.system_message is not None:
            new_system_content = [
                *request.system_message.content_blocks,
                {"type": "text", "text": f"\n\n{self.system_prompt}"},
            ]
        else:
            new_system_content = [{"type": "text", "text": self.system_prompt}]

        new_system_message = SystemMessage(
            content=cast("list[str | dict[str, str]]", new_system_content)
        )
        return await handler(request.override(system_message=new_system_message))
```

### 5. Integration in wiki_react_agents.py

**Modifications to `create_structure_agent()` and `create_page_agent()`:**

The memory middleware is a proper class that extracts `repository_id` from agent state automatically.
Just instantiate it with `source_agent` - no need to pass repository_id!

```python
from langchain.agents import create_agent
from ..middleware.wiki_memory_middleware import WikiMemoryMiddleware

async def create_structure_agent():
    """Create structure agent with memory capabilities."""
    # Get existing MCP tools
    tools = await get_mcp_tools(["read_text_file", "read_multiple_files"])

    # Create agent with memory middleware added to existing middleware stack
    # Memory middleware extracts repository_id from agent state automatically
    exploration_agent = create_agent(
        model=llm,
        tools=tools,  # MCP tools only - memory tools come from middleware
        system_prompt=system_prompt,
        middleware=[
            WikiMemoryMiddleware("structure_agent"),  # NEW - just pass source_agent!
            TodoListMiddleware(),
            SummarizationMiddleware(model="gpt-4o-mini"),
            PatchToolCallsMiddleware(),
            ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
            ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        ]
    )

    return StructuredAgentWrapper(
        agent=exploration_agent,
        llm=llm,
        schema=LLMWikiStructureSchema,
    )


async def create_page_agent():
    """Create page agent with memory capabilities."""
    tools = await get_mcp_tools(["list_directory_with_sizes", "read_text_file", "read_multiple_files"])

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        middleware=[
            WikiMemoryMiddleware("page_agent"),  # NEW - just pass source_agent!
            TodoListMiddleware(),
            SummarizationMiddleware(model="gpt-4o-mini"),
            PatchToolCallsMiddleware(),
            ModelRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
            ToolRetryMiddleware(max_retries=3, backoff_factor=2.0, initial_delay=1.0),
        ]
    )
```

### 6. Integration in wiki_workflow.py

**Handle force_regenerate - no service in state needed:**

The workflow only needs to handle memory purging for force_regenerate. This is done via the
`WikiMemoryService.purge_for_repository()` class method - no service instance needed in state.

```python
from typing import TypedDict, Optional
from uuid import UUID
import operator

from ...services.wiki_memory_service import WikiMemoryService

class WikiWorkflowState(TypedDict):
    """State for wiki generation workflow."""
    repository_id: UUID  # Middleware extracts this from state automatically
    repository_url: str
    branch: str
    repo_dir: str
    force_regenerate: bool  # NEW: Flag to purge memories
    # ... existing fields
    structure: Optional[WikiStructure]
    pages: Annotated[list, operator.add]


async def extract_structure_node(state: WikiWorkflowState) -> dict:
    """Extract wiki structure using structure agent."""
    repository_id = state["repository_id"]

    # Purge memories if force regenerate (using class method - no service instance needed)
    if state.get("force_regenerate", False):
        await WikiMemoryService.purge_for_repository(repository_id)
        logger.info(f"Purged memories for repository {repository_id} (force_regenerate=True)")

    # Create structure agent - parameterless, middleware extracts repository_id from state
    agent = await create_structure_agent()

    # Invoke agent - state contains repository_id, middleware will extract it
    result = await agent.ainvoke({"messages": [...]}, config={"state": state})
    # ... rest of implementation


async def generate_pages_node(state: WikiWorkflowState) -> dict:
    """Generate individual wiki pages using page agent."""

    # Create page agent - parameterless, middleware extracts repository_id from state
    agent = await create_page_agent()

    # Generate pages - state contains repository_id, middleware will extract it
    for page in pages_to_generate:
        result = await agent.ainvoke({"messages": [...]}, config={"state": state})
        # ... rest of implementation
```

### 7. Workflow Invocation

**No service factory needed - middleware is self-contained:**

```python
# Usage in wiki_agent.py or wherever wiki generation is triggered
async def generate_wiki(repository_id: UUID, force_regenerate: bool = False):
    """Generate wiki with memory support.

    The memory middleware is self-contained and creates its own service
    internally. No service factory or state injection needed.
    """
    initial_state = {
        "repository_id": repository_id,
        "force_regenerate": force_regenerate,
        # NOTE: No wiki_memory_service - middleware is self-contained
        # ... other state
    }

    result = await wiki_workflow.ainvoke(initial_state)
    return result
```

---

## Implementation Steps

### Step 1: Create WikiMemory Model
**File:** `src/models/wiki_memory.py`
- Create `MemoryType` enum (structural_decision, pattern_found, cross_reference)
- Create `WikiMemory` Beanie document extending `BaseDocument`
- Define indexes for repository_id, memory_type, file_paths

### Step 2: Create WikiMemoryRepository
**File:** `src/repository/wiki_memory_repository.py`
- Extend `BaseRepository[WikiMemory]`
- Implement `vector_search()` - MongoDB Atlas Vector Search pipeline
- Implement `find_by_file_paths()` - find memories by related files
- Implement `delete_by_repository()` - purge all memories
- **No business logic - data access only**

### Step 3: Create WikiMemoryService
**File:** `src/services/wiki_memory_service.py`
- Constructor injection: `WikiMemoryRepository`, `EmbeddingTool`
- `store_memory()` - validate, truncate, generate embedding, store
- `search_memories()` - generate query embedding, vector search
- `get_memories_by_files()` - find related memories
- `purge_memories()` - instance method for purging
- `purge_for_repository()` - **class method** for workflow to call directly
- Standardized `Dict[str, Any]` response format

### Step 4: Create WikiMemoryMiddleware
**File:** `src/agents/middleware/wiki_memory_middleware.py`
- **Class** `WikiMemoryMiddleware(AgentMiddleware)` - NOT a factory function
- Constructor takes only `source_agent` - extracts `repository_id` from agent state
- Internal `_get_service()` for lazy service initialization
- Tools defined in `__init__` with closure over `self`
- `awrap_model_call()` extracts repository_id from state + injects system prompt
- Tools: `store_memory`, `recall_memories`, `get_file_memories`

### Step 5: Register WikiMemory Document
**File:** `src/repository/database.py`
- Add `from ..models.wiki_memory import WikiMemory` import
- Add `WikiMemory` to `document_models` list in `init_beanie()`

### Step 6: Update Exports
**Files:**
- `src/models/__init__.py` - export `WikiMemory`, `MemoryType`
- `src/repository/__init__.py` - export `WikiMemoryRepository`
- `src/services/__init__.py` - export `WikiMemoryService`

### Step 7: Integrate into wiki_react_agents.py
**File:** `src/agents/wiki_react_agents.py`
- Import `create_wiki_memory_middleware`
- Add memory middleware to middleware stack (first position)
- **NO service parameter needed** - middleware is self-contained

### Step 8: Integrate into wiki_workflow.py
**File:** `src/agents/wiki_workflow.py`
- Add `force_regenerate: bool` to `WikiWorkflowState`
- In `extract_structure_node()`: call `WikiMemoryService.purge_for_repository()` if force_regenerate
- **NO wiki_memory_service in state** - middleware creates its own

### Step 9: Create MongoDB Vector Search Index
**MongoDB Atlas:**
- Create index `wiki_memories_index` on collection `wiki_memories`
- Vector field: `embedding` (384 dimensions, cosine similarity)
- Filter fields: `repository_id`, `memory_type`

### Step 10: Update Serena Memory
- Update `wiki_agent_memory_feature` with final implementation details

---

## Verification Plan

### Unit Tests
```bash
# Run after implementation
pytest tests/unit/test_wiki_memory_service.py -v
pytest tests/unit/test_wiki_memory_repository.py -v
```

### Integration Tests
1. **Store memory and verify in MongoDB:**
   - Use MongoDB MCP tool to query `wiki_memories` collection
   - Verify document structure and embedding field populated

2. **Vector search verification:**
   - Store 3+ memories with different content
   - Search with query, verify semantic relevance ordering

### End-to-End Tests
1. **Generate wiki with memory tools:**
   ```bash
   # Trigger wiki generation via API or direct call
   # Monitor logs for memory tool usage
   # Query wiki_memories collection to verify memories stored
   ```

2. **Regenerate wiki (non-force):**
   - Generate wiki again for same repository
   - Verify agent uses `recall_memories` tool
   - Verify existing memories are preserved

3. **Force regenerate:**
   - Call with `force_regenerate=True`
   - Verify memories purged before generation
   - Verify new memories stored after generation

### Manual Verification Commands
```bash
# Check memories in MongoDB
mongosh --eval 'db.wiki_memories.find({repository_id: "<uuid>"}).pretty()'

# Check vector search index exists
mongosh --eval 'db.wiki_memories.getSearchIndexes()'

# Run specific test file
pytest tests/integration/test_wiki_memory.py -v -s
```

---

## Key Architectural Principles Honored

| Principle | Implementation |
|-----------|----------------|
| **Proper Class Middleware** | `WikiMemoryMiddleware` is a class (not factory function) like `TodoListMiddleware` |
| **State-Driven Configuration** | Middleware extracts `repository_id` from agent state - no need to pass it |
| **Self-Contained** | Middleware initializes its own service via lazy `_get_service()` |
| **Repository via Service Only** | Middleware tools call service, service calls repository - never direct access |
| **Business Logic in Service** | Embedding generation, truncation, validation, error handling all in service |
| **Constructor Injection** | WikiMemoryService still uses constructor injection for its dependencies |
| **LangGraph Middleware** | `AgentMiddleware` base class (same as `TodoListMiddleware`) |
| **Tools via Closure** | Tools defined in `__init__` with closure over `self` - access instance state |
| **System Prompt Injection** | `awrap_model_call()` extracts state + injects memory guidance |
| **Class Method for Workflow** | `WikiMemoryService.purge_for_repository()` allows workflow to purge without instance |
| **Clean Agent API** | `WikiMemoryMiddleware("structure_agent")` - simple instantiation |
| **Existing Patterns** | Follows WikiGenerationService, BaseRepository, TodoListMiddleware patterns |
