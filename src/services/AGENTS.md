# AGENTS.md - Services Layer Guide

## 1. Overview

The services layer contains business logic with **constructor dependency injection**. Services orchestrate operations between repositories (data access) and agents (AI workflows) without directly touching the database.

**Location:** `src/services/`

**Core services:**
- `WikiGenerationService` - Wiki creation and management
- `RepositoryService` - Repository CRUD and webhook configuration
- `DocumentProcessingService` - Document analysis and embeddings

## 2. Setup

No direct dependencies to install - services receive all dependencies via constructor injection:
- Repositories from `src/repository/`
- Agents from `src/agents/`
- Tools from `src/tools/`

Dependencies are wired in `src/dependencies.py` and injected via FastAPI's `Depends()`.

## 3. Build/Tests

```bash
# Run service unit tests
pytest tests/unit/test_services.py -v

# Run with coverage
pytest tests/unit/test_services.py --cov=src/services
```

## 4. Code Style

### Constructor Injection Pattern
```python
class WikiGenerationService:
    def __init__(
        self,
        wiki_structure_repo: WikiStructureRepository,
        code_document_repo: CodeDocumentRepository,
        wiki_agent: WikiGenerationAgent,
        context_tool: ContextTool,
        llm_tool: LLMTool,
    ):
        self._wiki_structure_repo = wiki_structure_repo
        self._code_document_repo = code_document_repo
        self._wiki_agent = wiki_agent
        self._context_tool = context_tool
        self._llm_tool = llm_tool
```

### Lazy Loading Pattern (for optional dependencies)
```python
async def _get_repo_repository(self):
    if self._repo_repository is None:
        from ..repository.repository_repository import RepositoryRepository
        self._repo_repository = RepositoryRepository(Repository)
    return self._repo_repository
```

### Agent Delegation Pattern
```python
async def generate_wiki(self, repository_id: UUID) -> Dict[str, Any]:
    # Delegate AI work to agents - do NOT call LLM directly
    generation_result = await self._wiki_agent.generate_wiki(
        repository_id=str(repository_id)
    )
    return {"status": generation_result["status"], ...}
```

## 5. Security

- **Input validation:** Validate all inputs before processing
- **Output sanitization:** Sanitize data before returning to API layer
- **Permission checks:** Use auth service for authorization
- **Secrets:** Never log or return sensitive data (tokens, keys)

## 6. PR Checklist

- [ ] Constructor receives all dependencies (no direct instantiation)
- [ ] No `import motor` or `import pymongo` statements
- [ ] No direct database access (`get_database()`, `collection`)
- [ ] All operations return `Dict[str, Any]` with `status` key
- [ ] Proper error handling with `try/except` blocks
- [ ] Logging uses `logger.error()` for failures
- [ ] AI operations delegated to agents, not inline

## 7. Examples

### Return Format (always include status key)
```python
async def get_wiki_structure(self, repository_id: UUID) -> Dict[str, Any]:
    try:
        wiki = await self._wiki_structure_repo.find_one(...)
        if not wiki:
            return {
                "status": "error",
                "error": "Wiki not found",
                "error_type": "WikiNotFound",
            }
        return {
            "status": "success",
            "wiki_structure": wiki.model_dump(),
        }
    except Exception as e:
        logger.error(f"Get wiki structure failed: {e}")
        return {"status": "error", "error": str(e), "error_type": type(e).__name__}
```

### Repository Access (via injected repo)
```python
# Correct: Use injected repository
doc_count = await self._code_document_repo.count({"repository_id": str(id)})

# Wrong: Direct database access
# db = await get_database()  # NEVER do this in services
# doc_count = await db.documents.count_documents({...})
```

## 8. When Stuck

1. **DI not working:** Check `src/dependencies.py` for provider functions
2. **Repository method missing:** Check `src/repository/base.py` for inherited methods
3. **Agent not responding:** Verify agent is properly initialized in dependencies
4. **Type errors:** Services use `Dict[str, Any]` returns, not Pydantic models

## 9. House Rules

| Rule | Violation Example | Correct Approach |
|------|-------------------|------------------|
| NEVER import motor/pymongo | `from motor.motor_asyncio import...` | Use repository methods |
| NEVER instantiate repos directly | `repo = CodeDocumentRepository()` | Receive via constructor |
| NEVER access database connection | `db = await get_database()` | Call `self._repo.find_one()` |
| ALWAYS return Dict with status | `return wiki` | `return {"status": "success", "wiki": wiki}` |
| ALWAYS delegate AI to agents | `response = await llm.generate()` | `await self._agent.process()` |
