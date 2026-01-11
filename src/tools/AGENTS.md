# LangChain Tools Subsystem

## Overview

LangChain tools for LangGraph agents providing RAG (Retrieval-Augmented Generation), vector embeddings, multi-provider LLM operations, and Git repository analysis. These tools integrate with LangGraph workflows to enable AI-powered documentation generation.

Key tools:
- `LLMTool` - Multi-provider text generation, chat, and streaming (OpenAI, Gemini, Ollama)
- `EmbeddingTool` - Vector embedding generation, storage, and similarity search
- `ContextTool` - Semantic search and RAG context retrieval

## Setup

Required dependencies:
```bash
pip install langchain-core langchain-openai langchain-google-genai langchain-community
```

Environment variables:
```env
OPENAI_API_KEY=your-key      # For OpenAI provider
GOOGLE_API_KEY=your-key      # For Gemini provider
OLLAMA_BASE_URL=http://...   # For local Ollama
```

## Build/Tests

```bash
# Run tool tests
pytest tests/unit/test_tools.py -v

# Run with coverage
pytest tests/unit/test_tools.py --cov=src/tools --cov-report=term-missing
```

## Code Style

### Extend BaseTool
All tools must extend `langchain_core.tools.BaseTool`:
```python
from langchain_core.tools import BaseTool

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Tool description for LangGraph"
```

### Async-Only Implementation
Implement `_arun()` for async operations; raise `NotImplementedError` in `_run()`:
```python
async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
    if operation == "generate":
        return await self.generate(**kwargs)
    raise ValueError(f"Unknown operation: {operation}")

def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
    raise NotImplementedError("Tool only supports async operations")
```

### Pydantic Input Models
Define input schemas using Pydantic for validation:
```python
from pydantic import BaseModel, Field

class MyToolInput(BaseModel):
    query: str = Field(description="Search query text")
    limit: int = Field(default=10, description="Max results")
```

### Multi-Provider Support
Use lazy initialization for provider flexibility:
```python
def _get_provider(self, provider: Optional[str] = None):
    if not self._providers_initialized:
        self._setup_providers(self._settings)
        self._providers_initialized = True
    return self._providers.get(provider) or next(iter(self._providers.values()))
```

### Dependency Injection
Inject repositories via constructor, not global singletons:
```python
def __init__(self, code_document_repo: CodeDocumentRepository):
    super().__init__()
    self._code_document_repo = code_document_repo
```

## Security

- **API Keys**: Load from `get_settings()`, never hardcode credentials
- **File Paths**: Validate paths before file system operations
- **LLM Outputs**: Sanitize generated content before storage or display
- **Input Validation**: Use Pydantic models to validate all tool inputs
- **Error Handling**: Never expose internal errors or stack traces to users

## PR Checklist

- [ ] Tool extends `langchain_core.tools.BaseTool`
- [ ] `_arun()` implemented, `_run()` raises `NotImplementedError`
- [ ] Input schema defined with Pydantic model
- [ ] All providers tested (OpenAI, Gemini, Ollama)
- [ ] Dependencies injected via constructor
- [ ] Error handling returns structured `Dict[str, Any]` with `status` field
- [ ] Unit tests added in `tests/unit/test_tools.py`

## Examples

### Complete Tool Implementation
```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional

class GenerateInput(BaseModel):
    prompt: str = Field(description="Text prompt")
    provider: Optional[str] = Field(default=None)

class MyTool(BaseTool):
    name: str = "my_tool"
    description: str = "Tool for generation operations"

    def __init__(self, repo: MyRepository):
        super().__init__()
        self._repo = repo
        self._providers: Dict = {}
        self._initialized = False

    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        if operation == "generate":
            return await self.generate(**kwargs)
        raise ValueError(f"Unknown operation: {operation}")

    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError("Only async supported")

    async def generate(self, prompt: str, provider: Optional[str] = None) -> Dict[str, Any]:
        try:
            llm = self._get_provider(provider)
            result = await llm.ainvoke(prompt)
            return {"status": "success", "result": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}
```

## When Stuck

1. **Provider not initializing**: Check API keys in `.env` and `get_settings()` configuration
2. **Tool not registered**: Verify tool is added to LangGraph workflow in `src/agents/`
3. **Input validation fails**: Check Pydantic model field types and descriptions
4. **Async errors**: Ensure using `await` with all `ainvoke()` and `aembed_*()` calls
5. **Vector search empty**: Verify embeddings exist in MongoDB with `has_embedding()` check

## House Rules

1. **Async-only**: All tools use `_arun()`, never implement sync `_run()` logic
2. **Dependency injection**: Repositories passed via constructor, not imported globally
3. **Multi-provider**: Support OpenAI, Gemini, and Ollama; use lazy initialization
4. **Pydantic validation**: All inputs validated with Pydantic models
5. **Structured returns**: All methods return `Dict[str, Any]` with `status` field
6. **Error handling**: Catch exceptions, log errors, return structured error responses
7. **No singletons**: Use `get_*_tool()` from `src.dependencies` with FastAPI `Depends()`
