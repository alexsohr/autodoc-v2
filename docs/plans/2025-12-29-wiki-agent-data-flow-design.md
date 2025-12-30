# Wiki Agent Data Flow Redesign

**Date:** 2025-12-29
**Status:** Implemented
**Branch:** 003-updating-mongodb-to-beanie

## Problem

The wiki agent currently re-fetches data that is already available from the document processing step:
- File tree is rebuilt from CodeDocuments in the database
- README content is searched for in CodeDocuments
- Repository info is fetched separately

This is redundant since `process_documents` already returns:
- `clone_path` - where the repo was cloned
- `documentation_files` - list of `{path, content}` dicts
- `file_tree` - ASCII tree string

## Solution

Pass the document processing output through the orchestrator to the wiki agent, eliminating redundant database queries.

## Data Flow

```
DocumentAgent.process_repository()
    │
    ├── clone_path
    ├── documentation_files: [{path, content}, ...]
    └── file_tree: str
            │
            ▼
WorkflowOrchestrator._process_documents_node()
    │
    ├── Store clone_path in Repository document
    ├── Format documentation_files into readme_content string
    └── Store in state["results"]["document_processing"]
            │
            ▼
WorkflowOrchestrator._generate_wiki_node()
    │
    ├── Extract file_tree from state
    ├── Extract readme_content from state
    └── Pass to wiki_agent.generate_wiki()
            │
            ▼
WikiGenerationAgent.generate_wiki(repository_id, file_tree, readme_content)
    │
    └── Use provided values directly (no re-fetching)
```

## Documentation Files Format

All documentation files are concatenated into a single string with file citations:

```
--- README.md ---
# Project Title

Introduction content here...

--- docs/CONTRIBUTING.md ---
# Contributing Guidelines

Guidelines content here...

--- docs/API.md ---
# API Reference

API documentation here...
```

## Implementation Changes

### 1. Repository Model (`src/models/repository.py`)

Add `clone_path` field:

```python
clone_path: Optional[str] = Field(default=None, description="Local clone path")
```

### 2. WikiGenerationState (`src/agents/wiki_agent.py`)

Remove `repository_info` field:

```python
class WikiGenerationState(TypedDict):
    repository_id: str
    # repository_info removed - use repository_id to fetch if needed
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]
    generated_pages: List[Dict[str, Any]]
    current_page: Optional[str]
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
```

### 3. WikiGenerationAgent.generate_wiki (`src/agents/wiki_agent.py`)

Update method signature:

```python
async def generate_wiki(
    self,
    repository_id: str,
    file_tree: str = "",
    readme_content: str = "",
    force_regenerate: bool = False,
) -> Dict[str, Any]:
```

Initialize state with provided values:

```python
initial_state: WikiGenerationState = {
    "repository_id": repository_id,
    "file_tree": file_tree,
    "readme_content": readme_content,
    # ... rest of state
}
```

### 4. WikiGenerationAgent._analyze_repository_node (`src/agents/wiki_agent.py`)

Simplify to use pre-populated values:

- Skip file tree building (already provided)
- Skip README search (already provided)
- Only fetch Repository if needed for other validation

### 5. WorkflowOrchestrator._process_documents_node (`src/agents/workflow.py`)

After successful processing:

```python
# Store clone_path in Repository
await self._repository_repo.update_one(
    {"id": state["repository_id"]},
    {"$set": {"clone_path": processing_result["clone_path"]}}
)

# Format documentation files
doc_files = processing_result.get("documentation_files", [])
readme_content = self._format_documentation_files(doc_files)

# Store in state for wiki agent
state["results"]["document_processing"] = {
    "file_tree": processing_result["file_tree"],
    "readme_content": readme_content,
    "clone_path": processing_result["clone_path"],
}
```

### 6. WorkflowOrchestrator._generate_wiki_node (`src/agents/workflow.py`)

Extract and pass data:

```python
doc_result = state["results"].get("document_processing", {})
file_tree = doc_result.get("file_tree", "")
readme_content = doc_result.get("readme_content", "")

wiki_result = await self._wiki_agent.generate_wiki(
    repository_id=state["repository_id"],
    file_tree=file_tree,
    readme_content=readme_content,
    force_regenerate=state["force_update"],
)
```

### 7. Helper Method (`src/agents/workflow.py`)

Add helper to format documentation files:

```python
def _format_documentation_files(self, doc_files: List[Dict[str, str]]) -> str:
    """Format documentation files into a single string with citations."""
    if not doc_files:
        return ""

    parts = []
    for doc in doc_files:
        path = doc.get("path", "unknown")
        content = doc.get("content", "")
        parts.append(f"--- {path} ---\n{content}")

    return "\n\n".join(parts)
```

## Files Modified

| File | Changes |
|------|---------|
| `src/models/repository.py` | Add `clone_path` field |
| `src/agents/wiki_agent.py` | Update `WikiGenerationState`, `generate_wiki`, `_analyze_repository_node` |
| `src/agents/workflow.py` | Update `_process_documents_node`, `_generate_wiki_node`, add helper |

## Benefits

1. **Eliminates redundant queries** - No re-fetching of file tree and README
2. **Single source of truth** - Document processing output flows directly to wiki generation
3. **Simpler wiki agent** - Less responsibility, cleaner code
4. **Clone path tracked** - Repository now knows where it was cloned
5. **All docs available** - Wiki agent gets all documentation files, not just README
