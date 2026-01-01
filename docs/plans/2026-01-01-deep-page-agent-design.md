# Deep Page Agent Design

## Overview

Replace the simple `page_worker_node` (which just reads predefined files) with a deep agent that autonomously explores the codebase using MCP filesystem tools to generate comprehensive wiki pages.

## Architecture Decision

**Selected: Option A - Mirrored Deep Agent Pattern**

Create `deep_page_agent.py` following the exact same pattern as `deep_structure_agent.py`:
- Same MCP filesystem tools setup with `FilesystemBackend` fallback
- Same `create_deep_agent()` + custom finalize tool pattern
- Page-specific system prompt focused on generating documentation for a single topic

## DeepPageAgent Class

```python
class DeepPageAgent:
    def __init__(
        self,
        clone_path: str,
        page_definition: Dict[str, Any],  # title, description, topics, file_paths
        repository_context: Dict[str, Any],  # repo name, languages, description
        model: str = "openai:gpt-4o-mini",
        max_iterations: int = 15,
    ):
        self.clone_path = clone_path
        self.page_definition = page_definition
        self.repository_context = repository_context
        self.model = model
        self.max_iterations = max_iterations
        self._result: Optional[Dict[str, Any]] = None
```

## Output Schema

```python
class PageSection(BaseModel):
    heading: str
    content: str  # Markdown content
    code_examples: List[Dict[str, str]]  # language, code, description

class PageContent(BaseModel):
    title: str
    summary: str
    sections: List[PageSection]
    key_files: List[str]  # Files referenced in page
```

## System Prompt

The agent uses a comprehensive prompt that requires:
- Minimum 5 source files for comprehensive coverage
- Source citations with line numbers for every claim
- Mermaid diagrams (strict vertical orientation)
- Tables for structured data (APIs, config, models)
- Code snippets from actual codebase
- Details block listing all source files used

Key prompt elements:
- Start with file hints, expand outward through exploration
- Search for related patterns (imports, usages, tests)
- Real code examples only (no fabrication)
- Strict citation requirements

## Integration with page_worker_node

```python
async def page_worker_node(state: PageWorkerState) -> Dict[str, Any]:
    agent = DeepPageAgent(
        clone_path=state["clone_path"],
        page_definition={
            "title": page["title"],
            "description": page.get("description", ""),
            "topics": page.get("topics", []),
            "file_paths": page.get("file_paths", [])
        },
        repository_context=state["repository_context"],
        model=state.get("model", "openai:gpt-4o-mini"),
        max_iterations=15
    )
    result = await agent.run()
    return {"generated_pages": [{"page_id": page["id"], "content": result}]}
```

## MCP Tools

Available filesystem tools:
- `read_file` / `read_text_file` - Read file contents
- `search_files` - Search for patterns across codebase
- `list_directory` / `directory_tree` - Explore structure
- `get_file_info` - File metadata

Fallback to `FilesystemBackend` when MCP not available.

## Trade-offs

**Pros:**
- Autonomous exploration produces higher quality documentation
- Real code examples with accurate line citations
- Discovers related files beyond initial hints
- Consistent pattern with deep_structure_agent.py

**Cons:**
- Slower (~30-60s per page vs ~5s for simple approach)
- Higher cost (multiple LLM calls per page)
- More complex error handling needed
