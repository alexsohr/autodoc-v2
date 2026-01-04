# Deep Agent for Wiki Structure Generation

**Date:** 2025-12-30
**Status:** Approved
**Author:** Claude (with user collaboration)

## Overview

Replace the `_generate_structure_node` in `WikiGenerationAgent` with a LangGraph Deep Agent that can autonomously explore the cloned repository filesystem to generate a more accurate wiki structure.

## Background

### Current Implementation
- `_generate_structure_node` receives `file_tree` (ASCII text) and `readme_content`
- Uses single LLM call with structured output (`WikiStructureSchema`)
- No ability to inspect actual file contents or understand code semantics

### Problem
- Static file tree doesn't capture code relationships
- Cannot read config files, imports, or understand architecture
- Wiki structure is based on surface-level analysis

## Solution: Deep Agent

### Why Deep Agent over ReAct?

| Aspect | ReAct Agent | Deep Agent |
|--------|-------------|------------|
| Filesystem tools | Must build manually | Built-in: `ls`, `read_file`, `glob`, `grep` |
| Planning | None | Built-in: `write_todos` |
| Context management | Manual | Auto-summarization at 170k tokens |
| Subagents | Must implement | Built-in `task` tool |
| Complexity | Lightweight | Feature-rich, autonomous |

Deep Agent is ideal because:
1. Filesystem interaction is central to the task
2. Planning helps methodical exploration
3. Context management handles large codebases
4. Built-in tools match our requirements exactly

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    _generate_structure_node                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Fetch Repository from DB (get clone_path)                   │
│                         ↓                                        │
│  2. Create Deep Agent with:                                      │
│     • FilesystemBackend(root_dir=clone_path)                    │
│     • System prompt with context (file_tree, readme)            │
│     • Custom tool: finalize_wiki_structure                      │
│                         ↓                                        │
│  3. Agent explores repository autonomously:                      │
│     • Uses built-in: ls, read_file, glob, grep                  │
│     • Plans exploration with write_todos                        │
│     • Reads key files (package.json, pyproject.toml, etc.)      │
│                         ↓                                        │
│  4. Agent calls finalize_wiki_structure(structure)              │
│     • This tool captures the final WikiStructure                │
│                         ↓                                        │
│  5. Extract structure, update state, return                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Dependencies

```bash
pip install deepagents
```

Add to `pyproject.toml`:
```toml
dependencies = [
    # ... existing deps
    "deepagents>=0.2.0",
]
```

### 2. Deep Agent Configuration

```python
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI

backend = FilesystemBackend(root_dir=clone_path)
model = ChatOpenAI(model="gpt-4o", temperature=0)

agent = create_deep_agent(
    model=model,
    backend=backend,
    tools=[finalize_wiki_structure],
    system_prompt=STRUCTURE_GENERATION_PROMPT,
)
```

### 3. Structured Output via Tool Closure

```python
captured_structure = {}

@tool
def finalize_wiki_structure(
    title: str,
    description: str,
    pages: List[dict]
) -> str:
    """Finalize the wiki structure after exploration."""
    nonlocal captured_structure
    captured_structure = {
        "title": title,
        "description": description,
        "pages": pages
    }
    return "Wiki structure captured successfully"
```

### 4. System Prompt

```python
STRUCTURE_GENERATION_PROMPT = """You are an expert technical writer analyzing a repository to design a wiki structure.

## Context Provided
- Repository: {owner}/{repo}
- Initial file tree: {file_tree}
- README content: {readme_content}

## Your Task
Explore this repository to understand its architecture, then design a comprehensive wiki structure.

## Exploration Strategy
1. Start by examining the file tree and README provided
2. Use `ls` and `glob` to discover project structure
3. Read key files to understand the codebase:
   - Config files: package.json, pyproject.toml, Cargo.toml, etc.
   - Entry points: main.py, index.ts, App.tsx, etc.
   - Core modules and their purposes
4. Use `grep` to find patterns like class definitions, API routes, exports
5. Use `write_todos` to track your exploration progress

## Output Requirements
When you have sufficient understanding, call `finalize_wiki_structure` with:
- 8-12 pages covering key aspects
- Logical sections (Architecture, Features, API, Deployment, etc.)
- Each page with relevant source file paths
- Cross-references between related pages

Focus on what would help a new developer understand this codebase.
"""
```

### 5. Error Handling

| Error Case | Handling |
|------------|----------|
| Repository not found | Return state with error message |
| No clone_path | Return state with error message |
| Clone path deleted | Return state with error message |
| Agent timeout | 5 minute limit, graceful failure |
| Agent didn't call finalize tool | Detect empty capture, return error |
| Unexpected exception | Catch-all with logging |

### 6. Node Implementation

```python
async def _generate_structure_node(
    self, state: WikiGenerationState
) -> WikiGenerationState:
    try:
        state["current_step"] = "generating_structure"
        state["progress"] = 30.0

        # Validate repository
        repository = await self._repository_repo.find_one(
            {"_id": UUID(state["repository_id"])}
        )
        if not repository or not repository.clone_path:
            state["error_message"] = "Repository not found or not cloned"
            return state

        clone_path = Path(repository.clone_path)
        if not clone_path.exists():
            state["error_message"] = f"Clone path does not exist: {clone_path}"
            return state

        # Setup output capture
        captured_structure = {}

        @tool
        def finalize_wiki_structure(title: str, description: str, pages: List[dict]) -> str:
            nonlocal captured_structure
            captured_structure = {"title": title, "description": description, "pages": pages}
            return "Wiki structure captured successfully"

        # Create and run agent
        agent = create_deep_agent(
            backend=FilesystemBackend(root_dir=str(clone_path)),
            tools=[finalize_wiki_structure],
            system_prompt=self._format_structure_prompt(state, repository),
        )

        await asyncio.wait_for(
            agent.ainvoke({"messages": [{"role": "user", "content": "Analyze and create wiki structure."}]}),
            timeout=300.0
        )

        # Validate output
        if not captured_structure or not captured_structure.get("pages"):
            state["error_message"] = "Agent did not produce valid wiki structure"
            return state

        state["wiki_structure"] = captured_structure
        state["progress"] = 50.0
        return state

    except asyncio.TimeoutError:
        state["error_message"] = "Structure generation timed out"
        return state
    except Exception as e:
        logger.error(f"Structure generation failed: {e}")
        state["error_message"] = f"Structure generation failed: {str(e)}"
        return state
```

## Testing Strategy

### Unit Tests
- Mock `create_deep_agent` to test node logic
- Test error handling paths (no repo, no clone_path, timeout)
- Test closure capture mechanism

### Integration Tests
- Use small sample repository fixture
- Verify agent explores files and produces valid structure
- Test timeout behavior

### E2E Tests
- Existing BDD tests should continue to pass
- Add scenario for Deep Agent exploration

## Migration Path

1. Add `deepagents` dependency
2. Implement new `_generate_structure_node`
3. Keep old implementation as `_generate_structure_node_legacy` (temporary)
4. Feature flag to switch between implementations
5. Validate with integration tests
6. Remove legacy implementation

## Open Questions

None - design approved.

## References

- [Deep Agents GitHub](https://github.com/langchain-ai/deepagents)
- [Deep Agents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangChain Blog: Deep Agents](https://blog.langchain.com/deep-agents/)
