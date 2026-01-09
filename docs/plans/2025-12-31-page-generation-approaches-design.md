# Wiki Page Generation Approaches - Design Document

**Date:** 2025-12-31
**Status:** Draft (Updated)
**Author:** Claude (AI Assistant)
**Related Trace:** `019b7110-56e8-7bf0-83e3-36a4e178f9f1`

## Executive Summary

This document analyzes approaches for generating wiki page content after the Deep Agent identifies the documentation structure. The current system successfully generates a wiki structure (sections and pages with associated file paths) but does not yet generate the actual page content.

**Updated Recommendation:**
1. **Phase 1 (Structure Discovery):** Use context-efficient exploration with chunked file reading (`head`/`tail` parameters) and `grep` for dependency discovery
2. **Phase 2 (Page Generation):** Use Approach 3 (Hybrid with Content Caching) with parallel page workers

This two-phase approach optimizes both context usage during exploration and parallelism during content generation.

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Context-Efficient Exploration Strategy](#2-context-efficient-exploration-strategy) *(NEW)*
3. [Approach 1: Sub-agents During Structure Generation](#3-approach-1-sub-agents-during-structure-generation)
4. [Approach 2: Parallel Agents After Structure Completion](#4-approach-2-parallel-agents-after-structure-completion)
5. [Approach 3: Hybrid with Content Caching](#5-approach-3-hybrid-with-content-caching)
6. [Comparative Analysis](#6-comparative-analysis)
7. [Final Recommendation](#7-final-recommendation)
8. [Implementation Plan](#8-implementation-plan)
9. [Next Steps](#9-next-steps)

---

## 1. Current State Analysis

### 1.1 Execution Flow

Based on LangSmith trace analysis (`019b7110-56e8-7bf0-83e3-36a4e178f9f1`):

```
workflow.full_analysis_workflow (58.88s total)
    │
    ├── document_agent.document_processing_workflow (0.59s)
    │       └── clone_repository, build_tree, extract_docs, etc.
    │
    └── wiki_agent.wiki_generation_workflow (58.19s)
            ├── analyze_repository
            ├── generate_structure (58.18s) ← Deep Agent runs here
            │       └── deep_agent (56.70s, 24,249 tokens)
            │               ├── 3 LLM calls (ChatOpenAI)
            │               ├── 6 MCP tool calls (read_text_file, read_multiple_files)
            │               └── finalize_wiki_structure tool call
            ├── generate_pages (0.00s) ← NOT WORKING
            ├── store_wiki
            └── finalize
```

### 1.2 Key Observations

| Metric | Value | Notes |
|--------|-------|-------|
| Total execution time | 58.88s | Almost entirely Deep Agent |
| Deep Agent duration | 56.70s | 3 LLM calls + 6 file reads |
| Token usage | 24,249 | prompt=21,893, completion=2,356 |
| Files read | ~15 files | Via MCP tools (read_text_file, read_multiple_files) |
| Pages generated | 0 | generate_pages node not producing content |

### 1.3 Current Architecture

```python
# deep_structure_agent.py - Current finalize tool schema
class WikiPageInput(BaseModel):
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    description: str = Field(description="Brief description of page content")
```

**Problem:** The `file_paths` are captured, but the actual file contents read by the Deep Agent are discarded after structure generation. The `_generate_pages_node` attempts to fetch files from the `CodeDocument` database, but:

1. File paths from Deep Agent may not match database records
2. Database query uses `repository_id` + `file_path`, which may have path format mismatches
3. No LLM calls appear in the trace for page generation

### 1.4 Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Agent Framework | deepagents | 0.3.1 |
| Workflow Orchestration | LangGraph | (latest) |
| File Access | MCP Filesystem Tools | - |
| LLM | OpenAI GPT-4o-mini | - |

**Key Capabilities:**
- **deepagents**: Built-in sub-agent spawning via `task()` tool, filesystem middleware, todo lists
- **LangGraph**: `Send` API for dynamic parallel fan-out/fan-in patterns
- **MCP Tools**: `read_text_file`, `read_multiple_files`, `search_files`

---

## 2. Context-Efficient Exploration Strategy

This section describes an optimized approach for the structure discovery phase that significantly reduces context token usage while improving the agent's understanding of codebase relationships.

### 2.1 Problem Statement

The current Deep Agent reads full file contents during exploration:

```
Current Flow:
1. Agent reads src/api.py (500 lines) → All 500 lines in context
2. Agent reads src/routes.py (300 lines) → All 300 lines in context
3. Agent reads src/models.py (400 lines) → All 400 lines in context
...
Total: ~15 files × ~300 lines avg = 4,500 lines (~18K tokens) in context
```

**The problem:** To determine page structure, the agent only needs high-level understanding (imports, class names, function signatures), not full implementation details.

### 2.2 Proposed Solution: Chunked Reading + Dependency Discovery

```
Optimized Flow:
1. Agent reads src/api.py (first 50 lines) → Imports, class signatures
   → Understands: "Flask API with User, Post routes"
   → Greps for "from src.api import" → Finds 3 dependent files
   → Decision: "I understand this component, move on"

2. Agent reads src/routes.py (first 50 lines) → Route definitions
   → Greps for "@app.route" patterns → Finds all endpoints
   → Decision: "This belongs in API Reference section"

Total: ~15 files × ~50 lines avg = 750 lines (~3K tokens)
```

**Result: 60-80% reduction in context tokens** for the exploration phase.

### 2.3 MCP Filesystem Server Configuration

The project already uses `@modelcontextprotocol/server-filesystem`:

```env
# .env configuration (already configured)
MCP_FILESYSTEM_ENABLED=true
MCP_FILESYSTEM_COMMAND=npx
MCP_FILESYSTEM_ARGS=-y,@modelcontextprotocol/server-filesystem
```

**GitHub Repository:** https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem

### 2.3.1 How the Deep Agent Uses MCP Tools

The Deep Agent uses MCP tools **directly** - no wrapper methods involved:

```python
# deep_structure_agent.py
mcp_tools = list(mcp_client._tools.values())  # Raw MCP tools
agent_kwargs["tools"] = list(mcp_tools) + [finalize_tool]  # Passed directly to agent
```

The `MCPFilesystemClient` wrapper class only handles:
1. **Initialization** - connecting to the MCP server
2. **Tool caching** - storing tools in `_tools` dict

The wrapper methods (`read_file()`, etc.) are used by other parts of the codebase (storage adapters), but **not by the Deep Agent**. The agent calls MCP tools directly via the LangChain tool interface.

### 2.4 Key MCP Tools for Context-Efficient Exploration

#### 2.4.1 `read_text_file` with `head`/`tail` Parameters

The MCP filesystem server supports partial file reading natively:

```typescript
// Tool signature from MCP filesystem server
read_text_file: {
  path: string,      // File path to read
  head?: number,     // Read only first N lines (optional)
  tail?: number      // Read only last N lines (optional)
}
```

**No wrapper changes needed.** The Deep Agent uses MCP tools directly:

```python
# deep_structure_agent.py - MCP tools passed directly to agent
mcp_tools = list(mcp_client._tools.values())
agent_kwargs["tools"] = list(mcp_tools) + [finalize_tool]
```

The agent can call the tool with parameters directly:
```
read_text_file(path="/repo/src/api.py", head=50)
```

#### 2.4.2 `search_files` for File Discovery

The MCP `search_files` tool searches file/directory **names** using glob patterns:

```typescript
search_files: {
  path: string,              // Starting directory
  pattern: string,           // Glob pattern (e.g., "*.py", "**/*controller*")
  excludePatterns?: string[] // Patterns to exclude
}
```

**Use cases:**
- Find all Python files: `search_files("/repo", "**/*.py")`
- Find controllers: `search_files("/repo", "**/*controller*.py")`
- Find config files: `search_files("/repo", "**/config*")`

#### 2.4.3 `grep` for Content Search (via deepagents)

The MCP filesystem server does **not** include content search (grep). However, deepagents provides a built-in `grep` tool via `FilesystemMiddleware`:

```python
# deepagents built-in grep tool
grep: {
  pattern: string,    # Regex pattern to search
  path: string,       # File or directory to search in
  include?: string    # File pattern to include (e.g., "*.py")
}
```

**Use cases:**
- Find imports: `grep("from src\\.api import", "/repo", "*.py")`
- Find class definitions: `grep("class\\s+\\w+Controller", "/repo")`
- Find function calls: `grep("process_data\\(", "/repo")`

### 2.5 Tool Strategy for Structure Discovery

| Task | Tool | Why |
|------|------|-----|
| Understand file purpose | `read_text_file` with `head=50` | Imports, docstrings, class names in first 50 lines |
| Find file dependencies | `grep` for import patterns | Discovers which files import this one |
| Find related files | `search_files` with name patterns | Finds similarly named files (e.g., `*_controller.py`) |
| Deep dive (when needed) | `read_text_file` with `head=100-200` | Only when first 50 lines aren't enough |
| Verify understanding | `grep` for specific patterns | Confirm architectural assumptions |

### 2.6 Updated System Prompt for Deep Agent

The existing `get_structure_prompt()` function in `deep_structure_agent.py` already provides the core structure. The context-efficient exploration strategy enhances the **exploration_instructions** section.

**Current Exploration Instructions (MCP mode):**
```python
exploration_instructions = f"""## Exploration Strategy
The repository is located at: {clone_path}

The file tree above already shows the complete directory structure - use it to identify files to read.

You have access to MCP filesystem tools for reading file contents:
1. Use `read_text_file` to read individual files with absolute paths
2. Use `read_multiple_files` to read several files at once (more efficient)
3. Use `search_files` to find files matching a pattern if needed

Focus on reading key files to understand the codebase:
- Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
- Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
- Core modules and their purposes

IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files."""
```

**Enhanced Exploration Instructions (Context-Efficient):**
```python
exploration_instructions = f"""## Exploration Strategy
The repository is located at: {clone_path}

The file tree above already shows the complete directory structure - use it to identify files to read.

### Context-Efficient Reading
You have access to MCP filesystem tools. Use them efficiently to minimize context usage:

1. **Initial Exploration - Read Headers First**
   Use `read_text_file` with `head=50` to read only the first 50 lines of files.
   This captures imports, docstrings, and class/function signatures.
   ```
   read_text_file(path="{clone_path}/src/api.py", head=50)
   ```

2. **Targeted Deep Reads**
   Only read full files when necessary. If you need more context:
   - Use `head=100` or `head=150` for larger files
   - Use `tail=50` to see the end of a file (exports, main block)

3. **Dependency Discovery with grep**
   Use `grep` to find relationships between files:
   - Find imports: `grep(pattern="from src.api import", path="{clone_path}", include="*.py")`
   - Find classes: `grep(pattern="class.*Controller", path="{clone_path}")`
   - Find routes: `grep(pattern="@app.route|@router", path="{clone_path}")`

4. **File Discovery with search_files**
   Use `search_files` to find files by pattern:
   - Find controllers: `search_files(path="{clone_path}", pattern="**/*controller*.py")`
   - Find tests: `search_files(path="{clone_path}", pattern="**/test_*.py")`

### What to Read
Focus on understanding the codebase architecture:
- Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
- Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
- Core modules and their purposes

### Exploration Workflow
1. Start with config files (usually small, read in full)
2. Read entry points with `head=50` to understand structure
3. Use `grep` to discover dependencies and relationships
4. Read additional files only as needed for page decisions

IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files."""
```

**Key Changes:**
1. Added `head` parameter guidance for initial reads
2. Added `grep` patterns for dependency discovery
3. Added exploration workflow (config → headers → grep → targeted reads)
4. Preserved all existing context (clone_path, absolute paths, file types)

### 2.7 Expected Benefits

| Metric | Current | With Context-Efficient Exploration |
|--------|---------|-----------------------------------|
| Context tokens (exploration) | ~18K | ~3-5K |
| Tool calls | 6-10 | 15-25 |
| Understanding quality | Full content | Structural + targeted |
| Dependency awareness | Limited | High (via grep) |
| Time (exploration) | ~60s | ~45-60s |

### 2.8 Integration with Page Generation Approaches

This context-efficient exploration strategy is **Phase 1** of the recommended approach:

```
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Context-Efficient Structure Discovery                     │
│                                                                     │
│ Tools: get_file_structure, read_file_head, find_imports,          │
│        find_definitions, search_files                               │
│                                                                     │
│ Output: wiki_structure with file_paths per page                    │
│ Context: ~3-5K tokens (reduced from ~18K)                          │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Content Capture for Page Generation                       │
│                                                                     │
│ After structure is known:                                           │
│ - Read full content of files needed for each page                  │
│ - Cache in finalize_wiki_structure call (Approach 3)               │
│ - OR pass to parallel page workers (Approach 2)                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Parallel Page Generation                                  │
│                                                                     │
│ LangGraph Send API for parallel execution                          │
│ Each page worker has pre-loaded file contents                      │
│ No additional file I/O required                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. Approach 1: Sub-agents During Structure Generation

### 3.1 Concept

Leverage the Deep Agent's built-in sub-agent spawning capability to generate page content as each page is identified during repository exploration. The main Deep Agent acts as an orchestrator, delegating page generation to specialized sub-agents while maintaining its exploration context.

### 3.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Deep Agent (Orchestrator)                    │
│                                                                      │
│  System Prompt: "Explore repo, identify pages, delegate generation" │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ Exploration Loop:                                            │    │
│  │                                                              │    │
│  │  1. Read file_tree, README                                   │    │
│  │  2. Read key files (config, entry points, etc.)              │    │
│  │  3. Identify page: "API Reference"                           │    │
│  │       │                                                      │    │
│  │       └──→ task("page-writer", {                             │    │
│  │                title: "API Reference",                       │    │
│  │                files: ["src/api.py", "src/routes.py"],       │    │
│  │                context: <file contents already in memory>    │    │
│  │            })                                                │    │
│  │              │                                               │    │
│  │              ▼                                               │    │
│  │       ┌──────────────────────────────────────┐               │    │
│  │       │ Sub-agent: page-writer               │               │    │
│  │       │ - Receives file contents             │               │    │
│  │       │ - Generates markdown page            │               │    │
│  │       │ - Returns content to orchestrator    │               │    │
│  │       └──────────────────────────────────────┘               │    │
│  │              │                                               │    │
│  │              ▼                                               │    │
│  │  4. Continue exploration, identify next page...              │    │
│  │  5. Repeat until all pages identified and generated          │    │
│  │  6. Call finalize_wiki_structure with complete pages         │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.3 Implementation

#### 3.3.1 Define Page Writer Sub-agent

```python
# In deep_structure_agent.py

page_writer_subagent = {
    "name": "page-writer",
    "description": (
        "Use this agent to generate wiki page content. Provide the page title, "
        "description, and the relevant source file contents. The agent will "
        "produce comprehensive markdown documentation."
    ),
    "system_prompt": """You are an expert technical writer. Generate comprehensive
wiki documentation for the given page topic.

Your output should be well-structured markdown including:
- Clear introduction explaining the topic
- Code examples with explanations
- API/function documentation where relevant
- Usage examples
- Related concepts and cross-references

Base your documentation ONLY on the provided source files. Do not invent information.""",
    "tools": [],  # No tools needed - content provided in prompt
    "model": "openai:gpt-4o-mini",
}
```

#### 3.3.2 Modify Deep Agent Creation

```python
def create_structure_agent(
    clone_path: str,
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    model: Optional[str] = None,
    mcp_tools: Optional[List[Any]] = None,
) -> Any:
    # ... existing code ...

    # Add page-writer subagent
    subagents = [page_writer_subagent]

    agent = create_deep_agent(
        system_prompt=system_prompt,
        tools=tools,
        subagents=subagents,  # NEW
        backend=backend,
        model=model_instance,
    )

    return agent
```

#### 3.3.3 Modify System Prompt

```python
def get_structure_prompt(...) -> str:
    return f"""You are an expert technical writer analyzing a repository to design
and generate wiki documentation.

## Repository
- Owner: {owner}
- Name: {repo}

## File Tree
{file_tree}

## README
{readme_content}

## Your Task
1. Explore this repository to understand its architecture
2. Identify 8-12 documentation pages needed
3. For EACH page you identify:
   a. Read the relevant source files
   b. Use the `task` tool to delegate to "page-writer" sub-agent:
      - Provide page title and description
      - Include the full content of relevant files
      - Wait for generated content
   c. Store the returned content
4. After all pages are generated, call `finalize_wiki_structure`

## Sub-agent Usage
When delegating to page-writer, format your task like:
```
Generate documentation for: [Page Title]

Description: [What this page should cover]

Source Files:
--- file: src/api.py ---
[full file content]
--- end file ---

--- file: src/routes.py ---
[full file content]
--- end file ---
```

The sub-agent will return markdown content for the page.

## Important
- Generate pages AS you explore, don't wait until the end
- You can run up to 3 sub-agents in parallel for efficiency
- Include the generated content in your finalize_wiki_structure call
"""
```

#### 3.3.4 Modify Finalize Tool Schema

```python
class WikiPageInput(BaseModel):
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    description: str = Field(description="Brief description of page content")
    content: str = Field(description="Generated markdown content for this page")  # NEW
```

### 3.4 Reasoning

#### Why This Approach Works

1. **Context Efficiency**: The Deep Agent has already read the files into its context. Passing this content to sub-agents means no redundant file reads.

2. **Natural Workflow**: The exploration and generation happen in a single cognitive flow. As the agent understands a component, it immediately documents it while the understanding is fresh.

3. **Built-in Capability**: deepagents v0.3.1 has native sub-agent support via the `task()` tool. No additional infrastructure needed.

4. **Parallel Potential**: The Deep Agent can spawn up to 3 sub-agents in parallel, allowing some concurrency.

#### Why This Approach Has Limitations

1. **Sequential Bottleneck**: Even with parallel sub-agents, the main Deep Agent must wait for each batch to complete before continuing exploration. The workflow becomes: explore → spawn 3 sub-agents → wait → explore more → spawn 3 more → wait...

2. **Prompt Complexity**: The system prompt becomes complex, requiring the agent to juggle exploration, delegation, and result collection. This increases the chance of errors or incomplete execution.

3. **Token Accumulation**: All sub-agent results flow back to the main agent, potentially causing context overflow for large repositories with many pages.

4. **Debugging Difficulty**: When something fails, it's harder to isolate whether the issue is in exploration, delegation, or content generation.

5. **No Retry Granularity**: If one page fails to generate properly, you can't easily retry just that page—you'd need to re-run the entire Deep Agent.

### 3.5 Estimated Performance

| Metric | Estimate | Notes |
|--------|----------|-------|
| Total Time | 90-120s | Sequential with some parallelism |
| Token Usage | 40-60K | Main agent + all sub-agents |
| File Reads | 1x | Files read once during exploration |
| Parallelism | Limited (3 concurrent) | deepagents recommendation |

---

## 4. Approach 2: Parallel Agents After Structure Completion

### 4.1 Concept

Maintain the current two-phase architecture but implement the second phase (page generation) using LangGraph's `Send` API for true parallel execution. After the Deep Agent produces the wiki structure, spawn independent page-generation workers for all pages simultaneously.

### 4.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: Structure Generation                     │
│                         (Current Implementation)                     │
│                                                                      │
│  Deep Agent                                                          │
│      │                                                               │
│      ├── Read files via MCP tools                                    │
│      ├── Analyze repository structure                                │
│      └── finalize_wiki_structure() ──→ wiki_structure = {            │
│                                          title: "...",               │
│                                          pages: [                    │
│                                            {title, slug, section,    │
│                                             file_paths, description},│
│                                            ...                       │
│                                          ]                           │
│                                        }                             │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: Parallel Page Generation                 │
│                            (New Implementation)                      │
│                                                                      │
│  generate_structure node                                             │
│          │                                                           │
│          ▼                                                           │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │ fan_out_to_page_workers(state) -> list[Send]                  │  │
│  │                                                                │  │
│  │   pages = state["wiki_structure"]["pages"]                     │  │
│  │   return [                                                     │  │
│  │       Send("page_worker", {                                    │  │
│  │           "page_info": page,                                   │  │
│  │           "clone_path": state["clone_path"],                   │  │
│  │           "repository_id": state["repository_id"]              │  │
│  │       })                                                       │  │
│  │       for page in pages                                        │  │
│  │   ]                                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
│          │                                                           │
│          ▼                                                           │
│  ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐  │
│  │ page  │ page  │ page  │ page  │ page  │ page  │ page  │ page  │  │
│  │ worker│ worker│ worker│ worker│ worker│ worker│ worker│ worker│  │
│  │  #1   │  #2   │  #3   │  #4   │  #5   │  #6   │  #7   │  #8   │  │
│  └───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┘  │
│      │       │       │       │       │       │       │       │       │
│      └───────┴───────┴───────┼───────┴───────┴───────┴───────┘       │
│                              ▼                                       │
│                    aggregate_pages_node                              │
│                              │                                       │
│                              ▼                                       │
│                       store_wiki_node                                │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.3 Implementation

#### 4.3.1 Define Page Worker State

```python
# In wiki_agent.py

from typing import TypedDict, Annotated
import operator

class PageWorkerState(TypedDict):
    """State for individual page generation worker."""
    page_info: Dict[str, Any]  # {title, slug, section, file_paths, description}
    clone_path: str
    repository_id: str
    generated_content: str  # Output

class WikiGenerationState(TypedDict):
    """Main workflow state - updated."""
    repository_id: str
    clone_path: str
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]
    # Use operator.add to aggregate results from parallel workers
    generated_pages: Annotated[List[Dict[str, Any]], operator.add]
    messages: Annotated[List[BaseMessage], add_messages]
    current_step: str
    progress: float
    error_message: Optional[str]
```

#### 4.3.2 Implement Fan-out Function

```python
from langgraph.types import Send

def fan_out_to_page_workers(state: WikiGenerationState) -> list[Send]:
    """Create parallel page generation tasks."""

    if not state.get("wiki_structure") or not state["wiki_structure"].get("pages"):
        return []  # No pages to generate

    pages = state["wiki_structure"]["pages"]
    clone_path = state.get("clone_path", "")
    repository_id = state["repository_id"]

    sends = []
    for page in pages:
        sends.append(
            Send("page_worker", {
                "page_info": page,
                "clone_path": clone_path,
                "repository_id": repository_id,
                "generated_content": ""
            })
        )

    logger.info(f"Fanning out to {len(sends)} page workers")
    return sends
```

#### 4.3.3 Implement Page Worker Node

```python
async def page_worker_node(state: PageWorkerState) -> Dict[str, Any]:
    """Generate content for a single wiki page."""

    page_info = state["page_info"]
    clone_path = state["clone_path"]
    file_paths = page_info.get("file_paths", [])

    # Read files from disk using MCP tools or direct file access
    file_contents = {}
    for file_path in file_paths[:5]:  # Limit to 5 files
        full_path = os.path.join(clone_path, file_path)
        try:
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()[:4000]  # Truncate large files
                    file_contents[file_path] = content
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")

    if not file_contents:
        logger.warning(f"No files found for page: {page_info['title']}")
        return {"generated_pages": []}

    # Build prompt for page generation
    files_markdown = "\n\n".join([
        f"### File: {path}\n```\n{content}\n```"
        for path, content in file_contents.items()
    ])

    prompt = f"""Generate comprehensive wiki documentation for the following page.

## Page Information
- **Title:** {page_info['title']}
- **Section:** {page_info['section']}
- **Description:** {page_info['description']}

## Source Files
{files_markdown}

## Requirements
1. Write clear, professional technical documentation
2. Include code examples from the source files
3. Explain the purpose and usage of each component
4. Use proper markdown formatting
5. Be comprehensive but concise

Generate the page content now:
"""

    # Call LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    # Return result for aggregation
    page_result = {
        **page_info,
        "content": response.content
    }

    return {"generated_pages": [page_result]}
```

#### 4.3.4 Implement Aggregation Node

```python
async def aggregate_pages_node(state: WikiGenerationState) -> WikiGenerationState:
    """Collect all generated pages."""

    generated_pages = state.get("generated_pages", [])

    logger.info(f"Aggregated {len(generated_pages)} pages")

    # Update wiki structure with generated content
    if state.get("wiki_structure"):
        state["wiki_structure"]["pages"] = generated_pages

    state["current_step"] = "pages_generated"
    state["progress"] = 90.0

    return state
```

#### 4.3.5 Update Workflow Definition

```python
def _create_workflow(self) -> StateGraph:
    """Create the wiki generation workflow graph."""

    workflow = StateGraph(WikiGenerationState)

    # Add nodes
    workflow.add_node("analyze_repository", self._analyze_repository_node)
    workflow.add_node("generate_structure", self._generate_structure_node)
    workflow.add_node("page_worker", page_worker_node)  # NEW
    workflow.add_node("aggregate_pages", aggregate_pages_node)  # NEW
    workflow.add_node("store_wiki", self._store_wiki_node)
    workflow.add_node("handle_error", self._handle_error_node)

    # Define workflow edges
    workflow.add_edge(START, "analyze_repository")
    workflow.add_edge("analyze_repository", "generate_structure")

    # Fan-out to parallel page workers
    workflow.add_conditional_edges(
        "generate_structure",
        fan_out_to_page_workers,
        ["page_worker"]
    )

    # Fan-in: all workers converge at aggregation
    workflow.add_edge("page_worker", "aggregate_pages")
    workflow.add_edge("aggregate_pages", "store_wiki")
    workflow.add_edge("store_wiki", END)

    # Error handling
    workflow.add_edge("handle_error", END)

    return workflow.compile()
```

### 4.4 Reasoning

#### Why This Approach Works

1. **True Parallelism**: LangGraph's `Send` API creates genuinely concurrent executions. All 10 pages generate simultaneously, bounded only by API rate limits and system resources.

2. **Clean Separation**: Structure generation and page generation are completely decoupled. Each can be developed, tested, and debugged independently.

3. **Individual Retry**: If page 7 fails, you can potentially retry just that page without re-running the entire workflow.

4. **Scalability**: Adding more pages doesn't increase sequential execution time—just parallel load.

5. **Observability**: Each page worker appears as a separate trace in LangSmith, making debugging straightforward.

6. **State Aggregation**: LangGraph's `Annotated[List[...], operator.add]` pattern cleanly collects results from all parallel workers.

#### Why This Approach Has Limitations

1. **Redundant File Reads**: The Deep Agent read files during exploration, but those contents are not preserved. Each page worker must re-read its relevant files from disk.

2. **File Path Accuracy**: Page workers rely on `file_paths` from the wiki structure. If the Deep Agent recorded paths differently than they exist on disk (e.g., forward vs. backslashes, relative vs. absolute), reads will fail.

3. **MCP Tool Access**: If using MCP filesystem tools, each page worker needs access to the MCP client. This may require passing the client or reinitializing it per worker.

4. **No Context Sharing**: Each page worker starts with a blank slate. It doesn't know what other pages are being generated, which could lead to redundant explanations or missing cross-references.

5. **Rate Limiting**: With 10+ parallel LLM calls, you may hit API rate limits, requiring retry logic or request throttling.

### 4.5 Estimated Performance

| Metric | Estimate | Notes |
|--------|----------|-------|
| Total Time | 70-90s | 60s structure + 15-20s parallel pages |
| Token Usage | 50-80K | Structure agent + all page workers |
| File Reads | 2x | Once in Deep Agent, once per page worker |
| Parallelism | Full (all pages concurrent) | Limited by API rate limits |

---

## 5. Approach 3: Hybrid with Content Caching

### 5.1 Concept

Combine the strengths of both approaches: capture file contents during the Deep Agent's exploration phase (like Approach 1), then pass pre-loaded content to parallel page workers (like Approach 2). This eliminates redundant file reads while maintaining full parallelism.

### 5.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│           PHASE 1: Structure Generation with Content Capture         │
│                                                                      │
│  Deep Agent                                                          │
│      │                                                               │
│      ├── read_text_file("src/api.py")                               │
│      │       └── content captured in file_contents_cache            │
│      │                                                               │
│      ├── read_multiple_files(["src/routes.py", "src/models.py"])    │
│      │       └── contents captured in file_contents_cache           │
│      │                                                               │
│      └── finalize_wiki_structure_with_content(                       │
│              title: "Project Wiki",                                  │
│              pages: [                                                │
│                {                                                     │
│                  title: "API Reference",                             │
│                  slug: "api-reference",                              │
│                  section: "API",                                     │
│                  file_paths: ["src/api.py", "src/routes.py"],        │
│                  file_contents: {  ◀─── NEW: Content included        │
│                    "src/api.py": "from flask import...",            │
│                    "src/routes.py": "def get_users():..."           │
│                  },                                                  │
│                  description: "REST API endpoints..."                │
│                },                                                    │
│                ...                                                   │
│              ]                                                       │
│          )                                                           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│           PHASE 2: Parallel Page Generation (No File I/O)           │
│                                                                      │
│  fan_out_to_page_workers(state) -> list[Send]                       │
│      │                                                               │
│      │   # Each Send includes pre-loaded file_contents              │
│      │   return [                                                    │
│      │       Send("page_worker", {                                   │
│      │           "page_info": page,                                  │
│      │           "file_contents": page["file_contents"]  ◀── Cached │
│      │       })                                                      │
│      │       for page in pages                                       │
│      │   ]                                                           │
│      │                                                               │
│      ▼                                                               │
│  ┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐  │
│  │ page  │ page  │ page  │ page  │ page  │ page  │ page  │ page  │  │
│  │ worker│ worker│ worker│ worker│ worker│ worker│ worker│ worker│  │
│  │       │       │       │       │       │       │       │       │  │
│  │ Has   │ Has   │ Has   │ Has   │ Has   │ Has   │ Has   │ Has   │  │
│  │content│content│content│content│content│content│content│content│  │
│  └───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┴───┬───┘  │
│      │       │       │       │       │       │       │       │       │
│      └───────┴───────┴───────┼───────┴───────┴───────┴───────┘       │
│                              ▼                                       │
│                    aggregate_pages_node                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 Implementation Options

#### Option A: Modify Finalize Tool Schema

The simplest approach—extend the existing tool to capture file contents.

```python
# deep_structure_agent.py

class WikiPageInput(BaseModel):
    """Schema for a wiki page in the structure."""
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    file_contents: Dict[str, str] = Field(
        description="Mapping of file paths to their contents. Include the full "
                    "content of each file you read that's relevant to this page.",
        default_factory=dict
    )
    description: str = Field(description="Brief description of page content")


class FinalizeWikiStructureInput(BaseModel):
    """Schema for the finalize_wiki_structure tool."""
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    pages: List[WikiPageInput] = Field(description="List of wiki pages with content")
```

**Prompt Modification:**

```python
def get_structure_prompt(...) -> str:
    return f"""...existing prompt...

## Output Requirements
When you call `finalize_wiki_structure`, include:
- **title**: Wiki title
- **description**: Wiki description
- **pages**: List of pages, each with:
  - title, slug, section, description
  - file_paths: List of relevant files
  - file_contents: Dictionary mapping file paths to their FULL contents
    - Include the complete content of each file you read
    - This allows page generation without re-reading files

Example file_contents format:
{{
  "src/api.py": "from flask import Flask\\n\\napp = Flask(__name__)\\n...",
  "src/routes.py": "from . import app\\n\\n@app.route('/users')\\n..."
}}

IMPORTANT: Include file_contents for each page. This is critical for efficient
page generation.
"""
```

#### Option B: Use DeepAgents Filesystem State

Leverage the DeepAgents filesystem middleware to persist file contents in agent state.

```python
# deep_structure_agent.py

def get_structure_prompt(...) -> str:
    return f"""...existing prompt...

## File Caching Strategy
As you read files during exploration, ALSO write them to your filesystem for caching:

1. When you read a file (e.g., src/api.py):
   - Read it using read_text_file or read_multiple_files
   - Also write it to cache: write_file("/cache/src_api.py", <content>)

2. In your finalize_wiki_structure call:
   - Reference cached paths: file_cache_paths: ["/cache/src_api.py", ...]

This allows page generators to access content without disk I/O.
"""

# In the workflow, extract cached files from agent state
def extract_cached_files(agent_result) -> Dict[str, str]:
    files_state = agent_result.get("files", {})
    cache_files = {
        path: content
        for path, content in files_state.items()
        if path.startswith("/cache/")
    }
    return cache_files
```

#### Option C: Shared LangGraph Store

Use LangGraph's Store abstraction for cross-node data sharing.

```python
from langgraph.store.memory import InMemoryStore

# Create shared store
file_store = InMemoryStore()

# In Deep Agent's MCP tool wrapper - intercept reads and cache
class CachingMCPWrapper:
    def __init__(self, mcp_client, store, namespace):
        self.mcp_client = mcp_client
        self.store = store
        self.namespace = namespace

    async def read_text_file(self, path: str) -> str:
        content = await self.mcp_client.read_text_file(path)
        # Cache in store
        await self.store.aput(
            (self.namespace,),
            path,
            {"content": content}
        )
        return content

# In page worker - read from store
async def page_worker_node(state, *, store):
    page_info = state["page_info"]
    file_contents = {}

    for file_path in page_info["file_paths"]:
        cached = await store.aget((namespace,), file_path)
        if cached:
            file_contents[file_path] = cached["content"]

    # Generate page with cached contents
    ...
```

#### Option D: Middleware-based Content Capture

Create custom middleware that automatically captures all file reads.

```python
from deepagents.middleware.filesystem import FilesystemMiddleware

class ContentCapturingMiddleware(FilesystemMiddleware):
    """Middleware that captures all file contents read during agent execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_contents: Dict[str, str] = {}

    async def after_tool(self, tool_name: str, tool_input: dict, tool_output: str):
        if tool_name in ("read_text_file", "read_file"):
            path = tool_input.get("path", "")
            self.captured_contents[path] = tool_output
        elif tool_name == "read_multiple_files":
            # Parse multiple file outputs
            paths = tool_input.get("paths", [])
            # ... parse and store each file's content

        return tool_output

    def get_captured_contents(self) -> Dict[str, str]:
        return self.captured_contents.copy()
```

### 5.4 Recommended Implementation (Option A - Modified Schema)

Option A is recommended because:
1. Minimal code changes—just schema and prompt modifications
2. No new infrastructure (stores, middleware, etc.)
3. LLM naturally includes content when instructed
4. Content travels with the page definition through the workflow

**Complete Implementation:**

```python
# ============== deep_structure_agent.py ==============

class WikiPageInput(BaseModel):
    """Schema for a wiki page in the structure."""
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    file_contents: Dict[str, str] = Field(
        default_factory=dict,
        description="Full contents of relevant files. Map file path to content string."
    )
    description: str = Field(description="Brief description of page content")


def get_structure_prompt(owner, repo, file_tree, readme_content, clone_path, use_mcp_tools):
    return f"""You are an expert technical writer analyzing a repository.

## Repository: {owner}/{repo}

## File Tree
{file_tree}

## README
{readme_content}

## Your Task
1. Explore this repository to understand its architecture
2. Read key files to understand the codebase
3. Design a comprehensive wiki structure with 8-12 pages
4. Call finalize_wiki_structure with your findings

## Critical: Include File Contents
When you call finalize_wiki_structure, you MUST include the file_contents
for each page. This is the actual content of files you read.

Format:
{{
  "pages": [
    {{
      "title": "API Reference",
      "slug": "api-reference",
      "section": "API",
      "file_paths": ["src/api.py", "src/routes.py"],
      "file_contents": {{
        "src/api.py": "<paste the full content you read>",
        "src/routes.py": "<paste the full content you read>"
      }},
      "description": "REST API endpoints and routing"
    }}
  ]
}}

This eliminates redundant file reads during page generation.
"""


# ============== wiki_agent.py ==============

from langgraph.types import Send

class PageWorkerState(TypedDict):
    page_info: Dict[str, Any]
    generated_content: str

def fan_out_to_page_workers(state: WikiGenerationState) -> list[Send]:
    """Fan out to parallel page workers with pre-loaded content."""
    pages = state.get("wiki_structure", {}).get("pages", [])

    return [
        Send("page_worker", {
            "page_info": page,
            "generated_content": ""
        })
        for page in pages
    ]

async def page_worker_node(state: PageWorkerState) -> Dict[str, Any]:
    """Generate page content using pre-loaded file contents."""
    page_info = state["page_info"]

    # File contents already available - no I/O needed!
    file_contents = page_info.get("file_contents", {})

    if not file_contents:
        logger.warning(f"No file contents for page: {page_info['title']}")
        return {"generated_pages": []}

    # Build prompt with cached contents
    files_markdown = "\n\n".join([
        f"### File: {path}\n```\n{content[:4000]}\n```"
        for path, content in file_contents.items()
    ])

    prompt = f"""Generate wiki documentation for this page.

## Page: {page_info['title']}
**Section:** {page_info['section']}
**Description:** {page_info['description']}

## Source Files
{files_markdown}

Write comprehensive markdown documentation based on these files.
"""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = await llm.ainvoke([HumanMessage(content=prompt)])

    return {
        "generated_pages": [{
            **page_info,
            "content": response.content,
            "file_contents": None  # Clear to save memory in final state
        }]
    }
```

### 5.5 Reasoning

#### Why This Approach Is Optimal

1. **Single File Read**: Files are read exactly once during exploration. The Deep Agent already reads them to understand the codebase—we just capture that work.

2. **Full Parallelism**: Page workers receive pre-loaded content and can execute fully in parallel, just like Approach 2.

3. **No External Dependencies**: Unlike Option C (Store) or Option D (Middleware), the recommended Option A requires no new infrastructure—just schema changes.

4. **Natural LLM Behavior**: LLMs are good at following instructions like "include the file contents you read." The content is already in context; we're just asking it to repeat it in a structured format.

5. **Graceful Degradation**: If the LLM forgets to include some file contents, page workers can fall back to reading from disk.

6. **Reduced Latency**: Parallel page generation has zero I/O wait time—all data is immediately available.

#### Trade-offs to Consider

1. **Larger State Object**: The wiki_structure now contains file contents, which could be several MB for large repositories. This increases:
   - Memory usage during workflow execution
   - State serialization time if using checkpointing
   - Network transfer if using distributed execution

2. **Token Usage in Deep Agent**: The LLM must "output" the file contents in its finalize call, consuming completion tokens. For 10 pages with 5 files each at 2KB average:
   - ~100KB of file content
   - ~25,000 tokens additional output
   - Cost: ~$0.03-0.05 extra with GPT-4o-mini

3. **Prompt Engineering Sensitivity**: The Deep Agent must be carefully prompted to include file_contents. If it doesn't, pages won't generate properly. Testing and iteration required.

4. **Content Truncation**: Very large files must be truncated either during capture or during page generation. Need consistent truncation strategy.

### 5.6 Estimated Performance

| Metric | Estimate | Notes |
|--------|----------|-------|
| Total Time | 65-80s | 60s structure + 10-15s parallel pages |
| Token Usage | 45-60K | Slightly higher due to content in output |
| File Reads | 1x | Files read only during exploration |
| Parallelism | Full | All pages concurrent, no I/O blocking |

---

## 6. Comparative Analysis

### 6.1 Feature Comparison

| Feature | Approach 1 | Approach 2 | Approach 3 |
|---------|------------|------------|------------|
| File reads | 1x | 2x | 1x |
| Parallelism | Limited (3) | Full | Full |
| Complexity | High | Medium | Low-Medium |
| State size | Normal | Normal | Larger |
| Debugging | Hard | Easy | Easy |
| Retry granularity | None | Per-page | Per-page |
| Infrastructure | None | None | None |

### 6.2 Performance Comparison

| Metric | Approach 1 | Approach 2 | Approach 3 |
|--------|------------|------------|------------|
| Estimated time | 90-120s | 70-90s | 65-80s |
| Token usage | 40-60K | 50-80K | 45-60K |
| I/O operations | Low | High | Low |
| Memory usage | Medium | Low | Higher |
| API calls | Many sequential | Many parallel | Many parallel |

### 6.3 Risk Assessment

| Risk | Approach 1 | Approach 2 | Approach 3 |
|------|------------|------------|------------|
| Prompt complexity | High | Low | Medium |
| Single point of failure | Yes | No | No |
| Token overflow | Medium | Low | Medium |
| File path mismatches | Low | High | Low |
| Rate limiting | Low | High | High |

### 6.4 Development Effort

| Aspect | Approach 1 | Approach 2 | Approach 3 |
|--------|------------|------------|------------|
| Schema changes | Minor | Minor | Moderate |
| Prompt engineering | Major | Minor | Moderate |
| Workflow changes | Minor | Major | Major |
| Testing complexity | High | Medium | Medium |
| Debugging tools | Limited | Good | Good |

---

## 7. Final Recommendation

### 7.1 Primary Recommendation: Approach 3 (Hybrid with Content Caching)

**Rationale:**

1. **Best Performance**: Achieves near-optimal execution time by combining single file reads with full parallelism.

2. **Lowest Risk**: Avoids the complexity of Approach 1's prompt engineering and the I/O issues of Approach 2's file re-reading.

3. **Simplest Implementation**: Option A (schema modification) requires minimal code changes—primarily prompt updates and schema extensions.

4. **Future-Proof**: The cached content pattern can be extended for other use cases (search indexing, cross-referencing, etc.).

5. **Observable**: Each page worker is independently traceable, making debugging straightforward.

### 7.2 Implementation Priority

1. **Phase 1**: Implement Option A (schema modification)
   - Update `WikiPageInput` schema
   - Modify system prompt
   - Test with sample repository

2. **Phase 2**: Implement parallel page workers
   - Add `Send`-based fan-out
   - Implement page worker node
   - Add aggregation node

3. **Phase 3**: Optimize and harden
   - Add fallback file reading for missing contents
   - Implement retry logic for failed pages
   - Add progress tracking and logging

### 7.3 Alternative: Approach 2 as Fallback

If Approach 3 proves difficult (e.g., LLM doesn't reliably include file_contents), Approach 2 is a solid fallback:
- More straightforward implementation
- Slightly worse performance (2x file reads)
- Well-documented LangGraph patterns to follow

---

## 8. Implementation Plan

### 8.1 Phase 1: Context-Efficient Exploration (Week 1)

1. **Update Deep Agent Prompt**
   - Modify `get_structure_prompt()` in `deep_structure_agent.py`
   - Add exploration strategy guidelines (use `head=50` for initial reads)
   - Include examples showing how to use `grep` for dependency discovery
   - Encourage iterative exploration: read headers first, then targeted sections

   **Note:** No wrapper changes needed. The MCP filesystem server's `read_text_file` tool
   already supports `head`/`tail` parameters natively. The Deep Agent uses MCP tools
   directly via `mcp_client._tools.values()` passed to deepagents.

### 8.2 Phase 2: Content Caching (Week 2)

1. **Schema Updates**
   - Extend `WikiPageInput` with `file_contents: Dict[str, str]`
   - Update `FinalizeWikiStructureInput` schema
   - Add validation for required fields

2. **Prompt Engineering**
   - Update system prompt to instruct LLM to include file contents
   - Add examples of expected output format
   - Test with sample repositories

3. **Fallback Implementation**
   - Add fallback file reading in page workers
   - Handle missing file_contents gracefully

### 8.3 Phase 3: Parallel Page Generation (Week 3)

1. **LangGraph Updates**
   - Implement `fan_out_to_page_workers()` function
   - Add `page_worker_node()` for content generation
   - Implement `aggregate_pages_node()` for result collection
   - Update workflow graph with Send-based fan-out

2. **Page Worker Implementation**
   - Design page generation prompt
   - Implement LLM call with cached content
   - Add error handling and retries

3. **Integration Testing**
   - End-to-end test with sample repository
   - Verify parallel execution in LangSmith traces
   - Benchmark performance improvements

### 8.4 Phase 4: Optimization & Hardening (Week 4)

1. **Performance Optimization**
   - Tune chunk sizes for file reading
   - Optimize prompt lengths
   - Add request throttling for rate limits

2. **Error Handling**
   - Implement per-page retry logic
   - Add timeout handling
   - Create fallback strategies

3. **Monitoring & Observability**
   - Add structured logging for each phase
   - Create LangSmith dashboards
   - Document debugging procedures

---

## 9. Next Steps

1. **Validate Assumptions**: Run the Deep Agent with modified prompts to verify it can include file_contents reliably

2. **Prototype Page Worker**: Implement a minimal page worker node to test LangGraph's `Send` pattern

3. **Benchmark**: Compare actual performance of Approach 3 vs Approach 2 on a sample repository

4. **Document API Contracts**: Define exact schema for page worker input/output

5. **Consider Edge Cases**:
   - Binary files (should be excluded)
   - Very large files (truncation strategy)
   - Many pages (rate limiting)
   - Failed page generation (retry strategy)

---

## Appendix A: LangGraph Send Pattern Reference

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import TypedDict, Annotated
import operator

class State(TypedDict):
    items: list[str]
    results: Annotated[list[str], operator.add]

def splitter(state: State) -> list[Send]:
    return [Send("worker", {"item": item}) for item in state["items"]]

def worker(state: dict) -> dict:
    result = f"processed: {state['item']}"
    return {"results": [result]}

def aggregator(state: State) -> State:
    return state

graph = StateGraph(State)
graph.add_node("splitter", lambda s: s)  # Pass-through
graph.add_node("worker", worker)
graph.add_node("aggregator", aggregator)

graph.add_edge(START, "splitter")
graph.add_conditional_edges("splitter", splitter, ["worker"])
graph.add_edge("worker", "aggregator")
graph.add_edge("aggregator", END)

app = graph.compile()
```

## Appendix B: DeepAgents Subagent Pattern Reference

```python
from deepagents import create_deep_agent

subagents = [
    {
        "name": "specialist",
        "description": "Handles specialized tasks",
        "system_prompt": "You are a specialist...",
        "tools": [...],
        "model": "openai:gpt-4o-mini",
    }
]

agent = create_deep_agent(
    system_prompt="Main agent instructions...",
    subagents=subagents,
    tools=[...],
)

# Agent can call subagents via task() tool
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

---

*Document generated: 2025-12-31*
*For questions or updates, refer to the project repository.*
