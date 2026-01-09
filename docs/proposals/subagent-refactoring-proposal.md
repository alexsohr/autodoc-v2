# Proposal: Refactoring Wiki Generation to Use Subagents

## Executive Summary

This proposal outlines a major simplification of the wiki generation architecture by eliminating the complex page worker system (batch processing, queue management, fan-out/fan-in patterns) and replacing it with Deep Agents' native subagent delegation mechanism.

**Current state**: Structure agent generates wiki structure, then a complex LangGraph workflow with batch processing, parallel workers, and MongoDB queue manages page generation.

**Proposed state**: A single unified deep agent generates wiki structure AND page content, spawning specialized page-generation subagents for each page using the `task()` tool built into Deep Agents.

---

## Current Architecture (Complex)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        WikiGenerationAgent                               │
│  (LangGraph StateGraph with complex batch/parallel orchestration)        │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│ _analyze_repository  │ ──► │ _generate_structure  │
│      (node)          │     │   (calls deep agent) │
└──────────────────────┘     └──────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       BATCH PROCESSING LOOP                               │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  prepare_batch_node → fan_out_to_page_workers → page_worker_node(s) │ │
│  │         ▲                                              │            │ │
│  │         │                                              ▼            │ │
│  │  check_more_pages ◄─────────────────── batch_complete_node          │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐     ┌──────────────────────┐
│ aggregate_pages_node │ ──► │  _store_wiki_node    │
└──────────────────────┘     └──────────────────────┘

External Dependencies:
- WikiPageQueueRepository (MongoDB queue for resilience)
- WikiPageQueueItem (Beanie document)
- PageWorkerState (LangGraph TypedDict)
- deep_page_agent.py (standalone agent for page content)
- MCP client per-worker connections
```

### Problems with Current Architecture

1. **Excessive Complexity**: 8+ LangGraph nodes, 2 state classes, batch processing logic, conditional routing
2. **Context Fragmentation**: Page agents have no visibility into structure agent's exploration
3. **Queue Overhead**: MongoDB queue adds latency, failure modes, stale claim handling
4. **Duplicate MCP Connections**: Each page worker creates its own MCP client connection
5. **State Management Issues**: Aggregation with `operator.add`, edge cases with empty queues
6. **Retry Logic Scattered**: Retry handling in multiple places (wiki_agent, queue repository)
7. **Two Separate Deep Agents**: Structure agent and page agent have duplicate setup code

---

## Proposed Architecture (Simple)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      WikiDeepAgent (unified)                             │
│              (single create_deep_agent with subagents)                   │
└─────────────────────────────────────────────────────────────────────────┘
           │
           │  Configured with:
           │  - MCP filesystem tools
           │  - finalize_wiki_structure tool
           │  - page-generator subagent definition
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PARENT AGENT WORKFLOW                             │
│  1. Explore repository (read files, understand structure)               │
│  2. Design wiki structure (sections, pages)                             │
│  3. For EACH page, call: task(name="page-generator", task="...")        │
│     └──► Subagent explores & returns page content                       │
│  4. Call finalize_wiki with complete structure + all page contents      │
└─────────────────────────────────────────────────────────────────────────┘

Subagent Definition (dictionary):
{
    "name": "page-generator",
    "description": "Generate detailed wiki page content by exploring source files",
    "system_prompt": <page_prompt_template>,
    "tools": [mcp_filesystem_tools, finalize_page],
    "model": "google_genai:gemini-2.5-flash-lite"
}
```

### Benefits

1. **Dramatic Simplification**: Single agent, no LangGraph workflow nodes
2. **Context Isolation**: Subagents get clean context, parent sees only results
3. **No Queue Needed**: Deep Agents handles task delegation internally
4. **Single MCP Connection**: Parent agent shares tools with subagents
5. **Built-in Retries**: Deep Agents middleware can handle retries
6. **Unified Codebase**: One agent file instead of three

---

## Files to REMOVE (Decommission)

### 1. `src/agents/deep_page_agent.py` (ENTIRE FILE - 750+ lines)

**Reason**: Page generation becomes a subagent definition, not a standalone agent.

Contents being removed:
- `ToolCallIdPatchMiddleware` class
- `PageSection` model
- `PageContent` model
- `FinalizePageInput` model
- `create_page_finalize_tool()` function
- `get_page_prompt()` function
- `create_page_agent()` function
- `_extract_content_from_messages()` function
- `run_page_agent()` function

### 2. `src/repository/wiki_page_queue_repository.py` (ENTIRE FILE)

**Reason**: No queue needed - Deep Agents handles task delegation.

Contents being removed:
- `WikiPageQueueRepository` class with 14 methods:
  - `enqueue_pages()`
  - `claim_batch()`
  - `mark_completed()`
  - `mark_failed()`
  - `get_queue_status()`
  - `get_completed_pages()`
  - `get_failed_pages()`
  - `reset_stale_claims()`
  - `clear_queue()`
  - `get_pending_count()`
  - `is_queue_complete()`
  - `retry_failed_pages()`
  - `get_item_by_page_slug()`

### 3. `tests/unit/test_deep_page_agent.py` (ENTIRE FILE)

**Reason**: Tests for removed code.

Test classes being removed:
- `TestPageModels`
- `TestFinalizePageTool`
- `TestGetPagePrompt`
- `TestFallbackExtraction`

---

## Files to MODIFY

### 1. `src/agents/wiki_agent.py`

**Remove** (approximately 400 lines):
- `PageWorkerState` class (lines 56-74)
- `prepare_batch_node()` function (lines 77-164)
- `fan_out_to_page_workers()` function (lines 167-214)
- `batch_complete_node()` function (lines 217-269)
- `check_more_pages()` function (lines 272-305)
- `page_worker_node()` function (lines 308-414)
- `aggregate_pages_node()` function (lines 417-472)
- Complex workflow in `_create_workflow()` (simplify to direct call)

**Simplify** `WikiGenerationState`:
```python
class WikiGenerationState(TypedDict):
    """Simplified state for wiki generation"""
    repository_id: str
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]  # Now includes page contents
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    clone_path: Optional[str]
```

**Remove fields from state**:
- `generated_pages: Annotated[List[Dict[str, Any]], operator.add]`
- `failed_pages: Annotated[List[Dict[str, Any]], operator.add]`
- `current_page: Optional[str]`
- `messages: List[BaseMessage]`
- `pages_queued: Optional[List[Dict[str, Any]]]`
- `current_batch: Optional[List[Dict[str, Any]]]`

### 2. `src/agents/deep_structure_agent.py`

**Transform into**: `src/agents/wiki_deep_agent.py` (rename)

**Changes**:
- Rename `create_structure_agent()` → `create_wiki_agent()`
- Rename `run_structure_agent()` → `run_wiki_agent()`
- Add subagent configuration for page generation
- Expand system prompt to include page generation workflow
- Modify `finalize_wiki_structure` tool to accept pages with content

### 3. `src/models/wiki.py`

**Remove**:
- `WikiPageQueueStatus` enum (lines ~420-430)
- `WikiPageQueueItem` class (lines 435-515)

**Keep unchanged**:
- `PageImportance`
- `WikiPageDetail`
- `WikiSection`
- `WikiStructure`
- All Create/Request/Response models

### 4. `src/utils/config_loader.py`

**Remove settings** (no longer needed):
- `max_concurrent_page_workers: int` (line 143-145)
- `max_page_retries: int` (line 146-148)

### 5. `src/repository/database.py`

**Remove from document_models list**:
```python
# Remove this line:
WikiPageQueueItem,
```

---

## New Code to ADD

### 1. Page Generator Subagent Definition (in `wiki_deep_agent.py`)

```python
def get_page_subagent_config(
    clone_path: str,
    mcp_tools: List[Any],
) -> Dict[str, Any]:
    """Create page generator subagent configuration.

    This subagent is spawned by the parent wiki agent for each page.
    It receives the page title, description, and file hints, then
    explores the codebase and returns structured page content.
    """
    # Import finalize tool creator
    from .page_tools import create_page_finalize_tool

    captured_content: Dict[str, Any] = {}
    finalize_tool = create_page_finalize_tool(captured_content)

    return {
        "name": "page-generator",
        "description": (
            "Generate detailed wiki page content. Use this for each page "
            "in the wiki structure. Provide the page title, description, "
            "and relevant file paths. The subagent will explore the files "
            "and return comprehensive documentation."
        ),
        "system_prompt": get_page_system_prompt(clone_path),
        "tools": list(mcp_tools) + [finalize_tool],
        "model": "google_genai:gemini-2.5-flash-lite",
    }
```

### 2. Unified Wiki Agent Creation

```python
def create_wiki_agent(
    clone_path: str,
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    model: Optional[str] = None,
    mcp_tools: Optional[List[Any]] = None,
) -> Any:
    """Create unified wiki generation agent with page subagent."""
    from deepagents import create_deep_agent

    # Capture for final wiki structure
    captured_structure: Dict[str, Any] = {}
    finalize_tool = create_finalize_tool(captured_structure)

    # System prompt instructs agent to:
    # 1. Explore repository
    # 2. Design wiki structure
    # 3. For each page, call task(name="page-generator", task="...")
    # 4. Finalize with complete structure
    system_prompt = get_wiki_system_prompt(
        owner, repo, file_tree, readme_content, clone_path
    )

    # Configure page generator subagent
    page_subagent = get_page_subagent_config(clone_path, mcp_tools or [])

    agent = create_deep_agent(
        model=model or "google_genai:gemini-2.5-flash-lite",
        system_prompt=system_prompt,
        tools=(mcp_tools or []) + [finalize_tool],
        subagents=[page_subagent],
        middleware=[ToolCallIdPatchMiddleware()],
    )

    agent._structure_capture = captured_structure
    return agent
```

### 3. Updated System Prompt (key additions)

```python
def get_wiki_system_prompt(...) -> str:
    return f"""
You are a documentation expert creating a comprehensive wiki for {owner}/{repo}.

## WORKFLOW

### Phase 1: Repository Exploration
- Read key files to understand the codebase
- Identify main components, patterns, and architecture

### Phase 2: Wiki Structure Design
- Create sections (e.g., Getting Started, Architecture, API Reference)
- Define pages within each section with titles, descriptions, file hints

### Phase 3: Page Content Generation
For EACH page in your structure, delegate to the page-generator subagent:

```
task(
    name="page-generator",
    task="Generate wiki page for: [Page Title]
    Description: [Page Description]
    Relevant files: [file1.py, file2.py, ...]
    Repository: {owner}/{repo}
    Clone path: {clone_path}"
)
```

The subagent will explore the files and return the page content.
Collect all page contents.

### Phase 4: Finalize
Call finalize_wiki_structure with the complete structure including all page contents.

## IMPORTANT
- Generate ALL pages before finalizing
- Each page should have substantial content (1000+ words)
- Include code examples, diagrams, and cross-references
"""
```

---

## Migration Steps

### Step 1: Create New Unified Agent
1. Create `src/agents/wiki_deep_agent.py` based on `deep_structure_agent.py`
2. Add subagent configuration for page generation
3. Update system prompt with Phase 3 workflow
4. Create `src/agents/page_tools.py` with shared page finalization logic

### Step 2: Simplify WikiGenerationAgent
1. Remove all batch/worker nodes from `wiki_agent.py`
2. Simplify `WikiGenerationState`
3. Update `_generate_structure_node` to call new unified agent
4. Remove `_analyze_repository_node` (unified agent handles this)

### Step 3: Remove Queue Infrastructure
1. Delete `wiki_page_queue_repository.py`
2. Remove `WikiPageQueueItem` from `wiki.py`
3. Remove queue settings from `config_loader.py`
4. Remove from `database.py` document registration

### Step 4: Delete Old Code
1. Delete `deep_page_agent.py`
2. Delete `test_deep_page_agent.py`

### Step 5: Update Tests
1. Update `test_wiki_agent_deep_agent.py`
2. Create new tests for unified agent
3. Update integration tests

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Subagent timeout | Configure per-subagent timeouts in Deep Agents |
| Page generation failures | Parent agent can retry task() calls |
| Context window limits | Page subagent has isolated context |
| Model rate limits | Add rate limiting middleware |
| MCP connection issues | Single connection shared by parent/subagents |

---

## Estimated Impact

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Lines of code | ~1500 | ~500 | -67% |
| Files | 5 | 2 | -60% |
| LangGraph nodes | 8 | 0 | -100% |
| State classes | 2 | 1 | -50% |
| MongoDB collections | 1 extra | 0 | -100% |
| MCP connections per run | N (workers) | 1 | -90% |

---

## Appendix: File Inventory

### Files to DELETE
```
src/agents/deep_page_agent.py           # 750+ lines
src/repository/wiki_page_queue_repository.py  # 200+ lines
tests/unit/test_deep_page_agent.py      # 200+ lines
```

### Files to MODIFY
```
src/agents/wiki_agent.py                # Remove ~400 lines
src/agents/deep_structure_agent.py      # Rename + expand
src/models/wiki.py                      # Remove queue models
src/utils/config_loader.py              # Remove queue settings
src/repository/database.py              # Remove queue document
```

### Files to CREATE
```
src/agents/wiki_deep_agent.py           # Unified agent (~300 lines)
src/agents/page_tools.py                # Shared page tools (~100 lines)
```
