# Hybrid Page Generation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Approach 3 (Hybrid with Content Caching) to generate wiki pages using file contents captured during the Deep Agent's structure exploration phase, then generate pages in parallel using LangGraph's Send API.

**Architecture:** The Deep Agent reads files during repository exploration and includes their contents in the `finalize_wiki_structure` call. These cached contents are passed to parallel page workers via LangGraph's `Send` API, eliminating redundant file I/O. Page workers generate content simultaneously, results are aggregated automatically via LangGraph's reducer pattern, and stored to the database.

**Tech Stack:** LangGraph (Send API, StateGraph, Annotated reducers), langchain-openai (ChatOpenAI), Pydantic (schemas), asyncio, deepagents MCP filesystem tools

---

## IMPORTANT: LangGraph Patterns (from official documentation)

### Pattern 1: Worker State Must Share Aggregation Key with Main State

From LangGraph orchestrator-worker documentation:
```python
# Main graph state
class State(TypedDict):
    topic: str
    sections: list[Section]
    completed_sections: Annotated[list, operator.add]  # Shared key with reducer
    final_report: str

# Worker state - MUST include the same shared key with same reducer!
class WorkerState(TypedDict):
    section: Section
    completed_sections: Annotated[list, operator.add]  # SAME key, SAME reducer
```

### Pattern 2: Send Payload Becomes Worker Input

From documentation: "Each worker agent only sees the contents of the Send payload."
```python
def assign_workers(state: State):
    return [Send("worker", {"section": s}) for s in state["sections"]]

def worker(state: WorkerState):  # Receives Send payload as state
    return {"completed_sections": [result]}  # Writes to shared key
```

### Pattern 3: Async Node Functions
```python
async def node(state: State):
    response = await llm.ainvoke([...])
    return {"key": value}
```

---

## Phase 1: Schema and State Updates

### Task 1: Add `file_contents` Field to `WikiPageInput` Schema

**Files:**
- Modify: `src/agents/deep_structure_agent.py:19-25`
- Test: `tests/unit/test_deep_structure_agent.py` (create if not exists)

**Step 1: Write the failing test**

Create test file if it doesn't exist:

```python
# tests/unit/test_deep_structure_agent.py
"""Unit tests for deep_structure_agent module."""

import pytest
from pydantic import ValidationError

from src.agents.deep_structure_agent import WikiPageInput, FinalizeWikiStructureInput


class TestWikiPageInput:
    """Tests for WikiPageInput schema."""

    def test_wiki_page_input_with_file_contents(self):
        """Test that WikiPageInput accepts file_contents field."""
        page = WikiPageInput(
            title="API Reference",
            slug="api-reference",
            section="API",
            file_paths=["src/api.py", "src/routes.py"],
            file_contents={
                "src/api.py": "from flask import Flask\napp = Flask(__name__)",
                "src/routes.py": "@app.route('/users')\ndef get_users(): pass",
            },
            description="REST API endpoints documentation",
        )

        assert page.title == "API Reference"
        assert page.file_contents["src/api.py"].startswith("from flask")
        assert len(page.file_contents) == 2

    def test_wiki_page_input_file_contents_defaults_to_empty(self):
        """Test that file_contents defaults to empty dict when not provided."""
        page = WikiPageInput(
            title="Overview",
            slug="overview",
            section="Overview",
            file_paths=["README.md"],
            description="Project overview",
        )

        assert page.file_contents == {}

    def test_wiki_page_input_requires_mandatory_fields(self):
        """Test that mandatory fields are required."""
        with pytest.raises(ValidationError):
            WikiPageInput(
                title="Test",
                # Missing slug, section, file_paths, description
            )
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_deep_structure_agent.py -v`
Expected: FAIL with `TypeError` or `ValidationError` because `file_contents` field doesn't exist

**Step 3: Write minimal implementation**

Modify `src/agents/deep_structure_agent.py`:

```python
class WikiPageInput(BaseModel):
    """Schema for a wiki page in the structure"""
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    file_contents: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of file paths to their full contents. Include the content of each file you read that's relevant to this page."
    )
    description: str = Field(description="Brief description of page content")
```

Add import at top of file if not present:
```python
from typing import Any, Dict, List, Optional
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_deep_structure_agent.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_deep_structure_agent.py src/agents/deep_structure_agent.py
git commit -m "feat(schema): add file_contents field to WikiPageInput for content caching"
```

---

### Task 2: Update WikiGenerationState with `clone_path` and Annotated `generated_pages`

**Files:**
- Modify: `src/agents/wiki_agent.py:29-42`
- Test: `tests/unit/test_wiki_agent.py` (create if needed)

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_agent.py
"""Unit tests for wiki_agent module."""

import operator
import pytest
from typing import Annotated, get_type_hints, get_origin, get_args

from src.agents.wiki_agent import WikiGenerationState


class TestWikiGenerationState:
    """Tests for WikiGenerationState TypedDict."""

    def test_wiki_generation_state_has_clone_path(self):
        """Test that WikiGenerationState includes clone_path field."""
        hints = get_type_hints(WikiGenerationState)
        assert "clone_path" in hints, "WikiGenerationState must have clone_path field"

    def test_generated_pages_uses_annotated_reducer(self):
        """Test that generated_pages uses Annotated with operator.add for LangGraph aggregation."""
        hints = get_type_hints(WikiGenerationState, include_extras=True)
        generated_pages_type = hints.get("generated_pages")

        # Check it's Annotated
        assert get_origin(generated_pages_type) is Annotated, \
            "generated_pages must use Annotated for LangGraph reducer"

        # Check the metadata includes operator.add
        args = get_args(generated_pages_type)
        assert len(args) >= 2, "Annotated should have base type and metadata"
        assert operator.add in args, "Annotated must include operator.add for aggregation"

    def test_wiki_generation_state_can_be_instantiated(self):
        """Test that WikiGenerationState can be created with all fields."""
        state: WikiGenerationState = {
            "repository_id": "test-repo-id",
            "file_tree": "src/\n  main.py",
            "readme_content": "# Test Project",
            "wiki_structure": None,
            "generated_pages": [],
            "current_page": None,
            "current_step": "init",
            "error_message": None,
            "progress": 0.0,
            "start_time": "2025-01-01T00:00:00Z",
            "messages": [],
            "clone_path": "/path/to/repo",
        }

        assert state["clone_path"] == "/path/to/repo"
        assert state["generated_pages"] == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestWikiGenerationState -v`
Expected: FAIL because `clone_path` not in TypedDict and `generated_pages` not Annotated

**Step 3: Write minimal implementation**

Modify `src/agents/wiki_agent.py`. First update imports:

```python
from typing import Annotated, Any, Dict, List, Optional
import operator
```

Then update the TypedDict:

```python
class WikiGenerationState(TypedDict):
    """State for wiki generation workflow.

    Note: generated_pages uses Annotated with operator.add for LangGraph
    parallel result aggregation from page worker nodes.
    """

    repository_id: str
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]
    # CRITICAL: Use Annotated with operator.add for parallel worker result aggregation
    generated_pages: Annotated[List[Dict[str, Any]], operator.add]
    current_page: Optional[str]
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
    clone_path: Optional[str]  # Path to cloned repository for file access
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestWikiGenerationState -v`
Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/test_wiki_agent.py src/agents/wiki_agent.py
git commit -m "feat(state): add clone_path and Annotated generated_pages for parallel aggregation"
```

---

### Task 3: Create PageWorkerState with Shared Aggregation Key

**Files:**
- Modify: `src/agents/wiki_agent.py` (add after WikiGenerationState)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

from src.agents.wiki_agent import PageWorkerState


class TestPageWorkerState:
    """Tests for PageWorkerState TypedDict.

    CRITICAL: PageWorkerState must include the same `generated_pages` key
    with the same Annotated reducer as WikiGenerationState. This is required
    for LangGraph's parallel worker result aggregation.
    """

    def test_page_worker_state_has_required_fields(self):
        """Test that PageWorkerState has all required fields."""
        hints = get_type_hints(PageWorkerState)

        assert "page_info" in hints
        assert "clone_path" in hints
        assert "generated_pages" in hints, "PageWorkerState MUST have generated_pages for aggregation"

    def test_page_worker_state_generated_pages_uses_same_reducer(self):
        """Test that generated_pages uses same Annotated reducer as main state."""
        hints = get_type_hints(PageWorkerState, include_extras=True)
        generated_pages_type = hints.get("generated_pages")

        # Must be Annotated with operator.add - same as WikiGenerationState
        assert get_origin(generated_pages_type) is Annotated
        args = get_args(generated_pages_type)
        assert operator.add in args

    def test_page_worker_state_can_be_instantiated(self):
        """Test that PageWorkerState can be created with all fields."""
        state: PageWorkerState = {
            "page_info": {
                "title": "API Reference",
                "slug": "api-reference",
                "section": "API",
                "file_paths": ["src/api.py"],
                "file_contents": {"src/api.py": "content here"},
                "description": "API docs",
            },
            "clone_path": "/tmp/repo",
            "generated_pages": [],  # Initialized empty, worker appends result
        }

        assert state["page_info"]["title"] == "API Reference"
        assert state["generated_pages"] == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestPageWorkerState -v`
Expected: FAIL with `ImportError` because `PageWorkerState` doesn't exist

**Step 3: Write minimal implementation**

Add to `src/agents/wiki_agent.py` after `WikiGenerationState` definition:

```python
class PageWorkerState(TypedDict):
    """State for individual page generation worker.

    CRITICAL: This state MUST include `generated_pages` with the SAME
    Annotated reducer as WikiGenerationState. This is required for
    LangGraph to properly aggregate results from parallel workers.

    The worker receives this state via Send() and writes its result
    to generated_pages, which gets merged into the main state.
    """
    page_info: Dict[str, Any]  # Page definition from wiki_structure
    clone_path: Optional[str]  # For fallback file access
    # MUST match WikiGenerationState for aggregation!
    generated_pages: Annotated[List[Dict[str, Any]], operator.add]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestPageWorkerState -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(state): add PageWorkerState with shared generated_pages reducer"
```

---

### Task 4: Update `_generate_structure_node` to Set `clone_path` in State

**Files:**
- Modify: `src/agents/wiki_agent.py:346-422` (_generate_structure_node method)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.wiki_agent import WikiGenerationAgent


class TestGenerateStructureNode:
    """Tests for _generate_structure_node method."""

    @pytest.fixture
    def agent(self):
        """Create WikiGenerationAgent instance with mocked dependencies."""
        with patch("src.agents.wiki_agent.WikiGenerationAgent._get_repository_repo"):
            agent = WikiGenerationAgent()
            agent._repository_repo = AsyncMock()
            return agent

    @pytest.mark.asyncio
    async def test_generate_structure_sets_clone_path_in_state(self, agent):
        """Test that _generate_structure_node sets clone_path in returned state."""
        # Setup mock repository
        mock_repo = MagicMock()
        mock_repo.clone_path = "/tmp/test-repo"
        mock_repo.org = "test-org"
        mock_repo.name = "test-repo"
        agent._repository_repo.find_one = AsyncMock(return_value=mock_repo)

        # Mock run_structure_agent to return valid structure
        with patch("src.agents.wiki_agent.run_structure_agent") as mock_run:
            mock_run.return_value = {
                "title": "Test Wiki",
                "description": "Test description",
                "pages": [{"title": "Page 1", "slug": "page-1", "section": "Overview", "file_paths": [], "file_contents": {}, "description": "Test"}],
            }

            # Mock Path.exists
            with patch("pathlib.Path.exists", return_value=True):
                initial_state = {
                    "repository_id": str(uuid4()),
                    "file_tree": "src/",
                    "readme_content": "# Test",
                    "wiki_structure": None,
                    "generated_pages": [],
                    "current_page": None,
                    "current_step": "init",
                    "error_message": None,
                    "progress": 0.0,
                    "start_time": "2025-01-01T00:00:00Z",
                    "messages": [],
                    "clone_path": None,
                }

                result_state = await agent._generate_structure_node(initial_state)

                assert result_state["clone_path"] == "/tmp/test-repo"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestGenerateStructureNode -v`
Expected: FAIL because `clone_path` is not being set in returned state

**Step 3: Write minimal implementation**

Modify `_generate_structure_node` in `src/agents/wiki_agent.py`. Find the section after validating `clone_path` exists (around line 376) and add:

```python
            clone_path = Path(repository.clone_path)
            if not clone_path.exists():
                state["error_message"] = f"Clone path does not exist: {clone_path}"
                return state

            # IMPORTANT: Set clone_path in state for page workers to access files
            state["clone_path"] = str(clone_path)

            owner = repository.org or "unknown"
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestGenerateStructureNode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(workflow): set clone_path in state during structure generation"
```

---

## Phase 2: Prompt Engineering for Content Capture

### Task 5: Update `get_structure_prompt` to Instruct File Content Inclusion

**Files:**
- Modify: `src/agents/deep_structure_agent.py:69-181` (get_structure_prompt function)
- Test: `tests/unit/test_deep_structure_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_deep_structure_agent.py

from src.agents.deep_structure_agent import get_structure_prompt


class TestGetStructurePrompt:
    """Tests for get_structure_prompt function."""

    def test_prompt_includes_file_contents_instruction(self):
        """Test that prompt instructs agent to include file_contents."""
        prompt = get_structure_prompt(
            owner="test-org",
            repo="test-repo",
            file_tree="src/\n  main.py",
            readme_content="# Test",
            clone_path="/tmp/repo",
            use_mcp_tools=True,
        )

        # Should instruct to include file_contents
        assert "file_contents" in prompt
        assert "Include the content" in prompt or "include the content" in prompt.lower()

    def test_prompt_shows_file_contents_example(self):
        """Test that prompt shows example of file_contents format."""
        prompt = get_structure_prompt(
            owner="test-org",
            repo="test-repo",
            file_tree="src/\n  main.py",
            readme_content="# Test",
            clone_path="/tmp/repo",
            use_mcp_tools=True,
        )

        # Should show example dictionary format
        assert '"file_contents"' in prompt or "file_contents:" in prompt
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_deep_structure_agent.py::TestGetStructurePrompt -v`
Expected: FAIL because current prompt doesn't mention file_contents

**Step 3: Write minimal implementation**

Modify `get_structure_prompt` in `src/agents/deep_structure_agent.py`. Update the "Output Requirements" section at the end:

```python
    return f"""You are an expert technical writer analyzing a repository to design a wiki structure.

## Repository
- Owner: {owner}
- Name: {repo}

## Initial File Tree
{file_tree}

## README Content
{readme_content}

## Your Task
Explore this repository to understand its architecture, then design a comprehensive wiki structure.

{exploration_instructions}

## Output Requirements
When you have sufficient understanding, call `finalize_wiki_structure` with:
- **title**: A descriptive wiki title for this project
- **description**: A one-paragraph description of the wiki
- **pages**: A list of 8-12 pages, each with:
  - title: Page title
  - slug: URL-friendly identifier (lowercase, hyphens)
  - section: One of "Overview", "Architecture", "Features", "API", "Deployment", "Development"
  - file_paths: List of relevant source files
  - file_contents: Dictionary mapping file paths to their FULL contents (CRITICAL)
  - description: What this page covers

## CRITICAL: Include File Contents
For each page, you MUST include the `file_contents` field with the actual content of files you read.
This eliminates redundant file reads during page generation.

Example format:
```json
{{
  "title": "API Reference",
  "slug": "api-reference",
  "section": "API",
  "file_paths": ["src/api.py", "src/routes.py"],
  "file_contents": {{
    "src/api.py": "from flask import Flask\\n\\napp = Flask(__name__)\\n...<full content>",
    "src/routes.py": "@app.route('/users')\\ndef get_users():\\n    ...<full content>"
  }},
  "description": "REST API endpoints and routing"
}}
```

Include the COMPLETE content of each file in file_contents (not truncated).
Focus on what would help a new developer understand and work with this codebase.
"""
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_deep_structure_agent.py::TestGetStructurePrompt -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/deep_structure_agent.py tests/unit/test_deep_structure_agent.py
git commit -m "feat(prompt): update structure prompt to require file_contents inclusion"
```

---

## Phase 3: Parallel Page Generation with LangGraph Send

### Task 6: Implement `fan_out_to_page_workers` Function

**Files:**
- Modify: `src/agents/wiki_agent.py` (add new function before WikiGenerationAgent class)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

from src.agents.wiki_agent import fan_out_to_page_workers
from langgraph.types import Send


class TestFanOutToPageWorkers:
    """Tests for fan_out_to_page_workers function."""

    def test_fan_out_creates_send_for_each_page(self):
        """Test that fan_out creates a Send message for each page."""
        state = {
            "wiki_structure": {
                "title": "Test Wiki",
                "pages": [
                    {"title": "Page 1", "slug": "page-1", "section": "Overview", "file_paths": [], "file_contents": {}, "description": "Test 1"},
                    {"title": "Page 2", "slug": "page-2", "section": "API", "file_paths": [], "file_contents": {}, "description": "Test 2"},
                    {"title": "Page 3", "slug": "page-3", "section": "Dev", "file_paths": [], "file_contents": {}, "description": "Test 3"},
                ],
            },
            "clone_path": "/tmp/repo",
        }

        sends = fan_out_to_page_workers(state)

        assert len(sends) == 3
        assert all(isinstance(s, Send) for s in sends)
        assert all(s.node == "page_worker" for s in sends)

    def test_fan_out_send_payload_includes_generated_pages_empty_list(self):
        """Test that Send payload initializes generated_pages as empty list for aggregation."""
        state = {
            "wiki_structure": {
                "title": "Test Wiki",
                "pages": [
                    {"title": "Page 1", "slug": "page-1", "section": "Overview", "file_paths": [], "file_contents": {"a.py": "content"}, "description": "Test"},
                ],
            },
            "clone_path": "/my/clone/path",
        }

        sends = fan_out_to_page_workers(state)

        assert len(sends) == 1
        # CRITICAL: Send payload must include generated_pages for aggregation
        assert sends[0].arg["generated_pages"] == []
        assert sends[0].arg["clone_path"] == "/my/clone/path"
        assert sends[0].arg["page_info"]["title"] == "Page 1"

    def test_fan_out_returns_empty_list_when_no_pages(self):
        """Test that fan_out returns empty list when no pages exist."""
        state = {
            "wiki_structure": None,
            "clone_path": "/tmp/repo",
        }

        sends = fan_out_to_page_workers(state)

        assert sends == []
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestFanOutToPageWorkers -v`
Expected: FAIL with `ImportError` because `fan_out_to_page_workers` doesn't exist

**Step 3: Write minimal implementation**

Add to `src/agents/wiki_agent.py`. First, add the import at the top:
```python
from langgraph.types import Send
```

Then add the function before the `WikiGenerationAgent` class:

```python
def fan_out_to_page_workers(state: WikiGenerationState) -> list[Send]:
    """Create parallel page generation tasks using LangGraph Send.

    Each Send creates a separate page_worker execution with its own state.
    The worker state includes `generated_pages` with the same reducer as
    the main state, allowing LangGraph to aggregate results automatically.

    Args:
        state: Current workflow state with wiki_structure and clone_path

    Returns:
        List of Send objects, one per page to generate
    """
    if not state.get("wiki_structure"):
        return []

    pages = state["wiki_structure"].get("pages", [])
    if not pages:
        return []

    clone_path = state.get("clone_path")

    sends = []
    for page in pages:
        # Each Send payload becomes the worker's input state
        # CRITICAL: Must include generated_pages for aggregation to work
        sends.append(
            Send("page_worker", {
                "page_info": page,
                "clone_path": clone_path,
                "generated_pages": [],  # Will be populated by worker, then aggregated
            })
        )

    logger.info(f"Fanning out to {len(sends)} parallel page workers")
    return sends
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestFanOutToPageWorkers -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(workflow): implement fan_out_to_page_workers with correct Send payload"
```

---

### Task 7: Implement `page_worker_node` Async Function

**Files:**
- Modify: `src/agents/wiki_agent.py` (add new async function)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

from src.agents.wiki_agent import page_worker_node


class TestPageWorkerNode:
    """Tests for page_worker_node function."""

    @pytest.mark.asyncio
    async def test_page_worker_generates_content_from_cached_files(self):
        """Test that page_worker uses file_contents to generate page content."""
        state: PageWorkerState = {
            "page_info": {
                "title": "API Reference",
                "slug": "api-reference",
                "section": "API",
                "file_paths": ["src/api.py"],
                "file_contents": {
                    "src/api.py": "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/users')\ndef get_users():\n    return []",
                },
                "description": "REST API documentation",
            },
            "clone_path": "/tmp/repo",
            "generated_pages": [],
        }

        with patch("src.agents.wiki_agent.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "# API Reference\n\nThis module provides..."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            result = await page_worker_node(state)

            # Worker returns dict with generated_pages list containing the result
            assert "generated_pages" in result
            assert len(result["generated_pages"]) == 1
            assert result["generated_pages"][0]["content"] == "# API Reference\n\nThis module provides..."
            assert result["generated_pages"][0]["title"] == "API Reference"

    @pytest.mark.asyncio
    async def test_page_worker_returns_empty_when_no_file_contents(self):
        """Test that page_worker returns empty generated_pages when no file_contents."""
        state: PageWorkerState = {
            "page_info": {
                "title": "Empty Page",
                "slug": "empty-page",
                "section": "Overview",
                "file_paths": [],
                "file_contents": {},
                "description": "Page with no content",
            },
            "clone_path": None,
            "generated_pages": [],
        }

        result = await page_worker_node(state)

        assert result["generated_pages"] == []

    @pytest.mark.asyncio
    async def test_page_worker_clears_file_contents_in_result(self):
        """Test that worker clears file_contents in result to save memory."""
        state: PageWorkerState = {
            "page_info": {
                "title": "Test Page",
                "slug": "test-page",
                "section": "Overview",
                "file_paths": ["test.py"],
                "file_contents": {"test.py": "print('hello')"},
                "description": "Test",
            },
            "clone_path": "/tmp/repo",
            "generated_pages": [],
        }

        with patch("src.agents.wiki_agent.ChatOpenAI") as mock_llm_class:
            mock_llm = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = "# Test Page\n\nContent here."
            mock_llm.ainvoke = AsyncMock(return_value=mock_response)
            mock_llm_class.return_value = mock_llm

            result = await page_worker_node(state)

            # file_contents should be cleared to save memory
            assert result["generated_pages"][0].get("file_contents") is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestPageWorkerNode -v`
Expected: FAIL with `ImportError` because `page_worker_node` doesn't exist

**Step 3: Write minimal implementation**

Add to `src/agents/wiki_agent.py` after `fan_out_to_page_workers`:

```python
async def page_worker_node(state: PageWorkerState) -> Dict[str, Any]:
    """Generate content for a single wiki page using cached file contents.

    This node runs in parallel for each page via LangGraph's Send API.
    It uses pre-loaded file contents from the Deep Agent's exploration phase.

    IMPORTANT: Returns {"generated_pages": [page_result]} which gets
    aggregated with other workers via the operator.add reducer.

    Args:
        state: PageWorkerState with page_info, clone_path, and generated_pages

    Returns:
        Dict with 'generated_pages' list containing the page with generated content
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage

    page_info = state["page_info"]
    clone_path = state.get("clone_path")

    # Get file contents from cache (captured during structure generation)
    file_contents = page_info.get("file_contents", {})

    # Fallback: read from disk if file_contents is empty but file_paths exist
    if not file_contents and page_info.get("file_paths") and clone_path:
        import os
        for file_path in page_info["file_paths"][:5]:  # Limit to 5 files
            full_path = os.path.join(clone_path, file_path)
            try:
                if os.path.exists(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()[:8000]  # Limit file size
                        file_contents[file_path] = content
            except Exception as e:
                logger.warning(f"Failed to read fallback file {file_path}: {e}")

    if not file_contents:
        logger.warning(f"No file contents available for page: {page_info['title']}")
        return {"generated_pages": []}

    # Build prompt with cached file contents
    files_markdown = "\n\n".join([
        f"### File: {path}\n```\n{content[:6000]}\n```"
        for path, content in file_contents.items()
    ])

    prompt = f"""Generate comprehensive wiki documentation for this page.

## Page Information
- **Title:** {page_info['title']}
- **Section:** {page_info['section']}
- **Description:** {page_info['description']}

## Source Files
{files_markdown}

## Requirements
1. Write clear, professional technical documentation in Markdown
2. Include code examples extracted from the source files
3. Explain the purpose and usage of each component
4. Use proper headings, lists, and code blocks
5. Be comprehensive but concise
6. Do NOT include a title heading (it will be added automatically)

Generate the page content now:
"""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        # Build result page (clear file_contents to save memory)
        page_result = {
            **page_info,
            "content": response.content,
            "file_contents": None,  # Clear to reduce state size
        }

        logger.info(f"Generated content for page: {page_info['title']}")

        # Return in format for aggregation via operator.add
        return {"generated_pages": [page_result]}

    except Exception as e:
        logger.error(f"Failed to generate page {page_info['title']}: {e}")
        return {"generated_pages": []}
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestPageWorkerNode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(workflow): implement page_worker_node async function for parallel generation"
```

---

### Task 8: Implement `aggregate_pages_node` Function

**Files:**
- Modify: `src/agents/wiki_agent.py` (add new function)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

from src.agents.wiki_agent import aggregate_pages_node


class TestAggregatePagesNode:
    """Tests for aggregate_pages_node function.

    Note: LangGraph automatically aggregates generated_pages from all workers
    via the operator.add reducer BEFORE this node runs. This node receives
    the already-aggregated results.
    """

    @pytest.mark.asyncio
    async def test_aggregate_updates_progress_and_step(self):
        """Test that aggregate_pages_node updates progress and current_step."""
        # State with already-aggregated generated_pages (done by LangGraph)
        state = {
            "generated_pages": [
                {"title": "Page 1", "content": "Content 1"},
                {"title": "Page 2", "content": "Content 2"},
            ],
            "wiki_structure": {"title": "Test", "pages": []},
            "current_step": "generating_pages",
            "progress": 60.0,
            "messages": [],
        }

        result = await aggregate_pages_node(state)

        assert result["current_step"] == "pages_generated"
        assert result["progress"] == 90.0

    @pytest.mark.asyncio
    async def test_aggregate_preserves_generated_pages(self):
        """Test that aggregate_pages_node preserves the aggregated pages."""
        state = {
            "generated_pages": [
                {"title": "Page 1", "content": "Content 1"},
                {"title": "Page 2", "content": "Content 2"},
                {"title": "Page 3", "content": "Content 3"},
            ],
            "wiki_structure": {"title": "Test", "pages": []},
            "current_step": "generating_pages",
            "progress": 60.0,
            "messages": [],
        }

        result = await aggregate_pages_node(state)

        # Pages should be preserved, not modified
        assert len(result["generated_pages"]) == 3
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestAggregatePagesNode -v`
Expected: FAIL with `ImportError` because `aggregate_pages_node` doesn't exist

**Step 3: Write minimal implementation**

Add to `src/agents/wiki_agent.py` after `page_worker_node`:

```python
async def aggregate_pages_node(state: WikiGenerationState) -> Dict[str, Any]:
    """Aggregate results from all parallel page workers.

    NOTE: LangGraph automatically aggregates generated_pages from all workers
    via the operator.add reducer BEFORE this node runs. This node receives
    the already-merged results.

    This node:
    1. Logs the aggregation results
    2. Updates progress and step status
    3. Prepares state for the store_wiki node

    Args:
        state: WikiGenerationState with aggregated generated_pages

    Returns:
        Updated state dict with progress and step updates
    """
    generated_pages = state.get("generated_pages", [])

    logger.info(f"Aggregated {len(generated_pages)} generated pages from parallel workers")

    return {
        "current_step": "pages_generated",
        "progress": 90.0,
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestAggregatePagesNode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(workflow): implement aggregate_pages_node for post-worker status update"
```

---

### Task 9: Update `_create_workflow` to Use Send Pattern

**Files:**
- Modify: `src/agents/wiki_agent.py:75-105` (_create_workflow method)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

class TestCreateWorkflow:
    """Tests for _create_workflow method."""

    def test_workflow_includes_page_worker_node(self):
        """Test that workflow graph includes page_worker node."""
        with patch("src.agents.wiki_agent.WikiGenerationAgent._get_repository_repo"):
            agent = WikiGenerationAgent()

            graph = agent.workflow.get_graph()
            node_names = [n.name for n in graph.nodes.values()]

            assert "page_worker" in node_names, "Workflow must include page_worker node"

    def test_workflow_includes_aggregate_pages_node(self):
        """Test that workflow graph includes aggregate_pages node."""
        with patch("src.agents.wiki_agent.WikiGenerationAgent._get_repository_repo"):
            agent = WikiGenerationAgent()

            graph = agent.workflow.get_graph()
            node_names = [n.name for n in graph.nodes.values()]

            assert "aggregate_pages" in node_names, "Workflow must include aggregate_pages node"

    def test_workflow_does_not_include_old_generate_pages_node(self):
        """Test that old sequential generate_pages node is removed."""
        with patch("src.agents.wiki_agent.WikiGenerationAgent._get_repository_repo"):
            agent = WikiGenerationAgent()

            graph = agent.workflow.get_graph()
            node_names = [n.name for n in graph.nodes.values()]

            # Old node should not exist
            assert "generate_pages" not in node_names, "Old generate_pages node should be removed"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent.py::TestCreateWorkflow -v`
Expected: FAIL because workflow doesn't include new nodes

**Step 3: Write minimal implementation**

Replace the `_create_workflow` method in `src/agents/wiki_agent.py`:

```python
def _create_workflow(self) -> StateGraph:
    """Create the wiki generation workflow graph with parallel page generation.

    Workflow:
        START -> analyze_repository -> generate_structure
              -> [parallel page_workers via Send] -> aggregate_pages
              -> store_wiki -> END

    The Send API creates parallel page_worker executions, each processing
    one page. Results are automatically aggregated via the operator.add
    reducer on generated_pages.

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create workflow graph
    workflow = StateGraph(WikiGenerationState)

    # Add nodes
    workflow.add_node("analyze_repository", self._analyze_repository_node)
    workflow.add_node("generate_structure", self._generate_structure_node)
    workflow.add_node("page_worker", page_worker_node)  # Parallel page generator
    workflow.add_node("aggregate_pages", aggregate_pages_node)  # Post-aggregation status
    workflow.add_node("store_wiki", self._store_wiki_node)
    workflow.add_node("handle_error", self._handle_error_node)

    # Define workflow edges
    workflow.add_edge(START, "analyze_repository")
    workflow.add_edge("analyze_repository", "generate_structure")

    # Fan-out: generate_structure -> parallel page_workers via Send
    # Each Send creates a separate page_worker execution
    workflow.add_conditional_edges(
        "generate_structure",
        fan_out_to_page_workers,
        ["page_worker"]
    )

    # Fan-in: all page_workers -> aggregate_pages
    # LangGraph aggregates generated_pages via operator.add before this node
    workflow.add_edge("page_worker", "aggregate_pages")

    # Continue to storage
    workflow.add_edge("aggregate_pages", "store_wiki")
    workflow.add_edge("store_wiki", END)

    # Error handling
    workflow.add_edge("handle_error", END)

    app = workflow.compile().with_config({"run_name": "wiki_agent.wiki_generation_workflow"})
    logger.debug(f"Wiki generation workflow (parallel):\n {app.get_graph().draw_mermaid()}")
    return app
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestCreateWorkflow -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(workflow): update workflow to use LangGraph Send pattern for parallel pages"
```

---

## Phase 4: Update Store Node and Cleanup

### Task 10: Update `_store_wiki_node` to Use Aggregated Pages

**Files:**
- Modify: `src/agents/wiki_agent.py` (_store_wiki_node method)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Write the failing test**

```python
# Add to tests/unit/test_wiki_agent.py

class TestStoreWikiNode:
    """Tests for _store_wiki_node method."""

    @pytest.fixture
    def agent(self):
        """Create WikiGenerationAgent with mocked dependencies."""
        with patch("src.agents.wiki_agent.WikiGenerationAgent._get_repository_repo"):
            with patch("src.agents.wiki_agent.WikiGenerationAgent._get_wiki_repo"):
                agent = WikiGenerationAgent()
                agent._repository_repo = AsyncMock()
                agent._wiki_repo = AsyncMock()
                return agent

    @pytest.mark.asyncio
    async def test_store_wiki_uses_generated_pages_not_original(self, agent):
        """Test that _store_wiki_node uses generated_pages from parallel workers."""
        from src.models.wiki import WikiStructure

        agent._wiki_repo.create = AsyncMock(return_value=MagicMock())
        agent._repository_repo.find_one = AsyncMock(return_value=MagicMock(
            id=uuid4(),
            name="test-repo",
        ))

        state = {
            "repository_id": str(uuid4()),
            "wiki_structure": {
                "title": "Test Wiki",
                "description": "Test description",
                "pages": [
                    # Original pages - no content
                    {"title": "Page 1", "slug": "page-1", "section": "Overview", "file_paths": [], "description": "Desc 1"},
                ],
            },
            "generated_pages": [
                # Aggregated pages from workers - WITH content
                {"title": "Page 1", "slug": "page-1", "section": "Overview", "content": "Generated content 1", "file_paths": [], "description": "Desc 1"},
                {"title": "Page 2", "slug": "page-2", "section": "API", "content": "Generated content 2", "file_paths": [], "description": "Desc 2"},
            ],
            "current_step": "pages_generated",
            "progress": 90.0,
            "messages": [],
            "error_message": None,
        }

        await agent._store_wiki_node(state)

        # Verify create was called with generated_pages (2 pages), not original (1 page)
        agent._wiki_repo.create.assert_called_once()
        call_args = agent._wiki_repo.create.call_args
        saved_wiki = call_args[0][0]

        assert len(saved_wiki.pages) == 2, "Should use generated_pages (2), not original pages (1)"
        assert all(page.content for page in saved_wiki.pages), "All pages should have content"
```

**Step 2: Run test to verify behavior**

Run: `pytest tests/unit/test_wiki_agent.py::TestStoreWikiNode -v`
Expected: May fail if current implementation uses original pages

**Step 3: Modify _store_wiki_node**

Find `_store_wiki_node` and ensure it uses `generated_pages`:

```python
async def _store_wiki_node(
    self, state: WikiGenerationState
) -> WikiGenerationState:
    """Store generated wiki structure to database.

    Uses `generated_pages` from parallel workers (aggregated by LangGraph)
    rather than the original pages from wiki_structure.
    """
    try:
        state["current_step"] = "storing_wiki"

        if not state.get("wiki_structure"):
            state["error_message"] = "No wiki structure to store"
            return state

        # IMPORTANT: Use generated_pages from parallel workers, not original pages
        pages_to_store = state.get("generated_pages", [])

        if not pages_to_store:
            # Fallback to original pages if parallel generation failed
            pages_to_store = state["wiki_structure"].get("pages", [])
            logger.warning("No generated pages found, falling back to original structure")

        # Get repository
        repository = await self._repository_repo.find_one(
            {"_id": UUID(state["repository_id"])}
        )

        if not repository:
            state["error_message"] = "Repository not found"
            return state

        # Create WikiStructure with generated content
        wiki_structure = WikiStructure(
            repository_id=repository.id,
            title=state["wiki_structure"]["title"],
            description=state["wiki_structure"].get("description", ""),
            pages=[
                WikiPageDetail(
                    id=str(uuid4()),
                    title=page["title"],
                    slug=page["slug"],
                    section=page.get("section", "Overview"),
                    content=page.get("content", ""),
                    file_paths=page.get("file_paths", []),
                    description=page.get("description", ""),
                )
                for page in pages_to_store
            ],
        )

        await self._wiki_repo.create(wiki_structure)

        state["progress"] = 100.0
        state["current_step"] = "completed"
        state["messages"].append(
            AIMessage(content=f"Wiki saved with {len(pages_to_store)} pages")
        )

        return state

    except Exception as e:
        logger.error(f"Failed to store wiki: {e}")
        state["error_message"] = f"Failed to store wiki: {str(e)}"
        return state
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_agent.py::TestStoreWikiNode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent.py
git commit -m "feat(workflow): update store_wiki_node to use aggregated generated_pages"
```

---

### Task 11: Initialize State Properly in `generate_wiki` Method

**Files:**
- Modify: `src/agents/wiki_agent.py` (generate_wiki method)
- Test: `tests/unit/test_wiki_agent.py`

**Step 1: Verify and update generate_wiki**

Ensure `generate_wiki` initializes state with all required fields:

```python
async def generate_wiki(self, repository_id: str) -> WikiGenerationResult:
    """Generate wiki documentation for a repository."""
    from datetime import datetime

    initial_state: WikiGenerationState = {
        "repository_id": repository_id,
        "file_tree": "",
        "readme_content": "",
        "wiki_structure": None,
        "generated_pages": [],  # CRITICAL: Initialize for parallel aggregation
        "current_page": None,
        "current_step": "starting",
        "error_message": None,
        "progress": 0.0,
        "start_time": datetime.utcnow().isoformat(),
        "messages": [],
        "clone_path": None,  # Will be set by _generate_structure_node
    }

    # ... rest of method
```

**Step 2: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "feat(workflow): ensure generate_wiki initializes all state fields"
```

---

### Task 12: Remove Old `_generate_pages_node` Method

**Files:**
- Modify: `src/agents/wiki_agent.py`

**Step 1: Remove or deprecate the old method**

The old `_generate_pages_node` method is no longer called. Either:
- Delete it entirely, OR
- Add deprecation comment

```python
# DEPRECATED: Replaced by parallel page_worker_node with Send pattern
# Keeping for reference during transition
# async def _generate_pages_node(self, state: WikiGenerationState) -> WikiGenerationState:
#     ...
```

**Step 2: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 3: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "refactor(workflow): remove deprecated sequential _generate_pages_node"
```

---

## Phase 5: Integration Testing

### Task 13: Create Integration Test for Full Parallel Workflow

**Files:**
- Modify: `tests/integration/test_wiki_generation.py`

**Step 1: Write integration test**

```python
# Add to tests/integration/test_wiki_generation.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.wiki_agent import WikiGenerationAgent


class TestHybridPageGeneration:
    """Integration tests for hybrid parallel page generation workflow."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_workflow_generates_pages_in_parallel(self):
        """Test complete workflow with parallel page generation."""
        mock_repo = MagicMock()
        mock_repo.id = uuid4()
        mock_repo.name = "test-repo"
        mock_repo.org = "test-org"
        mock_repo.clone_path = "/tmp/test-repo"

        with patch("src.agents.wiki_agent.WikiGenerationAgent._get_repository_repo") as mock_repo_getter:
            with patch("src.agents.wiki_agent.WikiGenerationAgent._get_wiki_repo") as mock_wiki_getter:
                mock_repo_repo = AsyncMock()
                mock_repo_repo.find_one = AsyncMock(return_value=mock_repo)
                mock_repo_getter.return_value = mock_repo_repo

                mock_wiki_repo = AsyncMock()
                mock_wiki_repo.create = AsyncMock()
                mock_wiki_getter.return_value = mock_wiki_repo

                # Mock Deep Agent with file_contents
                with patch("src.agents.wiki_agent.run_structure_agent") as mock_deep:
                    mock_deep.return_value = {
                        "title": "Test Wiki",
                        "description": "Test",
                        "pages": [
                            {
                                "title": "Overview",
                                "slug": "overview",
                                "section": "Overview",
                                "file_paths": ["README.md"],
                                "file_contents": {"README.md": "# Test"},
                                "description": "Overview",
                            },
                            {
                                "title": "API",
                                "slug": "api",
                                "section": "API",
                                "file_paths": ["api.py"],
                                "file_contents": {"api.py": "def hello(): pass"},
                                "description": "API",
                            },
                        ],
                    }

                    # Mock ChatOpenAI for parallel workers
                    with patch("src.agents.wiki_agent.ChatOpenAI") as mock_llm:
                        mock_response = MagicMock()
                        mock_response.content = "Generated content"
                        mock_llm.return_value.ainvoke = AsyncMock(return_value=mock_response)

                        with patch("pathlib.Path.exists", return_value=True):
                            agent = WikiGenerationAgent()
                            result = await agent.generate_wiki(str(mock_repo.id))

                            # Verify wiki stored with generated content
                            mock_wiki_repo.create.assert_called_once()
                            saved = mock_wiki_repo.create.call_args[0][0]

                            assert len(saved.pages) == 2
                            assert all(p.content for p in saved.pages)
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_wiki_generation.py::TestHybridPageGeneration -v`

**Step 3: Commit**

```bash
git add tests/integration/test_wiki_generation.py
git commit -m "test(integration): add integration test for parallel page generation workflow"
```

---

## Phase 6: Final Verification

### Task 14: Run Full Test Suite

```bash
pytest tests/ -v --tb=short
```

Fix any failing tests.

### Task 15: Manual E2E Test

1. Start dev server: `.\scripts\dev-run.ps1`
2. Trigger wiki generation via API
3. Check LangSmith for parallel `page_worker` traces
4. Verify wiki in MongoDB has content

---

## Summary of Key LangGraph Patterns Used

| Pattern | Implementation |
|---------|----------------|
| **Send API** | `fan_out_to_page_workers` returns `list[Send]` |
| **Worker State** | `PageWorkerState` includes `generated_pages: Annotated[List, operator.add]` |
| **Reducer** | Both states share same key with `operator.add` for auto-aggregation |
| **Async nodes** | `page_worker_node` uses `async def` and `await llm.ainvoke()` |
| **Conditional edges** | `add_conditional_edges("generate_structure", fan_out, ["page_worker"])` |

---

## Expected Performance

| Metric | Sequential | Parallel (Hybrid) |
|--------|-----------|-------------------|
| File reads | 2x | 1x |
| Page generation | Sequential | Fully parallel |
| Total time | ~90-120s | ~65-80s |

---

*Plan revised: 2025-12-31 with LangGraph/LangChain documentation verification*
