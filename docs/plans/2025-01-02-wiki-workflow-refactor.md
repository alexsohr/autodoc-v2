# Wiki Workflow Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace autonomous deep agents with a predictable LangGraph Map-Reduce workflow for wiki documentation generation.

**Architecture:** The new workflow uses LangGraph's Send API for parallel page generation. Structure extraction happens first (deterministic), then pages fan out for parallel generation, aggregate results, and finalize into a complete wiki. All prompts are preserved in YAML.

**Tech Stack:** LangGraph StateGraph, Send API, Pydantic structured output, Beanie ODM, existing LLMTool

---

## Pre-Implementation: Code Cleanup

### Task 0: Archive Deep Agent Code

**Purpose:** Remove deprecated deep agent implementation before starting new work.

**Files:**
- Archive: `src/agents/deep_structure_agent.py` → `archive/deep_structure_agent.py`
- Delete test: `tests/unit/test_deep_page_agent.py`
- Keep: `src/agents/page_tools.py` (has useful prompt content, already extracted to YAML)

**Step 1: Create archive directory**

```bash
mkdir -p archive
```

**Step 2: Move deprecated file to archive**

```bash
git mv src/agents/deep_structure_agent.py archive/deep_structure_agent.py
```

**Step 3: Delete orphaned test file**

```bash
git rm tests/unit/test_deep_page_agent.py
```

**Step 4: Commit cleanup**

```bash
git add -A
git commit -m "chore: archive deep_structure_agent, remove orphaned tests

Preparing for wiki workflow refactor. Deep agent approach replaced
with predictable LangGraph workflow."
```

---

## Phase 1: State and Models

### Task 1: Explore Existing State Classes

**Purpose:** Identify reusable state components before creating new ones.

**Exploration checklist:**
- [ ] Check `WikiGenerationState` in `src/agents/wiki_agent.py` for reusable fields
- [ ] Check `wiki.py` models for existing page content structures
- [ ] Check if `WikiPageDetail` can store generated content or needs extension

**Files to examine:**
- `src/agents/wiki_agent.py:WikiGenerationState`
- `src/models/wiki.py:WikiPageDetail`
- `src/models/wiki.py:WikiStructure`

**Step 1: Document findings**

After exploration, document which existing classes can be reused vs extended.

---

### Task 2: Extend WikiPageDetail for Generated Content

**Purpose:** Add field to store generated markdown content on existing model.

**Files:**
- Modify: `src/models/wiki.py:WikiPageDetail`
- Test: `tests/unit/test_wiki_models.py`

**Step 1: Write failing test for content field**

```python
# tests/unit/test_wiki_models.py
def test_wiki_page_detail_with_generated_content():
    """WikiPageDetail should store generated markdown content."""
    page = WikiPageDetail(
        id="getting-started",
        title="Getting Started",
        description="How to get started",
        importance=PageImportance.HIGH,
        content="# Getting Started\n\nThis is the content.",
    )
    assert page.content == "# Getting Started\n\nThis is the content."
    assert page.id == "getting-started"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_models.py::test_wiki_page_detail_with_generated_content -v
```
Expected: FAIL - `content` field doesn't exist or validation fails

**Step 3: Add content field to WikiPageDetail**

Modify `src/models/wiki.py:WikiPageDetail`:

```python
class WikiPageDetail(BaseModel):
    """Represents a single page within a wiki section."""

    id: str = Field(..., description="Unique identifier for the page (kebab-case)")
    title: str = Field(..., description="Display title for the page")
    description: str = Field(..., description="Brief description of page content")
    importance: PageImportance = Field(
        default=PageImportance.MEDIUM,
        description="Priority level for this page"
    )
    file_paths: List[str] = Field(
        default_factory=list,
        description="Relevant source files to reference"
    )
    content: Optional[str] = Field(
        default=None,
        description="Generated markdown content for this page"
    )

    # Keep existing validators...
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_models.py::test_wiki_page_detail_with_generated_content -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/wiki.py tests/unit/test_wiki_models.py
git commit -m "feat(wiki): add content field to WikiPageDetail

Stores generated markdown content for each wiki page."
```

---

### Task 3: Create WikiWorkflowState

**Purpose:** Define state for the new LangGraph workflow, reusing existing types.

**Files:**
- Create: `src/agents/wiki_workflow.py`
- Test: `tests/unit/test_wiki_workflow.py`

**Step 1: Write failing test for state structure**

```python
# tests/unit/test_wiki_workflow.py
import operator
from typing import Annotated, Optional, List
from src.agents.wiki_workflow import WikiWorkflowState
from src.models.wiki import WikiStructure, WikiPageDetail

def test_wiki_workflow_state_structure():
    """WikiWorkflowState should have required fields with correct types."""
    state = WikiWorkflowState(
        repository_id="test-repo-id",
        clone_path="/tmp/repo",
        file_tree="src/\n  main.py",
        readme_content="# Test Repo",
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )
    assert state["repository_id"] == "test-repo-id"
    assert state["pages"] == []
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_workflow.py::test_wiki_workflow_state_structure -v
```
Expected: FAIL - module/class doesn't exist

**Step 3: Create WikiWorkflowState**

Create `src/agents/wiki_workflow.py`:

```python
"""
Wiki generation workflow using LangGraph Map-Reduce pattern.

This module replaces the autonomous deep agent approach with a predictable
workflow that:
1. Extracts wiki structure (deterministic)
2. Generates pages in parallel (fan-out)
3. Aggregates results (fan-in)
4. Finalizes wiki storage
"""

import operator
from typing import Annotated, Optional, List, TypedDict
from src.models.wiki import WikiStructure, WikiPageDetail


class WikiWorkflowState(TypedDict):
    """State for the wiki generation workflow.

    Attributes:
        repository_id: UUID of the repository being documented
        clone_path: Local filesystem path to cloned repository
        file_tree: String representation of repository file structure
        readme_content: Content of repository README file
        structure: Extracted wiki structure (sections and pages)
        pages: List of pages with generated content (reducer: append)
        error: Error message if workflow fails
        current_step: Current workflow step for observability
    """
    repository_id: str
    clone_path: str
    file_tree: str
    readme_content: str
    structure: Optional[WikiStructure]
    pages: Annotated[List[WikiPageDetail], operator.add]
    error: Optional[str]
    current_step: str
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_workflow.py::test_wiki_workflow_state_structure -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add WikiWorkflowState for LangGraph workflow

TypedDict state with reducer for parallel page collection."
```

---

## Phase 2: Structure Extraction Node

### Task 4: Explore Existing LLM Structured Output

**Purpose:** Understand how to use existing LLMTool for structured output.

**Exploration checklist:**
- [ ] Check `LLMTool.generate_structured` signature and return type
- [ ] Check how it handles Pydantic models
- [ ] Check error handling patterns

**Files to examine:**
- `src/tools/llm_tool.py:generate_structured`

**Findings to document:**
- Method signature: `async def generate_structured(self, prompt, schema, provider=None, system_message=None)`
- Returns: `{"status": "success", "structured_output": data}` or `{"status": "error", ...}`
- Schema can be Pydantic model class

---

### Task 5: Create Structure Extraction Node

**Purpose:** Node that extracts wiki structure using LLM structured output.

**Files:**
- Modify: `src/agents/wiki_workflow.py`
- Test: `tests/unit/test_wiki_workflow.py`

**Step 1: Write failing test for extract_structure node**

```python
# tests/unit/test_wiki_workflow.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.agents.wiki_workflow import extract_structure_node, WikiWorkflowState
from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

@pytest.mark.asyncio
async def test_extract_structure_node_success():
    """extract_structure_node should return structure from LLM."""
    initial_state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/\n  main.py\n  utils.py",
        readme_content="# Test Project\nA test project.",
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )

    mock_structure = WikiStructure(
        title="Test Project",
        description="A test project",
        sections=[
            WikiSection(
                id="overview",
                title="Overview",
                pages=[
                    WikiPageDetail(
                        id="getting-started",
                        title="Getting Started",
                        description="How to get started",
                        importance=PageImportance.HIGH,
                    )
                ]
            )
        ]
    )

    with patch("src.agents.wiki_workflow.LLMTool") as MockLLMTool:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "status": "success",
            "structured_output": mock_structure.model_dump()
        }
        MockLLMTool.return_value = mock_llm

        result = await extract_structure_node(initial_state)

        assert result["structure"] is not None
        assert result["structure"].title == "Test Project"
        assert result["current_step"] == "structure_extracted"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_workflow.py::test_extract_structure_node_success -v
```
Expected: FAIL - function doesn't exist

**Step 3: Implement extract_structure_node**

Add to `src/agents/wiki_workflow.py`:

```python
from src.tools.llm_tool import LLMTool
from src.utils.config_loader import get_settings

# Load prompt from YAML
import yaml
from pathlib import Path

def _load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent.parent / "prompts" / "wiki_prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

PROMPTS = _load_prompts()


async def extract_structure_node(state: WikiWorkflowState) -> dict:
    """Extract wiki structure from repository analysis.

    Uses LLM with structured output to generate a WikiStructure
    based on the repository's file tree and README content.

    Args:
        state: Current workflow state with file_tree and readme_content

    Returns:
        Dict with 'structure' and updated 'current_step'
    """
    settings = get_settings()
    llm_tool = LLMTool()

    # Build prompt from template
    system_prompt = PROMPTS["structure_agent"]["system_prompt"]
    user_prompt = f"""Analyze this repository and create a wiki structure.

## File Tree
```
{state["file_tree"]}
```

## README Content
```
{state["readme_content"]}
```

Create a comprehensive wiki structure with sections and pages."""

    result = await llm_tool.generate_structured(
        prompt=user_prompt,
        schema=WikiStructure,
        system_message=system_prompt,
    )

    if result["status"] == "error":
        return {
            "error": f"Structure extraction failed: {result.get('error', 'Unknown error')}",
            "current_step": "error",
        }

    # Parse structured output back to WikiStructure
    structure_data = result["structured_output"]
    if isinstance(structure_data, dict):
        structure = WikiStructure.model_validate(structure_data)
    else:
        structure = structure_data

    return {
        "structure": structure,
        "current_step": "structure_extracted",
    }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_workflow.py::test_extract_structure_node_success -v
```
Expected: PASS

**Step 5: Add error case test**

```python
@pytest.mark.asyncio
async def test_extract_structure_node_error():
    """extract_structure_node should handle LLM errors gracefully."""
    initial_state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )

    with patch("src.agents.wiki_workflow.LLMTool") as MockLLMTool:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "status": "error",
            "error": "Rate limit exceeded"
        }
        MockLLMTool.return_value = mock_llm

        result = await extract_structure_node(initial_state)

        assert result["error"] is not None
        assert "Rate limit" in result["error"]
        assert result["current_step"] == "error"
```

**Step 6: Run all structure tests**

```bash
pytest tests/unit/test_wiki_workflow.py -k "extract_structure" -v
```
Expected: All PASS

**Step 7: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add extract_structure_node for wiki workflow

Uses LLMTool.generate_structured with WikiStructure schema.
Loads prompts from wiki_prompts.yaml."
```

---

## Phase 3: Page Generation Nodes

### Task 6: Explore Existing Page Generation Patterns

**Purpose:** Understand existing page generation to reuse patterns.

**Exploration checklist:**
- [ ] Check `page_tools.py` for prompt patterns
- [ ] Check if file reading utilities exist
- [ ] Check how repository tools work

**Files to examine:**
- `src/agents/page_tools.py`
- `src/tools/repository_tool.py`

---

### Task 7: Create Fan-Out Function

**Purpose:** Create function that returns Send objects for parallel page generation.

**Files:**
- Modify: `src/agents/wiki_workflow.py`
- Test: `tests/unit/test_wiki_workflow.py`

**Step 1: Write failing test for fan_out_pages**

```python
# tests/unit/test_wiki_workflow.py
from langgraph.types import Send
from src.agents.wiki_workflow import fan_out_pages, WikiWorkflowState
from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

def test_fan_out_pages_creates_send_objects():
    """fan_out_pages should create Send object for each page."""
    structure = WikiStructure(
        title="Test",
        description="Test wiki",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(id="page1", title="Page 1", description="First page", importance=PageImportance.HIGH),
                    WikiPageDetail(id="page2", title="Page 2", description="Second page", importance=PageImportance.MEDIUM),
                ]
            )
        ]
    )

    state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
    )

    sends = fan_out_pages(state)

    assert len(sends) == 2
    assert all(isinstance(s, Send) for s in sends)
    assert sends[0].node == "generate_page"
    assert sends[0].arg["page"].id == "page1"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_workflow.py::test_fan_out_pages_creates_send_objects -v
```
Expected: FAIL - function doesn't exist

**Step 3: Implement fan_out_pages**

Add to `src/agents/wiki_workflow.py`:

```python
from langgraph.types import Send
from typing import List


def fan_out_pages(state: WikiWorkflowState) -> List[Send]:
    """Create parallel tasks for each page in the wiki structure.

    This is the fan-out step of the Map-Reduce pattern. Each page
    gets its own Send object targeting the generate_page node.

    Args:
        state: Current state with extracted structure

    Returns:
        List of Send objects for parallel page generation
    """
    if state.get("error") or not state.get("structure"):
        # Skip fan-out if there's an error or no structure
        return [Send("finalize", {})]

    structure = state["structure"]
    sends = []

    for page in structure.get_all_pages():
        sends.append(Send("generate_page", {
            "page": page,
            "clone_path": state["clone_path"],
            "file_tree": state["file_tree"],
            "readme_content": state["readme_content"],
            "repository_id": state["repository_id"],
        }))

    return sends
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_workflow.py::test_fan_out_pages_creates_send_objects -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add fan_out_pages for parallel page generation

Creates Send objects for each page in structure."
```

---

### Task 8: Create Page Generation Node

**Purpose:** Node that generates content for a single page.

**Files:**
- Modify: `src/agents/wiki_workflow.py`
- Test: `tests/unit/test_wiki_workflow.py`

**Step 1: Write failing test for generate_page_node**

```python
# tests/unit/test_wiki_workflow.py
@pytest.mark.asyncio
async def test_generate_page_node_success():
    """generate_page_node should generate content for a single page."""
    page = WikiPageDetail(
        id="getting-started",
        title="Getting Started",
        description="How to get started with the project",
        importance=PageImportance.HIGH,
        file_paths=["src/main.py"],
    )

    page_state = {
        "page": page,
        "clone_path": "/tmp/repo",
        "file_tree": "src/\n  main.py",
        "readme_content": "# Test",
        "repository_id": "test-repo",
    }

    with patch("src.agents.wiki_workflow.LLMTool") as MockLLMTool:
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = {
            "status": "success",
            "content": "# Getting Started\n\nThis is the generated content."
        }
        MockLLMTool.return_value = mock_llm

        result = await generate_page_node(page_state)

        assert "pages" in result
        assert len(result["pages"]) == 1
        assert result["pages"][0].id == "getting-started"
        assert result["pages"][0].content is not None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_workflow.py::test_generate_page_node_success -v
```
Expected: FAIL - function doesn't exist

**Step 3: Implement generate_page_node**

Add to `src/agents/wiki_workflow.py`:

```python
async def generate_page_node(state: dict) -> dict:
    """Generate content for a single wiki page.

    This node runs in parallel for each page. It receives page details
    and repository context, then uses LLM to generate markdown content.

    Args:
        state: Dict containing 'page', 'clone_path', 'file_tree', etc.

    Returns:
        Dict with 'pages' list containing the page with generated content
    """
    page: WikiPageDetail = state["page"]
    clone_path = state["clone_path"]
    file_tree = state["file_tree"]

    llm_tool = LLMTool()

    # Build page generation prompt
    system_prompt = PROMPTS["page_generation_full"]["system_prompt"]

    # Read relevant files if specified
    file_contents = ""
    if page.file_paths:
        for file_path in page.file_paths[:5]:  # Limit to 5 files
            full_path = Path(clone_path) / file_path
            if full_path.exists() and full_path.is_file():
                try:
                    content = full_path.read_text(encoding="utf-8", errors="ignore")
                    file_contents += f"\n\n### File: {file_path}\n```\n{content[:5000]}\n```"
                except Exception:
                    pass

    user_prompt = f"""Generate comprehensive documentation for this wiki page.

## Page Details
- Title: {page.title}
- Description: {page.description}
- Importance: {page.importance.value}

## Repository File Tree
```
{file_tree}
```

## Relevant Source Files
{file_contents if file_contents else "No specific files referenced."}

Generate the markdown content for this page."""

    result = await llm_tool.generate(
        prompt=user_prompt,
        system_message=system_prompt,
    )

    if result["status"] == "error":
        # Return page with error note in content
        page_with_content = page.model_copy(update={
            "content": f"*Error generating content: {result.get('error', 'Unknown')}*"
        })
    else:
        page_with_content = page.model_copy(update={
            "content": result["content"]
        })

    return {"pages": [page_with_content]}
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_workflow.py::test_generate_page_node_success -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add generate_page_node for content generation

Generates markdown content for single page using LLM."
```

---

## Phase 4: Aggregation and Finalization

### Task 9: Explore Existing Storage Logic

**Purpose:** Understand existing wiki storage to reuse.

**Exploration checklist:**
- [ ] Check `_store_wiki_node` in wiki_agent.py
- [ ] Check how WikiStructure is stored
- [ ] Check repository patterns for wiki storage

**Files to examine:**
- `src/agents/wiki_agent.py:_store_wiki_node`
- `src/repository/` for wiki repositories

---

### Task 10: Create Finalize Node

**Purpose:** Combine pages into final wiki and store to database.

**Files:**
- Modify: `src/agents/wiki_workflow.py`
- Test: `tests/unit/test_wiki_workflow.py`

**Step 1: Write failing test for finalize_node**

```python
# tests/unit/test_wiki_workflow.py
@pytest.mark.asyncio
async def test_finalize_node_combines_pages():
    """finalize_node should combine pages into structure and store."""
    structure = WikiStructure(
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(id="page1", title="Page 1", description="First"),
                    WikiPageDetail(id="page2", title="Page 2", description="Second"),
                ]
            )
        ]
    )

    pages_with_content = [
        WikiPageDetail(id="page1", title="Page 1", description="First", content="# Page 1 Content"),
        WikiPageDetail(id="page2", title="Page 2", description="Second", content="# Page 2 Content"),
    ]

    state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=pages_with_content,
        error=None,
        current_step="pages_generated",
    )

    with patch("src.agents.wiki_workflow.WikiStructure") as MockWikiStructure:
        # Mock database storage
        mock_wiki = MagicMock()
        mock_wiki.save = AsyncMock()
        MockWikiStructure.return_value = mock_wiki

        result = await finalize_node(state)

        assert result["current_step"] == "completed"
        assert result.get("error") is None
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_workflow.py::test_finalize_node_combines_pages -v
```
Expected: FAIL - function doesn't exist

**Step 3: Implement finalize_node**

Add to `src/agents/wiki_workflow.py`:

```python
async def finalize_node(state: WikiWorkflowState) -> dict:
    """Finalize wiki by combining pages and storing to database.

    This is the fan-in step. It takes all generated pages, updates
    the structure with content, and persists to MongoDB.

    Args:
        state: State with structure and generated pages

    Returns:
        Dict with updated current_step
    """
    if state.get("error"):
        return {"current_step": "error"}

    structure = state.get("structure")
    pages = state.get("pages", [])

    if not structure:
        return {
            "error": "No structure available for finalization",
            "current_step": "error",
        }

    # Create page lookup by ID
    page_content_map = {p.id: p.content for p in pages if p.content}

    # Update structure sections with generated content
    updated_sections = []
    for section in structure.sections:
        updated_pages = []
        for page in section.pages:
            content = page_content_map.get(page.id)
            if content:
                updated_page = page.model_copy(update={"content": content})
            else:
                updated_page = page
            updated_pages.append(updated_page)

        updated_section = section.model_copy(update={"pages": updated_pages})
        updated_sections.append(updated_section)

    # Create final wiki structure
    final_wiki = WikiStructure(
        repository_id=state["repository_id"],
        title=structure.title,
        description=structure.description,
        sections=updated_sections,
    )

    # Store to database
    try:
        await final_wiki.save()
    except Exception as e:
        return {
            "error": f"Failed to save wiki: {str(e)}",
            "current_step": "error",
        }

    return {
        "current_step": "completed",
    }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_workflow.py::test_finalize_node_combines_pages -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add finalize_node for wiki storage

Combines generated pages and persists WikiStructure to MongoDB."
```

---

## Phase 5: Graph Assembly

### Task 11: Assemble LangGraph Workflow

**Purpose:** Connect all nodes into complete workflow graph.

**Files:**
- Modify: `src/agents/wiki_workflow.py`
- Test: `tests/unit/test_wiki_workflow.py`

**Step 1: Write failing test for workflow compilation**

```python
# tests/unit/test_wiki_workflow.py
def test_wiki_workflow_compiles():
    """wiki_workflow should compile without errors."""
    from src.agents.wiki_workflow import create_wiki_workflow

    workflow = create_wiki_workflow()

    assert workflow is not None
    # Check graph has expected nodes
    assert "extract_structure" in str(workflow.nodes)
    assert "generate_page" in str(workflow.nodes)
    assert "finalize" in str(workflow.nodes)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_wiki_workflow.py::test_wiki_workflow_compiles -v
```
Expected: FAIL - function doesn't exist

**Step 3: Implement create_wiki_workflow**

Add to `src/agents/wiki_workflow.py`:

```python
from langgraph.graph import StateGraph, START, END


def create_wiki_workflow():
    """Create and compile the wiki generation workflow.

    The workflow follows a Map-Reduce pattern:
    1. extract_structure: Analyze repo and create wiki structure
    2. fan_out_pages: Create parallel tasks for each page
    3. generate_page: Generate content (runs in parallel)
    4. finalize: Combine results and store to database

    Returns:
        Compiled LangGraph workflow
    """
    builder = StateGraph(WikiWorkflowState)

    # Add nodes
    builder.add_node("extract_structure", extract_structure_node)
    builder.add_node("generate_page", generate_page_node)
    builder.add_node("finalize", finalize_node)

    # Add edges
    builder.add_edge(START, "extract_structure")
    builder.add_conditional_edges(
        "extract_structure",
        fan_out_pages,
        ["generate_page", "finalize"]  # finalize for error case
    )
    builder.add_edge("generate_page", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create singleton workflow instance
wiki_workflow = create_wiki_workflow()
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_wiki_workflow.py::test_wiki_workflow_compiles -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): assemble LangGraph workflow with Map-Reduce pattern

Connects extract_structure -> fan_out -> generate_page -> finalize."
```

---

## Phase 6: Integration

### Task 12: Update WikiGenerationAgent to Use New Workflow

**Purpose:** Replace deep agent calls with new workflow invocation.

**Files:**
- Modify: `src/agents/wiki_agent.py`
- Test: `tests/integration/test_wiki_workflow_integration.py`

**Step 1: Write integration test**

```python
# tests/integration/test_wiki_workflow_integration.py
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

@pytest.mark.integration
@pytest.mark.asyncio
async def test_wiki_agent_uses_new_workflow():
    """WikiGenerationAgent should use new wiki_workflow."""
    from src.agents.wiki_agent import WikiGenerationAgent

    agent = WikiGenerationAgent()

    # Mock the workflow
    with patch("src.agents.wiki_agent.wiki_workflow") as mock_workflow:
        mock_workflow.ainvoke = AsyncMock(return_value={
            "current_step": "completed",
            "structure": MagicMock(),
            "pages": [],
            "error": None,
        })

        result = await agent.generate_wiki(
            repository_id="test-repo",
            clone_path="/tmp/repo",
            file_tree="src/\n  main.py",
            readme_content="# Test",
        )

        mock_workflow.ainvoke.assert_called_once()
        assert result["status"] == "success"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/integration/test_wiki_workflow_integration.py -v
```
Expected: FAIL - method doesn't exist or uses old approach

**Step 3: Update WikiGenerationAgent**

Modify `src/agents/wiki_agent.py` to replace `_generate_wiki_node`:

```python
# Add import at top
from src.agents.wiki_workflow import wiki_workflow, WikiWorkflowState

# Replace _generate_wiki_node method:
async def _generate_wiki_node(self, state: WikiGenerationState) -> dict:
    """Generate wiki using the new workflow.

    Invokes the LangGraph wiki_workflow which handles:
    - Structure extraction
    - Parallel page generation
    - Finalization and storage
    """
    try:
        # Prepare state for new workflow
        workflow_state = WikiWorkflowState(
            repository_id=state["repository_id"],
            clone_path=state.get("clone_path", ""),
            file_tree=state["file_tree"],
            readme_content=state["readme_content"],
            structure=None,
            pages=[],
            error=None,
            current_step="init",
        )

        # Invoke workflow
        result = await wiki_workflow.ainvoke(workflow_state)

        if result.get("error"):
            return {
                "error_message": result["error"],
                "current_step": "error",
            }

        return {
            "wiki_structure": result.get("structure"),
            "current_step": "wiki_generated",
            "progress": 0.9,
        }

    except Exception as e:
        return {
            "error_message": f"Wiki generation failed: {str(e)}",
            "current_step": "error",
        }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/integration/test_wiki_workflow_integration.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/integration/
git commit -m "feat(wiki): integrate new workflow into WikiGenerationAgent

Replaces deep agent with LangGraph Map-Reduce workflow."
```

---

### Task 13: Remove Deep Agent Dependencies

**Purpose:** Clean up imports and dependencies on deprecated code.

**Files:**
- Modify: `src/agents/wiki_agent.py`
- Modify: `pyproject.toml` (if needed)

**Step 1: Search for remaining deep agent imports**

```bash
grep -r "deep_structure_agent\|create_structure_agent\|run_structure_agent" src/
```

**Step 2: Remove any found imports**

Update files to remove deprecated imports.

**Step 3: Run full test suite**

```bash
pytest tests/ -v --ignore=archive/
```
Expected: All PASS

**Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove deep agent dependencies

Cleanup after wiki workflow refactor."
```

---

## Phase 7: Validation

### Task 14: End-to-End Test

**Purpose:** Validate complete workflow with real repository.

**Files:**
- Create: `tests/e2e/test_wiki_generation_e2e.py`

**Step 1: Create E2E test**

```python
# tests/e2e/test_wiki_generation_e2e.py
import pytest
from pathlib import Path

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_wiki_generation_e2e():
    """End-to-end test of wiki generation workflow."""
    from src.agents.wiki_workflow import wiki_workflow, WikiWorkflowState

    # Use a simple test repository (could be fixture)
    test_repo_path = Path(__file__).parent / "fixtures" / "sample_repo"

    if not test_repo_path.exists():
        pytest.skip("Test repository fixture not available")

    file_tree = "src/\n  main.py\n  utils.py\nREADME.md"
    readme = "# Sample Project\nA sample project for testing."

    state = WikiWorkflowState(
        repository_id="e2e-test-repo",
        clone_path=str(test_repo_path),
        file_tree=file_tree,
        readme_content=readme,
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )

    result = await wiki_workflow.ainvoke(state)

    assert result["current_step"] == "completed"
    assert result.get("error") is None
    assert result.get("structure") is not None
    assert len(result.get("pages", [])) > 0
```

**Step 2: Run E2E test (manual)**

```bash
pytest tests/e2e/test_wiki_generation_e2e.py -v -s
```

**Step 3: Commit**

```bash
git add tests/e2e/
git commit -m "test: add e2e test for wiki generation workflow"
```

---

### Task 15: Final Cleanup and Documentation

**Purpose:** Update documentation and clean up temporary code.

**Files:**
- Update: `CLAUDE.md` if needed
- Update: `docs/proposals/` to mark as implemented
- Delete: Temporary test fixtures if any

**Step 1: Update proposal status**

Edit `docs/proposals/2025-01-02-wiki-agent-orchestration-proposal.md`:
- Change status from "Draft for Review" to "Implemented"

**Step 2: Run full test suite**

```bash
pytest tests/ -v
```
Expected: All PASS

**Step 3: Final commit**

```bash
git add -A
git commit -m "docs: mark wiki orchestration proposal as implemented

All phases completed:
- Deep agent archived
- Map-Reduce workflow implemented
- Integration complete
- E2E tests passing"
```

---

## Summary

### Files Created
- `src/agents/wiki_workflow.py` - New LangGraph workflow
- `tests/unit/test_wiki_workflow.py` - Unit tests
- `tests/integration/test_wiki_workflow_integration.py` - Integration tests
- `tests/e2e/test_wiki_generation_e2e.py` - E2E tests

### Files Modified
- `src/models/wiki.py` - Added content field to WikiPageDetail
- `src/agents/wiki_agent.py` - Updated to use new workflow

### Files Archived/Deleted
- `src/agents/deep_structure_agent.py` → `archive/`
- `tests/unit/test_deep_page_agent.py` - Deleted

### Reused Components
- `WikiStructure`, `WikiSection`, `WikiPageDetail` from `src/models/wiki.py`
- `LLMTool.generate_structured` from `src/tools/llm_tool.py`
- Error handling patterns from `wiki_agent.py`
- Prompts from `src/prompts/wiki_prompts.yaml`

### Key Patterns
- LangGraph StateGraph with TypedDict state
- Send API for parallel page generation
- State reducer (operator.add) for collecting pages
- Pydantic structured output for structure extraction
- YAML-based prompt management
