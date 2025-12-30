# Deep Agent Wiki Structure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `_generate_structure_node` with a Deep Agent that autonomously explores the cloned repository to generate accurate wiki structures.

**Architecture:** The Deep Agent uses `FilesystemBackend` to access the cloned repository. It explores using built-in tools (`ls`, `read_file`, `glob`, `grep`) and captures structured output via a custom `finalize_wiki_structure` tool using a closure pattern.

**Tech Stack:** `deepagents>=0.2.0`, `langgraph`, `langchain-openai`

---

## Task 1: Add deepagents Dependency

**Files:**
- Modify: `pyproject.toml:28-56`

**Step 1: Add deepagents to dependencies**

Add `deepagents` to the dependencies list in `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
    "langchain-mcp-adapters>=0.1.0",
    "deepagents>=0.2.0",
]
```

**Step 2: Add deepagents to mypy overrides**

Add to the mypy overrides section:

```toml
[[tool.mypy.overrides]]
module = [
    # ... existing modules ...
    "deepagents.*",
]
ignore_missing_imports = true
```

**Step 3: Install the dependency**

Run: `pip install -e ".[dev]"`
Expected: Successfully installed deepagents

**Step 4: Verify import works**

Run: `python -c "from deepagents import create_deep_agent; print('OK')"`
Expected: `OK`

**Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add deepagents dependency for wiki structure generation"
```

---

## Task 2: Create Deep Agent Structure Generator Module

**Files:**
- Create: `src/agents/deep_structure_agent.py`
- Test: `tests/unit/test_deep_structure_agent.py`

**Step 1: Write failing test for the module structure**

Create `tests/unit/test_deep_structure_agent.py`:

```python
"""Unit tests for Deep Agent structure generator"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestDeepStructureAgent:
    """Tests for DeepStructureAgent"""

    @pytest.mark.asyncio
    async def test_create_structure_agent_returns_agent(self):
        """Test that create_structure_agent returns a configured agent"""
        from src.agents.deep_structure_agent import create_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="├── src/\n│   └── main.py",
                readme_content="# Test Project",
            )

            assert agent is not None
            mock_create.assert_called_once()


    @pytest.mark.asyncio
    async def test_finalize_tool_captures_structure(self):
        """Test that finalize_wiki_structure tool captures the structure"""
        from src.agents.deep_structure_agent import create_finalize_tool

        captured = {}
        tool = create_finalize_tool(captured)

        # Simulate agent calling the tool
        result = tool.invoke({
            "title": "Test Wiki",
            "description": "A test wiki",
            "pages": [{"title": "Overview", "slug": "overview", "section": "intro", "file_paths": [], "description": "Overview page"}]
        })

        assert captured["title"] == "Test Wiki"
        assert captured["description"] == "A test wiki"
        assert len(captured["pages"]) == 1
        assert "success" in result.lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_deep_structure_agent.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'src.agents.deep_structure_agent'"

**Step 3: Create the deep_structure_agent module**

Create `src/agents/deep_structure_agent.py`:

```python
"""Deep Agent for wiki structure generation.

This module provides a Deep Agent that autonomously explores a cloned repository
to generate an accurate wiki structure.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class WikiPageInput(BaseModel):
    """Schema for a wiki page in the structure"""
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    description: str = Field(description="Brief description of page content")


class FinalizeWikiStructureInput(BaseModel):
    """Schema for finalizing the wiki structure"""
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    pages: List[Dict[str, Any]] = Field(
        description="List of wiki pages with title, slug, section, file_paths, description"
    )


def create_finalize_tool(capture_dict: Dict[str, Any]):
    """Create the finalize_wiki_structure tool with closure for capturing output.

    Args:
        capture_dict: Dictionary that will be mutated to store the captured structure

    Returns:
        A langchain tool that captures the wiki structure
    """
    @tool(args_schema=FinalizeWikiStructureInput)
    def finalize_wiki_structure(
        title: str,
        description: str,
        pages: List[Dict[str, Any]]
    ) -> str:
        """Finalize the wiki structure after exploring the repository.

        Call this tool when you have finished analyzing the repository and are
        ready to submit the final wiki structure. Include 8-12 pages covering
        the key aspects of the codebase.
        """
        capture_dict["title"] = title
        capture_dict["description"] = description
        capture_dict["pages"] = pages
        logger.info(
            "Wiki structure finalized",
            title=title,
            page_count=len(pages)
        )
        return f"Wiki structure captured successfully with {len(pages)} pages."

    return finalize_wiki_structure


def get_structure_prompt(
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str
) -> str:
    """Generate the system prompt for the structure agent.

    Args:
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README file content

    Returns:
        Formatted system prompt
    """
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

## Exploration Strategy
1. Start by examining the file tree and README provided above
2. Use `ls` to explore directory contents in detail
3. Use `glob` to find specific file patterns (e.g., `**/*.py`, `**/test_*.py`)
4. Read key files to understand the codebase:
   - Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
   - Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
   - Core modules and their purposes
5. Use `grep` to find patterns like class definitions, API routes, exports
6. Use `write_todos` to track your exploration progress

## Output Requirements
When you have sufficient understanding, call `finalize_wiki_structure` with:
- **title**: A descriptive wiki title for this project
- **description**: A one-paragraph description of the wiki
- **pages**: A list of 8-12 pages, each with:
  - title: Page title
  - slug: URL-friendly identifier (lowercase, hyphens)
  - section: One of "Overview", "Architecture", "Features", "API", "Deployment", "Development"
  - file_paths: List of relevant source files
  - description: What this page covers

Focus on what would help a new developer understand and work with this codebase.
"""


def create_structure_agent(
    clone_path: str,
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    model: Optional[str] = None,
) -> Any:
    """Create a Deep Agent configured for wiki structure generation.

    Args:
        clone_path: Path to the cloned repository
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README content
        model: Optional model override (default: uses deepagents default)

    Returns:
        Configured Deep Agent
    """
    # This will be populated by the finalize tool
    captured_structure: Dict[str, Any] = {}

    # Create the finalize tool with capture closure
    finalize_tool = create_finalize_tool(captured_structure)

    # Create backend pointing to cloned repo
    backend = FilesystemBackend(root_dir=clone_path)

    # Generate system prompt
    system_prompt = get_structure_prompt(owner, repo, file_tree, readme_content)

    # Create the agent
    agent_kwargs = {
        "backend": backend,
        "tools": [finalize_tool],
        "system_prompt": system_prompt,
    }

    if model:
        from langchain_openai import ChatOpenAI
        agent_kwargs["model"] = ChatOpenAI(model=model, temperature=0)

    agent = create_deep_agent(**agent_kwargs)

    # Attach the capture dict to the agent for retrieval
    agent._structure_capture = captured_structure

    return agent


async def run_structure_agent(
    clone_path: str,
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    timeout: float = 300.0,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the structure agent and return the captured wiki structure.

    Args:
        clone_path: Path to the cloned repository
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README content
        timeout: Maximum execution time in seconds
        model: Optional model override

    Returns:
        Captured wiki structure dict or None if failed
    """
    agent = create_structure_agent(
        clone_path=clone_path,
        owner=owner,
        repo=repo,
        file_tree=file_tree,
        readme_content=readme_content,
        model=model,
    )

    try:
        await asyncio.wait_for(
            agent.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": "Analyze this repository and create a comprehensive wiki structure. "
                              "Explore the codebase thoroughly, then call finalize_wiki_structure with your findings."
                }]
            }),
            timeout=timeout
        )

        # Retrieve captured structure
        structure = getattr(agent, "_structure_capture", {})

        if structure and structure.get("pages"):
            return structure

        logger.warning("Agent did not produce wiki structure")
        return None

    except asyncio.TimeoutError:
        logger.error("Structure agent timed out", timeout=timeout)
        return None
    except Exception as e:
        logger.error("Structure agent failed", error=str(e))
        return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_deep_structure_agent.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/agents/deep_structure_agent.py tests/unit/test_deep_structure_agent.py
git commit -m "feat: add deep structure agent module for wiki generation"
```

---

## Task 3: Add More Unit Tests for Edge Cases

**Files:**
- Modify: `tests/unit/test_deep_structure_agent.py`

**Step 1: Add tests for error handling and edge cases**

Append to `tests/unit/test_deep_structure_agent.py`:

```python
    @pytest.mark.asyncio
    async def test_run_structure_agent_timeout(self):
        """Test that run_structure_agent handles timeout gracefully"""
        from src.agents.deep_structure_agent import run_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            # Mock agent that never completes
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_create.return_value = mock_agent

            result = await run_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="",
                readme_content="",
                timeout=0.1,
            )

            assert result is None


    @pytest.mark.asyncio
    async def test_run_structure_agent_empty_output(self):
        """Test handling when agent doesn't call finalize tool"""
        from src.agents.deep_structure_agent import run_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
            mock_agent._structure_capture = {}  # Empty - agent didn't call tool
            mock_create.return_value = mock_agent

            result = await run_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="",
                readme_content="",
            )

            assert result is None


    @pytest.mark.asyncio
    async def test_run_structure_agent_success(self):
        """Test successful structure generation"""
        from src.agents.deep_structure_agent import run_structure_agent

        expected_structure = {
            "title": "Test Project Wiki",
            "description": "Documentation for test project",
            "pages": [
                {"title": "Overview", "slug": "overview", "section": "Overview", "file_paths": ["README.md"], "description": "Project overview"}
            ]
        }

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
            mock_agent._structure_capture = expected_structure
            mock_create.return_value = mock_agent

            result = await run_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="├── README.md",
                readme_content="# Test",
            )

            assert result == expected_structure
            assert result["title"] == "Test Project Wiki"
            assert len(result["pages"]) == 1


    def test_get_structure_prompt_includes_context(self):
        """Test that prompt includes all context"""
        from src.agents.deep_structure_agent import get_structure_prompt

        prompt = get_structure_prompt(
            owner="my-org",
            repo="my-repo",
            file_tree="├── src/",
            readme_content="# My Project"
        )

        assert "my-org" in prompt
        assert "my-repo" in prompt
        assert "├── src/" in prompt
        assert "# My Project" in prompt
        assert "finalize_wiki_structure" in prompt
```

**Step 2: Run all tests**

Run: `pytest tests/unit/test_deep_structure_agent.py -v`
Expected: 6 passed

**Step 3: Commit**

```bash
git add tests/unit/test_deep_structure_agent.py
git commit -m "test: add edge case tests for deep structure agent"
```

---

## Task 4: Integrate Deep Agent into WikiGenerationAgent

**Files:**
- Modify: `src/agents/wiki_agent.py:381-431` (the `_generate_structure_node` method)

**Step 1: Write failing integration test**

Create or append to a test file `tests/unit/test_wiki_agent_deep_agent.py`:

```python
"""Tests for WikiGenerationAgent with Deep Agent integration"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestWikiAgentDeepAgentIntegration:
    """Tests for Deep Agent integration in WikiGenerationAgent"""

    @pytest.mark.asyncio
    async def test_generate_structure_node_uses_deep_agent(self):
        """Test that _generate_structure_node uses the deep agent"""
        from src.agents.wiki_agent import WikiGenerationAgent

        # Mock dependencies
        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()
        mock_wiki_repo = MagicMock()
        mock_repo_repo = MagicMock()
        mock_code_doc_repo = MagicMock()

        # Mock repository with clone_path
        mock_repository = MagicMock()
        mock_repository.clone_path = "/tmp/test-repo"
        mock_repository.org = "test-org"
        mock_repository.name = "test-repo"
        mock_repo_repo.find_one = AsyncMock(return_value=mock_repository)

        agent = WikiGenerationAgent(
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool,
            wiki_structure_repo=mock_wiki_repo,
            repository_repo=mock_repo_repo,
            code_document_repo=mock_code_doc_repo,
        )

        state = {
            "repository_id": str(uuid4()),
            "file_tree": "├── src/",
            "readme_content": "# Test",
            "wiki_structure": None,
            "generated_pages": [],
            "current_page": None,
            "current_step": "starting",
            "error_message": None,
            "progress": 0.0,
            "start_time": "2024-01-01T00:00:00Z",
            "messages": [],
        }

        expected_structure = {
            "title": "Test Wiki",
            "description": "Test description",
            "pages": [{"title": "Overview", "slug": "overview", "section": "Overview", "file_paths": [], "description": "Test"}]
        }

        with patch("src.agents.wiki_agent.run_structure_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = expected_structure

            # Need to also patch Path.exists to return True
            with patch("pathlib.Path.exists", return_value=True):
                result = await agent._generate_structure_node(state)

        assert result["wiki_structure"] == expected_structure
        assert result["error_message"] is None
        assert result["progress"] == 50.0


    @pytest.mark.asyncio
    async def test_generate_structure_node_handles_no_clone_path(self):
        """Test error handling when clone_path is missing"""
        from src.agents.wiki_agent import WikiGenerationAgent

        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()
        mock_wiki_repo = MagicMock()
        mock_repo_repo = MagicMock()
        mock_code_doc_repo = MagicMock()

        # Repository without clone_path
        mock_repository = MagicMock()
        mock_repository.clone_path = None
        mock_repo_repo.find_one = AsyncMock(return_value=mock_repository)

        agent = WikiGenerationAgent(
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool,
            wiki_structure_repo=mock_wiki_repo,
            repository_repo=mock_repo_repo,
            code_document_repo=mock_code_doc_repo,
        )

        state = {
            "repository_id": str(uuid4()),
            "file_tree": "",
            "readme_content": "",
            "wiki_structure": None,
            "generated_pages": [],
            "current_page": None,
            "current_step": "starting",
            "error_message": None,
            "progress": 0.0,
            "start_time": "2024-01-01T00:00:00Z",
            "messages": [],
        }

        result = await agent._generate_structure_node(state)

        assert result["error_message"] is not None
        assert "clone" in result["error_message"].lower() or "path" in result["error_message"].lower()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_agent_deep_agent.py -v`
Expected: FAIL (import error or assertion error depending on current state)

**Step 3: Update the _generate_structure_node method**

Modify `src/agents/wiki_agent.py`. Replace the `_generate_structure_node` method (lines ~381-431):

```python
async def _generate_structure_node(
    self, state: WikiGenerationState
) -> WikiGenerationState:
    """Generate wiki structure using Deep Agent for repository exploration."""
    from pathlib import Path
    from src.agents.deep_structure_agent import run_structure_agent

    try:
        state["current_step"] = "generating_structure"
        state["progress"] = 30.0

        # Fetch repository from database
        repository = await self._repository_repo.find_one(
            {"_id": UUID(state["repository_id"])}
        )
        if not repository:
            state["error_message"] = "Repository not found"
            return state

        # Validate clone_path exists
        if not repository.clone_path:
            state["error_message"] = "Repository not cloned - no local path available"
            return state

        clone_path = Path(repository.clone_path)
        if not clone_path.exists():
            state["error_message"] = f"Clone path does not exist: {clone_path}"
            return state

        owner = repository.org or "unknown"
        repo_name = repository.name or "unknown"

        # Run the Deep Agent to explore and generate structure
        logger.info(
            "Running Deep Agent for wiki structure",
            repository_id=state["repository_id"],
            clone_path=str(clone_path),
        )

        wiki_structure = await run_structure_agent(
            clone_path=str(clone_path),
            owner=owner,
            repo=repo_name,
            file_tree=state["file_tree"],
            readme_content=state["readme_content"],
            timeout=300.0,  # 5 minute timeout
        )

        if not wiki_structure:
            state["error_message"] = "Deep Agent failed to generate wiki structure"
            return state

        if not wiki_structure.get("pages"):
            state["error_message"] = "Deep Agent produced empty wiki structure"
            return state

        state["wiki_structure"] = wiki_structure
        state["progress"] = 50.0

        # Add success message
        state["messages"].append(
            AIMessage(
                content=f"Generated wiki structure with {len(wiki_structure.get('pages', []))} pages using Deep Agent exploration"
            )
        )

        return state

    except Exception as e:
        logger.error(f"Structure generation failed: {e}")
        state["error_message"] = f"Structure generation failed: {str(e)}"
        return state
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_wiki_agent_deep_agent.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add src/agents/wiki_agent.py tests/unit/test_wiki_agent_deep_agent.py
git commit -m "feat: integrate Deep Agent into wiki structure generation"
```

---

## Task 5: Add Integration Test with Sample Repository

**Files:**
- Create: `tests/integration/test_deep_agent_structure.py`
- Create: `tests/fixtures/sample_repo/` (sample repository structure)

**Step 1: Create a minimal sample repository fixture**

Create directory structure:

```bash
mkdir -p tests/fixtures/sample_repo/src
mkdir -p tests/fixtures/sample_repo/docs
```

Create `tests/fixtures/sample_repo/README.md`:

```markdown
# Sample Project

A sample project for testing wiki generation.

## Features

- Feature A: Does something useful
- Feature B: Does something else

## Installation

```bash
pip install sample-project
```

## Usage

```python
from sample import main
main.run()
```
```

Create `tests/fixtures/sample_repo/pyproject.toml`:

```toml
[project]
name = "sample-project"
version = "1.0.0"
description = "A sample project"

[project.dependencies]
requests = ">=2.0"
```

Create `tests/fixtures/sample_repo/src/main.py`:

```python
"""Main module for sample project."""

def run():
    """Run the sample project."""
    print("Running sample project")

def process_data(data: dict) -> dict:
    """Process input data."""
    return {"processed": True, **data}
```

Create `tests/fixtures/sample_repo/src/utils.py`:

```python
"""Utility functions."""

def helper_function(x: int) -> int:
    """A helper function."""
    return x * 2
```

**Step 2: Write integration test**

Create `tests/integration/test_deep_agent_structure.py`:

```python
"""Integration tests for Deep Agent wiki structure generation.

These tests use a real sample repository fixture to verify the agent works.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_repo"


class TestDeepAgentStructureIntegration:
    """Integration tests for Deep Agent structure generation"""

    @pytest.fixture
    def sample_repo_path(self) -> str:
        """Get path to sample repository fixture"""
        assert FIXTURES_DIR.exists(), f"Fixture directory not found: {FIXTURES_DIR}"
        return str(FIXTURES_DIR)

    @pytest.fixture
    def sample_file_tree(self) -> str:
        """Generate file tree for sample repo"""
        return """├── README.md
├── pyproject.toml
├── docs/
└── src/
    ├── main.py
    └── utils.py"""

    @pytest.fixture
    def sample_readme(self, sample_repo_path: str) -> str:
        """Read sample README"""
        readme_path = Path(sample_repo_path) / "README.md"
        if readme_path.exists():
            return readme_path.read_text()
        return "# Sample Project"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_agent_generates_structure_for_sample_repo(
        self,
        sample_repo_path: str,
        sample_file_tree: str,
        sample_readme: str,
    ):
        """Test that Deep Agent generates valid structure for sample repo"""
        from src.agents.deep_structure_agent import run_structure_agent

        # Skip if no API key configured (CI environment)
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        result = await run_structure_agent(
            clone_path=sample_repo_path,
            owner="test-org",
            repo="sample-project",
            file_tree=sample_file_tree,
            readme_content=sample_readme,
            timeout=120.0,  # 2 minute timeout for test
        )

        # Verify structure was generated
        assert result is not None, "Agent should produce a structure"
        assert "title" in result
        assert "description" in result
        assert "pages" in result
        assert len(result["pages"]) >= 1, "Should have at least one page"

        # Verify page structure
        for page in result["pages"]:
            assert "title" in page
            assert "slug" in page
            assert "section" in page
            assert "description" in page

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_deep_agent_with_mocked_backend(
        self,
        sample_repo_path: str,
        sample_file_tree: str,
        sample_readme: str,
    ):
        """Test agent creation with mocked deep agent (no API call)"""
        from src.agents.deep_structure_agent import create_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_structure_agent(
                clone_path=sample_repo_path,
                owner="test-org",
                repo="sample-project",
                file_tree=sample_file_tree,
                readme_content=sample_readme,
            )

            # Verify agent was created with correct backend
            call_kwargs = mock_create.call_args.kwargs
            assert "backend" in call_kwargs
            assert "tools" in call_kwargs
            assert "system_prompt" in call_kwargs
            assert len(call_kwargs["tools"]) == 1  # finalize tool
```

**Step 3: Run integration tests**

Run: `pytest tests/integration/test_deep_agent_structure.py -v -m "not slow"`
Expected: 1 passed (the mocked test)

Run: `pytest tests/integration/test_deep_agent_structure.py -v -m slow` (only if API key available)
Expected: 1 passed or skipped

**Step 4: Commit**

```bash
git add tests/fixtures/sample_repo/ tests/integration/test_deep_agent_structure.py
git commit -m "test: add integration tests for Deep Agent structure generation"
```

---

## Task 6: Update Existing Tests

**Files:**
- Modify: `tests/unit/test_services.py` (if wiki agent tests exist)

**Step 1: Check for existing wiki agent tests and update mocks**

Run: `grep -r "WikiGenerationAgent" tests/`

If tests exist that mock `_generate_structured_wiki_structure`, update them to mock `run_structure_agent` instead.

**Step 2: Run full test suite**

Run: `pytest tests/unit/ -v --ignore=tests/unit/test_deep_structure_agent.py --ignore=tests/unit/test_wiki_agent_deep_agent.py`
Expected: All existing tests pass

**Step 3: Run new tests**

Run: `pytest tests/unit/test_deep_structure_agent.py tests/unit/test_wiki_agent_deep_agent.py -v`
Expected: All new tests pass

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "test: update existing tests for Deep Agent integration"
```

---

## Task 7: Clean Up Legacy Code

**Files:**
- Modify: `src/agents/wiki_agent.py`

**Step 1: Remove unused method _generate_structured_wiki_structure**

The `_generate_structured_wiki_structure` method (lines ~595-626) is no longer used. Remove it.

**Step 2: Verify no references remain**

Run: `grep -r "_generate_structured_wiki_structure" src/`
Expected: No results

**Step 3: Run tests to verify nothing broke**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "refactor: remove legacy _generate_structured_wiki_structure method"
```

---

## Task 8: Final Verification

**Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Run type checking**

Run: `python -m mypy src/agents/deep_structure_agent.py src/agents/wiki_agent.py`
Expected: No errors (or only pre-existing ones)

**Step 3: Run linting**

Run: `python -m black src/agents/deep_structure_agent.py src/agents/wiki_agent.py`
Run: `python -m isort src/agents/deep_structure_agent.py src/agents/wiki_agent.py`

**Step 4: Final commit**

```bash
git add -u
git commit -m "style: format deep agent code"
```

---

## Summary

After completing all tasks, you will have:

1. ✅ Added `deepagents` dependency
2. ✅ Created `deep_structure_agent.py` module with:
   - `create_finalize_tool()` - closure-based output capture
   - `get_structure_prompt()` - prompt generation
   - `create_structure_agent()` - agent factory
   - `run_structure_agent()` - async runner with timeout
3. ✅ Integrated Deep Agent into `WikiGenerationAgent._generate_structure_node`
4. ✅ Added comprehensive unit tests
5. ✅ Added integration tests with sample repo fixture
6. ✅ Cleaned up legacy code

The wiki structure generation now uses an autonomous Deep Agent that can explore the actual repository files to produce more accurate wiki structures.
