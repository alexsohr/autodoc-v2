# Deep Page Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace simple page_worker_node with a deep agent that autonomously explores codebase using MCP filesystem tools.

**Architecture:** Mirror deep_structure_agent.py pattern exactly - create_page_agent() + run_page_agent() functions with MCP tools and finalize tool for structured output capture.

**Tech Stack:** deepagents library, LangGraph, MCP filesystem tools, Pydantic models

---

## Task 1: Create PageContent Output Models

**Files:**
- Create: `src/agents/deep_page_agent.py`

**Step 1: Create the file with Pydantic models**

```python
"""Deep agent for generating wiki page content with autonomous exploration."""

import asyncio
import os
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class PageSection(BaseModel):
    """A section within the wiki page."""
    heading: str = Field(description="Section heading (H2 or H3)")
    content: str = Field(description="Markdown content for this section")


class PageContent(BaseModel):
    """Structured output for a wiki page."""
    title: str = Field(description="Page title")
    content: str = Field(description="Full markdown content of the page")
    source_files: List[str] = Field(
        default_factory=list,
        description="Source files used to generate this page (minimum 5)"
    )
```

**Step 2: Verify file created**

Run: `python -c "from src.agents.deep_page_agent import PageContent, PageSection; print('Models OK')"`
Expected: `Models OK`

**Step 3: Commit**

```bash
git add src/agents/deep_page_agent.py
git commit -m "feat(deep_page_agent): add PageContent and PageSection models"
```

---

## Task 2: Create Page Finalize Tool

**Files:**
- Modify: `src/agents/deep_page_agent.py`

**Step 1: Add finalize tool input model and creation function**

Add after PageContent class:

```python
class FinalizePageInput(BaseModel):
    """Input schema for the finalize_page tool."""
    title: str = Field(description="Page title")
    content: str = Field(description="Full markdown content including details block, headings, diagrams, tables, and citations")
    source_files: List[str] = Field(description="List of source files used (minimum 5)")


def create_page_finalize_tool(captured_content: Dict[str, Any]) -> StructuredTool:
    """Create finalize tool that captures page content.

    Args:
        captured_content: Dict that will be populated with the finalized content

    Returns:
        StructuredTool for the agent to call when done
    """
    def finalize_page(title: str, content: str, source_files: List[str]) -> str:
        """Submit the final page content. Call this when documentation is complete."""
        if len(source_files) < 5:
            return f"Error: Must cite at least 5 source files. You provided {len(source_files)}. Explore more files and try again."

        captured_content["title"] = title
        captured_content["content"] = content
        captured_content["source_files"] = source_files

        logger.info(
            "Page content finalized",
            title=title,
            content_length=len(content),
            num_source_files=len(source_files)
        )
        return "Page content finalized successfully."

    return StructuredTool.from_function(
        func=finalize_page,
        name="finalize_page",
        description="Submit the final wiki page content. Call this when you have completed the documentation with at least 5 source files cited.",
        args_schema=FinalizePageInput,
    )
```

**Step 2: Verify tool creation works**

Run: `python -c "from src.agents.deep_page_agent import create_page_finalize_tool; t = create_page_finalize_tool({}); print(f'Tool: {t.name}')"`
Expected: `Tool: finalize_page`

**Step 3: Commit**

```bash
git add src/agents/deep_page_agent.py
git commit -m "feat(deep_page_agent): add create_page_finalize_tool function"
```

---

## Task 3: Create Page System Prompt Function

**Files:**
- Modify: `src/agents/deep_page_agent.py`

**Step 1: Add get_page_prompt function**

Add after create_page_finalize_tool:

```python
def get_page_prompt(
    page_title: str,
    page_description: str,
    file_hints: List[str],
    clone_path: str,
    repo_name: str,
    repo_description: str,
    use_mcp_tools: bool = True,
) -> str:
    """Generate the system prompt for the page agent.

    Args:
        page_title: Title of the wiki page to generate
        page_description: Description of what the page should cover
        file_hints: Initial file paths that are likely relevant
        clone_path: Path to the cloned repository
        repo_name: Name of the repository
        repo_description: Description of the repository
        use_mcp_tools: Whether MCP filesystem tools are available

    Returns:
        System prompt string
    """
    file_hints_str = "\n".join(f"- {f}" for f in file_hints) if file_hints else "- No specific files provided, explore to find relevant ones"

    tool_instructions = ""
    if use_mcp_tools:
        tool_instructions = f"""
## Available Tools
- `read_text_file(path, head=N)`: Read file contents. Use head=50 to read first 50 lines efficiently.
- `search_files(path, pattern)`: Search for files matching a pattern.
- `list_directory(path)`: List directory contents.
- `directory_tree(path)`: Get directory tree structure.

All paths should be absolute, starting with: {clone_path}

Example: read_text_file(path="{clone_path}/src/main.py", head=50)
"""

    return f'''You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about a specific feature, system, or module within a given software project.

## Repository Context
- **Repository:** {repo_name}
- **Description:** {repo_description}
- **Clone Path:** {clone_path}

## Your Assignment
- **Page Title:** {page_title}
- **Page Description:** {page_description}

## Starting Files (hints - explore outward from here)
{file_hints_str}
{tool_instructions}
## Output Requirements

CRITICAL STARTING INSTRUCTION:
The very first thing in your content MUST be a `<details>` block listing ALL source files you used. There MUST be AT LEAST 5 source files listed.

Format it exactly like this:
<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- file1.py
- file2.py
- (at least 5 files)
</details>

Immediately after the `<details>` block, the main title should be an H1 heading: `# {page_title}`.

## Content Structure

1. **Introduction:** 1-2 paragraphs explaining purpose and scope.

2. **Detailed Sections:** Use H2 (`##`) and H3 (`###`) headings. For each section:
   - Explain architecture, components, data flow
   - Identify key functions, classes, API endpoints

3. **Mermaid Diagrams:** EXTENSIVELY use diagrams:
   - Use "graph TD" (top-down) - NEVER "graph LR"
   - Maximum node width: 3-4 words
   - No parentheses or slashes in node text
   - For sequence diagrams: define ALL participants first

4. **Tables:** Summarize structured information (API params, config options, data fields)

5. **Code Snippets:** Include relevant code from source files with language identifiers.

6. **Source Citations (CRITICAL):**
   - For EVERY piece of information, cite the source file and line numbers
   - Format: `Sources: [filename.ext:start_line-end_line]()`
   - Multiple files: `Sources: [file1.ext:1-10](), [file2.ext:5]()`
   - You MUST cite AT LEAST 5 different source files throughout

7. **Technical Accuracy:** All information must come SOLELY from source files. Do not invent or infer.

## When Done

Call `finalize_page` with:
- title: The page title
- content: Full markdown content (including details block)
- source_files: List of all source files used (minimum 5)
'''
```

**Step 2: Verify prompt generation**

Run: `python -c "from src.agents.deep_page_agent import get_page_prompt; p = get_page_prompt('Test', 'Desc', ['a.py'], '/tmp', 'repo', 'desc'); print(len(p), 'chars')"`
Expected: Shows character count (should be ~2500+)

**Step 3: Commit**

```bash
git add src/agents/deep_page_agent.py
git commit -m "feat(deep_page_agent): add get_page_prompt function"
```

---

## Task 4: Create Page Agent Factory Function

**Files:**
- Modify: `src/agents/deep_page_agent.py`

**Step 1: Add create_page_agent function**

Add after get_page_prompt. Mirror create_structure_agent exactly:

```python
def create_page_agent(
    clone_path: str,
    page_title: str,
    page_description: str,
    file_hints: List[str],
    repo_name: str,
    repo_description: str,
    model: Optional[str] = None,
    mcp_tools: Optional[List[Any]] = None,
) -> Any:
    """Create a Deep Agent configured for wiki page generation.

    Args:
        clone_path: Path to the cloned repository
        page_title: Title of the page to generate
        page_description: Description of page content
        file_hints: Initial file paths as starting points
        repo_name: Repository name
        repo_description: Repository description
        model: Optional model override (default: uses deepagents default)
        mcp_tools: Optional MCP filesystem tools

    Returns:
        Configured Deep Agent
    """
    from deepagents import FilesystemBackend, create_deep_agent

    # This will be populated by the finalize tool
    captured_content: Dict[str, Any] = {}

    # Create the finalize tool with capture closure
    finalize_tool = create_page_finalize_tool(captured_content)

    # Generate system prompt
    system_prompt = get_page_prompt(
        page_title=page_title,
        page_description=page_description,
        file_hints=file_hints,
        clone_path=clone_path,
        repo_name=repo_name,
        repo_description=repo_description,
        use_mcp_tools=mcp_tools is not None,
    )

    # Build agent kwargs
    agent_kwargs = {
        "system_prompt": system_prompt,
    }

    # Use MCP tools if provided, otherwise fall back to FilesystemBackend
    if mcp_tools:
        logger.info(f"Using MCP filesystem tools ({len(mcp_tools)} tools) for page generation")
        agent_kwargs["tools"] = list(mcp_tools) + [finalize_tool]
    else:
        logger.warning(
            "MCP filesystem not available, using FilesystemBackend. "
            "Note: Windows absolute paths may not work correctly."
        )
        backend = FilesystemBackend(root_dir=clone_path)
        agent_kwargs["backend"] = backend
        agent_kwargs["tools"] = [finalize_tool]

    if model:
        from langchain.chat_models import init_chat_model

        # Add openai: prefix if no provider specified
        if ":" not in model:
            model_string = f"openai:{model}"
            provider = "openai"
        else:
            model_string = model
            provider = model.split(":")[0]

        model_kwargs = {"temperature": 0}

        # Pass API key from environment
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            model_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        elif provider == "google" and os.environ.get("GOOGLE_API_KEY"):
            model_kwargs["api_key"] = os.environ["GOOGLE_API_KEY"]
        elif provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            model_kwargs["api_key"] = os.environ["ANTHROPIC_API_KEY"]

        logger.info(f"Initializing page agent with model: {model_string}")
        agent_kwargs["model"] = init_chat_model(model_string, **model_kwargs)

    agent = create_deep_agent(**agent_kwargs)

    # Attach the capture dict to the agent for retrieval
    agent._page_capture = captured_content

    return agent
```

**Step 2: Verify agent creation (import check only)**

Run: `python -c "from src.agents.deep_page_agent import create_page_agent; print('create_page_agent OK')"`
Expected: `create_page_agent OK`

**Step 3: Commit**

```bash
git add src/agents/deep_page_agent.py
git commit -m "feat(deep_page_agent): add create_page_agent factory function"
```

---

## Task 5: Create Run Page Agent Function

**Files:**
- Modify: `src/agents/deep_page_agent.py`

**Step 1: Add run_page_agent function**

Add after create_page_agent. Mirror run_structure_agent:

```python
async def run_page_agent(
    clone_path: str,
    page_title: str,
    page_description: str,
    file_hints: List[str],
    repo_name: str,
    repo_description: str,
    timeout: float = 120.0,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the page agent and return the captured page content.

    Args:
        clone_path: Path to the cloned repository
        page_title: Title of the page to generate
        page_description: Description of page content
        file_hints: Initial file paths as starting points
        repo_name: Repository name
        repo_description: Repository description
        timeout: Maximum execution time in seconds (default 120s for pages)
        model: Optional model override

    Returns:
        Captured page content dict or None if failed
    """
    from langsmith import traceable

    from src.services.mcp_filesystem_client import get_mcp_filesystem_client

    # Try to get MCP filesystem tools
    mcp_tools = None
    try:
        mcp_client = get_mcp_filesystem_client()
        if mcp_client.is_initialized:
            mcp_tools = list(mcp_client._tools.values())
            logger.info(f"Retrieved {len(mcp_tools)} MCP filesystem tools for page agent")
        else:
            logger.warning("MCP filesystem client not initialized for page agent")
    except Exception as e:
        logger.warning(f"Could not get MCP filesystem tools for page agent: {e}")

    agent = create_page_agent(
        clone_path=clone_path,
        page_title=page_title,
        page_description=page_description,
        file_hints=file_hints,
        repo_name=repo_name,
        repo_description=repo_description,
        model=model,
        mcp_tools=mcp_tools,
    )

    # Build the user message
    user_message = (
        f"Generate comprehensive wiki documentation for: {page_title}\n\n"
        f"Start by exploring the hinted files, then search for related code. "
        f"You MUST use at least 5 source files and cite them with line numbers. "
        f"When done, call finalize_page with your complete documentation."
    )

    # Wrap with traceable for LangSmith visibility
    safe_title = page_title.replace(" ", "_").replace("/", "_")[:30]

    @traceable(name=f"page_agent_{safe_title}", run_type="chain")
    async def _invoke_agent():
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"run_name": f"page_agent_{safe_title}"}
        )

    try:
        logger.info(
            "Starting page agent",
            page_title=page_title,
            timeout=timeout,
            model=model,
            has_mcp_tools=mcp_tools is not None,
            num_file_hints=len(file_hints),
        )

        result = await asyncio.wait_for(_invoke_agent(), timeout=timeout)

        logger.info(f"Page agent completed, result keys: {list(result.keys()) if result else 'None'}")

        # Retrieve captured content
        content = getattr(agent, "_page_capture", {})

        if content and content.get("content"):
            logger.info(
                "Page content generated",
                page_title=page_title,
                content_length=len(content.get("content", "")),
                num_source_files=len(content.get("source_files", [])),
            )
            return content

        logger.warning(f"Page agent did not produce content, captured: {content}")
        return None

    except asyncio.TimeoutError:
        logger.error(f"Page agent timed out after {timeout}s", page_title=page_title)
        return None
    except Exception as e:
        logger.exception(f"Page agent failed: {e}", page_title=page_title)
        return None
```

**Step 2: Verify function imports correctly**

Run: `python -c "from src.agents.deep_page_agent import run_page_agent; print('run_page_agent OK')"`
Expected: `run_page_agent OK`

**Step 3: Commit**

```bash
git add src/agents/deep_page_agent.py
git commit -m "feat(deep_page_agent): add run_page_agent function"
```

---

## Task 6: Update page_worker_node to Use Deep Agent

**Files:**
- Modify: `src/agents/wiki_agent.py:111-217`

**Step 1: Replace page_worker_node implementation**

Replace the entire function body (lines 111-217):

```python
async def page_worker_node(state: PageWorkerState) -> Dict[str, Any]:
    """Generate content for a single wiki page using deep agent exploration.

    This node runs in parallel for each page via LangGraph's Send API.
    It uses a deep agent with MCP filesystem tools to explore and document.

    IMPORTANT: Returns {"generated_pages": [page_result]} which gets
    aggregated with other workers via the operator.add reducer.

    Args:
        state: PageWorkerState with page_info, clone_path, and repository context

    Returns:
        Dict with 'generated_pages' list containing the page with generated content
    """
    from .deep_page_agent import run_page_agent
    from ..utils.config_loader import get_settings

    page_info = state["page_info"]
    clone_path = state.get("clone_path")

    # Get repository context from state
    repo_name = state.get("repo_name", "unknown")
    repo_description = state.get("repo_description", "")

    # Validate page_info has required fields
    required_keys = ["title", "section", "description", "slug"]
    missing_keys = [k for k in required_keys if k not in page_info]
    if missing_keys:
        logger.warning(
            "page_info missing required fields",
            page_title=page_info.get("title", "UNKNOWN"),
            missing_keys=missing_keys
        )
        return {"generated_pages": []}

    if not clone_path:
        logger.error("No clone_path provided for page worker", page_title=page_info["title"])
        return {"generated_pages": []}

    logger.info(
        "Page worker starting with deep agent",
        page_title=page_info["title"],
        num_file_hints=len(page_info.get("file_paths", []))
    )

    try:
        settings = get_settings()

        # Run deep page agent
        result = await run_page_agent(
            clone_path=clone_path,
            page_title=page_info["title"],
            page_description=page_info.get("description", ""),
            file_hints=page_info.get("file_paths", []),
            repo_name=repo_name,
            repo_description=repo_description,
            timeout=120.0,  # 2 minutes per page
            model=settings.openai_model,
        )

        if result and result.get("content"):
            page_result = {
                "title": page_info["title"],
                "slug": page_info["slug"],
                "section": page_info["section"],
                "description": page_info["description"],
                "file_paths": result.get("source_files", page_info.get("file_paths", [])),
                "content": result["content"],
            }

            logger.info(
                "Deep agent generated content for page",
                page_title=page_info["title"],
                content_chars=len(result["content"]),
                source_files=len(result.get("source_files", []))
            )

            return {"generated_pages": [page_result]}
        else:
            logger.warning(
                "Deep agent returned no content for page",
                page_title=page_info["title"]
            )
            return {"generated_pages": []}

    except Exception as e:
        logger.error("Failed to generate page with deep agent", page_title=page_info["title"], error=str(e))
        return {"generated_pages": []}
```

**Step 2: Update PageWorkerState TypedDict if needed**

Check if PageWorkerState needs repo_name and repo_description. If not present, add them:

```python
class PageWorkerState(TypedDict):
    page_info: Dict[str, Any]
    clone_path: Optional[str]
    repo_name: str  # Add if missing
    repo_description: str  # Add if missing
    generated_pages: Annotated[List[Dict[str, Any]], operator.add]
```

**Step 3: Verify imports work**

Run: `python -c "from src.agents.wiki_agent import page_worker_node; print('page_worker_node OK')"`
Expected: `page_worker_node OK`

**Step 4: Commit**

```bash
git add src/agents/wiki_agent.py src/agents/deep_page_agent.py
git commit -m "feat(wiki_agent): integrate deep page agent into page_worker_node"
```

---

## Task 7: Update distribute_pages_node to Pass Repository Context

**Files:**
- Modify: `src/agents/wiki_agent.py` (distribute_pages_node function)

**Step 1: Find and update distribute_pages_node**

The distribute_pages_node needs to pass repo_name and repo_description to each page worker. Find where it creates Send() calls and add the context:

```python
# In distribute_pages_node, update the Send payload:
Send(
    "page_worker",
    {
        "page_info": page,
        "clone_path": state.get("clone_path"),
        "repo_name": state.get("repo_name", state.get("owner", "") + "/" + state.get("repo", "")),
        "repo_description": state.get("repo_description", ""),
        "generated_pages": [],
    }
)
```

**Step 2: Verify the workflow still compiles**

Run: `python -c "from src.agents.wiki_agent import create_wiki_graph; g = create_wiki_graph(); print('Graph OK')"`
Expected: `Graph OK`

**Step 3: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "feat(wiki_agent): pass repository context to page workers"
```

---

## Task 8: Add Unit Tests for Deep Page Agent

**Files:**
- Create: `tests/unit/test_deep_page_agent.py`

**Step 1: Create test file**

```python
"""Unit tests for the deep page agent."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.deep_page_agent import (
    PageContent,
    PageSection,
    create_page_finalize_tool,
    get_page_prompt,
)


class TestPageModels:
    """Test Pydantic models."""

    def test_page_section_model(self):
        section = PageSection(
            heading="Overview",
            content="This is the overview."
        )
        assert section.heading == "Overview"
        assert section.content == "This is the overview."

    def test_page_content_model(self):
        content = PageContent(
            title="Test Page",
            content="# Test\n\nContent here",
            source_files=["a.py", "b.py", "c.py", "d.py", "e.py"]
        )
        assert content.title == "Test Page"
        assert len(content.source_files) == 5


class TestFinalizePageTool:
    """Test the finalize_page tool."""

    def test_finalize_captures_content(self):
        captured = {}
        tool = create_page_finalize_tool(captured)

        result = tool.func(
            title="Test",
            content="# Test\n\nContent",
            source_files=["a.py", "b.py", "c.py", "d.py", "e.py"]
        )

        assert "finalized successfully" in result
        assert captured["title"] == "Test"
        assert captured["content"] == "# Test\n\nContent"
        assert len(captured["source_files"]) == 5

    def test_finalize_rejects_insufficient_sources(self):
        captured = {}
        tool = create_page_finalize_tool(captured)

        result = tool.func(
            title="Test",
            content="# Test",
            source_files=["a.py", "b.py"]  # Only 2, need 5
        )

        assert "Error" in result
        assert "at least 5" in result
        assert "title" not in captured  # Should not be captured


class TestGetPagePrompt:
    """Test prompt generation."""

    def test_prompt_includes_page_info(self):
        prompt = get_page_prompt(
            page_title="Authentication System",
            page_description="How auth works",
            file_hints=["src/auth.py", "src/login.py"],
            clone_path="/tmp/repo",
            repo_name="myrepo",
            repo_description="A test repo",
            use_mcp_tools=True,
        )

        assert "Authentication System" in prompt
        assert "How auth works" in prompt
        assert "src/auth.py" in prompt
        assert "/tmp/repo" in prompt
        assert "myrepo" in prompt
        assert "read_text_file" in prompt  # MCP tools

    def test_prompt_without_mcp_tools(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp/repo",
            repo_name="repo",
            repo_description="desc",
            use_mcp_tools=False,
        )

        assert "read_text_file" not in prompt
        assert "Test" in prompt

    def test_prompt_requires_5_sources(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        assert "AT LEAST 5" in prompt or "at least 5" in prompt.lower()
```

**Step 2: Run tests**

Run: `pytest tests/unit/test_deep_page_agent.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add tests/unit/test_deep_page_agent.py
git commit -m "test(deep_page_agent): add unit tests for models, finalize tool, and prompt"
```

---

## Task 9: Run Full Test Suite

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `pytest tests/ -v --ignore=tests/performance --ignore=tests/security -x`
Expected: All tests pass (or identify any failures to fix)

**Step 2: If failures, fix and re-run**

Address any import errors or integration issues discovered.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: address test failures from deep page agent integration"
```

---

## Task 10: Manual Integration Test

**Files:**
- None (manual verification)

**Step 1: Start development server**

Run: `.\scripts\dev-run.ps1`
Expected: Server starts successfully with MCP filesystem client initialized

**Step 2: Trigger wiki generation on a small repo**

Use the API or existing test to trigger wiki generation. Observe:
- Page agents start with "Starting page agent" log
- MCP filesystem tools are used
- Each page logs "Page content generated" with source file count
- Generated content includes `<details>` block with source files

**Step 3: Document results**

Note any issues for follow-up.
