# Wiki Workflow React Agent Refactoring Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace plain LLM calls with React agents that have MCP filesystem tool access for structure extraction and page generation.

**Architecture:** Sequential LangGraph workflow where each node invokes a React agent (via `create_react_agent`) with bound MCP filesystem tools. The agents can explore the codebase to generate accurate, source-grounded documentation.

**Tech Stack:** LangGraph `create_react_agent`, `langchain-mcp-adapters`, `MCPFilesystemClient`, Pydantic structured output

---

## Problem Statement

### Current Implementation Issues (from trace 019b8095-2e6f-7561-8691-084bda0eb3a4)

1. **No Tool Access**: `extract_structure_node` uses `LLMTool.generate_structured()` - plain LLM with no filesystem access
2. **No Tool Access**: `generate_pages_node` uses `LLMTool.generate_text()` - plain LLM with no filesystem access
3. **Prompts Expect Tools**: The prompts reference `finalize_page`, `read_text_file`, `search_files` tools that aren't bound
4. **Limited Context**: LLM only sees file_tree and readme, can't explore actual source code

### Desired Architecture

**Key Distinction: Reading Strategy**

| Node | Reading Strategy | Purpose |
|------|-----------------|---------|
| `extract_structure_node` | **Context-efficient** (headers only, 50-100 lines) | Understand file purpose, assign to pages |
| `generate_pages_node` | **Full file reads** (complete content) | Generate detailed documentation with citations |

```
START
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ extract_structure_node                                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ React Agent + MCP Filesystem Tools                      │ │
│ │                                                         │ │
│ │ CONTEXT-EFFICIENT READING:                              │ │
│ │ - Use read_file with head=50 or head=100                │ │
│ │ - Focus on imports, class/function signatures, docstrings│
│ │ - Understand structure, NOT implementation details      │ │
│ │ - Goal: Determine which pages each file belongs to      │ │
│ │                                                         │ │
│ │ Output: WikiStructure with file_paths assigned per page │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────────┐
│ generate_pages_node (SEQUENTIAL LOOP)                        │
│ FOR each page in structure:                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ React Agent + MCP Filesystem Tools                      │ │
│ │                                                         │ │
│ │ FULL FILE READING:                                      │ │
│ │ - Read COMPLETE content of all file_paths for this page │ │
│ │ - Understand implementation details, logic, data flow   │ │
│ │ - Extract code snippets for documentation               │ │
│ │ - Generate accurate source citations with line numbers  │ │
│ │                                                         │ │
│ │ Output: Full markdown documentation with citations      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ aggregate_node                           │
│ - Collects all generated page content   │
│ - Merges content back into WikiPageDetail│
│ - Validates completeness                │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│ finalize_node                            │
│ - Creates final WikiStructure document  │
│ - Saves to MongoDB via repository       │
└─────────────────────────────────────────┘
  │
  ▼
 END
```

---

## Research Summary (from Context7)

### Creating React Agents with MCP Tools

```python
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Get MCP tools
client = MultiServerMCPClient({...})
tools = await client.get_tools()

# Create React agent with tools
agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=tools,
    prompt="System prompt here"
)

# Invoke agent
result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
```

### Structured Output from React Agent

```python
from pydantic import BaseModel

class WikiStructure(BaseModel):
    title: str
    sections: list[Section]

agent = create_react_agent(
    model="anthropic:claude-3-7-sonnet-latest",
    tools=tools,
    response_format=WikiStructure  # Forces structured output
)
```

### Custom State with React Agent

```python
from langgraph.prebuilt.chat_agent_executor import AgentState

class CustomState(AgentState):
    clone_path: str
    file_tree: str

def prompt(state: CustomState) -> list[AnyMessage]:
    return [{"role": "system", "content": f"Working in {state['clone_path']}"}] + state["messages"]

agent = create_react_agent(
    model=model,
    tools=tools,
    state_schema=CustomState,
    prompt=prompt
)
```

---

## Implementation Tasks

### Task 1: Create React Agent Factory Module

**Files:**
- Create: `src/agents/wiki_react_agents.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_react_agents.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.fixture
def mock_mcp_tools():
    """Mock MCP filesystem tools."""
    return [
        MagicMock(name="read_file"),
        MagicMock(name="list_directory"),
        MagicMock(name="search_files"),
    ]

@pytest.mark.asyncio
async def test_create_structure_agent_returns_compiled_graph(mock_mcp_tools):
    """Structure agent should be a compiled LangGraph."""
    from src.agents.wiki_react_agents import create_structure_agent

    with patch("src.agents.wiki_react_agents.get_mcp_tools", return_value=mock_mcp_tools):
        agent = await create_structure_agent()

    assert agent is not None
    assert hasattr(agent, "ainvoke")

@pytest.mark.asyncio
async def test_create_page_agent_returns_compiled_graph(mock_mcp_tools):
    """Page agent should be a compiled LangGraph."""
    from src.agents.wiki_react_agents import create_page_agent

    with patch("src.agents.wiki_react_agents.get_mcp_tools", return_value=mock_mcp_tools):
        agent = await create_page_agent()

    assert agent is not None
    assert hasattr(agent, "ainvoke")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_react_agents.py -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# src/agents/wiki_react_agents.py
"""
React agent factories for wiki generation.

Creates React agents with MCP filesystem tools for:
- Structure extraction (explores codebase, designs wiki structure)
- Page generation (reads source files, generates documentation)
"""

from typing import List, Any, Optional
from pathlib import Path

import structlog
import yaml
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from src.services.mcp_filesystem_client import MCPFilesystemClient
from src.tools.llm_tool import LLMTool
from src.utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


def _load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent.parent / "prompts" / "wiki_prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPTS = _load_prompts()


async def get_mcp_tools() -> List[Any]:
    """Get MCP filesystem tools.

    Returns:
        List of LangChain-compatible tools from MCP client.
    """
    mcp_client = MCPFilesystemClient.get_instance()
    if not mcp_client.is_initialized:
        await mcp_client.initialize()
    return mcp_client.get_tools()


def get_llm() -> BaseChatModel:
    """Get configured LLM for agents."""
    llm_tool = LLMTool()
    return llm_tool._get_model()


# Schema for structured structure extraction output
class LLMPageSchema(BaseModel):
    """Schema for LLM to generate page details."""
    id: str = Field(description="URL-friendly page identifier (lowercase, hyphens)")
    title: str = Field(description="Page title")
    description: str = Field(description="Brief description of what this page covers")
    importance: str = Field(description="Page importance: 'high', 'medium', or 'low'", default="medium")
    file_paths: List[str] = Field(default_factory=list, description="Relevant source file paths")


class LLMSectionSchema(BaseModel):
    """Schema for LLM to generate section details."""
    id: str = Field(description="URL-friendly section identifier")
    title: str = Field(description="Section title")
    description: str = Field(description="Brief section description")
    order: int = Field(description="Display order (1-based)")
    pages: List[LLMPageSchema] = Field(default_factory=list, description="Pages in this section")


class LLMWikiStructureSchema(BaseModel):
    """Schema for complete wiki structure from LLM."""
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    sections: List[LLMSectionSchema] = Field(default_factory=list, description="Wiki sections")


async def create_structure_agent(
    clone_path: str = "",
    file_tree: str = "",
    readme_content: str = "",
) -> Any:
    """Create a React agent for wiki structure extraction.

    The agent can explore the codebase using MCP filesystem tools
    and returns a structured WikiStructure.

    Args:
        clone_path: Path to cloned repository
        file_tree: Pre-computed file tree string
        readme_content: README content

    Returns:
        Compiled React agent graph
    """
    tools = await get_mcp_tools()
    llm = get_llm()

    # Build system prompt with context
    system_prompt = PROMPTS["structure_agent"]["system_prompt"].format(
        owner="",  # Will be extracted from clone_path
        repo="",
        file_tree=file_tree,
        readme_content=readme_content,
        clone_path=clone_path,
        exploration_instructions=PROMPTS.get("structure_agent", {}).get("exploration_instructions_mcp", ""),
    )

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        response_format=LLMWikiStructureSchema,
    )

    return agent


async def create_page_agent(
    clone_path: str = "",
) -> Any:
    """Create a React agent for page content generation.

    The agent has FULL MCP filesystem tool access to read complete
    source files and generate detailed documentation with citations.

    Unlike the structure agent (context-efficient reading), this agent
    should read ENTIRE files to understand implementation details.

    Args:
        clone_path: Path to cloned repository

    Returns:
        Compiled React agent graph
    """
    tools = await get_mcp_tools()
    llm = get_llm()

    # Use the page_generation_react prompt which emphasizes full file reading
    system_prompt = PROMPTS.get("page_generation_react", {}).get("system_prompt", "").format(
        clone_path=clone_path,
        page_title="{page_title}",  # Will be filled in user message
    )

    # Fallback if prompt not found
    if not system_prompt:
        system_prompt = f"""You are an expert technical writer generating wiki documentation.

## Available Filesystem Tools

You have FULL access to filesystem tools:
- `read_file(path)`: Read COMPLETE file content (no head limit!)
- `list_directory(path)`: List directory contents
- `search_files(path, pattern)`: Search for files

All paths must be absolute, starting with: {clone_path}

## CRITICAL: Full File Reading Strategy

For documentation, you MUST read files COMPLETELY:

**DO:**
- Read the ENTIRE content of files assigned to this page
- Understand implementation details, data flow, logic
- Extract actual code snippets for examples
- Note specific line numbers for citations
- Read related files if needed for full context

**DON'T:**
- Use head parameter (you need full content)
- Skip reading files
- Guess at implementation details
- Cite lines without reading them

## Documentation Requirements

Generate comprehensive markdown including:
- Clear explanations of functionality
- Mermaid diagrams for architecture (use graph TD, never LR)
- Code snippets from actual source files
- Source citations: `Sources: [filename:line-range]()`
"""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    return agent
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_react_agents.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_react_agents.py tests/unit/test_wiki_react_agents.py
git commit -m "feat(wiki): add React agent factory with MCP tools"
```

---

### Task 2: Refactor extract_structure_node to use React Agent

**Files:**
- Modify: `src/agents/wiki_workflow.py:157-247` (extract_structure_node function)

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_workflow.py - ADD this test
@pytest.mark.asyncio
async def test_extract_structure_node_uses_react_agent():
    """Structure node should invoke React agent with MCP tools."""
    from src.agents.wiki_workflow import extract_structure_node

    state = {
        "repository_id": "test-repo-id",
        "clone_path": "/tmp/test-repo",
        "file_tree": "src/\n  main.py",
        "readme_content": "# Test Repo",
        "structure": None,
        "pages": [],
        "error": None,
        "current_step": "init",
    }

    with patch("src.agents.wiki_workflow.create_structure_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [],
            "structured_response": {
                "title": "Test Wiki",
                "description": "Test",
                "sections": []
            }
        }
        mock_create.return_value = mock_agent

        result = await extract_structure_node(state)

    mock_create.assert_called_once()
    mock_agent.ainvoke.assert_called_once()
    assert result.get("structure") is not None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_workflow.py::test_extract_structure_node_uses_react_agent -v`
Expected: FAIL (current implementation doesn't use create_structure_agent)

**Step 3: Update extract_structure_node implementation**

```python
async def extract_structure_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Extract wiki structure using React agent with MCP tools.

    The React agent can explore the codebase using filesystem tools
    and returns a structured WikiStructure.

    Args:
        state: Current workflow state with file_tree and readme_content

    Returns:
        Dict with 'structure' and updated 'current_step'
    """
    from src.agents.wiki_react_agents import create_structure_agent, LLMWikiStructureSchema

    clone_path = state.get("clone_path", "")
    file_tree = state.get("file_tree", "")
    readme_content = state.get("readme_content", "")

    # Create React agent with MCP tools
    agent = await create_structure_agent(
        clone_path=clone_path,
        file_tree=file_tree,
        readme_content=readme_content,
    )

    # Build user message for the agent
    user_message = f"""Analyze this repository and create a comprehensive wiki structure.

## Repository Context
- Clone Path: {clone_path}

## File Tree
```
{file_tree}
```

## README
```
{readme_content}
```

Explore the codebase using the filesystem tools to understand:
1. Project architecture and structure
2. Key modules and their purposes
3. Important files and their relationships

Then design a wiki with 8-12 pages covering:
- Overview and Getting Started
- Architecture and core concepts
- Key features and functionality
- API reference (if applicable)
- Development and deployment guides

Use the filesystem tools to examine actual source files before finalizing the structure.
"""

    try:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_message}]
        })

        # Extract structured response
        structured_output = result.get("structured_response")
        if not structured_output:
            return {
                "error": "Agent did not return structured output",
                "current_step": "error",
            }

        # Convert to WikiStructure
        if isinstance(structured_output, dict):
            structure = _convert_llm_structure_to_wiki_structure(
                structured_output,
                state["repository_id"],
            )
        else:
            structure = _convert_llm_structure_to_wiki_structure(
                structured_output.model_dump(),
                state["repository_id"],
            )

        return {
            "structure": structure,
            "current_step": "structure_extracted",
        }

    except Exception as e:
        logger.error("Structure extraction failed", error=str(e))
        return {
            "error": f"Structure extraction failed: {str(e)}",
            "current_step": "error",
        }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_workflow.py::test_extract_structure_node_uses_react_agent -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "refactor(wiki): use React agent for structure extraction"
```

---

### Task 3: Refactor generate_pages_node to use React Agent (Sequential)

**Files:**
- Modify: `src/agents/wiki_workflow.py:250-340` (generate_pages_node function)

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_workflow.py - ADD this test
@pytest.mark.asyncio
async def test_generate_pages_node_uses_react_agent_per_page():
    """Page generation should invoke React agent for each page sequentially."""
    from src.agents.wiki_workflow import generate_pages_node
    from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

    # Create a structure with 2 pages
    structure = WikiStructure(
        repository_id=UUID("12345678-1234-5678-1234-567812345678"),
        title="Test Wiki",
        description="Test",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                description="Test section",
                order=1,
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First page",
                        importance=PageImportance.HIGH,
                        file_paths=["src/main.py"],
                    ),
                    WikiPageDetail(
                        id="page-2",
                        title="Page 2",
                        description="Second page",
                        importance=PageImportance.MEDIUM,
                        file_paths=["src/utils.py"],
                    ),
                ]
            )
        ]
    )

    state = {
        "repository_id": "12345678-1234-5678-1234-567812345678",
        "clone_path": "/tmp/test-repo",
        "file_tree": "src/\n  main.py\n  utils.py",
        "readme_content": "# Test",
        "structure": structure,
        "pages": [],
        "error": None,
        "current_step": "structure_extracted",
    }

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        # Return different content for each invocation
        mock_agent.ainvoke.side_effect = [
            {"messages": [MagicMock(content="# Page 1 Content")]},
            {"messages": [MagicMock(content="# Page 2 Content")]},
        ]
        mock_create.return_value = mock_agent

        result = await generate_pages_node(state)

    # Should create agent once (reused for all pages)
    mock_create.assert_called_once()
    # Should invoke agent twice (once per page)
    assert mock_agent.ainvoke.call_count == 2
    # Should have 2 pages with content
    assert len(result.get("pages", [])) == 2
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_workflow.py::test_generate_pages_node_uses_react_agent_per_page -v`
Expected: FAIL

**Step 3: Update generate_pages_node implementation**

```python
async def generate_pages_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Generate content for all wiki pages sequentially using React agents.

    Creates a React agent with MCP filesystem tools and invokes it
    for each page in sequence. The agent can read source files to
    generate accurate, source-grounded documentation.

    Args:
        state: Current state with extracted structure

    Returns:
        Dict with 'pages' list and updated 'current_step'
    """
    from src.agents.wiki_react_agents import create_page_agent

    # Check for existing errors or missing structure
    if state.get("error"):
        return {
            "error": state.get("error"),
            "current_step": "error",
        }

    if not state.get("structure"):
        return {
            "error": "No structure available",
            "current_step": "error",
        }

    structure = state["structure"]
    clone_path = state["clone_path"]

    # Create React agent with MCP tools (reused for all pages)
    agent = await create_page_agent(clone_path=clone_path)

    generated_pages = []
    all_pages = structure.get_all_pages()

    logger.info(
        "Starting sequential page generation",
        total_pages=len(all_pages),
        clone_path=clone_path,
    )

    for idx, page in enumerate(all_pages):
        logger.info(
            "Generating page",
            page_id=page.id,
            page_title=page.title,
            progress=f"{idx + 1}/{len(all_pages)}",
        )

        # Build user message for this page
        file_list = "\n".join(f"- {clone_path}/{fp}" for fp in page.file_paths) if page.file_paths else "No specific files assigned"

        user_message = f"""Generate comprehensive documentation for this wiki page.

## Page Details
- Title: {page.title}
- Description: {page.description}
- Importance: {page.importance.value if hasattr(page.importance, 'value') else page.importance}

## Files to Read (MUST READ ALL COMPLETELY)
{file_list}

## CRITICAL Instructions

1. **READ ALL FILES COMPLETELY** - Use read_file WITHOUT the head parameter
   - You need full implementation details, not just headers
   - Read each file in the list above in its entirety

2. **Understand the implementation** - After reading:
   - Data structures and their purposes
   - Function logic and data flow
   - Error handling patterns
   - Integration with other components

3. **Generate comprehensive documentation** including:
   - Clear explanation with implementation details
   - Mermaid diagrams (use graph TD, never LR)
   - Actual code snippets from source files
   - Source citations: Sources: [filename:line-range]()

4. **Start output with:** # {page.title}

Remember: All file paths are absolute. Read files COMPLETELY, not just headers!
"""

        try:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            # Extract content from last message
            messages = result.get("messages", [])
            content = ""
            if messages:
                last_message = messages[-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)

            page_with_content = page.model_copy(update={"content": content})

        except Exception as e:
            logger.error(
                "Page generation failed",
                page_id=page.id,
                error=str(e),
            )
            page_with_content = page.model_copy(update={
                "content": f"*Error generating content: {str(e)}*"
            })

        generated_pages.append(page_with_content)

    logger.info(
        "Page generation complete",
        total_pages=len(generated_pages),
        pages_with_content=len([p for p in generated_pages if p.content]),
    )

    return {
        "pages": generated_pages,
        "current_step": "pages_generated",
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_workflow.py::test_generate_pages_node_uses_react_agent_per_page -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "refactor(wiki): use React agent for page generation"
```

---

### Task 4: Create Aggregate Node

**Files:**
- Modify: `src/agents/wiki_workflow.py` (add aggregate_node function)

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_workflow.py - ADD this test
@pytest.mark.asyncio
async def test_aggregate_node_merges_content_into_structure():
    """Aggregate node should merge page content back into structure."""
    from src.agents.wiki_workflow import aggregate_node
    from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

    # Structure without content
    structure = WikiStructure(
        repository_id=UUID("12345678-1234-5678-1234-567812345678"),
        title="Test Wiki",
        description="Test",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                description="Test",
                order=1,
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First",
                        importance=PageImportance.HIGH,
                        file_paths=[],
                        content=None,
                    ),
                ]
            )
        ]
    )

    # Pages with generated content
    pages_with_content = [
        WikiPageDetail(
            id="page-1",
            title="Page 1",
            description="First",
            importance=PageImportance.HIGH,
            file_paths=[],
            content="# Page 1\n\nThis is the content.",
        )
    ]

    state = {
        "repository_id": "12345678-1234-5678-1234-567812345678",
        "structure": structure,
        "pages": pages_with_content,
        "error": None,
        "current_step": "pages_generated",
    }

    result = await aggregate_node(state)

    assert result.get("current_step") == "aggregated"
    # Verify content was merged into structure
    updated_structure = result.get("structure")
    assert updated_structure is not None
    all_pages = updated_structure.get_all_pages()
    assert len(all_pages) == 1
    assert all_pages[0].content == "# Page 1\n\nThis is the content."
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_workflow.py::test_aggregate_node_merges_content_into_structure -v`
Expected: FAIL

**Step 3: Implement aggregate_node**

```python
async def aggregate_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Aggregate generated page content back into the wiki structure.

    Merges the content from generated pages back into the corresponding
    WikiPageDetail objects in the structure.

    Args:
        state: Current state with structure and generated pages

    Returns:
        Dict with updated 'structure' and 'current_step'
    """
    if state.get("error"):
        return {
            "error": state.get("error"),
            "current_step": "error",
        }

    structure = state.get("structure")
    pages = state.get("pages", [])

    if not structure:
        return {
            "error": "No structure to aggregate into",
            "current_step": "error",
        }

    # Build lookup of page content by id
    content_by_id = {page.id: page.content for page in pages if page.content}

    logger.info(
        "Aggregating page content",
        total_pages=len(pages),
        pages_with_content=len(content_by_id),
    )

    # Update structure with generated content
    updated_sections = []
    for section in structure.sections:
        updated_pages = []
        for page in section.pages:
            content = content_by_id.get(page.id, page.content)
            updated_page = page.model_copy(update={"content": content})
            updated_pages.append(updated_page)

        updated_section = section.model_copy(update={"pages": updated_pages})
        updated_sections.append(updated_section)

    updated_structure = structure.model_copy(update={"sections": updated_sections})

    # Count pages with content
    all_pages = updated_structure.get_all_pages()
    pages_with_content = [p for p in all_pages if p.content]

    logger.info(
        "Aggregation complete",
        total_pages=len(all_pages),
        pages_with_content=len(pages_with_content),
    )

    return {
        "structure": updated_structure,
        "current_step": "aggregated",
    }
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_workflow.py::test_aggregate_node_merges_content_into_structure -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add aggregate node to merge page content"
```

---

### Task 5: Update Workflow Graph with Aggregate Node

**Files:**
- Modify: `src/agents/wiki_workflow.py` (update create_wiki_workflow)

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_workflow.py - ADD this test
def test_workflow_includes_aggregate_node():
    """Workflow should have 4 nodes: extract_structure, generate_pages, aggregate, finalize."""
    from src.agents.wiki_workflow import create_wiki_workflow

    workflow = create_wiki_workflow()

    # Get node names from the graph
    graph = workflow.get_graph()
    node_names = [node.name for node in graph.nodes if node.name not in ("__start__", "__end__")]

    assert "extract_structure" in node_names
    assert "generate_pages" in node_names
    assert "aggregate" in node_names
    assert "finalize" in node_names
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_workflow.py::test_workflow_includes_aggregate_node -v`
Expected: FAIL (aggregate node doesn't exist yet)

**Step 3: Update create_wiki_workflow**

```python
def create_wiki_workflow() -> Any:
    """Create the wiki generation workflow.

    Returns:
        Compiled StateGraph for wiki generation
    """
    builder = StateGraph(WikiWorkflowState)

    # Add nodes
    builder.add_node("extract_structure", extract_structure_node)
    builder.add_node("generate_pages", generate_pages_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("finalize", finalize_node)

    # Define edges: START -> extract -> generate -> aggregate -> finalize -> END
    builder.add_edge(START, "extract_structure")
    builder.add_edge("extract_structure", "generate_pages")
    builder.add_edge("generate_pages", "aggregate")
    builder.add_edge("aggregate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_wiki_workflow.py::test_workflow_includes_aggregate_node -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "feat(wiki): add aggregate node to workflow graph"
```

---

### Task 6: Update finalize_node to Use Aggregated Structure

**Files:**
- Modify: `src/agents/wiki_workflow.py:finalize_node`

**Step 1: Write the failing test**

```python
# tests/unit/test_wiki_workflow.py - ADD this test
@pytest.mark.asyncio
async def test_finalize_node_saves_structure_with_content():
    """Finalize node should save the structure with all page content."""
    from src.agents.wiki_workflow import finalize_node
    from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

    structure = WikiStructure(
        repository_id=UUID("12345678-1234-5678-1234-567812345678"),
        title="Test Wiki",
        description="Test",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                description="Test",
                order=1,
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First",
                        importance=PageImportance.HIGH,
                        file_paths=[],
                        content="# Page 1 Content",
                    ),
                ]
            )
        ]
    )

    state = {
        "repository_id": "12345678-1234-5678-1234-567812345678",
        "structure": structure,
        "pages": [],  # Already merged into structure
        "error": None,
        "current_step": "aggregated",
    }

    with patch.object(WikiStructureRepository, "upsert", new_callable=AsyncMock) as mock_upsert:
        result = await finalize_node(state)

    mock_upsert.assert_called_once()
    saved_structure = mock_upsert.call_args[0][0]
    assert saved_structure.sections[0].pages[0].content == "# Page 1 Content"
    assert result.get("current_step") == "complete"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_wiki_workflow.py::test_finalize_node_saves_structure_with_content -v`
Expected: PASS or minor adjustment needed

**Step 3: Update finalize_node if needed**

The current implementation should work, but verify it uses the structure from state (which now has content merged in).

**Step 4: Commit**

```bash
git add src/agents/wiki_workflow.py tests/unit/test_wiki_workflow.py
git commit -m "refactor(wiki): ensure finalize uses aggregated structure"
```

---

### Task 7: Update Prompts for React Agent Usage

**Files:**
- Modify: `src/prompts/wiki_prompts.yaml`

**Step 1: Update structure_agent prompt with CONTEXT-EFFICIENT reading strategy**

```yaml
structure_agent:
  # ... existing fields ...

  exploration_instructions_mcp: |
    ## Available Filesystem Tools

    You have access to filesystem tools to explore the codebase:
    - `list_directory(path)`: List directory contents
    - `read_file(path, head=N)`: Read file content (use head parameter!)
    - `search_files(path, pattern)`: Search for files matching pattern
    - `get_file_info(path)`: Get file metadata

    All paths must be absolute, starting with: {clone_path}

    ## CRITICAL: Context-Efficient Reading Strategy

    Your goal is to UNDERSTAND THE STRUCTURE, not read every detail.

    **DO:**
    - Use `read_file(path, head=50)` to read only first 50 lines
    - Focus on imports, class definitions, function signatures
    - Read docstrings and module-level comments
    - Use this to determine what PAGE each file belongs to

    **DON'T:**
    - Read entire files (wastes context)
    - Read implementation details
    - Read test files in detail (just note they exist)

    ## Exploration Strategy

    1. Start with file tree and README for high-level understanding
    2. Use list_directory to explore key directories (src/, lib/, etc.)
    3. Use read_file with head=50 to examine file headers
    4. For each significant file, determine which wiki page it belongs to
    5. Assign file_paths to each page in your structure output
```

**Step 2: Add page_generation prompt with FULL reading strategy**

```yaml
page_generation_react:
  name: "Page Generation React Agent Prompt"
  description: |
    Prompt for React agent that generates page documentation with full file access.

  system_prompt: |
    You are an expert technical writer generating comprehensive wiki documentation.

    ## Available Filesystem Tools

    You have access to filesystem tools:
    - `read_file(path)`: Read COMPLETE file content
    - `list_directory(path)`: List directory contents
    - `search_files(path, pattern)`: Search for files

    All paths must be absolute, starting with: {clone_path}

    ## CRITICAL: Full File Reading Strategy

    For documentation, you need COMPLETE understanding:

    **DO:**
    - Read the ENTIRE content of files assigned to this page
    - Understand implementation details, data flow, logic
    - Extract actual code snippets for examples
    - Note specific line numbers for citations
    - Read related files if needed for context

    **DON'T:**
    - Skip reading files - you need the full content
    - Guess at implementation details
    - Cite lines without reading them

    ## Documentation Requirements

    1. Read ALL files in the provided file_paths list completely
    2. Create comprehensive documentation including:
       - Clear explanations of functionality
       - Mermaid diagrams for architecture (use graph TD, never LR)
       - Code snippets from actual source files
       - Source citations: `Sources: [filename:line-range]()`
    3. Start output with: # {page_title}
```

**Step 3: Commit**

```bash
git add src/prompts/wiki_prompts.yaml
git commit -m "feat(wiki): add context-efficient and full-read MCP prompts"
```

---

### Task 8: Integration Test

**Files:**
- Create: `tests/integration/test_wiki_workflow_integration.py`

**Step 1: Write integration test**

```python
# tests/integration/test_wiki_workflow_integration.py
"""Integration tests for wiki workflow with actual MCP tools."""

import pytest
from pathlib import Path
from unittest.mock import patch, AsyncMock

from src.agents.wiki_workflow import wiki_workflow, WikiWorkflowState


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_with_mocked_mcp():
    """Test full workflow with mocked MCP client."""
    # Create a simple test repository structure
    state: WikiWorkflowState = {
        "repository_id": "test-repo-123",
        "clone_path": str(Path(__file__).parent / "fixtures" / "test_repo"),
        "file_tree": """
src/
  main.py
  utils.py
README.md
""",
        "readme_content": "# Test Repository\n\nA simple test repo.",
        "structure": None,
        "pages": [],
        "error": None,
        "current_step": "init",
    }

    # Mock MCP tools
    mock_tools = [
        AsyncMock(name="list_directory"),
        AsyncMock(name="read_file"),
    ]

    with patch("src.agents.wiki_react_agents.get_mcp_tools", return_value=mock_tools):
        with patch("src.agents.wiki_react_agents.get_llm") as mock_llm:
            # Set up mock LLM responses
            mock_model = AsyncMock()
            mock_llm.return_value = mock_model

            # Skip actual invocation for now - just verify setup
            # result = await wiki_workflow.ainvoke(state)
            # assert result.get("current_step") == "complete"
            pass
```

**Step 2: Commit**

```bash
git add tests/integration/test_wiki_workflow_integration.py
git commit -m "test(wiki): add integration test skeleton for React agent workflow"
```

---

### Task 9: Clean Up Old Implementation

**Files:**
- Modify: `src/agents/wiki_workflow.py` (remove old LLMTool imports if unused)
- Archive: Old test files if any

**Step 1: Remove unused imports**

```python
# Remove from wiki_workflow.py if no longer used:
# from src.tools.llm_tool import LLMTool
```

**Step 2: Run all tests**

```bash
pytest tests/unit/test_wiki_workflow.py -v
pytest tests/unit/test_wiki_react_agents.py -v
```

**Step 3: Commit**

```bash
git add -A
git commit -m "chore(wiki): clean up old LLMTool usage"
```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MCP client not initialized | Add initialization check in `get_mcp_tools()` |
| Agent timeout on large repos | Add timeout parameter to agent invocation |
| Rate limiting on LLM calls | Use exponential backoff in page generation loop |
| Structure extraction fails | Fallback to simpler structure with file_tree only |
| Page generation produces low quality | Add validation step before aggregation |

---

## Success Metrics

1. **Tool Access**: Agents can successfully call MCP filesystem tools
2. **Structure Quality**: Structure reflects actual codebase organization
3. **Page Quality**: Pages contain source citations with file:line references
4. **Reliability**: 95%+ successful wiki generations
5. **Observability**: Clear traces showing agent tool calls in LangSmith

---

## Dependencies

```
langgraph>=0.2.0
langchain-mcp-adapters>=0.1.0
langchain-anthropic>=0.1.0
```

---

## Summary

This plan replaces plain LLM calls with React agents that have full MCP filesystem tool access. The key changes:

1. **Task 1**: Create `wiki_react_agents.py` with factory functions
2. **Task 2**: Refactor `extract_structure_node` to use React agent
3. **Task 3**: Refactor `generate_pages_node` to use React agent (sequential)
4. **Task 4**: Create `aggregate_node` to merge content
5. **Task 5**: Update workflow graph with aggregate node
6. **Task 6**: Update finalize_node
7. **Task 7**: Update prompts with MCP exploration instructions
8. **Task 8**: Add integration tests
9. **Task 9**: Clean up old implementation
