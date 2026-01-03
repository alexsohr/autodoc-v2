# tests/unit/test_wiki_react_agents.py
"""Tests for wiki React agent factories with MCP filesystem tools.

These tests verify that:
1. create_structure_agent() returns a compiled LangGraph agent
2. create_page_agent() returns a compiled LangGraph agent
3. Both agents have ainvoke() method for async execution
4. Agents are created with MCP tools bound
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_mcp_tools():
    """Mock MCP filesystem tools."""
    read_file_tool = MagicMock(name="read_text_file")
    read_file_tool.name = "read_text_file"

    list_dir_tool = MagicMock(name="list_directory")
    list_dir_tool.name = "list_directory"

    search_tool = MagicMock(name="search_files")
    search_tool.name = "search_files"

    return [read_file_tool, list_dir_tool, search_tool]


@pytest.fixture
def mock_llm():
    """Mock LLM from LLMTool."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=llm)
    llm.with_structured_output = MagicMock(return_value=llm)
    return llm


# =============================================================================
# Tests for get_mcp_tools
# =============================================================================


@pytest.mark.asyncio
async def test_get_mcp_tools_returns_list(mock_mcp_tools):
    """get_mcp_tools should return a list of MCP tools."""
    from src.agents.wiki_react_agents import get_mcp_tools

    with patch("src.agents.wiki_react_agents.MCPFilesystemClient") as MockMCPClient:
        mock_instance = MagicMock()
        mock_instance.is_initialized = True
        mock_instance.get_tools.return_value = mock_mcp_tools
        mock_instance.initialize = AsyncMock(return_value=True)
        MockMCPClient.get_instance.return_value = mock_instance

        tools = await get_mcp_tools()

        assert isinstance(tools, list)
        assert len(tools) == 3


@pytest.mark.asyncio
async def test_get_mcp_tools_initializes_if_needed(mock_mcp_tools):
    """get_mcp_tools should initialize MCP client if not initialized."""
    from src.agents.wiki_react_agents import get_mcp_tools

    with patch("src.agents.wiki_react_agents.MCPFilesystemClient") as MockMCPClient:
        mock_instance = MagicMock()
        mock_instance.is_initialized = False  # Not initialized
        mock_instance.get_tools.return_value = mock_mcp_tools
        mock_instance.initialize = AsyncMock(return_value=True)
        MockMCPClient.get_instance.return_value = mock_instance

        tools = await get_mcp_tools()

        mock_instance.initialize.assert_called_once()
        assert len(tools) == 3


# =============================================================================
# Tests for get_llm
# =============================================================================


def test_get_llm_returns_model(mock_llm):
    """get_llm should return a LangChain chat model."""
    from src.agents.wiki_react_agents import get_llm

    with patch("src.agents.wiki_react_agents.LLMTool") as MockLLMTool:
        mock_llm_tool = MagicMock()
        mock_llm_tool._get_llm_provider.return_value = mock_llm
        MockLLMTool.return_value = mock_llm_tool

        llm = get_llm()

        assert llm is not None
        MockLLMTool.assert_called_once()


# =============================================================================
# Tests for create_structure_agent
# =============================================================================


@pytest.mark.asyncio
async def test_create_structure_agent_returns_compiled_graph(mock_mcp_tools, mock_llm):
    """Structure agent should be a compiled LangGraph."""
    from src.agents.wiki_react_agents import create_structure_agent

    # Create a mock agent that has ainvoke
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock()

    with patch("src.agents.wiki_react_agents.get_mcp_tools", new_callable=AsyncMock, return_value=mock_mcp_tools), \
         patch("src.agents.wiki_react_agents.get_llm", return_value=mock_llm), \
         patch("src.agents.wiki_react_agents.create_react_agent", return_value=mock_agent):

        agent = await create_structure_agent()

        assert agent is not None
        assert hasattr(agent, "ainvoke")


@pytest.mark.asyncio
async def test_create_structure_agent_with_context(mock_mcp_tools, mock_llm):
    """Structure agent should accept context parameters."""
    from src.agents.wiki_react_agents import create_structure_agent

    # Create a mock agent that has ainvoke
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock()

    with patch("src.agents.wiki_react_agents.get_mcp_tools", new_callable=AsyncMock, return_value=mock_mcp_tools), \
         patch("src.agents.wiki_react_agents.get_llm", return_value=mock_llm), \
         patch("src.agents.wiki_react_agents.create_react_agent", return_value=mock_agent):

        agent = await create_structure_agent(
            clone_path="/tmp/test-repo",
            file_tree="src/\n  main.py\n  utils.py",
            readme_content="# Test Project",
        )

        assert agent is not None
        assert hasattr(agent, "ainvoke")


# =============================================================================
# Tests for create_page_agent
# =============================================================================


@pytest.mark.asyncio
async def test_create_page_agent_returns_compiled_graph(mock_mcp_tools, mock_llm):
    """Page agent should be a compiled LangGraph."""
    from src.agents.wiki_react_agents import create_page_agent

    # Create a mock agent that has ainvoke
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock()

    with patch("src.agents.wiki_react_agents.get_mcp_tools", new_callable=AsyncMock, return_value=mock_mcp_tools), \
         patch("src.agents.wiki_react_agents.get_llm", return_value=mock_llm), \
         patch("src.agents.wiki_react_agents.create_react_agent", return_value=mock_agent):

        agent = await create_page_agent()

        assert agent is not None
        assert hasattr(agent, "ainvoke")


@pytest.mark.asyncio
async def test_create_page_agent_with_clone_path(mock_mcp_tools, mock_llm):
    """Page agent should accept clone_path parameter."""
    from src.agents.wiki_react_agents import create_page_agent

    # Create a mock agent that has ainvoke
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock()

    with patch("src.agents.wiki_react_agents.get_mcp_tools", new_callable=AsyncMock, return_value=mock_mcp_tools), \
         patch("src.agents.wiki_react_agents.get_llm", return_value=mock_llm), \
         patch("src.agents.wiki_react_agents.create_react_agent", return_value=mock_agent):

        agent = await create_page_agent(
            clone_path="/tmp/test-repo",
        )

        assert agent is not None
        assert hasattr(agent, "ainvoke")


# =============================================================================
# Tests for Pydantic Schemas
# =============================================================================


def test_llm_page_schema_validation():
    """LLMPageSchema should validate page details."""
    from src.agents.wiki_react_agents import LLMPageSchema

    page = LLMPageSchema(
        id="getting-started",
        title="Getting Started",
        description="How to get started with the project",
        importance="high",
        file_paths=["src/main.py", "README.md"],
    )

    assert page.id == "getting-started"
    assert page.title == "Getting Started"
    assert page.importance == "high"
    assert len(page.file_paths) == 2


def test_llm_section_schema_validation():
    """LLMSectionSchema should validate section with pages."""
    from src.agents.wiki_react_agents import LLMSectionSchema, LLMPageSchema

    page = LLMPageSchema(
        id="intro",
        title="Introduction",
        description="Project introduction",
    )

    section = LLMSectionSchema(
        id="overview",
        title="Overview",
        description="Project overview section",
        order=1,
        pages=[page],
    )

    assert section.id == "overview"
    assert len(section.pages) == 1
    assert section.pages[0].id == "intro"


def test_llm_wiki_structure_schema_validation():
    """LLMWikiStructureSchema should validate complete wiki structure."""
    from src.agents.wiki_react_agents import (
        LLMWikiStructureSchema,
        LLMSectionSchema,
        LLMPageSchema,
    )

    page = LLMPageSchema(
        id="overview",
        title="Overview",
        description="Project overview",
    )

    section = LLMSectionSchema(
        id="getting-started",
        title="Getting Started",
        description="Getting started guide",
        order=1,
        pages=[page],
    )

    wiki = LLMWikiStructureSchema(
        title="Test Project Wiki",
        description="Documentation for Test Project",
        sections=[section],
    )

    assert wiki.title == "Test Project Wiki"
    assert len(wiki.sections) == 1
    assert wiki.sections[0].id == "getting-started"


def test_llm_page_schema_default_values():
    """LLMPageSchema should have sensible defaults."""
    from src.agents.wiki_react_agents import LLMPageSchema

    # Minimal page with just required fields
    page = LLMPageSchema(
        id="minimal-page",
        title="Minimal Page",
        description="A minimal page",
    )

    assert page.importance == "medium"  # default
    assert page.file_paths == []  # default empty list
