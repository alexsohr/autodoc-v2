# tests/unit/test_wiki_workflow.py
"""Tests for WikiWorkflowState TypedDict."""

import pytest
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


def test_wiki_workflow_state_with_structure():
    """WikiWorkflowState should accept WikiStructure for structure field."""
    from uuid import uuid4
    from src.models.wiki import WikiSection

    # Create a minimal wiki structure
    section = WikiSection(
        id="getting-started",
        title="Getting Started",
        pages=[
            WikiPageDetail(
                id="installation",
                title="Installation",
                description="How to install the project",
                importance="medium",
            )
        ],
    )

    structure = WikiStructure(
        id="test-wiki",
        repository_id=uuid4(),
        title="Test Wiki",
        description="A test wiki",
        sections=[section],
    )

    state = WikiWorkflowState(
        repository_id="test-repo-id",
        clone_path="/tmp/repo",
        file_tree="src/\n  main.py",
        readme_content="# Test Repo",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
    )

    assert state["structure"] is not None
    assert state["structure"].title == "Test Wiki"
    assert len(state["structure"].get_all_pages()) == 1


def test_wiki_workflow_state_pages_list():
    """WikiWorkflowState pages field should be a list of WikiPageDetail."""
    page = WikiPageDetail(
        id="test-page",
        title="Test Page",
        description="A test page",
        importance="high",
        content="# Test Content\n\nThis is test content.",
    )

    state = WikiWorkflowState(
        repository_id="test-repo-id",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[page],
        error=None,
        current_step="page_generated",
    )

    assert len(state["pages"]) == 1
    assert state["pages"][0].id == "test-page"
    assert state["pages"][0].has_content() is True


def test_wiki_workflow_state_error_handling():
    """WikiWorkflowState should support error field for workflow failures."""
    state = WikiWorkflowState(
        repository_id="test-repo-id",
        clone_path="/tmp/repo",
        file_tree="",
        readme_content="",
        structure=None,
        pages=[],
        error="Failed to extract wiki structure: LLM timeout",
        current_step="error",
    )

    assert state["error"] is not None
    assert "LLM timeout" in state["error"]
    assert state["current_step"] == "error"


def test_wiki_workflow_state_current_step_tracking():
    """WikiWorkflowState should track current workflow step for observability."""
    valid_steps = ["init", "structure_extracted", "generating_pages", "page_generated", "finalizing", "complete", "error"]

    for step in valid_steps:
        state = WikiWorkflowState(
            repository_id="test-repo-id",
            clone_path="/tmp/repo",
            file_tree="src/",
            readme_content="# Test",
            structure=None,
            pages=[],
            error=None,
            current_step=step,
        )
        assert state["current_step"] == step


# =============================================================================
# Tests for extract_structure_node
# =============================================================================


@pytest.mark.asyncio
async def test_extract_structure_node_success():
    """extract_structure_node should return structure from React agent."""
    from unittest.mock import AsyncMock, patch
    from src.agents.wiki_workflow import extract_structure_node
    from src.models.wiki import PageImportance

    # Use a valid UUID for repository_id
    test_repo_id = "00000000-0000-0000-0000-000000000000"

    initial_state = WikiWorkflowState(
        repository_id=test_repo_id,
        clone_path="/tmp/repo",
        file_tree="src/\n  main.py\n  utils.py",
        readme_content="# Test Project\nA test project.",
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )

    # Mock the React agent structured_response output
    mock_structured_response = {
        "title": "Test Project",
        "description": "A test project",
        "sections": [
            {
                "id": "overview",
                "title": "Overview",
                "pages": [
                    {
                        "id": "getting-started",
                        "title": "Getting Started",
                        "description": "How to get started",
                        "importance": "high",
                        "file_paths": ["src/main.py"],
                    }
                ]
            }
        ]
    }

    with patch("src.agents.wiki_workflow.create_structure_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [],
            "structured_response": mock_structured_response
        }
        mock_create.return_value = mock_agent

        result = await extract_structure_node(initial_state)

        assert result["structure"] is not None
        assert result["structure"].title == "Test Project"
        assert result["current_step"] == "structure_extracted"
        # Verify the structure was properly converted
        assert len(result["structure"].sections) == 1
        assert result["structure"].sections[0].id == "overview"
        assert len(result["structure"].sections[0].pages) == 1
        assert result["structure"].sections[0].pages[0].importance == PageImportance.HIGH


@pytest.mark.asyncio
async def test_extract_structure_node_error():
    """extract_structure_node should handle agent errors gracefully."""
    from unittest.mock import AsyncMock, patch
    from src.agents.wiki_workflow import extract_structure_node

    # Use a valid UUID for repository_id
    test_repo_id = "00000000-0000-0000-0000-000000000000"

    initial_state = WikiWorkflowState(
        repository_id=test_repo_id,
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )

    with patch("src.agents.wiki_workflow.create_structure_agent") as mock_create:
        mock_agent = AsyncMock()
        # Simulate agent raising an exception
        mock_agent.ainvoke.side_effect = Exception("Rate limit exceeded")
        mock_create.return_value = mock_agent

        result = await extract_structure_node(initial_state)

        assert result["error"] is not None
        assert "Rate limit" in result["error"]
        assert result["current_step"] == "error"


# =============================================================================
# Tests for generate_pages_node
# =============================================================================


@pytest.mark.asyncio
async def test_generate_pages_node_success():
    """generate_pages_node should generate content for all pages sequentially."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from src.agents.wiki_workflow import generate_pages_node
    from src.models.wiki import WikiSection, PageImportance

    structure = WikiStructure(
        id="test-wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
        title="Test",
        description="Test wiki",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page1",
                        title="Page 1",
                        description="First page",
                        importance=PageImportance.HIGH,
                    ),
                    WikiPageDetail(
                        id="page2",
                        title="Page 2",
                        description="Second page",
                        importance=PageImportance.MEDIUM,
                    ),
                ]
            )
        ]
    )

    state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/\n  main.py",
        readme_content="# Test",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
    )

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        # React agent returns messages with content
        mock_agent.ainvoke.return_value = {
            "messages": [MagicMock(content="# Generated Content\n\nThis is the generated content.")]
        }
        mock_create.return_value = mock_agent

        result = await generate_pages_node(state)

        assert "pages" in result
        assert len(result["pages"]) == 2
        assert result["pages"][0].id == "page1"
        assert result["pages"][0].content is not None
        assert result["pages"][0].has_content() is True
        assert result["pages"][1].id == "page2"
        assert result["current_step"] == "pages_generated"


@pytest.mark.asyncio
async def test_generate_pages_node_no_structure():
    """generate_pages_node should handle missing structure gracefully."""
    from src.agents.wiki_workflow import generate_pages_node

    state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="init",
    )

    result = await generate_pages_node(state)

    assert result["current_step"] == "error"
    assert "No structure" in result.get("error", "")


@pytest.mark.asyncio
async def test_generate_pages_node_with_existing_error():
    """generate_pages_node should propagate existing errors."""
    from src.agents.wiki_workflow import generate_pages_node

    state = WikiWorkflowState(
        repository_id="test-repo",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error="Previous error occurred",
        current_step="error",
    )

    result = await generate_pages_node(state)

    assert result["current_step"] == "error"
    assert "Previous error" in result.get("error", "")


@pytest.mark.asyncio
async def test_generate_pages_node_agent_error():
    """generate_pages_node should handle agent errors and include error message in content."""
    from unittest.mock import AsyncMock, patch
    from src.agents.wiki_workflow import generate_pages_node
    from src.models.wiki import WikiSection, PageImportance

    structure = WikiStructure(
        id="test-wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
        title="Test",
        description="Test wiki",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page1",
                        title="Page 1",
                        description="First page",
                        importance=PageImportance.HIGH,
                    ),
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

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        # Agent raises an exception
        mock_agent.ainvoke.side_effect = Exception("Rate limit exceeded")
        mock_create.return_value = mock_agent

        result = await generate_pages_node(state)

        # Should still return pages but with error content
        assert "pages" in result
        assert len(result["pages"]) == 1
        assert result["pages"][0].id == "page1"
        assert "Error generating content" in result["pages"][0].content
        assert result["current_step"] == "pages_generated"


# =============================================================================
# Tests for finalize_node
# =============================================================================


@pytest.mark.asyncio
async def test_finalize_node_combines_pages():
    """finalize_node should combine pages into structure and store."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from src.agents.wiki_workflow import finalize_node, WikiWorkflowState
    from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

    structure = WikiStructure(
        id="test-wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(id="page1", title="Page 1", description="First", importance=PageImportance.HIGH),
                    WikiPageDetail(id="page2", title="Page 2", description="Second", importance=PageImportance.MEDIUM),
                ]
            )
        ]
    )

    pages_with_content = [
        WikiPageDetail(id="page1", title="Page 1", description="First", importance=PageImportance.HIGH, content="# Page 1 Content"),
        WikiPageDetail(id="page2", title="Page 2", description="Second", importance=PageImportance.MEDIUM, content="# Page 2 Content"),
    ]

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=pages_with_content,
        error=None,
        current_step="pages_generated",
    )

    with patch("src.agents.wiki_workflow.WikiStructureRepository") as MockWikiRepo:
        # Mock repository upsert
        mock_repo_instance = MagicMock()
        mock_repo_instance.upsert = AsyncMock()
        MockWikiRepo.return_value = mock_repo_instance

        result = await finalize_node(state)

        assert result["current_step"] == "completed"
        assert result.get("error") is None
        # Verify repository was called
        mock_repo_instance.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_finalize_node_with_error_state():
    """finalize_node should return error state if error already exists."""
    from src.agents.wiki_workflow import finalize_node, WikiWorkflowState

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error="Previous error occurred",
        current_step="error",
    )

    result = await finalize_node(state)

    assert result["current_step"] == "error"


@pytest.mark.asyncio
async def test_finalize_node_no_structure():
    """finalize_node should handle missing structure gracefully."""
    from src.agents.wiki_workflow import finalize_node, WikiWorkflowState

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="pages_generated",
    )

    result = await finalize_node(state)

    assert result["current_step"] == "error"
    assert "No structure" in result.get("error", "")


@pytest.mark.asyncio
async def test_finalize_node_save_failure():
    """finalize_node should handle database save failures gracefully."""
    from unittest.mock import AsyncMock, patch, MagicMock
    from src.agents.wiki_workflow import finalize_node, WikiWorkflowState
    from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance

    structure = WikiStructure(
        id="test-wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(id="page1", title="Page 1", description="First", importance=PageImportance.HIGH),
                ]
            )
        ]
    )

    pages_with_content = [
        WikiPageDetail(id="page1", title="Page 1", description="First", importance=PageImportance.HIGH, content="# Page 1 Content"),
    ]

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=pages_with_content,
        error=None,
        current_step="pages_generated",
    )

    with patch("src.agents.wiki_workflow.WikiStructureRepository") as MockWikiRepo:
        # Mock repository upsert failure
        mock_repo_instance = MagicMock()
        mock_repo_instance.upsert = AsyncMock(side_effect=Exception("Database connection failed"))
        MockWikiRepo.return_value = mock_repo_instance

        result = await finalize_node(state)

        assert result["current_step"] == "error"
        assert "Failed to save wiki" in result.get("error", "")


# =============================================================================
# Tests for create_wiki_workflow (StateGraph assembly)
# =============================================================================


def test_wiki_workflow_compiles():
    """wiki_workflow should compile without errors."""
    from src.agents.wiki_workflow import create_wiki_workflow

    workflow = create_wiki_workflow()

    assert workflow is not None
    # Check graph has expected nodes
    assert "extract_structure" in str(workflow.nodes)
    assert "generate_pages" in str(workflow.nodes)
    assert "finalize" in str(workflow.nodes)


# =============================================================================
# Tests for React Agent Integration
# =============================================================================


@pytest.mark.asyncio
async def test_extract_structure_node_uses_react_agent():
    """Structure node should invoke React agent with MCP tools."""
    from src.agents.wiki_workflow import extract_structure_node
    from unittest.mock import patch, AsyncMock

    # Use a valid UUID for repository_id
    test_repo_id = "00000000-0000-0000-0000-000000000001"

    state = {
        "repository_id": test_repo_id,
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


@pytest.mark.asyncio
async def test_extract_structure_node_no_structured_output():
    """Structure node should handle missing structured_response gracefully."""
    from src.agents.wiki_workflow import extract_structure_node
    from unittest.mock import patch, AsyncMock

    # Use a valid UUID for repository_id
    test_repo_id = "00000000-0000-0000-0000-000000000002"

    state = {
        "repository_id": test_repo_id,
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
        # Agent returns messages but no structured_response
        mock_agent.ainvoke.return_value = {
            "messages": [{"role": "assistant", "content": "I analyzed the repo..."}],
            "structured_response": None
        }
        mock_create.return_value = mock_agent

        result = await extract_structure_node(state)

    assert result.get("error") is not None
    assert "did not return structured output" in result["error"]
    assert result["current_step"] == "error"


@pytest.mark.asyncio
async def test_generate_pages_node_uses_react_agent_per_page():
    """Page generation should invoke React agent for each page sequentially."""
    from src.agents.wiki_workflow import generate_pages_node
    from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance
    from uuid import UUID
    from unittest.mock import patch, AsyncMock, MagicMock

    # Create a structure with 2 pages
    structure = WikiStructure(
        id="wiki-test",
        repository_id="12345678-1234-5678-1234-567812345678",
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
