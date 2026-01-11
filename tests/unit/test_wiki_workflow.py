# tests/unit/test_wiki_workflow.py
"""Tests for wiki workflow including fan-out/fan-in pattern."""

import pytest
from langgraph.types import Send

from src.agents.wiki_workflow import (
    PageGenerationState,
    WikiWorkflowState,
    fan_out_to_page_workers,
    generate_single_page_node,
    should_fan_out,
)
from src.models.wiki import PageImportance, WikiPageDetail, WikiSection, WikiStructure


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
        force_regenerate=False,
    )
    assert state["repository_id"] == "test-repo-id"
    assert state["pages"] == []
    assert state["force_regenerate"] is False


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
        force_regenerate=False,
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
        force_regenerate=False,
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
        force_regenerate=False,
    )

    assert state["error"] is not None
    assert "LLM timeout" in state["error"]
    assert state["current_step"] == "error"


def test_wiki_workflow_state_current_step_tracking():
    """WikiWorkflowState should track current workflow step for observability."""
    valid_steps = [
        "init",
        "structure_extracted",
        "generating_pages",
        "page_generated",
        "finalizing",
        "complete",
        "error",
    ]

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
            force_regenerate=False,
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
        force_regenerate=False,
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
                ],
            }
        ],
    }

    with patch("src.agents.wiki_workflow.create_structure_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "messages": [],
            "structured_response": mock_structured_response,
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
        assert (
            result["structure"].sections[0].pages[0].importance == PageImportance.HIGH
        )


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
        force_regenerate=False,
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
# Tests for finalize_node
# =============================================================================


@pytest.mark.asyncio
async def test_finalize_node_saves_aggregated_structure():
    """Finalize node should save the structure that already has content from aggregate step."""
    from unittest.mock import AsyncMock, patch
    from uuid import UUID

    from src.agents.wiki_workflow import finalize_node
    from src.models.wiki import (
        PageImportance,
        WikiPageDetail,
        WikiSection,
        WikiStructure,
    )
    from src.repository.wiki_structure_repository import WikiStructureRepository

    # Structure with content already merged (from aggregate step)
    structure = WikiStructure(
        repository_id="12345678-1234-5678-1234-567812345678",
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
                        content="# Page 1 Content",  # Already merged by aggregate
                    ),
                ],
            )
        ],
    )

    state = {
        "repository_id": "12345678-1234-5678-1234-567812345678",
        "clone_path": "/tmp/test",
        "file_tree": "",
        "readme_content": "",
        "structure": structure,
        "pages": [],  # Empty - content already in structure
        "error": None,
        "current_step": "aggregated",
    }

    with patch.object(
        WikiStructureRepository, "upsert", new_callable=AsyncMock
    ) as mock_upsert:
        result = await finalize_node(state)

    mock_upsert.assert_called_once()
    # Verify the structure passed to upsert has the content
    call_kwargs = mock_upsert.call_args.kwargs
    saved_wiki = call_kwargs.get("wiki")
    assert saved_wiki is not None
    # mdformat adds trailing newline (CommonMark spec)
    assert saved_wiki.sections[0].pages[0].content.strip() == "# Page 1 Content"
    assert result.get("current_step") == "completed"


@pytest.mark.asyncio
async def test_finalize_node_with_error_state():
    """finalize_node should return error state if error already exists."""
    from src.agents.wiki_workflow import WikiWorkflowState, finalize_node

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error="Previous error occurred",
        current_step="error",
        force_regenerate=False,
    )

    result = await finalize_node(state)

    assert result["current_step"] == "error"


@pytest.mark.asyncio
async def test_finalize_node_no_structure():
    """finalize_node should handle missing structure gracefully."""
    from src.agents.wiki_workflow import WikiWorkflowState, finalize_node

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="pages_generated",
        force_regenerate=False,
    )

    result = await finalize_node(state)

    assert result["current_step"] == "error"
    assert "No structure" in result.get("error", "")


@pytest.mark.asyncio
async def test_finalize_node_save_failure():
    """finalize_node should handle database save failures gracefully."""
    from unittest.mock import AsyncMock, patch

    from src.agents.wiki_workflow import finalize_node
    from src.models.wiki import (
        PageImportance,
        WikiPageDetail,
        WikiSection,
        WikiStructure,
    )
    from src.repository.wiki_structure_repository import WikiStructureRepository

    # Structure with content already merged (from aggregate step)
    structure = WikiStructure(
        repository_id="00000000-0000-0000-0000-000000000000",
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page1",
                        title="Page 1",
                        description="First",
                        importance=PageImportance.HIGH,
                        content="# Page 1 Content",  # Already merged
                    ),
                ],
            )
        ],
    )

    state = {
        "repository_id": "00000000-0000-0000-0000-000000000000",
        "clone_path": "/tmp/repo",
        "file_tree": "src/",
        "readme_content": "# Test",
        "structure": structure,
        "pages": [],  # Empty - content already in structure
        "error": None,
        "current_step": "aggregated",
    }

    with patch.object(
        WikiStructureRepository,
        "upsert",
        new_callable=AsyncMock,
        side_effect=Exception("Database connection failed"),
    ) as mock_upsert:
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
    # Check graph has expected nodes for fan-out/fan-in pattern
    assert "extract_structure" in str(workflow.nodes)
    assert "generate_single_page" in str(
        workflow.nodes
    )  # Worker node for parallel generation
    assert "aggregate" in str(workflow.nodes)  # Fan-in node
    assert "finalize" in str(workflow.nodes)


# =============================================================================
# Tests for React Agent Integration
# =============================================================================


@pytest.mark.asyncio
async def test_extract_structure_node_uses_react_agent():
    """Structure node should invoke React agent with MCP tools."""
    from unittest.mock import AsyncMock, patch

    from src.agents.wiki_workflow import extract_structure_node

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
                "sections": [],
            },
        }
        mock_create.return_value = mock_agent

        result = await extract_structure_node(state)

    mock_create.assert_called_once()
    mock_agent.ainvoke.assert_called_once()
    assert result.get("structure") is not None


@pytest.mark.asyncio
async def test_extract_structure_node_no_structured_output():
    """Structure node should handle missing structured_response gracefully."""
    from unittest.mock import AsyncMock, patch

    from src.agents.wiki_workflow import extract_structure_node

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
            "structured_response": None,
        }
        mock_create.return_value = mock_agent

        result = await extract_structure_node(state)

    assert result.get("error") is not None
    assert "did not return structured output" in result["error"]
    assert result["current_step"] == "error"


# =============================================================================
# Tests for workflow graph structure
# =============================================================================


def test_workflow_includes_aggregate_node():
    """Workflow should have 4 nodes: extract_structure, generate_single_page, aggregate, finalize."""
    from src.agents.wiki_workflow import create_wiki_workflow

    workflow = create_wiki_workflow()

    # Get node names from the graph (nodes is a dict keyed by node name)
    graph = workflow.get_graph()
    node_names = [
        name for name in graph.nodes.keys() if name not in ("__start__", "__end__")
    ]

    assert "extract_structure" in node_names
    assert "generate_single_page" in node_names
    assert "aggregate" in node_names
    assert "finalize" in node_names


# =============================================================================
# Tests for aggregate_node
# =============================================================================


@pytest.mark.asyncio
async def test_aggregate_node_merges_content_into_structure():
    """Aggregate node should merge page content back into structure."""
    from uuid import UUID

    from src.agents.wiki_workflow import aggregate_node
    from src.models.wiki import (
        PageImportance,
        WikiPageDetail,
        WikiSection,
        WikiStructure,
    )

    # Structure without content
    structure = WikiStructure(
        repository_id="12345678-1234-5678-1234-567812345678",
        title="Test Wiki",
        description="Test",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First",
                        importance=PageImportance.HIGH,
                        file_paths=[],
                        content="",  # Empty content, not None
                    ),
                ],
            )
        ],
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
        "clone_path": "/tmp/test",
        "file_tree": "",
        "readme_content": "",
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
    # mdformat adds trailing newline (CommonMark spec)
    assert all_pages[0].content.strip() == "# Page 1\n\nThis is the content."


@pytest.mark.asyncio
async def test_aggregate_node_with_existing_error():
    """Aggregate node should propagate existing errors."""
    from src.agents.wiki_workflow import aggregate_node

    state = {
        "repository_id": "test-repo",
        "clone_path": "/tmp/test",
        "file_tree": "",
        "readme_content": "",
        "structure": None,
        "pages": [],
        "error": "Previous error occurred",
        "current_step": "error",
    }

    result = await aggregate_node(state)

    assert result["current_step"] == "error"
    assert "Previous error" in result.get("error", "")


@pytest.mark.asyncio
async def test_aggregate_node_no_structure():
    """Aggregate node should handle missing structure gracefully."""
    from src.agents.wiki_workflow import aggregate_node

    state = {
        "repository_id": "test-repo",
        "clone_path": "/tmp/test",
        "file_tree": "",
        "readme_content": "",
        "structure": None,
        "pages": [],
        "error": None,
        "current_step": "pages_generated",
    }

    result = await aggregate_node(state)

    assert result["current_step"] == "error"
    assert "No structure to aggregate" in result.get("error", "")


@pytest.mark.asyncio
async def test_aggregate_node_multiple_pages():
    """Aggregate node should merge content for multiple pages across sections."""
    from src.agents.wiki_workflow import aggregate_node
    from src.models.wiki import (
        PageImportance,
        WikiPageDetail,
        WikiSection,
        WikiStructure,
    )

    # Structure with multiple sections and pages
    structure = WikiStructure(
        repository_id="12345678-1234-5678-1234-567812345678",
        title="Test Wiki",
        description="Test",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First",
                        importance=PageImportance.HIGH,
                    ),
                    WikiPageDetail(
                        id="page-2",
                        title="Page 2",
                        description="Second",
                        importance=PageImportance.MEDIUM,
                    ),
                ],
            ),
            WikiSection(
                id="section-2",
                title="Section 2",
                pages=[
                    WikiPageDetail(
                        id="page-3",
                        title="Page 3",
                        description="Third",
                        importance=PageImportance.LOW,
                    ),
                ],
            ),
        ],
    )

    # Pages with generated content
    pages_with_content = [
        WikiPageDetail(
            id="page-1",
            title="Page 1",
            description="First",
            importance=PageImportance.HIGH,
            content="# Page 1 Content",
        ),
        WikiPageDetail(
            id="page-2",
            title="Page 2",
            description="Second",
            importance=PageImportance.MEDIUM,
            content="# Page 2 Content",
        ),
        WikiPageDetail(
            id="page-3",
            title="Page 3",
            description="Third",
            importance=PageImportance.LOW,
            content="# Page 3 Content",
        ),
    ]

    state = {
        "repository_id": "12345678-1234-5678-1234-567812345678",
        "clone_path": "/tmp/test",
        "file_tree": "",
        "readme_content": "",
        "structure": structure,
        "pages": pages_with_content,
        "error": None,
        "current_step": "pages_generated",
    }

    result = await aggregate_node(state)

    assert result.get("current_step") == "aggregated"
    updated_structure = result.get("structure")
    assert updated_structure is not None

    # Verify all pages have content
    # mdformat adds trailing newline (CommonMark spec)
    all_pages = updated_structure.get_all_pages()
    assert len(all_pages) == 3
    assert all_pages[0].content.strip() == "# Page 1 Content"
    assert all_pages[1].content.strip() == "# Page 2 Content"
    assert all_pages[2].content.strip() == "# Page 3 Content"

    # Verify structure is preserved
    assert len(updated_structure.sections) == 2
    assert updated_structure.sections[0].id == "section-1"
    assert updated_structure.sections[1].id == "section-2"


# =============================================================================
# Tests for PageGenerationState TypedDict
# =============================================================================


def test_page_generation_state_structure():
    """PageGenerationState should have required fields for parallel page generation."""
    page = WikiPageDetail(
        id="test-page",
        title="Test Page",
        description="A test page",
        importance=PageImportance.HIGH,
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/repo",
        structure_description="A test wiki about the project",
        repository_id="00000000-0000-0000-0000-000000000000",
    )

    assert state["page"] is page
    assert state["clone_path"] == "/tmp/repo"
    assert state["structure_description"] == "A test wiki about the project"
    assert state["repository_id"] == "00000000-0000-0000-0000-000000000000"


def test_page_generation_state_with_file_paths():
    """PageGenerationState page field should support file_paths."""
    page = WikiPageDetail(
        id="api-reference",
        title="API Reference",
        description="API documentation",
        importance=PageImportance.HIGH,
        file_paths=["src/api/main.py", "src/api/routes.py"],
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/my-repo",
        structure_description="Project documentation",
        repository_id="12345678-1234-5678-1234-567812345678",
    )

    assert len(state["page"].file_paths) == 2
    assert "src/api/main.py" in state["page"].file_paths


# =============================================================================
# Tests for fan_out_to_page_workers
# =============================================================================


def test_fan_out_to_page_workers_returns_send_list():
    """fan_out_to_page_workers should return List[Send] for each page."""
    from uuid import uuid4

    structure = WikiStructure(
        repository_id=str(uuid4()),
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First page",
                        importance=PageImportance.HIGH,
                    ),
                    WikiPageDetail(
                        id="page-2",
                        title="Page 2",
                        description="Second page",
                        importance=PageImportance.MEDIUM,
                    ),
                ],
            )
        ],
    )

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = fan_out_to_page_workers(state)

    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(send, Send) for send in result)

    # Check Send node name
    assert result[0].node == "generate_single_page"
    assert result[1].node == "generate_single_page"


def test_fan_out_to_page_workers_send_contains_correct_state():
    """fan_out_to_page_workers Send objects should contain correct PageGenerationState."""
    from uuid import uuid4

    page = WikiPageDetail(
        id="getting-started",
        title="Getting Started",
        description="How to get started",
        importance=PageImportance.HIGH,
        file_paths=["README.md"],
    )

    structure = WikiStructure(
        repository_id=str(uuid4()),
        title="Test Wiki",
        description="A test project wiki",
        sections=[WikiSection(id="intro", title="Introduction", pages=[page])],
    )

    state = WikiWorkflowState(
        repository_id="12345678-1234-5678-1234-567812345678",
        clone_path="/tmp/my-project",
        file_tree="src/",
        readme_content="# My Project",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = fan_out_to_page_workers(state)

    assert len(result) == 1
    send_arg = result[0].arg

    assert send_arg["page"].id == "getting-started"
    assert send_arg["clone_path"] == "/tmp/my-project"
    assert send_arg["structure_description"] == "A test project wiki"
    assert send_arg["repository_id"] == "12345678-1234-5678-1234-567812345678"


def test_fan_out_to_page_workers_returns_empty_on_error():
    """fan_out_to_page_workers should return empty list when error is present."""
    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error="Previous error occurred",
        current_step="error",
        force_regenerate=False,
    )

    result = fan_out_to_page_workers(state)

    assert result == []


def test_fan_out_to_page_workers_returns_empty_no_structure():
    """fan_out_to_page_workers should return empty list when structure is None."""
    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = fan_out_to_page_workers(state)

    assert result == []


def test_fan_out_multiple_sections():
    """fan_out_to_page_workers should handle pages across multiple sections."""
    from uuid import uuid4

    structure = WikiStructure(
        repository_id=str(uuid4()),
        title="Multi-Section Wiki",
        description="Wiki with multiple sections",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="p1",
                        title="P1",
                        description="Page 1",
                        importance=PageImportance.HIGH,
                    ),
                ],
            ),
            WikiSection(
                id="section-2",
                title="Section 2",
                pages=[
                    WikiPageDetail(
                        id="p2",
                        title="P2",
                        description="Page 2",
                        importance=PageImportance.MEDIUM,
                    ),
                    WikiPageDetail(
                        id="p3",
                        title="P3",
                        description="Page 3",
                        importance=PageImportance.LOW,
                    ),
                ],
            ),
        ],
    )

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = fan_out_to_page_workers(state)

    assert len(result) == 3
    page_ids = [send.arg["page"].id for send in result]
    assert "p1" in page_ids
    assert "p2" in page_ids
    assert "p3" in page_ids


# =============================================================================
# Tests for generate_single_page_node
# =============================================================================


@pytest.mark.asyncio
async def test_generate_single_page_node_success():
    """generate_single_page_node should return dict with pages list."""
    from unittest.mock import AsyncMock, patch

    page = WikiPageDetail(
        id="test-page",
        title="Test Page",
        description="A test page",
        importance=PageImportance.HIGH,
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/repo",
        structure_description="Test wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
    )

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        # Mock structured_response with a content attribute (PageContentResponse)
        from types import SimpleNamespace

        mock_structured = SimpleNamespace(content="# Test Page\n\nThis is the content.")
        mock_agent.ainvoke.return_value = {
            "messages": [],
            "structured_response": mock_structured,
        }
        mock_create.return_value = mock_agent

        result = await generate_single_page_node(state)

    assert "pages" in result
    assert isinstance(result["pages"], list)
    assert len(result["pages"]) == 1
    assert result["pages"][0].id == "test-page"
    assert result["pages"][0].content == "# Test Page\n\nThis is the content."


@pytest.mark.asyncio
async def test_generate_single_page_node_creates_fresh_agent():
    """generate_single_page_node should create a fresh agent for each call."""
    from unittest.mock import AsyncMock, patch

    page = WikiPageDetail(
        id="page-1",
        title="Page 1",
        description="First page",
        importance=PageImportance.MEDIUM,
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/repo",
        structure_description="Wiki description",
        repository_id="00000000-0000-0000-0000-000000000000",
    )

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        # Mock structured_response with a content attribute (PageContentResponse)
        from types import SimpleNamespace

        mock_structured = SimpleNamespace(content="Content")
        mock_agent.ainvoke.return_value = {"structured_response": mock_structured}
        mock_create.return_value = mock_agent

        await generate_single_page_node(state)

    # Verify create_page_agent was called (fresh agent per invocation)
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_generate_single_page_node_handles_error():
    """generate_single_page_node should handle agent errors gracefully."""
    from unittest.mock import AsyncMock, patch

    page = WikiPageDetail(
        id="error-page",
        title="Error Page",
        description="Page that will error",
        importance=PageImportance.LOW,
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/repo",
        structure_description="Test wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
    )

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = Exception("LLM rate limit exceeded")
        mock_create.return_value = mock_agent

        result = await generate_single_page_node(state)

    # Should still return a page, but with error content
    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["pages"][0].id == "error-page"
    assert "Error generating content" in result["pages"][0].content
    assert "rate limit" in result["pages"][0].content.lower()


@pytest.mark.asyncio
async def test_generate_single_page_node_no_content():
    """generate_single_page_node should handle empty content response."""
    from unittest.mock import AsyncMock, patch

    page = WikiPageDetail(
        id="empty-page",
        title="Empty Page",
        description="Page with no content",
        importance=PageImportance.LOW,
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/repo",
        structure_description="Test wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
    )

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()
        # Mock structured_response with empty content (PageContentResponse)
        from types import SimpleNamespace

        mock_structured = SimpleNamespace(content="")  # Empty content
        mock_agent.ainvoke.return_value = {
            "messages": [],
            "structured_response": mock_structured,
        }
        mock_create.return_value = mock_agent

        result = await generate_single_page_node(state)

    assert "pages" in result
    assert len(result["pages"]) == 1
    assert result["pages"][0].content == ""


@pytest.mark.asyncio
async def test_generate_single_page_node_with_file_paths():
    """generate_single_page_node should pass file_paths to agent prompt."""
    from unittest.mock import AsyncMock, patch

    page = WikiPageDetail(
        id="api-page",
        title="API Reference",
        description="API documentation",
        importance=PageImportance.HIGH,
        file_paths=["src/api/main.py", "src/api/routes.py"],
    )

    state = PageGenerationState(
        page=page,
        clone_path="/tmp/my-project",
        structure_description="Project wiki",
        repository_id="00000000-0000-0000-0000-000000000000",
    )

    captured_message = None

    with patch("src.agents.wiki_workflow.create_page_agent") as mock_create:
        mock_agent = AsyncMock()

        async def capture_invoke(invoke_arg):
            nonlocal captured_message
            captured_message = invoke_arg["messages"][0]["content"]
            # Return structured_response with content attribute (PageContentResponse)
            from types import SimpleNamespace

            mock_structured = SimpleNamespace(content="# API Reference")
            return {"structured_response": mock_structured}

        mock_agent.ainvoke.side_effect = capture_invoke
        mock_create.return_value = mock_agent

        await generate_single_page_node(state)

    # Verify file paths are in the prompt
    assert "/tmp/my-project/src/api/main.py" in captured_message
    assert "/tmp/my-project/src/api/routes.py" in captured_message


# =============================================================================
# Tests for should_fan_out
# =============================================================================


def test_should_fan_out_returns_aggregate_on_error():
    """should_fan_out should return 'aggregate' when error exists."""
    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error="Previous error",
        current_step="error",
        force_regenerate=False,
    )

    result = should_fan_out(state)

    assert result == "aggregate"


def test_should_fan_out_returns_aggregate_no_structure():
    """should_fan_out should return 'aggregate' when structure is None."""
    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=None,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = should_fan_out(state)

    assert result == "aggregate"


def test_should_fan_out_returns_send_list():
    """should_fan_out should return List[Send] when structure exists."""
    from uuid import uuid4

    structure = WikiStructure(
        repository_id=str(uuid4()),
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="page-1",
                        title="Page 1",
                        description="First",
                        importance=PageImportance.HIGH,
                    ),
                ],
            )
        ],
    )

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = should_fan_out(state)

    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], Send)
    assert result[0].node == "generate_single_page"


def test_should_fan_out_multiple_pages():
    """should_fan_out should return one Send per page."""
    from uuid import uuid4

    structure = WikiStructure(
        repository_id=str(uuid4()),
        title="Test Wiki",
        description="Test description",
        sections=[
            WikiSection(
                id="section-1",
                title="Section 1",
                pages=[
                    WikiPageDetail(
                        id="p1",
                        title="P1",
                        description="Page 1",
                        importance=PageImportance.HIGH,
                    ),
                    WikiPageDetail(
                        id="p2",
                        title="P2",
                        description="Page 2",
                        importance=PageImportance.MEDIUM,
                    ),
                ],
            ),
            WikiSection(
                id="section-2",
                title="Section 2",
                pages=[
                    WikiPageDetail(
                        id="p3",
                        title="P3",
                        description="Page 3",
                        importance=PageImportance.LOW,
                    ),
                ],
            ),
        ],
    )

    state = WikiWorkflowState(
        repository_id="00000000-0000-0000-0000-000000000000",
        clone_path="/tmp/repo",
        file_tree="src/",
        readme_content="# Test",
        structure=structure,
        pages=[],
        error=None,
        current_step="structure_extracted",
        force_regenerate=False,
    )

    result = should_fan_out(state)

    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(send, Send) for send in result)


# =============================================================================
# Tests for workflow graph includes new nodes
# =============================================================================


def test_workflow_includes_fan_out_nodes():
    """Workflow should include generate_single_page for fan-out pattern."""
    from src.agents.wiki_workflow import create_wiki_workflow

    workflow = create_wiki_workflow()
    graph = workflow.get_graph()
    node_names = [
        name for name in graph.nodes.keys() if name not in ("__start__", "__end__")
    ]

    # Must have the worker node for fan-out
    assert "generate_single_page" in node_names

    # Should NOT have old sequential generate_pages
    assert "generate_pages" not in node_names


def test_workflow_has_correct_edge_from_extract_structure():
    """Workflow extract_structure should have conditional edge for fan-out."""
    from src.agents.wiki_workflow import create_wiki_workflow

    workflow = create_wiki_workflow()
    graph = workflow.get_graph()

    # Find edges from extract_structure
    extract_node = graph.nodes.get("extract_structure")
    assert extract_node is not None

    # The graph should have edges that allow routing to generate_single_page or aggregate
    # This verifies the conditional edge setup
    edges = graph.edges
    edge_sources = [edge.source for edge in edges]
    assert "extract_structure" in edge_sources
