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
    """extract_structure_node should return structure from LLM."""
    from unittest.mock import AsyncMock, patch
    from src.agents.wiki_workflow import extract_structure_node, LLMWikiStructureSchema
    from src.models.wiki import WikiSection, PageImportance

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

    # Mock the LLM output using the LLMWikiStructureSchema format
    mock_llm_output = {
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

    with patch("src.agents.wiki_workflow.LLMTool") as MockLLMTool:
        mock_llm = AsyncMock()
        mock_llm.generate_structured.return_value = {
            "status": "success",
            "structured_output": mock_llm_output
        }
        MockLLMTool.return_value = mock_llm

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
    """extract_structure_node should handle LLM errors gracefully."""
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
