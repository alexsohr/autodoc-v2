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
