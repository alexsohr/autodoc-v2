"""Unit tests for WorkflowOrchestrator."""

import pytest

from src.agents.workflow import WorkflowOrchestrator


def test_format_documentation_files_single_file():
    """Test formatting a single documentation file."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    doc_files = [{"path": "README.md", "content": "# Hello World"}]

    result = orchestrator._format_documentation_files(doc_files)

    assert "--- README.md ---" in result
    assert "# Hello World" in result


def test_format_documentation_files_multiple_files():
    """Test formatting multiple documentation files."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    doc_files = [
        {"path": "README.md", "content": "# Project"},
        {"path": "docs/API.md", "content": "# API Reference"},
    ]

    result = orchestrator._format_documentation_files(doc_files)

    assert "--- README.md ---" in result
    assert "# Project" in result
    assert "--- docs/API.md ---" in result
    assert "# API Reference" in result


def test_format_documentation_files_empty_list():
    """Test formatting empty documentation files list."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)

    result = orchestrator._format_documentation_files([])

    assert result == ""


def test_format_documentation_files_missing_keys():
    """Test formatting with missing path or content keys."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    doc_files = [
        {"path": "README.md"},  # Missing content
        {"content": "# Orphan content"},  # Missing path
        {},  # Both missing
    ]

    result = orchestrator._format_documentation_files(doc_files)

    assert "--- README.md ---" in result
    assert "--- unknown ---" in result
    assert "# Orphan content" in result
