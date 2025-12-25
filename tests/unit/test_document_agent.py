"""Tests for DocumentProcessingAgent"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import os

from src.agents.document_agent import (
    DocumentProcessingAgent,
    DocumentProcessingState,
    DOC_FILE_PATTERNS,
)


class TestDocumentProcessingState:
    """Tests for DocumentProcessingState structure"""

    def test_state_has_required_fields(self):
        """State should have all required fields"""
        state: DocumentProcessingState = {
            "repository_id": "test-id",
            "repository_url": "https://github.com/test/repo",
            "branch": "main",
            "clone_path": None,
            "documentation_files": [],
            "file_tree": "",
            "excluded_dirs": [],
            "excluded_files": [],
            "current_step": "init",
            "error_message": None,
            "progress": 0.0,
            "start_time": "2025-01-01T00:00:00Z",
            "messages": [],
        }
        assert state["repository_id"] == "test-id"
        assert state["documentation_files"] == []
        assert state["file_tree"] == ""


class TestDocFilePatterns:
    """Tests for DOC_FILE_PATTERNS constant"""

    def test_includes_readme(self):
        """Should include README.md pattern"""
        assert "README.md" in DOC_FILE_PATTERNS

    def test_includes_claude_md(self):
        """Should include CLAUDE.md pattern"""
        assert "CLAUDE.md" in DOC_FILE_PATTERNS

    def test_includes_docs_folder(self):
        """Should include docs folder pattern"""
        assert "docs/**/*.md" in DOC_FILE_PATTERNS


class TestPatternMatching:
    """Tests for pattern matching helpers"""

    def test_matches_simple_filename(self):
        """Should match exact filename"""
        agent = self._create_mock_agent()
        assert agent._matches_pattern("README.md", "README.md") is True
        assert agent._matches_pattern("README.md", "OTHER.md") is False

    def test_matches_wildcard_pattern(self):
        """Should match wildcard patterns"""
        agent = self._create_mock_agent()
        assert agent._matches_pattern("test.min.js", "*.min.js") is True
        assert agent._matches_pattern("test.js", "*.min.js") is False

    def test_matches_glob_pattern(self):
        """Should match glob patterns"""
        agent = self._create_mock_agent()
        assert agent._matches_pattern("docs/api/guide.md", "docs/**/*.md") is True
        assert agent._matches_pattern("src/main.py", "docs/**/*.md") is False

    def _create_mock_agent(self):
        """Create agent with mocked dependencies"""
        mock_repo_tool = MagicMock()
        mock_repo_repo = MagicMock()
        with patch.object(DocumentProcessingAgent, '_create_workflow', return_value=MagicMock()):
            return DocumentProcessingAgent(mock_repo_tool, mock_repo_repo)
