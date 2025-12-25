"""Tests for DocumentProcessingAgent"""

import json
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


class TestLoadPatternsNode:
    """Tests for _load_patterns_node"""

    def test_loads_default_patterns(self):
        """Should load patterns from config when no autodoc.json exists"""
        import asyncio

        agent = self._create_mock_agent()
        state = self._create_initial_state()
        state["clone_path"] = "/tmp/nonexistent"

        result = asyncio.run(agent._load_patterns_node(state))

        assert len(result["excluded_dirs"]) > 0
        assert len(result["excluded_files"]) > 0
        assert ".git/" in result["excluded_dirs"]

    def test_overrides_from_autodoc_json(self):
        """Should override patterns from .autodoc/autodoc.json"""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create .autodoc/autodoc.json
            autodoc_dir = Path(tmpdir) / ".autodoc"
            autodoc_dir.mkdir()
            config = {
                "excluded_dirs": ["custom_dir/"],
                "excluded_files": ["custom.txt"]
            }
            (autodoc_dir / "autodoc.json").write_text(json.dumps(config))

            agent = self._create_mock_agent()
            state = self._create_initial_state()
            state["clone_path"] = tmpdir

            result = asyncio.run(agent._load_patterns_node(state))

            assert result["excluded_dirs"] == ["custom_dir/"]
            assert result["excluded_files"] == ["custom.txt"]

    def _create_mock_agent(self):
        mock_repo_tool = MagicMock()
        mock_repo_repo = MagicMock()
        with patch.object(DocumentProcessingAgent, '_create_workflow', return_value=MagicMock()):
            return DocumentProcessingAgent(mock_repo_tool, mock_repo_repo)

    def _create_initial_state(self) -> DocumentProcessingState:
        return {
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
