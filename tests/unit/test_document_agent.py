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


class TestBuildFileTree:
    """Tests for _build_file_tree helper"""

    def test_builds_simple_tree(self):
        """Should build ASCII tree from directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "src" / "main.py").touch()
            (Path(tmpdir) / "README.md").touch()

            agent = self._create_mock_agent()
            tree = agent._build_file_tree(tmpdir, [], [])

            assert "src/" in tree
            assert "main.py" in tree
            assert "README.md" in tree

    def test_excludes_directories(self):
        """Should exclude specified directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "node_modules").mkdir()
            (Path(tmpdir) / "src" / "main.py").touch()
            (Path(tmpdir) / "node_modules" / "pkg.js").touch()

            agent = self._create_mock_agent()
            tree = agent._build_file_tree(tmpdir, ["node_modules/"], [])

            assert "src/" in tree
            assert "node_modules" not in tree

    def test_excludes_files(self):
        """Should exclude specified files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "main.py").touch()
            (Path(tmpdir) / "test.min.js").touch()

            agent = self._create_mock_agent()
            tree = agent._build_file_tree(tmpdir, [], ["*.min.js"])

            assert "main.py" in tree
            assert "test.min.js" not in tree

    def _create_mock_agent(self):
        mock_repo_tool = MagicMock()
        mock_repo_repo = MagicMock()
        with patch.object(DocumentProcessingAgent, '_create_workflow', return_value=MagicMock()):
            return DocumentProcessingAgent(mock_repo_tool, mock_repo_repo)


class TestExtractDocsNode:
    """Tests for _extract_docs_node"""

    def test_extracts_readme(self):
        """Should extract README.md content"""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            readme_content = "# Test Project\n\nThis is a test."
            (Path(tmpdir) / "README.md").write_text(readme_content)

            agent = self._create_mock_agent()
            state = self._create_state_with_clone_path(tmpdir)

            result = asyncio.run(agent._extract_docs_node(state))

            assert len(result["documentation_files"]) == 1
            assert result["documentation_files"][0]["path"] == "README.md"
            assert result["documentation_files"][0]["content"] == readme_content

    def test_extracts_multiple_docs(self):
        """Should extract multiple doc files"""
        import asyncio

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("# README")
            (Path(tmpdir) / "CLAUDE.md").write_text("# Claude")
            (Path(tmpdir) / "docs").mkdir()
            (Path(tmpdir) / "docs" / "guide.md").write_text("# Guide")

            agent = self._create_mock_agent()
            state = self._create_state_with_clone_path(tmpdir)

            result = asyncio.run(agent._extract_docs_node(state))

            paths = [d["path"] for d in result["documentation_files"]]
            assert "README.md" in paths
            assert "CLAUDE.md" in paths
            assert "docs/guide.md" in paths or "docs\\guide.md" in paths

    def _create_mock_agent(self):
        mock_repo_tool = MagicMock()
        mock_repo_repo = MagicMock()
        with patch.object(DocumentProcessingAgent, '_create_workflow', return_value=MagicMock()):
            return DocumentProcessingAgent(mock_repo_tool, mock_repo_repo)

    def _create_state_with_clone_path(self, clone_path: str) -> DocumentProcessingState:
        return {
            "repository_id": "test-id",
            "repository_url": "https://github.com/test/repo",
            "branch": "main",
            "clone_path": clone_path,
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


class TestDocumentAgentIntegration:
    """Integration tests for DocumentProcessingAgent"""

    def test_full_workflow_with_temp_repo(self):
        """Test full workflow with a temporary repository structure"""
        import asyncio
        asyncio.run(self._test_full_workflow_with_temp_repo())

    async def _test_full_workflow_with_temp_repo(self):
        """Actual async test implementation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock repo structure
            (Path(tmpdir) / "src").mkdir()
            (Path(tmpdir) / "src" / "main.py").write_text("print('hello')")
            (Path(tmpdir) / "node_modules").mkdir()
            (Path(tmpdir) / "node_modules" / "pkg.js").write_text("// pkg")
            (Path(tmpdir) / "README.md").write_text("# Test Project")
            (Path(tmpdir) / "CLAUDE.md").write_text("# Instructions for Claude")
            (Path(tmpdir) / ".git").mkdir()

            # Create agent with mocks that simulate successful clone
            mock_repo_tool = MagicMock()
            mock_repo_tool._arun = AsyncMock(return_value={
                "status": "success",
                "clone_path": tmpdir
            })

            mock_repo_repo = MagicMock()
            mock_repo_repo.get_by_id = AsyncMock(return_value=MagicMock())
            mock_repo_repo.update = AsyncMock()

            agent = DocumentProcessingAgent(mock_repo_tool, mock_repo_repo)

            result = await agent.process_repository(
                repository_id="test-123",
                repository_url="https://github.com/test/repo",
                branch="main"
            )

            # Verify result structure
            assert result["status"] == "success"
            assert result["clone_path"] == tmpdir

            # Verify tree excludes node_modules and .git
            assert "node_modules" not in result["file_tree"]
            assert ".git" not in result["file_tree"]
            assert "src/" in result["file_tree"]
            assert "main.py" in result["file_tree"]

            # Verify docs were extracted
            doc_paths = [d["path"] for d in result["documentation_files"]]
            assert "README.md" in doc_paths
            assert "CLAUDE.md" in doc_paths
