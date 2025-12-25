# Document Agent Simplification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Simplify DocumentProcessingAgent to output only documentation files content and ASCII file tree structure.

**Architecture:** Replace the current 7-node workflow with a 4-node workflow that skips embedding generation and MongoDB storage. Output is passed in-memory to Wiki Agent.

**Tech Stack:** Python 3.12, LangGraph, fnmatch for pattern matching, pathlib for file operations.

---

## Task 1: Add Exclusion Pattern Defaults to Config

**Files:**
- Modify: `src/utils/config_loader.py:147` (after `supported_languages`)

**Step 1: Add excluded_dirs field to Settings class**

Add after line 163 (after the `supported_languages` field block):

```python
    # Document agent file filtering settings
    default_excluded_dirs: List[str] = Field(
        default=[
            ".venv/", "venv/", "env/", "virtualenv/",
            "node_modules/", "bower_components/", "jspm_packages/",
            ".git/", ".svn/", ".hg/", ".bzr/",
            ".idea/", ".vscode/", ".vscode-server/", ".vscode-server-insiders/",
            ".pytest_cache/", ".pytest/", ".next/",
        ],
        description="Default directories to exclude from file tree",
    )
```

**Step 2: Add excluded_files field to Settings class**

Add immediately after `default_excluded_dirs`:

```python
    default_excluded_files: List[str] = Field(
        default=[
            # Lock files
            "yarn.lock", "pnpm-lock.yaml", "npm-shrinkwrap.json", "poetry.lock",
            "Pipfile.lock", "requirements.txt.lock", "Cargo.lock", "composer.lock",
            ".lock",
            # OS files
            ".DS_Store", "Thumbs.db", "desktop.ini", "*.lnk",
            # Environment files
            ".env", ".env.*", "*.env", "*.cfg", "*.ini", ".flaskenv",
            # Git/CI files
            ".gitignore", ".gitattributes", ".gitmodules", ".github",
            ".gitlab-ci.yml",
            # Linter/formatter configs
            ".prettierrc", ".eslintrc", ".eslintignore", ".stylelintrc",
            ".editorconfig", ".jshintrc", ".pylintrc", ".flake8",
            "mypy.ini", "pyproject.toml", "tsconfig.json",
            # Build configs
            "webpack.config.js", "babel.config.js", "rollup.config.js",
            "jest.config.js", "karma.conf.js", "vite.config.js", "next.config.js",
            # Minified/bundled files
            "*.min.js", "*.min.css", "*.bundle.js", "*.bundle.css", "*.map",
            # Archives
            "*.gz", "*.zip", "*.tar", "*.tgz", "*.rar", "*.7z",
            "*.iso", "*.dmg", "*.img",
            # Installers/packages
            "*.msix", "*.appx", "*.appxbundle", "*.xap", "*.ipa",
            "*.deb", "*.rpm", "*.msi",
            # Binaries
            "*.exe", "*.dll", "*.so", "*.dylib", "*.o", "*.obj",
            "*.jar", "*.war", "*.ear", "*.jsm", "*.class",
            # Python compiled
            "*.pyc", "*.pyd", "*.pyo", "__pycache__",
        ],
        description="Default files to exclude from file tree",
    )
```

**Step 3: Verify the file still imports correctly**

Run: `python -c "from src.utils.config_loader import get_settings; s = get_settings(); print(len(s.default_excluded_dirs), len(s.default_excluded_files))"`

Expected: `11 55` (approximate counts)

**Step 4: Commit**

```bash
git add src/utils/config_loader.py
git commit -m "feat(config): add default exclusion patterns for document agent"
```

---

## Task 2: Create Test File for Document Agent

**Files:**
- Create: `tests/unit/test_document_agent.py`

**Step 1: Create basic test file structure**

```python
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
```

**Step 2: Run test to verify structure**

Run: `pytest tests/unit/test_document_agent.py -v`

Expected: FAIL (module not yet updated)

**Step 3: Commit test file**

```bash
git add tests/unit/test_document_agent.py
git commit -m "test(document_agent): add initial test structure"
```

---

## Task 3: Update DocumentProcessingState TypedDict

**Files:**
- Modify: `src/agents/document_agent.py:27-42`

**Step 1: Replace DocumentProcessingState definition**

Replace lines 27-42 with:

```python
class DocumentProcessingState(TypedDict):
    """State for document processing workflow"""

    repository_id: str
    repository_url: str
    branch: Optional[str]
    clone_path: Optional[str]

    # New simplified outputs
    documentation_files: List[Dict[str, str]]  # [{path, content}, ...]
    file_tree: str  # ASCII tree structure

    # Patterns (loaded from config, possibly overridden by .autodoc/autodoc.json)
    excluded_dirs: List[str]
    excluded_files: List[str]

    # Workflow tracking
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
```

**Step 2: Run state structure test**

Run: `pytest tests/unit/test_document_agent.py::TestDocumentProcessingState -v`

Expected: PASS

**Step 3: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "refactor(document_agent): update state to simplified structure"
```

---

## Task 4: Add DOC_FILE_PATTERNS Constant and Imports

**Files:**
- Modify: `src/agents/document_agent.py:7-24`

**Step 1: Update imports**

Replace lines 7-24 with:

```python
import fnmatch
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..models.repository import AnalysisStatus
from ..repository.repository_repository import RepositoryRepository
from ..tools.repository_tool import RepositoryTool
from ..utils.config_loader import get_settings

logger = logging.getLogger(__name__)

# Hardcoded patterns for documentation files to extract
DOC_FILE_PATTERNS = [
    # AI assistant instructions
    "CLAUDE.md",
    "claude.md",
    ".claude/CLAUDE.md",
    "agent.md",
    "AGENT.md",
    "llm.txt",
    "LLM.txt",
    "copilot-instructions.md",
    ".github/copilot-instructions.md",
    # Standard project docs
    "README.md",
    "README.txt",
    "README",
    "readme.md",
    "CONTRIBUTING.md",
    "ARCHITECTURE.md",
    "CHANGELOG.md",
    "CODEOWNERS",
    ".github/CODEOWNERS",
    # Docs folder (recursive)
    "docs/**/*.md",
    "doc/**/*.md",
]
```

**Step 2: Run pattern test**

Run: `pytest tests/unit/test_document_agent.py::TestDocFilePatterns -v`

Expected: PASS

**Step 3: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "feat(document_agent): add DOC_FILE_PATTERNS constant"
```

---

## Task 5: Simplify __init__ Method

**Files:**
- Modify: `src/agents/document_agent.py` (the `__init__` method)

**Step 1: Replace __init__ method**

Replace the `__init__` method (lines 51-74) with:

```python
    def __init__(
        self,
        repository_tool: RepositoryTool,
        repository_repo: RepositoryRepository,
    ):
        """Initialize document processing agent with dependency injection.

        Args:
            repository_tool: RepositoryTool instance for cloning repos.
            repository_repo: RepositoryRepository instance for status updates.
        """
        self.settings = get_settings()
        self._repository_tool = repository_tool
        self._repository_repo = repository_repo
        self.workflow = self._create_workflow()
```

**Step 2: Verify import still works**

Run: `python -c "from src.agents.document_agent import DocumentProcessingAgent; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "refactor(document_agent): simplify __init__ dependencies"
```

---

## Task 6: Add Helper Methods for Pattern Matching

**Files:**
- Modify: `src/agents/document_agent.py`
- Test: `tests/unit/test_document_agent.py`

**Step 1: Add test for _matches_pattern helper**

Add to `tests/unit/test_document_agent.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_document_agent.py::TestPatternMatching -v`

Expected: FAIL (method not implemented)

**Step 3: Add _matches_pattern method to DocumentProcessingAgent**

Add after `__init__`:

```python
    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a pattern.

        Supports:
        - Exact matches: "README.md"
        - Wildcards: "*.min.js"
        - Glob patterns: "docs/**/*.md"
        """
        # Normalize path separators
        path = path.replace("\\", "/")
        pattern = pattern.replace("\\", "/")

        # Handle ** glob patterns
        if "**" in pattern:
            # Convert glob pattern to regex-like matching
            parts = pattern.split("**")
            if len(parts) == 2:
                prefix, suffix = parts
                # Check if path starts with prefix (if any) and ends with suffix pattern
                if prefix and not path.startswith(prefix.rstrip("/")):
                    return False
                if suffix:
                    suffix = suffix.lstrip("/")
                    # Get the remaining path after prefix
                    remaining = path[len(prefix.rstrip("/")):].lstrip("/") if prefix else path
                    return fnmatch.fnmatch(remaining, f"*{suffix}") or fnmatch.fnmatch(path, f"*{suffix}")
                return True

        # Handle simple wildcard patterns
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_document_agent.py::TestPatternMatching -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/document_agent.py tests/unit/test_document_agent.py
git commit -m "feat(document_agent): add _matches_pattern helper method"
```

---

## Task 7: Add _load_patterns_node Method

**Files:**
- Modify: `src/agents/document_agent.py`
- Test: `tests/unit/test_document_agent.py`

**Step 1: Add test for load_patterns_node**

Add to `tests/unit/test_document_agent.py`:

```python
class TestLoadPatternsNode:
    """Tests for _load_patterns_node"""

    @pytest.mark.asyncio
    async def test_loads_default_patterns(self):
        """Should load patterns from config when no autodoc.json exists"""
        agent = self._create_mock_agent()
        state = self._create_initial_state()
        state["clone_path"] = "/tmp/nonexistent"

        result = await agent._load_patterns_node(state)

        assert len(result["excluded_dirs"]) > 0
        assert len(result["excluded_files"]) > 0
        assert ".git/" in result["excluded_dirs"]

    @pytest.mark.asyncio
    async def test_overrides_from_autodoc_json(self):
        """Should override patterns from .autodoc/autodoc.json"""
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

            result = await agent._load_patterns_node(state)

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_document_agent.py::TestLoadPatternsNode -v`

Expected: FAIL

**Step 3: Add _load_patterns_node method**

Add to DocumentProcessingAgent class:

```python
    async def _load_patterns_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Load exclusion patterns from config and optional .autodoc/autodoc.json override."""
        try:
            state["current_step"] = "loading_patterns"
            state["progress"] = 25.0

            # Start with defaults from config
            excluded_dirs = list(self.settings.default_excluded_dirs)
            excluded_files = list(self.settings.default_excluded_files)

            # Check for .autodoc/autodoc.json override
            if state["clone_path"]:
                autodoc_config_path = Path(state["clone_path"]) / ".autodoc" / "autodoc.json"
                if autodoc_config_path.exists():
                    try:
                        with open(autodoc_config_path, "r", encoding="utf-8") as f:
                            autodoc_config = json.load(f)

                        # Override with values from autodoc.json if present
                        if "excluded_dirs" in autodoc_config:
                            excluded_dirs = autodoc_config["excluded_dirs"]
                            logger.info(f"Loaded excluded_dirs from .autodoc/autodoc.json")
                        if "excluded_files" in autodoc_config:
                            excluded_files = autodoc_config["excluded_files"]
                            logger.info(f"Loaded excluded_files from .autodoc/autodoc.json")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid .autodoc/autodoc.json: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to read .autodoc/autodoc.json: {e}")

            state["excluded_dirs"] = excluded_dirs
            state["excluded_files"] = excluded_files

            state["messages"].append(
                AIMessage(content=f"Loaded {len(excluded_dirs)} dir exclusions and {len(excluded_files)} file exclusions")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Load patterns node failed: {str(e)}"
            return state
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_document_agent.py::TestLoadPatternsNode -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/document_agent.py tests/unit/test_document_agent.py
git commit -m "feat(document_agent): add _load_patterns_node method"
```

---

## Task 8: Add _build_file_tree Helper Method

**Files:**
- Modify: `src/agents/document_agent.py`
- Test: `tests/unit/test_document_agent.py`

**Step 1: Add test for _build_file_tree**

Add to `tests/unit/test_document_agent.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_document_agent.py::TestBuildFileTree -v`

Expected: FAIL

**Step 3: Add _build_file_tree method**

Add to DocumentProcessingAgent class:

```python
    def _build_file_tree(
        self, root_path: str, excluded_dirs: List[str], excluded_files: List[str]
    ) -> str:
        """Build ASCII tree representation of directory structure.

        Args:
            root_path: Root directory to build tree from.
            excluded_dirs: List of directory patterns to exclude.
            excluded_files: List of file patterns to exclude.

        Returns:
            ASCII tree string.
        """
        lines = []
        root = Path(root_path)

        def should_exclude_dir(dir_path: Path) -> bool:
            rel_path = str(dir_path.relative_to(root)).replace("\\", "/") + "/"
            dir_name = dir_path.name + "/"
            for pattern in excluded_dirs:
                pattern = pattern.replace("\\", "/")
                # Match against relative path or just directory name
                if self._matches_pattern(rel_path, pattern) or self._matches_pattern(dir_name, pattern):
                    return True
            return False

        def should_exclude_file(file_path: Path) -> bool:
            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
            file_name = file_path.name
            for pattern in excluded_files:
                if self._matches_pattern(rel_path, pattern) or self._matches_pattern(file_name, pattern):
                    return True
            return False

        def add_tree(path: Path, prefix: str = ""):
            entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))

            # Filter entries
            filtered = []
            for entry in entries:
                if entry.is_dir():
                    if not should_exclude_dir(entry):
                        filtered.append(entry)
                else:
                    if not should_exclude_file(entry):
                        filtered.append(entry)

            for i, entry in enumerate(filtered):
                is_last = i == len(filtered) - 1
                connector = "└── " if is_last else "├── "

                if entry.is_dir():
                    lines.append(f"{prefix}{connector}{entry.name}/")
                    extension = "    " if is_last else "│   "
                    add_tree(entry, prefix + extension)
                else:
                    lines.append(f"{prefix}{connector}{entry.name}")

        add_tree(root)
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_document_agent.py::TestBuildFileTree -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/document_agent.py tests/unit/test_document_agent.py
git commit -m "feat(document_agent): add _build_file_tree helper method"
```

---

## Task 9: Add _discover_and_build_tree_node Method

**Files:**
- Modify: `src/agents/document_agent.py`

**Step 1: Add _discover_and_build_tree_node method**

Add to DocumentProcessingAgent class:

```python
    async def _discover_and_build_tree_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Discover files and build ASCII tree structure."""
        try:
            state["current_step"] = "building_tree"
            state["progress"] = 40.0

            if not state["clone_path"]:
                state["error_message"] = "No clone path available for tree building"
                return state

            # Build the file tree
            file_tree = self._build_file_tree(
                state["clone_path"],
                state["excluded_dirs"],
                state["excluded_files"]
            )

            state["file_tree"] = file_tree
            state["progress"] = 50.0

            # Count lines for message
            line_count = len(file_tree.split("\n")) if file_tree else 0
            state["messages"].append(
                AIMessage(content=f"Built file tree with {line_count} entries")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Build tree node failed: {str(e)}"
            return state
```

**Step 2: Verify method is callable**

Run: `python -c "from src.agents.document_agent import DocumentProcessingAgent; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "feat(document_agent): add _discover_and_build_tree_node method"
```

---

## Task 10: Add _extract_docs_node Method

**Files:**
- Modify: `src/agents/document_agent.py`
- Test: `tests/unit/test_document_agent.py`

**Step 1: Add test for _extract_docs_node**

Add to `tests/unit/test_document_agent.py`:

```python
class TestExtractDocsNode:
    """Tests for _extract_docs_node"""

    @pytest.mark.asyncio
    async def test_extracts_readme(self):
        """Should extract README.md content"""
        with tempfile.TemporaryDirectory() as tmpdir:
            readme_content = "# Test Project\n\nThis is a test."
            (Path(tmpdir) / "README.md").write_text(readme_content)

            agent = self._create_mock_agent()
            state = self._create_state_with_clone_path(tmpdir)

            result = await agent._extract_docs_node(state)

            assert len(result["documentation_files"]) == 1
            assert result["documentation_files"][0]["path"] == "README.md"
            assert result["documentation_files"][0]["content"] == readme_content

    @pytest.mark.asyncio
    async def test_extracts_multiple_docs(self):
        """Should extract multiple doc files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "README.md").write_text("# README")
            (Path(tmpdir) / "CLAUDE.md").write_text("# Claude")
            (Path(tmpdir) / "docs").mkdir()
            (Path(tmpdir) / "docs" / "guide.md").write_text("# Guide")

            agent = self._create_mock_agent()
            state = self._create_state_with_clone_path(tmpdir)

            result = await agent._extract_docs_node(state)

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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_document_agent.py::TestExtractDocsNode -v`

Expected: FAIL

**Step 3: Add _extract_docs_node method**

Add to DocumentProcessingAgent class:

```python
    async def _extract_docs_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Extract documentation files content."""
        try:
            state["current_step"] = "extracting_docs"
            state["progress"] = 60.0

            if not state["clone_path"]:
                state["error_message"] = "No clone path available for doc extraction"
                return state

            root = Path(state["clone_path"])
            documentation_files = []

            for pattern in DOC_FILE_PATTERNS:
                if "**" in pattern:
                    # Glob pattern
                    for file_path in root.glob(pattern):
                        if file_path.is_file():
                            try:
                                content = file_path.read_text(encoding="utf-8")
                                rel_path = str(file_path.relative_to(root)).replace("\\", "/")
                                documentation_files.append({
                                    "path": rel_path,
                                    "content": content
                                })
                            except Exception as e:
                                logger.warning(f"Failed to read {file_path}: {e}")
                else:
                    # Exact file match
                    file_path = root / pattern
                    if file_path.is_file():
                        try:
                            content = file_path.read_text(encoding="utf-8")
                            rel_path = str(file_path.relative_to(root)).replace("\\", "/")
                            documentation_files.append({
                                "path": rel_path,
                                "content": content
                            })
                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {e}")

            # Deduplicate by path
            seen_paths = set()
            unique_docs = []
            for doc in documentation_files:
                if doc["path"] not in seen_paths:
                    seen_paths.add(doc["path"])
                    unique_docs.append(doc)

            state["documentation_files"] = unique_docs
            state["progress"] = 80.0

            state["messages"].append(
                AIMessage(content=f"Extracted {len(unique_docs)} documentation files")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Extract docs node failed: {str(e)}"
            return state
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_document_agent.py::TestExtractDocsNode -v`

Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/document_agent.py tests/unit/test_document_agent.py
git commit -m "feat(document_agent): add _extract_docs_node method"
```

---

## Task 11: Update _create_workflow Method

**Files:**
- Modify: `src/agents/document_agent.py`

**Step 1: Replace _create_workflow method**

Replace the existing `_create_workflow` method with:

```python
    def _create_workflow(self) -> StateGraph:
        """Create the document processing workflow graph.

        Returns:
            LangGraph StateGraph for document processing.
        """
        workflow = StateGraph(DocumentProcessingState)

        # Add nodes
        workflow.add_node("clone_repository", self._clone_repository_node)
        workflow.add_node("load_patterns", self._load_patterns_node)
        workflow.add_node("build_tree", self._discover_and_build_tree_node)
        workflow.add_node("extract_docs", self._extract_docs_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define workflow edges
        workflow.add_edge(START, "clone_repository")
        workflow.add_edge("clone_repository", "load_patterns")
        workflow.add_edge("load_patterns", "build_tree")
        workflow.add_edge("build_tree", "extract_docs")
        workflow.add_edge("extract_docs", END)

        # Error handling
        workflow.add_edge("handle_error", END)

        app = workflow.compile().with_config(
            {"run_name": "document_agent.document_processing_workflow"}
        )
        logger.debug(f"Document processing workflow:\n {app.get_graph().draw_mermaid()}")
        return app
```

**Step 2: Verify workflow compiles**

Run: `python -c "from src.agents.document_agent import DocumentProcessingAgent; from unittest.mock import MagicMock; a = DocumentProcessingAgent(MagicMock(), MagicMock()); print('Workflow created:', type(a.workflow))"`

Expected: `Workflow created: <class 'langgraph.graph.state.CompiledStateGraph'>` (or similar)

**Step 3: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "refactor(document_agent): update _create_workflow with new nodes"
```

---

## Task 12: Update process_repository Method

**Files:**
- Modify: `src/agents/document_agent.py`

**Step 1: Replace process_repository method**

Find and replace the `process_repository` method with:

```python
    async def process_repository(
        self,
        repository_id: str,
        repository_url: str,
        branch: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a repository and return documentation files + tree structure.

        Args:
            repository_id: Unique identifier for the repository.
            repository_url: URL of the repository to clone.
            branch: Optional branch to clone (defaults to default branch).

        Returns:
            Dict containing:
                - clone_path: Path to cloned repository
                - documentation_files: List of {path, content} dicts
                - file_tree: ASCII tree string
                - error_message: Error message if failed
        """
        initial_state: DocumentProcessingState = {
            "repository_id": repository_id,
            "repository_url": repository_url,
            "branch": branch,
            "clone_path": None,
            "documentation_files": [],
            "file_tree": "",
            "excluded_dirs": [],
            "excluded_files": [],
            "current_step": "initializing",
            "error_message": None,
            "progress": 0.0,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "messages": [HumanMessage(content=f"Processing repository: {repository_url}")],
        }

        try:
            # Update repository status to processing
            await self._update_repository_status(repository_id, AnalysisStatus.PROCESSING)

            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Check for errors
            if final_state.get("error_message"):
                await self._update_repository_status(repository_id, AnalysisStatus.FAILED)
                return {
                    "status": "failed",
                    "error": final_state["error_message"],
                    "clone_path": final_state.get("clone_path"),
                    "documentation_files": [],
                    "file_tree": "",
                }

            # Update status to completed
            await self._update_repository_status(repository_id, AnalysisStatus.COMPLETED)

            return {
                "status": "success",
                "clone_path": final_state["clone_path"],
                "documentation_files": final_state["documentation_files"],
                "file_tree": final_state["file_tree"],
            }

        except Exception as e:
            logger.error(f"Repository processing failed: {e}")
            await self._update_repository_status(repository_id, AnalysisStatus.FAILED)
            return {
                "status": "failed",
                "error": str(e),
                "clone_path": None,
                "documentation_files": [],
                "file_tree": "",
            }
```

**Step 2: Verify method signature**

Run: `python -c "from src.agents.document_agent import DocumentProcessingAgent; import inspect; print(inspect.signature(DocumentProcessingAgent.process_repository))"`

Expected: `(self, repository_id: str, repository_url: str, branch: Optional[str] = None) -> Dict[str, Any]`

**Step 3: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "refactor(document_agent): update process_repository for new output"
```

---

## Task 13: Remove Unused Methods and Clean Up

**Files:**
- Modify: `src/agents/document_agent.py`

**Step 1: Remove unused methods**

Delete the following methods from DocumentProcessingAgent:
- `_discover_files_node`
- `_discover_files_with_mcp`
- `_transform_mcp_tree_to_files`
- `_process_content_node`
- `_read_file_content`
- `_generate_embeddings_node`
- `_store_documents_node`
- `_cleanup_node`
- `_clean_content_for_embedding`
- `get_processing_status`

Keep these methods:
- `__init__`
- `_create_workflow`
- `process_repository`
- `_clone_repository_node`
- `_load_patterns_node`
- `_discover_and_build_tree_node`
- `_extract_docs_node`
- `_handle_error_node`
- `_update_repository_status`
- `_matches_pattern`
- `_build_file_tree`

**Step 2: Remove unused imports**

Remove from imports:
- `re` (if no longer used)
- `CodeDocument` model import
- `CodeDocumentRepository` import
- `MCPFilesystemClient` import
- `EmbeddingTool` import

**Step 3: Verify file is valid Python**

Run: `python -m py_compile src/agents/document_agent.py && echo "Syntax OK"`

Expected: `Syntax OK`

**Step 4: Run all document agent tests**

Run: `pytest tests/unit/test_document_agent.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/agents/document_agent.py
git commit -m "refactor(document_agent): remove unused methods and clean up"
```

---

## Task 14: Final Integration Test

**Files:**
- Test: `tests/unit/test_document_agent.py`

**Step 1: Add integration test**

Add to `tests/unit/test_document_agent.py`:

```python
class TestDocumentAgentIntegration:
    """Integration tests for DocumentProcessingAgent"""

    @pytest.mark.asyncio
    async def test_full_workflow_with_temp_repo(self):
        """Test full workflow with a temporary repository structure"""
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
```

**Step 2: Run integration test**

Run: `pytest tests/unit/test_document_agent.py::TestDocumentAgentIntegration -v`

Expected: PASS

**Step 3: Run all tests**

Run: `pytest tests/unit/test_document_agent.py -v`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/unit/test_document_agent.py
git commit -m "test(document_agent): add integration test for full workflow"
```

---

## Task 15: Final Cleanup and Documentation

**Step 1: Run full test suite**

Run: `pytest tests/ -v --ignore=tests/performance --ignore=tests/security`

Expected: All tests PASS (or at least document_agent tests pass)

**Step 2: Format code**

Run: `python -m black src/agents/document_agent.py tests/unit/test_document_agent.py`

**Step 3: Run type check**

Run: `python -m mypy src/agents/document_agent.py --ignore-missing-imports`

Expected: No errors (or only pre-existing ones)

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore(document_agent): format and finalize simplification"
```

**Step 5: Update design doc status**

Edit `docs/plans/2025-12-25-document-agent-simplification-design.md` and change:
```
**Status:** Approved
```
to:
```
**Status:** Implemented
```

```bash
git add docs/plans/2025-12-25-document-agent-simplification-design.md
git commit -m "docs: mark document agent design as implemented"
```
