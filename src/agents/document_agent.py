"""Document processing agent for LangGraph workflows

This module implements the document processing agent that handles
repository analysis, file processing, and content preparation for embedding.
"""

import fnmatch
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

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


class DocumentProcessingAgent:
    """LangGraph agent for document processing workflows

    Orchestrates the complete document processing pipeline from
    repository cloning to embedding generation and storage.
    """

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

    def _load_exclusion_patterns(self, clone_path: Optional[str]) -> tuple[List[str], List[str]]:
        """Load exclusion patterns from config and optional .autodoc/autodoc.json override.

        This helper loads patterns internally so nodes don't need to pass patterns
        through state between workflow steps.

        Args:
            clone_path: Path to the cloned repository.

        Returns:
            Tuple of (excluded_dirs, excluded_files).
        """
        # Start with defaults from config
        excluded_dirs = list(self.settings.default_excluded_dirs)
        excluded_files = list(self.settings.default_excluded_files)

        # Check for .autodoc/autodoc.json override
        if clone_path:
            autodoc_config_path = Path(clone_path) / ".autodoc" / "autodoc.json"
            if autodoc_config_path.exists():
                try:
                    with open(autodoc_config_path, "r", encoding="utf-8") as f:
                        autodoc_config = json.load(f)

                    # Override with values from autodoc.json if present
                    if "excluded_dirs" in autodoc_config:
                        excluded_dirs = autodoc_config["excluded_dirs"]
                        logger.info("Loaded excluded_dirs from .autodoc/autodoc.json")
                    if "excluded_files" in autodoc_config:
                        excluded_files = autodoc_config["excluded_files"]
                        logger.info("Loaded excluded_files from .autodoc/autodoc.json")
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid .autodoc/autodoc.json: {e}")
                except Exception as e:
                    logger.warning(f"Failed to read .autodoc/autodoc.json: {e}")

        return excluded_dirs, excluded_files

    def _build_file_tree(
        self, root_path: str, excluded_dirs: List[str], excluded_files: List[str]
    ) -> str:
        """Build flat list of absolute file paths in the repository.

        Args:
            root_path: Root directory to scan.
            excluded_dirs: List of directory patterns to exclude.
            excluded_files: List of file patterns to exclude.

        Returns:
            Newline-separated string of absolute file paths.
        """
        file_paths = []
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

        def collect_files(path: Path):
            try:
                entries = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return

            for entry in entries:
                if entry.is_dir():
                    if not should_exclude_dir(entry):
                        collect_files(entry)
                else:
                    if not should_exclude_file(entry):
                        # Use absolute path
                        file_paths.append(str(entry.resolve()))

        collect_files(root)
        return "\n".join(file_paths)

    def _create_workflow(self) -> StateGraph:
        """Create the document processing workflow graph.

        Workflow order:
        1. clone_repository - Clone the repository
        2. build_tree - Build file tree (loads patterns internally)
        3. extract_docs - Extract documentation files
        4. load_patterns - Load patterns into state for cleanup
        5. cleanup_excluded - Physically delete excluded files/dirs

        Returns:
            LangGraph StateGraph for document processing.
        """
        workflow = StateGraph(DocumentProcessingState)

        # Add nodes
        workflow.add_node("clone_repository", self._clone_repository_node)
        workflow.add_node("build_tree", self._discover_and_build_tree_node)
        workflow.add_node("extract_docs", self._extract_docs_node)
        workflow.add_node("load_patterns", self._load_patterns_node)
        workflow.add_node("cleanup_excluded", self._cleanup_excluded_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define workflow edges - new order
        workflow.add_edge(START, "clone_repository")
        workflow.add_edge("clone_repository", "build_tree")
        workflow.add_edge("build_tree", "extract_docs")
        workflow.add_edge("extract_docs", "load_patterns")
        workflow.add_edge("load_patterns", "cleanup_excluded")
        workflow.add_edge("cleanup_excluded", END)

        # Error handling
        workflow.add_edge("handle_error", END)

        app = workflow.compile().with_config(
            {"run_name": "document_agent.document_processing_workflow"}
        )
        logger.debug(f"Document processing workflow:\n {app.get_graph().draw_mermaid()}")
        return app

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

    async def _load_patterns_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Load exclusion patterns into state for cleanup step.
        
        This node runs after extract_docs and before cleanup_excluded.
        The patterns are loaded into state for the cleanup node to use.
        """
        try:
            state["current_step"] = "loading_patterns"
            state["progress"] = 80.0

            # Load patterns using helper
            excluded_dirs, excluded_files = self._load_exclusion_patterns(state["clone_path"])

            state["excluded_dirs"] = excluded_dirs
            state["excluded_files"] = excluded_files

            state["messages"].append(
                AIMessage(content=f"Loaded {len(excluded_dirs)} dir exclusions and {len(excluded_files)} file exclusions for cleanup")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Load patterns node failed: {str(e)}"
            return state

    async def _discover_and_build_tree_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Discover files and build ASCII tree structure.
        
        Loads exclusion patterns internally - does not require patterns from state.
        """
        try:
            state["current_step"] = "building_tree"
            state["progress"] = 25.0

            if not state["clone_path"]:
                state["error_message"] = "No clone path available for tree building"
                return state

            # Load patterns internally (not from state)
            excluded_dirs, excluded_files = self._load_exclusion_patterns(state["clone_path"])

            # Build the file tree
            file_tree = self._build_file_tree(
                state["clone_path"],
                excluded_dirs,
                excluded_files
            )

            state["file_tree"] = file_tree
            state["progress"] = 40.0

            # Count lines for message
            line_count = len(file_tree.split("\n")) if file_tree else 0
            state["messages"].append(
                AIMessage(content=f"Built file tree with {line_count} entries")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Build tree node failed: {str(e)}"
            return state

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

            # Deduplicate by path (case-insensitive for cross-platform compatibility)
            seen_paths = set()
            unique_docs = []
            for doc in documentation_files:
                path_lower = doc["path"].lower()
                if path_lower not in seen_paths:
                    seen_paths.add(path_lower)
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

    async def _cleanup_excluded_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Physically delete excluded files and directories from cloned repository.
        
        This node runs after load_patterns and uses the patterns from state.
        The cleanup makes the repo ready for Wiki Agent's file searches.
        """
        try:
            state["current_step"] = "cleanup_excluded"
            state["progress"] = 90.0

            if not state["clone_path"]:
                state["error_message"] = "No clone path available for cleanup"
                return state

            root = Path(state["clone_path"])
            excluded_dirs = state.get("excluded_dirs", [])
            excluded_files = state.get("excluded_files", [])

            deleted_dirs = 0
            deleted_files = 0

            # Helper to check if directory should be excluded
            def should_exclude_dir(dir_path: Path) -> bool:
                rel_path = str(dir_path.relative_to(root)).replace("\\", "/") + "/"
                dir_name = dir_path.name + "/"
                for pattern in excluded_dirs:
                    pattern = pattern.replace("\\", "/")
                    if self._matches_pattern(rel_path, pattern) or self._matches_pattern(dir_name, pattern):
                        return True
                return False

            # Helper to check if file should be excluded
            def should_exclude_file(file_path: Path) -> bool:
                rel_path = str(file_path.relative_to(root)).replace("\\", "/")
                file_name = file_path.name
                for pattern in excluded_files:
                    if self._matches_pattern(rel_path, pattern) or self._matches_pattern(file_name, pattern):
                        return True
                return False

            # Collect directories to delete (deepest first to avoid issues)
            dirs_to_delete = []
            for dir_path in root.rglob("*"):
                if dir_path.is_dir() and should_exclude_dir(dir_path):
                    dirs_to_delete.append(dir_path)

            # Sort by depth (deepest first)
            dirs_to_delete.sort(key=lambda p: len(p.parts), reverse=True)

            # Delete directories
            for dir_path in dirs_to_delete:
                try:
                    if dir_path.exists():
                        shutil.rmtree(dir_path)
                        deleted_dirs += 1
                except Exception as e:
                    logger.warning(f"Failed to delete directory {dir_path}: {e}")

            # Collect and delete files
            for file_path in root.rglob("*"):
                if file_path.is_file() and should_exclude_file(file_path):
                    try:
                        file_path.unlink()
                        deleted_files += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete file {file_path}: {e}")

            state["current_step"] = "success"
            state["progress"] = 100.0

            state["messages"].append(
                AIMessage(content=f"Cleanup complete: deleted {deleted_dirs} directories and {deleted_files} files")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Cleanup node failed: {str(e)}"
            return state

    async def _clone_repository_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Clone repository node"""
        try:
            state["current_step"] = "cloning_repository"
            state["progress"] = 10.0

            clone_result = await self._repository_tool._arun(
                "clone", repository_url=state["repository_url"], branch=state["branch"]
            )

            if clone_result["status"] != "success":
                state["error_message"] = (
                    f"Repository clone failed: {clone_result.get('error', 'Unknown error')}"
                )
                return state

            state["clone_path"] = clone_result["clone_path"]
            state["progress"] = 20.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Successfully cloned repository to {clone_result['clone_path']}"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Clone node failed: {str(e)}"
            return state

    async def _handle_error_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Handle error node"""
        try:
            # Log error details
            logger.error(
                f"Document processing failed for repository {state['repository_id']}: {state.get('error_message')}"
            )

            # Add error message
            state["messages"].append(
                AIMessage(
                    content=f"Processing failed: {state.get('error_message', 'Unknown error')}"
                )
            )

            state["current_step"] = "error_handling"

            return state

        except Exception as e:
            logger.error(f"Error handling node failed: {e}")
            return state


    async def _update_repository_status(
        self,
        repository_id: str,
        status: AnalysisStatus,
        error_message: Optional[str] = None,
        commit_sha: Optional[str] = None,
    ) -> None:
        """Update repository analysis status

        Args:
            repository_id: Repository ID
            status: New analysis status
            error_message: Optional error message
            commit_sha: Optional commit SHA
        """
        try:

            updates = {
                "analysis_status": status.value,
                "updated_at": datetime.now(timezone.utc),
            }

            if status == AnalysisStatus.COMPLETED:
                updates["last_analyzed"] = datetime.now(timezone.utc)
                if commit_sha:
                    updates["commit_sha"] = commit_sha

            if error_message:
                updates["error_message"] = error_message

            await self._repository_repo.update(UUID(repository_id), updates)

        except Exception as e:
            logger.error(f"Failed to update repository status: {e}")


