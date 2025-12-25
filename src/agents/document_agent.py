"""Document processing agent for LangGraph workflows

This module implements the document processing agent that handles
repository analysis, file processing, and content preparation for embedding.
"""

import fnmatch
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..models.code_document import CodeDocument
from ..models.repository import AnalysisStatus
from ..repository.code_document_repository import CodeDocumentRepository
from ..repository.repository_repository import RepositoryRepository
from ..services.mcp_filesystem_client import MCPFilesystemClient
from ..tools.embedding_tool import EmbeddingTool
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

    def _create_workflow(self) -> StateGraph:
        """Create the document processing workflow graph

        Returns:
            LangGraph StateGraph for document processing
        """
        # Create workflow graph
        workflow = StateGraph(DocumentProcessingState)

        # Add nodes
        workflow.add_node("clone_repository", self._clone_repository_node)
        workflow.add_node("discover_files", self._discover_files_node)
        workflow.add_node("process_content", self._process_content_node)
        workflow.add_node("generate_embeddings", self._generate_embeddings_node)
        workflow.add_node("store_documents", self._store_documents_node)
        workflow.add_node("cleanup", self._cleanup_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define workflow edges
        workflow.add_edge(START, "clone_repository")

        # Sequential processing flow
        workflow.add_edge("clone_repository", "discover_files")
        workflow.add_edge("discover_files", "process_content")
        workflow.add_edge("process_content", "store_documents")
        workflow.add_edge("store_documents", "generate_embeddings")
        workflow.add_edge("generate_embeddings", "cleanup")
        workflow.add_edge("cleanup", END)

        # Error handling
        workflow.add_edge("handle_error", "cleanup")
        app = workflow.compile().with_config({"run_name": "document_agent.document_processing_workflow"})
        logger.debug(
            f"Document processing workflow:\n {app.get_graph().draw_mermaid()}"
        )
        return app

    async def process_repository(
        self, repository_id: str, repository_url: str, branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process repository through complete document processing pipeline

        Args:
            repository_id: Repository identifier
            repository_url: Repository URL to process
            branch: Specific branch to process

        Returns:
            Dictionary with processing results
        """
        try:
            # Initialize state
            initial_state: DocumentProcessingState = {
                "repository_id": repository_id,
                "repository_url": repository_url,
                "branch": branch,
                "clone_path": None,
                "discovered_files": [],
                "processed_documents": [],
                "embeddings_generated": 0,
                "current_step": "starting",
                "error_message": None,
                "progress": 0.0,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "messages": [
                    HumanMessage(content=f"Process repository: {repository_url}")
                ],
            }

            # Update repository status to processing
            await self._update_repository_status(
                repository_id, AnalysisStatus.PROCESSING
            )

            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)

            # Update final repository status
            if result.get("error_message"):
                await self._update_repository_status(
                    repository_id,
                    AnalysisStatus.FAILED,
                    error_message=result["error_message"],
                )
            else:
                await self._update_repository_status(
                    repository_id,
                    AnalysisStatus.COMPLETED,
                    commit_sha=result.get("commit_sha"),
                )

            return {
                "status": "completed" if not result.get("error_message") else "failed",
                "repository_id": repository_id,
                "processed_files": len(result.get("processed_documents", [])),
                "embeddings_generated": result.get("embeddings_generated", 0),
                "processing_time": result.get("processing_time", 0),
                "error_message": result.get("error_message"),
            }

        except Exception as e:
            logger.error(
                f"Document processing failed for repository {repository_id}: {e}"
            )

            # Update repository status to failed
            await self._update_repository_status(
                repository_id, AnalysisStatus.FAILED, error_message=str(e)
            )

            return {
                "status": "failed",
                "repository_id": repository_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

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

    async def _discover_files_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Discover files node

        Uses MCP filesystem client if available for enhanced performance,
        otherwise falls back to the native RepositoryTool.
        """
        try:
            state["current_step"] = "discovering_files"
            state["progress"] = 30.0

            if not state["clone_path"]:
                state["error_message"] = "No clone path available for file discovery"
                return state

            # Try MCP client first if available and initialized
            if self._mcp_client and self._mcp_client.is_initialized:
                logger.info("Using MCP filesystem for file discovery")
                discovery_result = await self._discover_files_with_mcp(state["clone_path"])
            else:
                logger.info("Using native RepositoryTool for file discovery")
                discovery_result = await self._repository_tool._arun(
                    "discover_files", repository_path=state["clone_path"]
                )

            if discovery_result["status"] != "success":
                state["error_message"] = (
                    f"File discovery failed: {discovery_result.get('error', 'Unknown error')}"
                )
                return state

            state["discovered_files"] = discovery_result["discovered_files"]
            state["progress"] = 40.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Discovered {len(state['discovered_files'])} files for processing"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"File discovery node failed: {str(e)}"
            return state

    async def _discover_files_with_mcp(self, clone_path: str) -> Dict[str, Any]:
        """Discover files using MCP filesystem client.

        Args:
            clone_path: Path to the cloned repository.

        Returns:
            Dictionary with discovered files in the expected format.
        """
        try:
            # Get directory tree using MCP
            tree_result = await self._mcp_client.get_directory_tree(clone_path)

            if tree_result["status"] != "success":
                # Fallback to native tool
                logger.warning("MCP directory tree failed, falling back to native tool")
                return await self._repository_tool._arun(
                    "discover_files", repository_path=clone_path
                )

            # Transform MCP result to expected format
            discovered_files = self._transform_mcp_tree_to_files(
                tree_result.get("tree", {}), clone_path
            )

            return {
                "status": "success",
                "discovered_files": discovered_files,
                "total_discovered": len(discovered_files),
                "source": "mcp",
            }

        except Exception as e:
            logger.warning(f"MCP file discovery failed: {e}, falling back to native tool")
            return await self._repository_tool._arun(
                "discover_files", repository_path=clone_path
            )

    def _transform_mcp_tree_to_files(
        self, tree: Any, base_path: str, current_path: str = ""
    ) -> List[Dict[str, Any]]:
        """Transform MCP directory tree to flat file list.

        Args:
            tree: MCP directory tree structure.
            base_path: Base path of the repository.
            current_path: Current relative path in recursion.

        Returns:
            List of file dictionaries in expected format.
        """
        from pathlib import Path

        files = []
        supported_languages = self.settings.supported_languages

        # Default exclude patterns
        exclude_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".env", ".idea", ".vscode", "dist", "build", ".tox"
        }
        exclude_extensions = {
            ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".svg",
            ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z"
        }

        # Language extension mapping
        extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
        }

        def process_node(node: Any, rel_path: str) -> None:
            if isinstance(node, dict):
                for name, content in node.items():
                    new_path = f"{rel_path}/{name}" if rel_path else name

                    # Skip excluded directories
                    if name in exclude_dirs:
                        continue

                    if isinstance(content, dict):
                        # It's a directory, recurse
                        process_node(content, new_path)
                    else:
                        # It's a file
                        file_path = Path(base_path) / new_path
                        ext = file_path.suffix.lower()

                        # Skip excluded extensions
                        if ext in exclude_extensions:
                            continue

                        # Detect language
                        language = extension_to_language.get(ext, "unknown")
                        if language not in supported_languages:
                            continue

                        try:
                            stat = file_path.stat()
                            files.append({
                                "path": new_path.replace("\\", "/"),
                                "full_path": str(file_path),
                                "language": language,
                                "size": stat.st_size,
                                "modified_at": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                            })
                        except OSError:
                            pass

            elif isinstance(node, list):
                # Handle list of files/directories
                for item in node:
                    if isinstance(item, str):
                        file_path = Path(base_path) / rel_path / item
                        ext = file_path.suffix.lower()

                        if ext in exclude_extensions:
                            continue

                        language = extension_to_language.get(ext, "unknown")
                        if language not in supported_languages:
                            continue

                        try:
                            stat = file_path.stat()
                            full_rel_path = f"{rel_path}/{item}" if rel_path else item
                            files.append({
                                "path": full_rel_path.replace("\\", "/"),
                                "full_path": str(file_path),
                                "language": language,
                                "size": stat.st_size,
                                "modified_at": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                            })
                        except OSError:
                            pass
                    elif isinstance(item, dict):
                        process_node(item, rel_path)

        process_node(tree, current_path)
        return files

    async def _process_content_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Process file content node

        Uses MCP filesystem client for file reading if available,
        otherwise falls back to native Python file operations.
        """
        try:
            state["current_step"] = "processing_content"
            state["progress"] = 50.0

            if not state["discovered_files"]:
                state["error_message"] = "No files discovered for processing"
                return state

            processed_documents = []
            use_mcp = self._mcp_client and self._mcp_client.is_initialized

            if use_mcp:
                logger.info("Using MCP filesystem for file reading")
            else:
                logger.info("Using native file operations for file reading")

            # Process each discovered file
            for i, file_info in enumerate(state["discovered_files"]):
                try:
                    # Read file content
                    file_path = file_info["full_path"]
                    content = await self._read_file_content(file_path, use_mcp)

                    if content is None:
                        logger.warning(f"Could not read file {file_info['path']}")
                        continue

                    # Process content for embedding
                    processed_content = self._clean_content_for_embedding(
                        content, file_info["language"]
                    )

                    # Create document record
                    doc_data = {
                        "id": f"{state['repository_id']}_{file_info['path'].replace('/', '_').replace('.', '_')}",
                        "repository_id": state["repository_id"],
                        "file_path": file_info["path"],
                        "language": file_info["language"],
                        "content": content,
                        "processed_content": processed_content,
                        "metadata": {
                            "size": file_info["size"],
                            "modified_at": file_info["modified_at"],
                            "processing_time": datetime.now(timezone.utc).isoformat(),
                        },
                    }

                    processed_documents.append(doc_data)

                    # Update progress
                    progress = 50.0 + (i / len(state["discovered_files"])) * 20.0
                    state["progress"] = progress

                except Exception as e:
                    logger.warning(f"Failed to process file {file_info['path']}: {e}")
                    continue

            state["processed_documents"] = processed_documents
            state["progress"] = 70.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Processed {len(processed_documents)} files for embedding generation"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Content processing node failed: {str(e)}"
            return state

    async def _read_file_content(self, file_path: str, use_mcp: bool) -> Optional[str]:
        """Read file content using MCP or native operations.

        Args:
            file_path: Path to the file to read.
            use_mcp: Whether to use MCP client.

        Returns:
            File content as string, or None if reading failed.
        """
        try:
            if use_mcp:
                result = await self._mcp_client.read_file(file_path)
                if result["status"] == "success":
                    return result.get("content", "")
                else:
                    # Fallback to native on MCP failure
                    logger.debug(f"MCP read failed for {file_path}, using native")
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Failed to read file {file_path}: {e}")
            return None

    async def _generate_embeddings_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Generate embeddings node"""
        try:
            state["current_step"] = "generating_embeddings"
            state["progress"] = 80.0

            if not state["processed_documents"]:
                state["error_message"] = (
                    "No processed documents for embedding generation"
                )
                return state

            embedding_result = await self._embedding_tool._arun(
                "generate_and_store", documents=state["processed_documents"]
            )

            if embedding_result["status"] != "success":
                state["error_message"] = (
                    f"Embedding generation failed: {embedding_result.get('error', 'Unknown error')}"
                )
                return state

            state["embeddings_generated"] = embedding_result["processed_count"]
            state["progress"] = 90.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Generated embeddings for {state['embeddings_generated']} documents"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Embedding generation node failed: {str(e)}"
            return state

    async def _store_documents_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Store documents node"""
        try:
            state["current_step"] = "storing_documents"
            state["progress"] = 95.0

            if not state["processed_documents"]:
                state["progress"] = 100.0
                return state

            stored_count = 0

            for doc_data in state["processed_documents"]:
                try:
                    doc_data["repository_id"] = UUID(state["repository_id"])
                    code_document = CodeDocument(**doc_data)
                    await self._code_document_repo.insert(code_document)
                    stored_count += 1

                except Exception as e:
                    logger.warning(
                        f"Failed to store document {doc_data.get('id')}: {e}"
                    )
                    continue

            state["progress"] = 100.0

            state["messages"].append(
                AIMessage(content=f"Stored {stored_count} documents in database")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Document storage node failed: {str(e)}"
            return state

    async def _cleanup_node(
        self, state: DocumentProcessingState
    ) -> DocumentProcessingState:
        """Cleanup node"""
        try:
            state["current_step"] = "cleanup"

            # Cleanup cloned repository
            if state["clone_path"]:
                cleanup_result = await self._repository_tool._arun(
                    "cleanup", repository_path=state["clone_path"]
                )

                if cleanup_result["status"] == "success":
                    state["messages"].append(
                        AIMessage(content="Repository cleanup completed")
                    )

            # Calculate total processing time
            start_time = datetime.fromisoformat(state["start_time"])
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()

            state["processing_time"] = processing_time
            state["current_step"] = "completed"

            return state

        except Exception as e:
            logger.error(f"Cleanup node failed: {e}")
            state["error_message"] = f"Cleanup failed: {str(e)}"
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

    def _clean_content_for_embedding(self, content: str, language: str) -> str:
        """Clean content for embedding generation

        Args:
            content: Raw file content
            language: Programming language

        Returns:
            Cleaned content suitable for embedding
        """
        try:
            # Remove excessive whitespace
            cleaned = re.sub(r"\s+", " ", content.strip())

            # Language-specific cleaning
            if language == "python":
                # Remove docstring quotes but keep content
                cleaned = re.sub(
                    r'""".*?"""',
                    lambda m: m.group(0).replace('"""', ""),
                    cleaned,
                    flags=re.DOTALL,
                )
                cleaned = re.sub(
                    r"'''.*?'''",
                    lambda m: m.group(0).replace("'''", ""),
                    cleaned,
                    flags=re.DOTALL,
                )

            elif language in ["javascript", "typescript"]:
                # Remove excessive comments but keep meaningful ones
                cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
                cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

            elif language in ["java", "csharp"]:
                # Remove comments
                cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
                cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

            # Remove empty lines and normalize whitespace
            lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
            cleaned = "\n".join(lines)

            # Truncate if too long (for embedding efficiency)
            max_length = 8000  # Reasonable limit for embeddings
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + "..."

            return cleaned

        except Exception as e:
            logger.warning(f"Content cleaning failed: {e}")
            return content[:8000]  # Fallback to truncated original

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

    async def get_processing_status(self, repository_id: str) -> Dict[str, Any]:
        """Get current processing status for repository

        Args:
            repository_id: Repository ID

        Returns:
            Dictionary with current processing status
        """
        try:

            repository = await self._repository_repo.get(UUID(repository_id))
            if not repository:
                return {"status": "not_found", "error": "Repository not found"}

            doc_count = await self._code_document_repo.count({"repository_id": UUID(repository_id)})

            embedding_count = await self._code_document_repo.count(
                {"repository_id": UUID(repository_id), "embedding": {"$exists": True}}
            )

            return {
                "status": "success",
                "repository_id": repository_id,
                "analysis_status": repository.analysis_status.value if repository.analysis_status else "unknown",
                "last_analyzed": repository.last_analyzed,
                "total_documents": doc_count,
                "documents_with_embeddings": embedding_count,
                "embedding_coverage": (
                    (embedding_count / doc_count * 100) if doc_count > 0 else 0
                ),
                "error_message": repository.error_message,
            }

        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {"status": "error", "error": str(e), "repository_id": repository_id}

