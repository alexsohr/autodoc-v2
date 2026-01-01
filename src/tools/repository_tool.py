"""Repository tool for LangGraph workflows

This module implements the repository tool for cloning and analyzing
Git repositories as part of LangGraph workflows.
"""

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..models.code_document import CodeDocument
from ..models.repository import AnalysisStatus, Repository, RepositoryProvider
from ..utils.config_loader import get_settings
from ..utils.storage_adapters import StorageAdapterFactory

logger = logging.getLogger(__name__)


class RepositoryCloneInput(BaseModel):
    """Input schema for repository clone operation"""

    repository_url: str = Field(description="Git repository URL to clone")
    branch: Optional[str] = Field(default=None, description="Specific branch to clone")
    depth: Optional[int] = Field(
        default=1, description="Clone depth (1 for shallow clone)"
    )
    target_directory: Optional[str] = Field(
        default=None, description="Target directory for clone"
    )


class RepositoryAnalyzeInput(BaseModel):
    """Input schema for repository analysis operation"""

    repository_path: str = Field(description="Path to cloned repository")
    repository_id: str = Field(description="Repository ID for tracking")
    include_patterns: Optional[List[str]] = Field(
        default=None, description="File patterns to include"
    )
    exclude_patterns: Optional[List[str]] = Field(
        default=None, description="File patterns to exclude"
    )
    max_file_size: Optional[int] = Field(
        default=None, description="Maximum file size in bytes"
    )


class RepositoryTool(BaseTool):
    """LangGraph tool for repository operations

    Provides repository cloning and analysis capabilities for LangGraph workflows.
    Handles Git operations, file discovery, and content processing.
    """

    name: str = "repository_tool"
    description: str = "Tool for cloning Git repositories and analyzing their content"

    def __init__(self):
        """Initialize RepositoryTool.

        No dependencies to inject - this tool manages repository operations independently.
        """
        super().__init__()
        # Initialize settings and configuration
        settings = get_settings()
        self._clone_timeout = settings.clone_timeout_seconds
        self._storage_base_path = settings.storage_base_path
        self._max_file_size = settings.max_file_size_mb * 1024 * 1024
        self._supported_languages = set(settings.supported_languages)

    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Async run method for LangGraph"""
        if operation == "clone":
            return await self.clone_repository(**kwargs)
        elif operation == "analyze":
            return await self.analyze_repository(**kwargs)
        elif operation == "discover_files":
            return await self.discover_files(**kwargs)
        elif operation == "cleanup":
            return await self.cleanup_repository(**kwargs)
        else:
            raise ValueError(f"Unknown repository operation: {operation}")

    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Sync run method (not used in async workflows)"""
        raise NotImplementedError("Repository tool only supports async operations")

    async def _run_git_command(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> subprocess.CompletedProcess:
        """Run a git command in a Windows-compatible way.

        Uses asyncio.to_thread with subprocess.run instead of
        asyncio.create_subprocess_exec, which doesn't work on Windows with uvicorn.

        Args:
            cmd: Command and arguments to run
            cwd: Working directory
            timeout: Timeout in seconds

        Returns:
            CompletedProcess with returncode, stdout, stderr
        """
        def run_subprocess():
            return subprocess.run(
                cmd,
                capture_output=True,
                cwd=cwd,
                timeout=timeout or 30,
            )

        return await asyncio.to_thread(run_subprocess)

    async def clone_repository(
        self,
        repository_url: str,
        branch: Optional[str] = None,
        depth: Optional[int] = 1,
        target_directory: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Clone a Git repository

        Args:
            repository_url: Git repository URL
            branch: Specific branch to clone
            depth: Clone depth (1 for shallow clone)
            target_directory: Target directory (auto-generated if None)

        Returns:
            Dictionary with clone results
        """
        logger.info(
            f"Starting clone: url={repository_url!r}, branch={branch!r}, "
            f"depth={depth}, target_dir={target_directory!r}"
        )
        try:
            # Validate repository URL
            parsed_url = urlparse(repository_url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid repository URL: {repository_url}")

            # Create target directory
            if target_directory is None:
                if self._storage_base_path:
                    # Use configured storage base path with repos subdirectory
                    base_path = Path(self._storage_base_path) / "repos"
                    base_path.mkdir(parents=True, exist_ok=True)
                    target_directory = tempfile.mkdtemp(prefix="autodoc_repo_", dir=base_path)
                else:
                    # Fall back to system temp directory
                    target_directory = tempfile.mkdtemp(prefix="autodoc_repo_")
            else:
                os.makedirs(target_directory, exist_ok=True)

            target_path = Path(target_directory)

            # Build git clone command
            git_cmd = ["git", "clone"]

            if depth and depth > 0:
                git_cmd.extend(["--depth", str(depth)])

            if branch:
                git_cmd.extend(["--branch", branch])

            git_cmd.extend([repository_url, str(target_path)])

            logger.info(f"Executing git command: {' '.join(git_cmd)}")
            logger.debug(f"Working directory: {target_path.parent}")

            # Execute git clone using subprocess.run in a thread (Windows compatible)
            # asyncio.create_subprocess_exec doesn't work on Windows with uvicorn
            def run_git_clone():
                return subprocess.run(
                    git_cmd,
                    capture_output=True,
                    cwd=target_path.parent,
                    timeout=self._clone_timeout,
                )

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(run_git_clone),
                    timeout=self._clone_timeout + 5,  # Extra buffer for thread overhead
                )
            except (asyncio.TimeoutError, subprocess.TimeoutExpired):
                raise TimeoutError(
                    f"Repository clone timeout after {self._clone_timeout} seconds"
                )

            if result.returncode != 0:
                error_msg = result.stderr.decode("utf-8") if result.stderr else "Unknown git error"
                logger.error(f"Git clone returned {result.returncode}: {error_msg}")
                raise RuntimeError(f"Git clone failed: {error_msg}")

            logger.info(f"Git clone succeeded, getting repository info from {target_path}")
            # Get repository information
            repo_info = await self._get_repository_info(target_path)

            return {
                "status": "success",
                "clone_path": str(target_path),
                "repository_info": repo_info,
                "clone_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(
                f"Repository clone failed: {e!r} (type: {type(e).__name__})\n"
                f"Traceback:\n{traceback.format_exc()}"
            )

            # Cleanup on failure
            if target_directory and Path(target_directory).exists():
                shutil.rmtree(target_directory, ignore_errors=True)

            return {"status": "error", "error": str(e) or repr(e), "error_type": type(e).__name__}

    async def analyze_repository(
        self,
        repository_path: str,
        repository_id: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze repository content and structure

        Args:
            repository_path: Path to cloned repository
            repository_id: Repository ID for tracking
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            max_file_size: Maximum file size in bytes

        Returns:
            Dictionary with analysis results
        """
        try:
            repo_path = Path(repository_path)
            if not repo_path.exists() or not repo_path.is_dir():
                raise ValueError(f"Repository path does not exist: {repository_path}")

            # Discover files
            files = await self.discover_files(
                repository_path,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                max_file_size=max_file_size or self._max_file_size,
            )

            # Analyze repository structure
            structure_analysis = await self._analyze_repository_structure(repo_path)

            # Get Git information
            git_info = await self._get_git_info(repo_path)

            # Language statistics
            language_stats = self._calculate_language_stats(files["discovered_files"])

            return {
                "status": "success",
                "repository_id": repository_id,
                "analysis_time": datetime.now(timezone.utc).isoformat(),
                "files": files,
                "structure": structure_analysis,
                "git_info": git_info,
                "language_stats": language_stats,
                "total_files": len(files["discovered_files"]),
                "total_size": sum(f["size"] for f in files["discovered_files"]),
            }

        except Exception as e:
            logger.error(f"Repository analysis failed: {e}")
            return {
                "status": "error",
                "repository_id": repository_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def discover_files(
        self,
        repository_path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        max_file_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Discover and filter files in repository

        Args:
            repository_path: Path to repository
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            max_file_size: Maximum file size in bytes

        Returns:
            Dictionary with discovered files and statistics
        """
        repo_path = Path(repository_path)
        max_size = max_file_size or self._max_file_size

        # Default exclude patterns
        default_excludes = [
            ".git/**",
            "__pycache__/**",
            "node_modules/**",
            ".venv/**",
            "venv/**",
            ".env",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            "*.so",
            "*.dll",
            "*.exe",
            "*.bin",
            "*.jpg",
            "*.jpeg",
            "*.png",
            "*.gif",
            "*.bmp",
            "*.ico",
            "*.svg",
            "*.pdf",
            "*.zip",
            "*.tar",
            "*.gz",
            "*.rar",
            "*.7z",
        ]

        exclude_patterns = (exclude_patterns or []) + default_excludes
        include_patterns = include_patterns or ["**/*"]

        discovered_files = []
        skipped_files = []

        # Walk through repository
        for file_path in repo_path.rglob("*"):
            if not file_path.is_file():
                continue

            relative_path = file_path.relative_to(repo_path)
            relative_str = str(relative_path).replace("\\", "/")

            # Check exclude patterns
            if self._matches_patterns(relative_str, exclude_patterns):
                skipped_files.append(
                    {"path": relative_str, "reason": "excluded_pattern"}
                )
                continue

            # Check include patterns
            if not self._matches_patterns(relative_str, include_patterns):
                skipped_files.append({"path": relative_str, "reason": "not_included"})
                continue

            # Check file size
            try:
                file_size = file_path.stat().st_size
                if file_size > max_size:
                    skipped_files.append(
                        {"path": relative_str, "reason": "too_large", "size": file_size}
                    )
                    continue
            except OSError:
                skipped_files.append({"path": relative_str, "reason": "stat_error"})
                continue

            # Detect language
            language = self._detect_language(file_path)

            # Check if language is supported
            if language not in self._supported_languages:
                skipped_files.append(
                    {
                        "path": relative_str,
                        "reason": "unsupported_language",
                        "language": language,
                    }
                )
                continue

            # Add to discovered files
            discovered_files.append(
                {
                    "path": relative_str,
                    "full_path": str(file_path),
                    "language": language,
                    "size": file_size,
                    "modified_at": datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).isoformat(),
                }
            )

        return {
            "status": "success",
            "discovered_files": discovered_files,
            "skipped_files": skipped_files,
            "total_discovered": len(discovered_files),
            "total_skipped": len(skipped_files),
            "languages_found": list(set(f["language"] for f in discovered_files)),
        }

    async def _get_repository_info(self, repo_path: Path) -> Dict[str, Any]:
        """Get repository information from Git

        Args:
            repo_path: Path to cloned repository

        Returns:
            Dictionary with repository information
        """
        try:
            # Get remote origin URL
            result = await self._run_git_command(
                ["git", "config", "--get", "remote.origin.url"],
                cwd=repo_path,
            )
            origin_url = (
                result.stdout.decode("utf-8").strip() if result.returncode == 0 else ""
            )

            # Get current branch
            result = await self._run_git_command(
                ["git", "branch", "--show-current"],
                cwd=repo_path,
            )
            current_branch = (
                result.stdout.decode("utf-8").strip() if result.returncode == 0 else ""
            )

            # Get latest commit SHA
            result = await self._run_git_command(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_path,
            )
            commit_sha = (
                result.stdout.decode("utf-8").strip() if result.returncode == 0 else ""
            )

            # Get commit count
            result = await self._run_git_command(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=repo_path,
            )
            commit_count = (
                int(result.stdout.decode("utf-8").strip()) if result.returncode == 0 else 0
            )

            return {
                "origin_url": origin_url,
                "current_branch": current_branch,
                "commit_sha": commit_sha,
                "commit_count": commit_count,
                "clone_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.warning(f"Could not get full repository info: {e}")
            return {
                "origin_url": "",
                "current_branch": "",
                "commit_sha": "",
                "commit_count": 0,
                "error": str(e),
            }

    async def _analyze_repository_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze repository structure and patterns

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with structure analysis
        """
        try:
            structure = {
                "directories": [],
                "config_files": [],
                "documentation_files": [],
                "test_directories": [],
                "main_directories": [],
                "framework_indicators": [],
            }

            # Analyze directory structure
            for item in repo_path.rglob("*"):
                if item.is_dir():
                    relative_path = str(item.relative_to(repo_path)).replace("\\", "/")
                    structure["directories"].append(relative_path)

                    # Identify special directories
                    dir_name = item.name.lower()
                    if dir_name in ["test", "tests", "__tests__", "spec", "specs"]:
                        structure["test_directories"].append(relative_path)
                    elif dir_name in ["src", "lib", "app", "source"]:
                        structure["main_directories"].append(relative_path)

                elif item.is_file():
                    file_name = item.name.lower()
                    relative_path = str(item.relative_to(repo_path)).replace("\\", "/")

                    # Identify config files
                    if file_name in [
                        "package.json",
                        "requirements.txt",
                        "pyproject.toml",
                        "setup.py",
                        "pom.xml",
                        "build.gradle",
                        "cargo.toml",
                        "go.mod",
                        "composer.json",
                    ]:
                        structure["config_files"].append(relative_path)

                    # Identify documentation files
                    elif file_name.startswith("readme") or file_name.endswith(
                        (".md", ".rst", ".txt")
                    ):
                        if "doc" in relative_path.lower() or file_name.startswith(
                            "readme"
                        ):
                            structure["documentation_files"].append(relative_path)

                    # Framework indicators
                    if file_name in ["package.json", "requirements.txt"]:
                        framework_info = await self._detect_framework(item)
                        if framework_info:
                            structure["framework_indicators"].extend(framework_info)

            return structure

        except Exception as e:
            logger.error(f"Repository structure analysis failed: {e}")
            return {
                "directories": [],
                "config_files": [],
                "documentation_files": [],
                "test_directories": [],
                "main_directories": [],
                "framework_indicators": [],
                "error": str(e),
            }

    async def _get_git_info(self, repo_path: Path) -> Dict[str, Any]:
        """Get detailed Git information

        Args:
            repo_path: Path to repository

        Returns:
            Dictionary with Git information
        """
        try:
            git_info = {}

            # Get all branches
            result = await self._run_git_command(
                ["git", "branch", "-a"],
                cwd=repo_path,
            )
            if result.returncode == 0:
                branches = [
                    line.strip().replace("* ", "").replace("remotes/origin/", "")
                    for line in result.stdout.decode("utf-8").split("\n")
                    if line.strip() and not line.strip().startswith("HEAD")
                ]
                git_info["branches"] = list(set(branches))

            # Get recent commits
            result = await self._run_git_command(
                ["git", "log", "--oneline", "-10"],
                cwd=repo_path,
            )
            if result.returncode == 0:
                commits = []
                for line in result.stdout.decode("utf-8").split("\n"):
                    if line.strip():
                        parts = line.strip().split(" ", 1)
                        if len(parts) == 2:
                            commits.append({"sha": parts[0], "message": parts[1]})
                git_info["recent_commits"] = commits

            # Get repository size
            result = await self._run_git_command(
                ["git", "count-objects", "-vH"],
                cwd=repo_path,
            )
            if result.returncode == 0:
                for line in result.stdout.decode("utf-8").split("\n"):
                    if "size-pack" in line:
                        git_info["repository_size"] = line.split(":")[1].strip()
                        break

            return git_info

        except Exception as e:
            logger.warning(f"Could not get Git info: {e}")
            return {"error": str(e)}

    async def _detect_framework(self, config_file: Path) -> List[str]:
        """Detect framework from config file

        Args:
            config_file: Path to config file

        Returns:
            List of detected frameworks
        """
        frameworks = []

        try:
            if config_file.name == "package.json":
                # Node.js frameworks
                import json

                with open(config_file, "r", encoding="utf-8") as f:
                    package_data = json.load(f)

                dependencies = {
                    **package_data.get("dependencies", {}),
                    **package_data.get("devDependencies", {}),
                }

                framework_indicators = {
                    "react": "React",
                    "vue": "Vue.js",
                    "angular": "Angular",
                    "express": "Express.js",
                    "fastify": "Fastify",
                    "next": "Next.js",
                    "nuxt": "Nuxt.js",
                    "svelte": "Svelte",
                }

                for dep, framework in framework_indicators.items():
                    if any(dep in key.lower() for key in dependencies.keys()):
                        frameworks.append(framework)

            elif config_file.name in ["requirements.txt", "pyproject.toml", "setup.py"]:
                # Python frameworks
                content = config_file.read_text(encoding="utf-8").lower()

                framework_indicators = {
                    "django": "Django",
                    "flask": "Flask",
                    "fastapi": "FastAPI",
                    "starlette": "Starlette",
                    "tornado": "Tornado",
                    "pyramid": "Pyramid",
                    "bottle": "Bottle",
                }

                for indicator, framework in framework_indicators.items():
                    if indicator in content:
                        frameworks.append(framework)

        except Exception as e:
            logger.debug(f"Could not detect framework from {config_file}: {e}")

        return frameworks

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension

        Args:
            file_path: Path to file

        Returns:
            Detected language or 'unknown'
        """
        extension = file_path.suffix.lower()

        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "c",
            ".h": "c",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".clj": "clojure",
            ".hs": "haskell",
            ".erl": "erlang",
            ".ex": "elixir",
            ".html": "html",
            ".htm": "html",
            ".css": "css",
            ".scss": "scss",
            ".less": "less",
            ".sql": "sql",
            ".sh": "shell",
            ".bash": "bash",
            ".ps1": "powershell",
            ".dockerfile": "dockerfile",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".xml": "xml",
            ".md": "markdown",
            ".rst": "markdown",
            ".txt": "text",
        }

        # Special case for files without extension
        if not extension:
            file_name = file_path.name.lower()
            if file_name in ["dockerfile", "makefile", "rakefile"]:
                return file_name
            elif file_name.startswith("readme"):
                return "markdown"

        return language_map.get(extension, "unknown")

    def _matches_patterns(self, file_path: str, patterns: List[str]) -> bool:
        """Check if file path matches any of the given patterns

        Args:
            file_path: File path to check
            patterns: List of glob patterns

        Returns:
            True if file matches any pattern
        """
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(file_path, pattern):
                return True

            # Also check if any parent directory matches
            path_parts = file_path.split("/")
            for i in range(len(path_parts)):
                partial_path = "/".join(path_parts[: i + 1])
                if fnmatch.fnmatch(partial_path, pattern):
                    return True

        return False

    def _calculate_language_stats(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate language statistics

        Args:
            files: List of discovered files

        Returns:
            Dictionary with language statistics
        """
        language_counts = {}
        language_sizes = {}
        total_size = 0

        for file_info in files:
            language = file_info["language"]
            size = file_info["size"]

            language_counts[language] = language_counts.get(language, 0) + 1
            language_sizes[language] = language_sizes.get(language, 0) + size
            total_size += size

        # Calculate percentages
        language_percentages = {}
        for language, size in language_sizes.items():
            language_percentages[language] = (
                (size / total_size * 100) if total_size > 0 else 0
            )

        return {
            "language_counts": language_counts,
            "language_sizes": language_sizes,
            "language_percentages": language_percentages,
            "total_files": len(files),
            "total_size": total_size,
            "primary_language": (
                max(language_sizes.keys(), key=language_sizes.get)
                if language_sizes
                else "unknown"
            ),
        }

    async def cleanup_repository(self, repository_path: str) -> Dict[str, Any]:
        """Cleanup cloned repository

        Args:
            repository_path: Path to repository to cleanup

        Returns:
            Dictionary with cleanup results
        """
        try:
            repo_path = Path(repository_path)
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)

                return {
                    "status": "success",
                    "message": f"Repository cleaned up: {repository_path}",
                }
            else:
                return {
                    "status": "warning",
                    "message": f"Repository path does not exist: {repository_path}",
                }

        except Exception as e:
            logger.error(f"Repository cleanup failed: {e}")
            return {"status": "error", "error": str(e)}


# Tool instance for LangGraph
# Deprecated: Module-level singleton removed
# Use get_repository_tool() from src.dependencies with FastAPI's Depends() instead
# repository_tool = RepositoryTool()  # REMOVED - use dependency injection
