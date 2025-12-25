"""MCP Filesystem Client Service

This module provides a persistent MCP client for interacting with the
fast-filesystem-mcp server, enabling high-performance filesystem operations
for repository analysis.
"""

import asyncio
from typing import Any, Dict, List, Optional

import structlog
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..utils.config_loader import Settings, get_settings

logger = structlog.get_logger(__name__)


class MCPFilesystemClient:
    """Persistent MCP client for filesystem operations.

    This client manages a connection to the fast-filesystem-mcp server,
    providing methods for file discovery, reading, and search operations.
    It is designed to be initialized once at application startup and
    reused across all workflow executions.
    """

    _instance: Optional["MCPFilesystemClient"] = None
    _initialized: bool = False

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the MCP filesystem client.

        Args:
            settings: Application settings. If None, uses global settings.
        """
        self._settings = settings or get_settings()
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls, settings: Optional[Settings] = None) -> "MCPFilesystemClient":
        """Get the singleton instance of MCPFilesystemClient.

        Args:
            settings: Application settings for first initialization.

        Returns:
            The singleton MCPFilesystemClient instance.
        """
        if cls._instance is None:
            cls._instance = cls(settings)
        return cls._instance

    @property
    def is_enabled(self) -> bool:
        """Check if MCP filesystem is enabled in settings."""
        return self._settings.mcp_filesystem_enabled

    @property
    def is_initialized(self) -> bool:
        """Check if the client has been initialized."""
        return self._initialized and self._client is not None

    async def initialize(self) -> bool:
        """Initialize the MCP client connection.

        Returns:
            True if initialization was successful, False otherwise.
        """
        if not self.is_enabled:
            logger.info("MCP filesystem is disabled in settings")
            return False

        async with self._lock:
            if self._initialized:
                logger.debug("MCP filesystem client already initialized")
                return True

            try:
                logger.info(
                    "Initializing MCP filesystem client",
                    command=self._settings.mcp_filesystem_command,
                    args=self._settings.mcp_filesystem_args_list,
                )

                self._client = MultiServerMCPClient(
                    {
                        "filesystem": {
                            "transport": "stdio",
                            "command": self._settings.mcp_filesystem_command,
                            "args": self._settings.mcp_filesystem_args_list,
                        }
                    }
                )

                # Get available tools and cache them
                tools = await self._client.get_tools()
                self._tools = {tool.name: tool for tool in tools}

                self._initialized = True
                logger.info(
                    "MCP filesystem client initialized successfully",
                    available_tools=list(self._tools.keys()),
                )
                return True

            except Exception as e:
                logger.error(
                    "Failed to initialize MCP filesystem client",
                    error=str(e),
                    exc_info=True,
                )
                self._initialized = False
                return False

    async def shutdown(self) -> None:
        """Shutdown the MCP client and release resources."""
        async with self._lock:
            if self._client is not None:
                try:
                    # Close the client connection
                    if hasattr(self._client, "close"):
                        await self._client.close()
                    logger.info("MCP filesystem client shutdown complete")
                except Exception as e:
                    logger.error(
                        "Error during MCP filesystem client shutdown",
                        error=str(e),
                    )
                finally:
                    self._client = None
                    self._tools = {}
                    self._initialized = False

    def _get_tool(self, tool_name: str) -> Any:
        """Get a tool by name.

        Args:
            tool_name: The name of the tool to retrieve.

        Returns:
            The tool object.

        Raises:
            ValueError: If the tool is not found or client not initialized.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        if tool_name not in self._tools:
            raise ValueError(
                f"Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}"
            )

        return self._tools[tool_name]

    async def list_directory(
        self,
        path: str,
        recursive: bool = False,
        page: int = 1,
        page_size: int = 100,
    ) -> Dict[str, Any]:
        """List directory contents using fast_list_directory.

        Args:
            path: The directory path to list.
            recursive: Whether to list recursively.
            page: Page number for pagination.
            page_size: Number of items per page.

        Returns:
            Dictionary with directory listing results.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            tool = self._get_tool("fast_list_directory")
            result = await tool.ainvoke({
                "path": path,
                "recursive": recursive,
                "page": page,
                "pageSize": page_size,
            })
            return {"status": "success", "data": result}
        except Exception as e:
            logger.error("Error listing directory", path=path, error=str(e))
            return {"status": "error", "error": str(e)}

    async def read_file(self, path: str) -> Dict[str, Any]:
        """Read file contents using fast_read_file.

        Args:
            path: The file path to read.

        Returns:
            Dictionary with file contents or error.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            tool = self._get_tool("fast_read_file")
            result = await tool.ainvoke({"path": path})
            return {"status": "success", "content": result}
        except Exception as e:
            logger.error("Error reading file", path=path, error=str(e))
            return {"status": "error", "error": str(e)}

    async def get_file_info(self, path: str) -> Dict[str, Any]:
        """Get file information using fast_get_file_info.

        Args:
            path: The file path to get info for.

        Returns:
            Dictionary with file metadata.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            tool = self._get_tool("fast_get_file_info")
            result = await tool.ainvoke({"path": path})
            return {"status": "success", "info": result}
        except Exception as e:
            logger.error("Error getting file info", path=path, error=str(e))
            return {"status": "error", "error": str(e)}

    async def get_directory_tree(
        self,
        path: str,
        max_depth: int = 5,
    ) -> Dict[str, Any]:
        """Get directory tree using fast_get_directory_tree.

        Args:
            path: The root path for the tree.
            max_depth: Maximum depth to traverse.

        Returns:
            Dictionary with directory tree structure.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            tool = self._get_tool("fast_get_directory_tree")
            result = await tool.ainvoke({
                "path": path,
                "maxDepth": max_depth,
            })
            return {"status": "success", "tree": result}
        except Exception as e:
            logger.error("Error getting directory tree", path=path, error=str(e))
            return {"status": "error", "error": str(e)}

    async def search_files(
        self,
        path: str,
        pattern: str,
        recursive: bool = True,
    ) -> Dict[str, Any]:
        """Search for files by name pattern using fast_search_files.

        Args:
            path: The directory to search in.
            pattern: The file name pattern (glob-style).
            recursive: Whether to search recursively.

        Returns:
            Dictionary with matching files.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            tool = self._get_tool("fast_search_files")
            result = await tool.ainvoke({
                "path": path,
                "pattern": pattern,
                "recursive": recursive,
            })
            return {"status": "success", "files": result}
        except Exception as e:
            logger.error(
                "Error searching files",
                path=path,
                pattern=pattern,
                error=str(e),
            )
            return {"status": "error", "error": str(e)}

    async def search_code(
        self,
        path: str,
        pattern: str,
        file_pattern: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Search for code patterns using fast_search_code.

        Args:
            path: The directory to search in.
            pattern: The code pattern to search for.
            file_pattern: Optional file pattern filter.

        Returns:
            Dictionary with matching code locations.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            tool = self._get_tool("fast_search_code")
            params: Dict[str, Any] = {
                "path": path,
                "pattern": pattern,
            }
            if file_pattern:
                params["filePattern"] = file_pattern

            result = await tool.ainvoke(params)
            return {"status": "success", "matches": result}
        except Exception as e:
            logger.error(
                "Error searching code",
                path=path,
                pattern=pattern,
                error=str(e),
            )
            return {"status": "error", "error": str(e)}

    async def discover_files(
        self,
        path: str,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Discover files in a directory with filtering.

        This is a high-level method that combines list_directory with
        pattern filtering for repository file discovery.

        Args:
            path: The directory path to discover files in.
            include_patterns: File patterns to include (glob-style).
            exclude_patterns: File patterns to exclude (glob-style).

        Returns:
            Dictionary with discovered files list.
        """
        if not self._initialized:
            raise ValueError("MCP filesystem client not initialized")

        try:
            # Use directory tree for efficient discovery
            tree_result = await self.get_directory_tree(path, max_depth=10)

            if tree_result["status"] != "success":
                return tree_result

            # For now, just return the tree - transformation can be added later
            # based on how the DocumentAgent needs the data
            return {
                "status": "success",
                "tree": tree_result.get("tree"),
                "include_patterns": include_patterns,
                "exclude_patterns": exclude_patterns,
            }
        except Exception as e:
            logger.error(
                "Error discovering files",
                path=path,
                error=str(e),
            )
            return {"status": "error", "error": str(e)}


# Global instance accessor
_mcp_client: Optional[MCPFilesystemClient] = None


def get_mcp_filesystem_client() -> MCPFilesystemClient:
    """Get the global MCP filesystem client instance.

    Returns:
        The global MCPFilesystemClient instance.
    """
    global _mcp_client
    if _mcp_client is None:
        _mcp_client = MCPFilesystemClient.get_instance()
    return _mcp_client


async def init_mcp_filesystem() -> bool:
    """Initialize the global MCP filesystem client.

    Returns:
        True if initialization was successful, False otherwise.
    """
    client = get_mcp_filesystem_client()
    return await client.initialize()


async def close_mcp_filesystem() -> None:
    """Shutdown the global MCP filesystem client."""
    global _mcp_client
    if _mcp_client is not None:
        await _mcp_client.shutdown()
        _mcp_client = None
