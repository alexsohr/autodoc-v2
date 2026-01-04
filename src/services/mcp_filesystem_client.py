"""MCP Filesystem Client Service

This module provides a persistent MCP client for interacting with the
fast-filesystem-mcp server, enabling high-performance filesystem operations
for repository analysis.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..utils.config_loader import Settings, get_settings

logger = structlog.get_logger(__name__)


class MCPFilesystemClient:
    """MCP client for filesystem operations.

    This client manages a connection to the fast-filesystem-mcp server,
    providing access to filesystem tools for agents. Agents use the raw
    MCP tools directly via get_tools().

    Can be used as:
    - A singleton for shared usage (via get_instance())
    - Per-worker instances for concurrent usage (via create_for_worker())

    For concurrent workers (like page generation), use create_for_worker() to
    give each worker its own stdio connection, avoiding connection contention.
    """

    _instance: Optional["MCPFilesystemClient"] = None

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the MCP filesystem client.

        Args:
            settings: Application settings. If None, uses global settings.
        """
        self._settings = settings or get_settings()
        self._client: Optional[MultiServerMCPClient] = None
        self._tools: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._initialized: bool = False

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

    @classmethod
    async def create_for_worker(
        cls,
        settings: Optional[Settings] = None,
        worker_id: Optional[str] = None,
    ) -> Optional["MCPFilesystemClient"]:
        """Create a new MCP client instance for a worker.

        Unlike get_instance(), this creates a new connection for each call,
        giving each worker its own stdio connection to the MCP server.
        This prevents connection contention when multiple workers run concurrently.

        The caller is responsible for calling shutdown() when done.

        Args:
            settings: Application settings. If None, uses global settings.
            worker_id: Optional identifier for logging purposes.

        Returns:
            Initialized client or None if disabled/failed.
        """
        settings = settings or get_settings()
        if not settings.mcp_filesystem_enabled:
            logger.debug(
                "MCP filesystem disabled, skipping worker client creation",
                worker_id=worker_id,
            )
            return None

        client = cls(settings)
        if await client.initialize():
            logger.info(
                "Created MCP filesystem client for worker",
                worker_id=worker_id,
                num_tools=len(client._tools),
            )
            return client

        logger.warning(
            "Failed to create MCP filesystem client for worker",
            worker_id=worker_id,
        )
        return None

    @property
    def is_enabled(self) -> bool:
        """Check if MCP filesystem is enabled in settings."""
        return self._settings.mcp_filesystem_enabled

    @property
    def is_initialized(self) -> bool:
        """Check if the client has been initialized."""
        return self._initialized and self._client is not None

    def get_tools(self, names: Optional[List[str]] = None) -> List[Any]:
        """Get MCP tools, optionally filtered by name.

        Args:
            names: Optional list of tool names to include. If None, returns all tools.

        Returns:
            List of tool objects that can be passed to agents.
        """
        if not self._initialized:
            return []
        if names is None:
            return list(self._tools.values())
        return [self._tools[name] for name in names if name in self._tools]

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
                # Build args list, automatically including storage path for repo access
                args = self._settings.mcp_filesystem_args_list.copy()

                # Add the storage base path (where repos are cloned) to allowed directories
                storage_path = Path(self._settings.storage_base_path).resolve()
                if storage_path.exists():
                    # Add as absolute path for MCP server access control
                    storage_path_str = str(storage_path)
                    if storage_path_str not in args:
                        args.append(storage_path_str)
                        logger.info(
                            "Added storage path to MCP allowed directories",
                            path=storage_path_str,
                        )

                logger.info(
                    "Initializing MCP filesystem client",
                    command=self._settings.mcp_filesystem_command,
                    args=args,
                )

                self._client = MultiServerMCPClient(
                    {
                        "filesystem": {
                            "transport": "stdio",
                            "command": self._settings.mcp_filesystem_command,
                            "args": args,
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
