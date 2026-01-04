"""Storage adapter interface and base classes

This module defines the abstract base classes and interfaces for different
storage adapters (local filesystem, AWS S3, etc.) used by AutoDoc v2.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, BinaryIO, Dict, List, Optional, Union

from ..models.config import StorageConfig


class StorageError(Exception):
    """Base exception for storage operations"""

    pass


class FileNotFoundError(StorageError):
    """File not found in storage"""

    pass


class PermissionError(StorageError):
    """Permission denied for storage operation"""

    pass


class StorageFullError(StorageError):
    """Storage is full or quota exceeded"""

    pass


class FileMetadata:
    """File metadata information"""

    def __init__(
        self,
        path: str,
        size: int,
        modified_at: datetime,
        created_at: Optional[datetime] = None,
        content_type: Optional[str] = None,
        etag: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.path = path
        self.size = size
        self.modified_at = modified_at
        self.created_at = created_at or modified_at
        self.content_type = content_type
        self.etag = etag
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"FileMetadata(path={self.path}, size={self.size}, modified_at={self.modified_at})"


class StorageAdapter(ABC):
    """Abstract base class for storage adapters

    Defines the interface that all storage adapters must implement
    for consistent file operations across different storage backends.
    """

    def __init__(self, config: StorageConfig):
        """Initialize storage adapter with configuration

        Args:
            config: Storage configuration object
        """
        self.config = config
        self._initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the storage adapter

        This method should be called before any other operations.
        It should set up connections, validate configuration, etc.

        Raises:
            StorageError: If initialization fails
        """
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources and close connections

        This method should be called when the adapter is no longer needed.
        """
        pass

    # File operations

    @abstractmethod
    async def read_file(self, path: str) -> bytes:
        """Read file content as bytes

        Args:
            path: File path relative to storage root

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def read_text_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read file content as text

        Args:
            path: File path relative to storage root
            encoding: Text encoding (default: utf-8)

        Returns:
            File content as string

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def write_file(
        self,
        path: str,
        content: Union[bytes, str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write content to file

        Args:
            path: File path relative to storage root
            content: Content to write (bytes or string)
            metadata: Optional metadata to store with file

        Raises:
            PermissionError: If write access is denied
            StorageFullError: If storage is full
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def delete_file(self, path: str) -> None:
        """Delete a file

        Args:
            path: File path relative to storage root

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If delete access is denied
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def file_exists(self, path: str) -> bool:
        """Check if file exists

        Args:
            path: File path relative to storage root

        Returns:
            True if file exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get file metadata

        Args:
            path: File path relative to storage root

        Returns:
            File metadata object

        Raises:
            FileNotFoundError: If file doesn't exist
            StorageError: For other storage errors
        """
        pass

    # Directory operations

    @abstractmethod
    async def create_directory(self, path: str) -> None:
        """Create directory (and parent directories if needed)

        Args:
            path: Directory path relative to storage root

        Raises:
            PermissionError: If create access is denied
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def delete_directory(self, path: str, recursive: bool = False) -> None:
        """Delete directory

        Args:
            path: Directory path relative to storage root
            recursive: If True, delete directory and all contents

        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If delete access is denied
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def list_files(
        self, path: str = "", recursive: bool = False, pattern: Optional[str] = None
    ) -> List[str]:
        """List files in directory

        Args:
            path: Directory path relative to storage root (empty = root)
            recursive: If True, list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of file paths relative to storage root

        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If list access is denied
            StorageError: For other storage errors
        """
        pass

    @abstractmethod
    async def list_directories(
        self, path: str = "", recursive: bool = False
    ) -> List[str]:
        """List directories

        Args:
            path: Directory path relative to storage root (empty = root)
            recursive: If True, list directories recursively

        Returns:
            List of directory paths relative to storage root

        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If list access is denied
            StorageError: For other storage errors
        """
        pass

    # Batch operations

    async def copy_file(self, source_path: str, dest_path: str) -> None:
        """Copy file to new location

        Default implementation reads and writes. Subclasses can override
        for more efficient provider-specific copy operations.

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        content = await self.read_file(source_path)
        metadata = await self.get_file_metadata(source_path)
        await self.write_file(dest_path, content, metadata.metadata)

    async def move_file(self, source_path: str, dest_path: str) -> None:
        """Move file to new location

        Default implementation copies then deletes. Subclasses can override
        for more efficient provider-specific move operations.

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        await self.copy_file(source_path, dest_path)
        await self.delete_file(source_path)

    async def get_storage_usage(self, path: str = "") -> Dict[str, int]:
        """Get storage usage statistics

        Args:
            path: Path to analyze (empty = entire storage)

        Returns:
            Dictionary with 'total_files', 'total_size' keys
        """
        files = await self.list_files(path, recursive=True)
        total_files = len(files)
        total_size = 0

        for file_path in files:
            try:
                metadata = await self.get_file_metadata(file_path)
                total_size += metadata.size
            except (FileNotFoundError, StorageError):
                # Skip files that can't be accessed
                continue

        return {"total_files": total_files, "total_size": total_size}

    # Stream operations

    async def read_file_stream(
        self, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Read file as stream of chunks

        Default implementation reads entire file and yields chunks.
        Subclasses should override for true streaming support.

        Args:
            path: File path relative to storage root
            chunk_size: Size of each chunk in bytes

        Yields:
            Chunks of file content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            StorageError: For other storage errors
        """
        content = await self.read_file(path)
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    async def write_file_stream(
        self,
        path: str,
        content_stream: AsyncIterator[bytes],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write file from stream of chunks

        Default implementation collects all chunks then writes.
        Subclasses should override for true streaming support.

        Args:
            path: File path relative to storage root
            content_stream: Stream of content chunks
            metadata: Optional metadata to store with file

        Raises:
            PermissionError: If write access is denied
            StorageFullError: If storage is full
            StorageError: For other storage errors
        """
        content = b""
        async for chunk in content_stream:
            content += chunk

        await self.write_file(path, content, metadata)

    # Utility methods

    def normalize_path(self, path: str) -> str:
        """Normalize path for consistent handling

        Args:
            path: Input path

        Returns:
            Normalized path
        """
        # Remove leading/trailing slashes and normalize separators
        path = path.strip("/\\")
        path = path.replace("\\", "/")

        # Remove double slashes
        while "//" in path:
            path = path.replace("//", "/")

        return path

    def get_file_extension(self, path: str) -> str:
        """Get file extension from path

        Args:
            path: File path

        Returns:
            File extension (without dot) or empty string
        """
        return Path(path).suffix.lstrip(".")

    def get_content_type(self, path: str) -> str:
        """Get MIME content type for file

        Args:
            path: File path

        Returns:
            MIME content type string
        """
        extension = self.get_file_extension(path).lower()

        content_types = {
            "txt": "text/plain",
            "md": "text/markdown",
            "py": "text/x-python",
            "js": "application/javascript",
            "ts": "application/typescript",
            "json": "application/json",
            "yaml": "application/x-yaml",
            "yml": "application/x-yaml",
            "xml": "application/xml",
            "html": "text/html",
            "css": "text/css",
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "svg": "image/svg+xml",
        }

        return content_types.get(extension, "application/octet-stream")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on storage adapter

        Returns:
            Dictionary with health check results
        """
        try:
            # Try to list files in root to test connectivity
            await self.list_files("", recursive=False)

            return {
                "status": "healthy",
                "adapter_type": self.__class__.__name__,
                "base_path": self.config.base_path,
                "initialized": self._initialized,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "adapter_type": self.__class__.__name__,
                "base_path": self.config.base_path,
                "initialized": self._initialized,
                "error": str(e),
            }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(base_path={self.config.base_path})"


class StorageAdapterFactory:
    """Factory for creating storage adapters"""

    _adapters: Dict[str, type] = {}

    @classmethod
    def register_adapter(cls, storage_type: str, adapter_class: type) -> None:
        """Register a storage adapter class

        Args:
            storage_type: Storage type identifier
            adapter_class: Adapter class to register
        """
        cls._adapters[storage_type] = adapter_class

    @classmethod
    def create_adapter(cls, config: StorageConfig) -> StorageAdapter:
        """Create storage adapter from configuration

        Args:
            config: Storage configuration

        Returns:
            Storage adapter instance

        Raises:
            ValueError: If storage type is not supported
        """
        storage_type = (
            config.type.value if hasattr(config.type, "value") else str(config.type)
        )
        adapter_class = cls._adapters.get(storage_type)
        if adapter_class is None:
            raise ValueError(f"Unsupported storage type: {config.type}")

        return adapter_class(config)

    @classmethod
    def get_supported_types(cls) -> List[str]:
        """Get list of supported storage types

        Returns:
            List of supported storage type identifiers
        """
        return list(cls._adapters.keys())
