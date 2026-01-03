"""Local filesystem storage adapter

This module implements the StorageAdapter interface for local filesystem storage.
Provides file operations using the local filesystem with async support.
"""

import asyncio
import fnmatch
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import aiofiles
import aiofiles.os

from ..models.config import StorageConfig
from .storage_adapters import (
    FileMetadata,
    FileNotFoundError,
    PermissionError,
    StorageAdapter,
    StorageError,
)


class LocalStorageAdapter(StorageAdapter):
    """Local filesystem storage adapter

    Implements storage operations using the local filesystem.
    All paths are relative to the configured base_path.
    """

    def __init__(self, config: StorageConfig):
        """Initialize local storage adapter

        Args:
            config: Storage configuration with base_path
        """
        super().__init__(config)
        self.base_path = Path(config.base_path).resolve()
        self.create_dirs = config.get_connection_param("create_dirs", True)
        self.permissions = config.get_connection_param("permissions", "0755")

    async def initialize(self) -> None:
        """Initialize the local storage adapter

        Creates the base directory if it doesn't exist and create_dirs is True.

        Raises:
            StorageError: If initialization fails
        """
        try:
            if self.create_dirs and not self.base_path.exists():
                self.base_path.mkdir(parents=True, exist_ok=True)

                # Set permissions if specified
                if self.permissions:
                    mode = int(self.permissions, 8)
                    os.chmod(self.base_path, mode)

            # Verify base path exists and is accessible
            if not self.base_path.exists():
                raise StorageError(f"Base path does not exist: {self.base_path}")

            if not self.base_path.is_dir():
                raise StorageError(f"Base path is not a directory: {self.base_path}")

            # Test read/write access
            test_file = self.base_path / ".autodoc_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except OSError as e:
                raise StorageError(f"No write access to base path: {e}")

            self._initialized = True

        except OSError as e:
            raise StorageError(f"Failed to initialize local storage: {e}")

    async def cleanup(self) -> None:
        """Cleanup resources

        For local storage, there are no persistent connections to clean up.
        """
        self._initialized = False

    def _get_full_path(self, path: str) -> Path:
        """Get full filesystem path from relative path

        Args:
            path: Relative path

        Returns:
            Full filesystem path
        """
        normalized_path = self.normalize_path(path)
        if not normalized_path:
            return self.base_path

        full_path = self.base_path / normalized_path

        # Ensure path is within base_path (security check)
        try:
            full_path.resolve().relative_to(self.base_path.resolve())
        except ValueError:
            raise StorageError(f"Path outside base directory: {path}")

        return full_path

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
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if not full_path.is_file():
                raise StorageError(f"Path is not a file: {path}")

            async with aiofiles.open(full_path, "rb") as f:
                return await f.read()

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"File not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error reading file {path}: {e}")

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
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if not full_path.is_file():
                raise StorageError(f"Path is not a file: {path}")

            async with aiofiles.open(full_path, "r", encoding=encoding) as f:
                return await f.read()

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"File not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error reading file {path}: {e}")
        except UnicodeDecodeError as e:
            raise StorageError(
                f"Error decoding file {path} with encoding {encoding}: {e}"
            )

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
            metadata: Optional metadata (stored as extended attributes if supported)

        Raises:
            PermissionError: If write access is denied
            StorageError: For other storage errors
        """
        try:
            full_path = self._get_full_path(path)

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            # Write content
            if isinstance(content, str):
                async with aiofiles.open(full_path, "w", encoding="utf-8") as f:
                    await f.write(content)
            else:
                async with aiofiles.open(full_path, "wb") as f:
                    await f.write(content)

            # Set file permissions if specified
            if self.permissions:
                mode = int(self.permissions, 8)
                os.chmod(full_path, mode)

            # Store metadata as extended attributes (if supported and metadata provided)
            if metadata and hasattr(os, "setxattr"):
                try:
                    for key, value in metadata.items():
                        attr_name = f"user.autodoc.{key}"
                        attr_value = str(value).encode("utf-8")
                        os.setxattr(full_path, attr_name, attr_value)
                except OSError:
                    # Extended attributes not supported, ignore
                    pass

        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            elif e.errno == 28:  # No space left on device
                raise StorageError(f"Storage full: {path}")
            else:
                raise StorageError(f"Error writing file {path}: {e}")

    async def delete_file(self, path: str) -> None:
        """Delete a file

        Args:
            path: File path relative to storage root

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If delete access is denied
            StorageError: For other storage errors
        """
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if not full_path.is_file():
                raise StorageError(f"Path is not a file: {path}")

            await aiofiles.os.remove(full_path)

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"File not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error deleting file {path}: {e}")

    async def file_exists(self, path: str) -> bool:
        """Check if file exists

        Args:
            path: File path relative to storage root

        Returns:
            True if file exists, False otherwise
        """
        try:
            full_path = self._get_full_path(path)
            return full_path.exists() and full_path.is_file()
        except (OSError, StorageError):
            return False

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
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if not full_path.is_file():
                raise StorageError(f"Path is not a file: {path}")

            stat_result = full_path.stat()

            # Get extended attributes if supported
            extended_metadata = {}
            if hasattr(os, "listxattr"):
                try:
                    attrs = os.listxattr(full_path)
                    for attr in attrs:
                        if attr.startswith("user.autodoc."):
                            key = attr[len("user.autodoc.") :]
                            value = os.getxattr(full_path, attr).decode("utf-8")
                            extended_metadata[key] = value
                except OSError:
                    # Extended attributes not supported, ignore
                    pass

            return FileMetadata(
                path=path,
                size=stat_result.st_size,
                modified_at=datetime.fromtimestamp(stat_result.st_mtime),
                created_at=datetime.fromtimestamp(stat_result.st_ctime),
                content_type=self.get_content_type(path),
                etag=f"{stat_result.st_mtime}-{stat_result.st_size}",
                metadata=extended_metadata,
            )

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"File not found: {path}")
            else:
                raise StorageError(f"Error getting file metadata {path}: {e}")

    async def create_directory(self, path: str) -> None:
        """Create directory (and parent directories if needed)

        Args:
            path: Directory path relative to storage root

        Raises:
            PermissionError: If create access is denied
            StorageError: For other storage errors
        """
        try:
            full_path = self._get_full_path(path)
            full_path.mkdir(parents=True, exist_ok=True)

            # Set directory permissions if specified
            if self.permissions:
                mode = int(self.permissions, 8)
                os.chmod(full_path, mode)

        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error creating directory {path}: {e}")

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
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")

            if not full_path.is_dir():
                raise StorageError(f"Path is not a directory: {path}")

            if recursive:
                shutil.rmtree(full_path)
            else:
                full_path.rmdir()

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"Directory not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            elif e.errno == 39:  # Directory not empty
                raise StorageError(f"Directory not empty: {path}")
            else:
                raise StorageError(f"Error deleting directory {path}: {e}")

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
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")

            if not full_path.is_dir():
                raise StorageError(f"Path is not a directory: {path}")

            files = []

            if recursive:
                for root, dirs, filenames in os.walk(full_path):
                    for filename in filenames:
                        file_path = Path(root) / filename
                        relative_path = file_path.relative_to(self.base_path)
                        relative_str = str(relative_path).replace("\\", "/")

                        if pattern is None or fnmatch.fnmatch(filename, pattern):
                            files.append(relative_str)
            else:
                for item in full_path.iterdir():
                    if item.is_file():
                        relative_path = item.relative_to(self.base_path)
                        relative_str = str(relative_path).replace("\\", "/")

                        if pattern is None or fnmatch.fnmatch(item.name, pattern):
                            files.append(relative_str)

            return sorted(files)

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"Directory not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error listing files in {path}: {e}")

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
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"Directory not found: {path}")

            if not full_path.is_dir():
                raise StorageError(f"Path is not a directory: {path}")

            directories = []

            if recursive:
                for root, dirnames, files in os.walk(full_path):
                    for dirname in dirnames:
                        dir_path = Path(root) / dirname
                        relative_path = dir_path.relative_to(self.base_path)
                        relative_str = str(relative_path).replace("\\", "/")
                        directories.append(relative_str)
            else:
                for item in full_path.iterdir():
                    if item.is_dir():
                        relative_path = item.relative_to(self.base_path)
                        relative_str = str(relative_path).replace("\\", "/")
                        directories.append(relative_str)

            return sorted(directories)

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"Directory not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error listing directories in {path}: {e}")

    async def copy_file(self, source_path: str, dest_path: str) -> None:
        """Copy file to new location

        Uses efficient filesystem copy operation.

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        try:
            source_full_path = self._get_full_path(source_path)
            dest_full_path = self._get_full_path(dest_path)

            if not source_full_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            if not source_full_path.is_file():
                raise StorageError(f"Source path is not a file: {source_path}")

            # Create destination directory if needed
            dest_full_path.parent.mkdir(parents=True, exist_ok=True)

            # Use efficient copy operation
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.copy2, source_full_path, dest_full_path
            )

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"Source file not found: {source_path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(
                    f"Permission denied copying {source_path} to {dest_path}"
                )
            else:
                raise StorageError(
                    f"Error copying file {source_path} to {dest_path}: {e}"
                )

    async def move_file(self, source_path: str, dest_path: str) -> None:
        """Move file to new location

        Uses efficient filesystem move operation.

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        try:
            source_full_path = self._get_full_path(source_path)
            dest_full_path = self._get_full_path(dest_path)

            if not source_full_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")

            if not source_full_path.is_file():
                raise StorageError(f"Source path is not a file: {source_path}")

            # Create destination directory if needed
            dest_full_path.parent.mkdir(parents=True, exist_ok=True)

            # Use efficient move operation
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.move, source_full_path, dest_full_path
            )

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"Source file not found: {source_path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(
                    f"Permission denied moving {source_path} to {dest_path}"
                )
            else:
                raise StorageError(
                    f"Error moving file {source_path} to {dest_path}: {e}"
                )

    async def read_file_stream(
        self, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Read file as stream of chunks

        Args:
            path: File path relative to storage root
            chunk_size: Size of each chunk in bytes

        Yields:
            Chunks of file content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            StorageError: For other storage errors
        """
        try:
            full_path = self._get_full_path(path)

            if not full_path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            if not full_path.is_file():
                raise StorageError(f"Path is not a file: {path}")

            async with aiofiles.open(full_path, "rb") as f:
                while True:
                    chunk = await f.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

        except OSError as e:
            if e.errno == 2:  # No such file or directory
                raise FileNotFoundError(f"File not found: {path}")
            elif e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error reading file stream {path}: {e}")

    async def write_file_stream(
        self,
        path: str,
        content_stream: AsyncIterator[bytes],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write file from stream of chunks

        Args:
            path: File path relative to storage root
            content_stream: Stream of content chunks
            metadata: Optional metadata to store with file

        Raises:
            PermissionError: If write access is denied
            StorageError: For other storage errors
        """
        try:
            full_path = self._get_full_path(path)

            # Create parent directories if needed
            full_path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(full_path, "wb") as f:
                async for chunk in content_stream:
                    await f.write(chunk)

            # Set file permissions if specified
            if self.permissions:
                mode = int(self.permissions, 8)
                os.chmod(full_path, mode)

            # Store metadata as extended attributes (if supported and metadata provided)
            if metadata and hasattr(os, "setxattr"):
                try:
                    for key, value in metadata.items():
                        attr_name = f"user.autodoc.{key}"
                        attr_value = str(value).encode("utf-8")
                        os.setxattr(full_path, attr_name, attr_value)
                except OSError:
                    # Extended attributes not supported, ignore
                    pass

        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise PermissionError(f"Permission denied: {path}")
            elif e.errno == 28:  # No space left on device
                raise StorageError(f"Storage full: {path}")
            else:
                raise StorageError(f"Error writing file stream {path}: {e}")


# Register the local storage adapter
from .storage_adapters import StorageAdapterFactory

StorageAdapterFactory.register_adapter("local", LocalStorageAdapter)
