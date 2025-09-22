"""AWS S3 storage adapter

This module implements the StorageAdapter interface for AWS S3 storage.
Provides file operations using boto3 with async support.
"""

import asyncio
import fnmatch
from datetime import datetime
from io import BytesIO
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import boto3
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from ..models.config import StorageConfig
from .storage_adapters import (
    FileMetadata,
    FileNotFoundError,
    PermissionError,
    StorageAdapter,
    StorageError,
    StorageFullError,
)


class S3StorageAdapter(StorageAdapter):
    """AWS S3 storage adapter

    Implements storage operations using AWS S3 with async support.
    All paths are treated as S3 object keys with the base_path as prefix.
    """

    def __init__(self, config: StorageConfig):
        """Initialize S3 storage adapter

        Args:
            config: Storage configuration with S3 parameters
        """
        super().__init__(config)

        # Extract S3-specific configuration
        self.bucket_name = (
            config.get_connection_param("bucket_name") or config.base_path
        )
        self.region = config.get_connection_param("region", "us-east-1")
        self.access_key_id = config.get_connection_param("aws_access_key_id")
        self.secret_access_key = config.get_connection_param("aws_secret_access_key")
        self.endpoint_url = config.get_connection_param(
            "endpoint_url"
        )  # For S3-compatible services

        # S3 client configuration
        self.storage_class = config.get_connection_param("storage_class", "STANDARD")
        self.server_side_encryption = config.get_connection_param(
            "server_side_encryption", "AES256"
        )
        self.versioning_enabled = config.get_connection_param(
            "versioning_enabled", True
        )

        # Client instances (initialized in initialize())
        self.s3_client = None
        self.s3_resource = None
        self.bucket = None

    async def initialize(self) -> None:
        """Initialize the S3 storage adapter

        Creates S3 client, validates bucket access, and sets up bucket configuration.

        Raises:
            StorageError: If initialization fails
        """
        try:
            # Create boto3 session
            session_kwargs = {"region_name": self.region}

            if self.access_key_id and self.secret_access_key:
                session_kwargs.update(
                    {
                        "aws_access_key_id": self.access_key_id,
                        "aws_secret_access_key": self.secret_access_key,
                    }
                )

            session = boto3.Session(**session_kwargs)

            # Create S3 client and resource
            client_kwargs = {"region_name": self.region}
            if self.endpoint_url:
                client_kwargs["endpoint_url"] = self.endpoint_url

            self.s3_client = session.client("s3", **client_kwargs)
            self.s3_resource = session.resource("s3", **client_kwargs)
            self.bucket = self.s3_resource.Bucket(self.bucket_name)

            # Test bucket access
            await self._run_in_executor(
                self.s3_client.head_bucket, Bucket=self.bucket_name
            )

            # Configure bucket if needed
            await self._configure_bucket()

            self._initialized = True

        except NoCredentialsError:
            raise StorageError("AWS credentials not found")
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchBucket":
                raise StorageError(f"S3 bucket does not exist: {self.bucket_name}")
            elif error_code == "AccessDenied":
                raise StorageError(f"Access denied to S3 bucket: {self.bucket_name}")
            else:
                raise StorageError(f"Failed to access S3 bucket: {e}")
        except Exception as e:
            raise StorageError(f"Failed to initialize S3 storage: {e}")

    async def cleanup(self) -> None:
        """Cleanup S3 client resources"""
        if self.s3_client:
            await self._run_in_executor(self.s3_client.close)
        self._initialized = False

    async def _run_in_executor(self, func, *args, **kwargs):
        """Run synchronous boto3 operations in thread executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))

    async def _configure_bucket(self) -> None:
        """Configure bucket settings (versioning, encryption, etc.)"""
        try:
            # Enable versioning if requested
            if self.versioning_enabled:
                await self._run_in_executor(
                    self.s3_client.put_bucket_versioning,
                    Bucket=self.bucket_name,
                    VersioningConfiguration={"Status": "Enabled"},
                )

            # Configure server-side encryption
            if self.server_side_encryption:
                encryption_config = {
                    "Rules": [
                        {
                            "ApplyServerSideEncryptionByDefault": {
                                "SSEAlgorithm": self.server_side_encryption
                            }
                        }
                    ]
                }

                await self._run_in_executor(
                    self.s3_client.put_bucket_encryption,
                    Bucket=self.bucket_name,
                    ServerSideEncryptionConfiguration=encryption_config,
                )

        except ClientError as e:
            # Configuration errors are non-fatal
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code != "AccessDenied":
                # Log warning but don't fail initialization
                pass

    def _get_s3_key(self, path: str) -> str:
        """Get S3 object key from relative path

        Args:
            path: Relative path

        Returns:
            S3 object key
        """
        normalized_path = self.normalize_path(path)
        if not normalized_path:
            return ""

        # Use base_path as prefix if it's not the bucket name
        if self.config.base_path != self.bucket_name:
            base_prefix = self.normalize_path(self.config.base_path)
            if base_prefix and not normalized_path.startswith(base_prefix):
                normalized_path = f"{base_prefix}/{normalized_path}"

        return normalized_path

    async def read_file(self, path: str) -> bytes:
        """Read file content as bytes from S3

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
            s3_key = self._get_s3_key(path)

            response = await self._run_in_executor(
                self.s3_client.get_object, Bucket=self.bucket_name, Key=s3_key
            )

            return response["Body"].read()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File not found: {path}")
            elif error_code == "AccessDenied":
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error reading file {path}: {e}")

    async def read_text_file(self, path: str, encoding: str = "utf-8") -> str:
        """Read file content as text from S3

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
            content_bytes = await self.read_file(path)
            return content_bytes.decode(encoding)
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
        """Write content to S3 object

        Args:
            path: File path relative to storage root
            content: Content to write (bytes or string)
            metadata: Optional metadata to store with object

        Raises:
            PermissionError: If write access is denied
            StorageFullError: If storage quota is exceeded
            StorageError: For other storage errors
        """
        try:
            s3_key = self._get_s3_key(path)

            # Convert string to bytes if needed
            if isinstance(content, str):
                content_bytes = content.encode("utf-8")
            else:
                content_bytes = content

            # Prepare put_object parameters
            put_params = {
                "Bucket": self.bucket_name,
                "Key": s3_key,
                "Body": content_bytes,
                "ContentType": self.get_content_type(path),
                "StorageClass": self.storage_class,
            }

            # Add server-side encryption
            if self.server_side_encryption:
                put_params["ServerSideEncryption"] = self.server_side_encryption

            # Add metadata
            if metadata:
                # S3 metadata keys must be lowercase and contain only letters, numbers, and hyphens
                s3_metadata = {}
                for key, value in metadata.items():
                    clean_key = key.lower().replace("_", "-")
                    s3_metadata[clean_key] = str(value)
                put_params["Metadata"] = s3_metadata

            await self._run_in_executor(self.s3_client.put_object, **put_params)

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "AccessDenied":
                raise PermissionError(f"Permission denied: {path}")
            elif error_code in ["QuotaExceeded", "StorageQuotaExceeded"]:
                raise StorageFullError(f"Storage quota exceeded: {path}")
            else:
                raise StorageError(f"Error writing file {path}: {e}")

    async def delete_file(self, path: str) -> None:
        """Delete S3 object

        Args:
            path: File path relative to storage root

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If delete access is denied
            StorageError: For other storage errors
        """
        try:
            s3_key = self._get_s3_key(path)

            # Check if object exists first
            if not await self.file_exists(path):
                raise FileNotFoundError(f"File not found: {path}")

            await self._run_in_executor(
                self.s3_client.delete_object, Bucket=self.bucket_name, Key=s3_key
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File not found: {path}")
            elif error_code == "AccessDenied":
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error deleting file {path}: {e}")

    async def file_exists(self, path: str) -> bool:
        """Check if S3 object exists

        Args:
            path: File path relative to storage root

        Returns:
            True if file exists, False otherwise
        """
        try:
            s3_key = self._get_s3_key(path)

            await self._run_in_executor(
                self.s3_client.head_object, Bucket=self.bucket_name, Key=s3_key
            )

            return True

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                return False
            else:
                # Other errors (permission, etc.) should not return False
                raise StorageError(f"Error checking file existence {path}: {e}")

    async def get_file_metadata(self, path: str) -> FileMetadata:
        """Get S3 object metadata

        Args:
            path: File path relative to storage root

        Returns:
            File metadata object

        Raises:
            FileNotFoundError: If file doesn't exist
            StorageError: For other storage errors
        """
        try:
            s3_key = self._get_s3_key(path)

            response = await self._run_in_executor(
                self.s3_client.head_object, Bucket=self.bucket_name, Key=s3_key
            )

            # Extract metadata
            size = response.get("ContentLength", 0)
            modified_at = response.get("LastModified")
            content_type = response.get("ContentType")
            etag = response.get("ETag", "").strip('"')

            # S3 metadata
            s3_metadata = response.get("Metadata", {})

            return FileMetadata(
                path=path,
                size=size,
                modified_at=modified_at,
                created_at=modified_at,  # S3 doesn't have separate creation time
                content_type=content_type,
                etag=etag,
                metadata=s3_metadata,
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File not found: {path}")
            else:
                raise StorageError(f"Error getting file metadata {path}: {e}")

    async def create_directory(self, path: str) -> None:
        """Create directory (S3 doesn't have real directories)

        In S3, directories are simulated by creating an empty object with trailing slash.

        Args:
            path: Directory path relative to storage root
        """
        try:
            s3_key = self._get_s3_key(path)
            if not s3_key.endswith("/"):
                s3_key += "/"

            await self._run_in_executor(
                self.s3_client.put_object,
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=b"",
                ContentType="application/x-directory",
            )

        except ClientError as e:
            raise StorageError(f"Error creating directory {path}: {e}")

    async def delete_directory(self, path: str, recursive: bool = False) -> None:
        """Delete directory (and optionally all contents)

        Args:
            path: Directory path relative to storage root
            recursive: If True, delete directory and all contents

        Raises:
            FileNotFoundError: If directory doesn't exist
            PermissionError: If delete access is denied
            StorageError: For other storage errors
        """
        try:
            s3_prefix = self._get_s3_key(path)
            if not s3_prefix.endswith("/"):
                s3_prefix += "/"

            if recursive:
                # List and delete all objects with this prefix
                objects_to_delete = []

                paginator = self.s3_client.get_paginator("list_objects_v2")
                pages = paginator.paginate(Bucket=self.bucket_name, Prefix=s3_prefix)

                async for page in self._paginate_async(pages):
                    if "Contents" in page:
                        for obj in page["Contents"]:
                            objects_to_delete.append({"Key": obj["Key"]})

                # Delete objects in batches
                if objects_to_delete:
                    for i in range(0, len(objects_to_delete), 1000):  # S3 limit is 1000
                        batch = objects_to_delete[i : i + 1000]
                        await self._run_in_executor(
                            self.s3_client.delete_objects,
                            Bucket=self.bucket_name,
                            Delete={"Objects": batch},
                        )
            else:
                # Just delete the directory marker object
                await self._run_in_executor(
                    self.s3_client.delete_object, Bucket=self.bucket_name, Key=s3_prefix
                )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"Directory not found: {path}")
            elif error_code == "AccessDenied":
                raise PermissionError(f"Permission denied: {path}")
            else:
                raise StorageError(f"Error deleting directory {path}: {e}")

    async def list_files(
        self, path: str = "", recursive: bool = False, pattern: Optional[str] = None
    ) -> List[str]:
        """List files in S3 'directory'

        Args:
            path: Directory path relative to storage root (empty = root)
            recursive: If True, list files recursively
            pattern: Optional glob pattern to filter files

        Returns:
            List of file paths relative to storage root

        Raises:
            StorageError: For S3 errors
        """
        try:
            s3_prefix = self._get_s3_key(path)
            if s3_prefix and not s3_prefix.endswith("/"):
                s3_prefix += "/"

            files = []

            # Configure list parameters
            list_params = {"Bucket": self.bucket_name, "Prefix": s3_prefix}

            if not recursive:
                list_params["Delimiter"] = "/"

            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(**list_params)

            async for page in self._paginate_async(pages):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]

                        # Skip directory markers
                        if key.endswith("/"):
                            continue

                        # Convert back to relative path
                        relative_path = key
                        if s3_prefix:
                            relative_path = key[len(s3_prefix) :]

                        # Apply pattern filter
                        if pattern is None or fnmatch.fnmatch(
                            relative_path.split("/")[-1], pattern
                        ):
                            files.append(relative_path)

            return sorted(files)

        except ClientError as e:
            raise StorageError(f"Error listing files in {path}: {e}")

    async def list_directories(
        self, path: str = "", recursive: bool = False
    ) -> List[str]:
        """List directories in S3 'directory'

        Args:
            path: Directory path relative to storage root (empty = root)
            recursive: If True, list directories recursively

        Returns:
            List of directory paths relative to storage root

        Raises:
            StorageError: For S3 errors
        """
        try:
            s3_prefix = self._get_s3_key(path)
            if s3_prefix and not s3_prefix.endswith("/"):
                s3_prefix += "/"

            directories = set()

            # List with delimiter to get common prefixes (directories)
            list_params = {
                "Bucket": self.bucket_name,
                "Prefix": s3_prefix,
                "Delimiter": "/",
            }

            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(**list_params)

            async for page in self._paginate_async(pages):
                # Common prefixes represent directories
                if "CommonPrefixes" in page:
                    for prefix_info in page["CommonPrefixes"]:
                        prefix = prefix_info["Prefix"]

                        # Convert back to relative path
                        relative_path = prefix.rstrip("/")
                        if s3_prefix:
                            relative_path = relative_path[len(s3_prefix.rstrip("/")) :]
                            if relative_path.startswith("/"):
                                relative_path = relative_path[1:]

                        if relative_path:
                            directories.add(relative_path)

                # If recursive, also get subdirectories
                if recursive and "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if "/" in key:
                            # Extract directory parts
                            parts = key.split("/")
                            for i in range(1, len(parts)):
                                dir_path = "/".join(parts[:i])
                                if s3_prefix:
                                    dir_path = dir_path[len(s3_prefix.rstrip("/")) :]
                                    if dir_path.startswith("/"):
                                        dir_path = dir_path[1:]

                                if dir_path:
                                    directories.add(dir_path)

            return sorted(list(directories))

        except ClientError as e:
            raise StorageError(f"Error listing directories in {path}: {e}")

    async def read_file_stream(
        self, path: str, chunk_size: int = 8192
    ) -> AsyncIterator[bytes]:
        """Read file as stream of chunks from S3

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
            s3_key = self._get_s3_key(path)

            # Get object with streaming
            response = await self._run_in_executor(
                self.s3_client.get_object, Bucket=self.bucket_name, Key=s3_key
            )

            # Stream the body
            body = response["Body"]
            try:
                while True:
                    chunk = body.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            finally:
                body.close()

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"File not found: {path}")
            else:
                raise StorageError(f"Error reading file stream {path}: {e}")

    async def write_file_stream(
        self,
        path: str,
        content_stream: AsyncIterator[bytes],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Write file from stream of chunks to S3

        Args:
            path: File path relative to storage root
            content_stream: Stream of content chunks
            metadata: Optional metadata to store with object

        Raises:
            PermissionError: If write access is denied
            StorageError: For other storage errors
        """
        try:
            s3_key = self._get_s3_key(path)

            # Collect stream content (S3 requires content length)
            content_parts = []
            async for chunk in content_stream:
                content_parts.append(chunk)

            content_bytes = b"".join(content_parts)

            # Write using regular write_file method
            await self.write_file(path, content_bytes, metadata)

        except Exception as e:
            raise StorageError(f"Error writing file stream {path}: {e}")

    async def copy_file(self, source_path: str, dest_path: str) -> None:
        """Copy S3 object to new location

        Uses efficient S3 copy operation.

        Args:
            source_path: Source file path
            dest_path: Destination file path

        Raises:
            FileNotFoundError: If source file doesn't exist
            PermissionError: If access is denied
            StorageError: For other storage errors
        """
        try:
            source_s3_key = self._get_s3_key(source_path)
            dest_s3_key = self._get_s3_key(dest_path)

            # Check source exists
            if not await self.file_exists(source_path):
                raise FileNotFoundError(f"Source file not found: {source_path}")

            copy_source = {"Bucket": self.bucket_name, "Key": source_s3_key}

            await self._run_in_executor(
                self.s3_client.copy_object,
                CopySource=copy_source,
                Bucket=self.bucket_name,
                Key=dest_s3_key,
                StorageClass=self.storage_class,
            )

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            if error_code == "NoSuchKey":
                raise FileNotFoundError(f"Source file not found: {source_path}")
            elif error_code == "AccessDenied":
                raise PermissionError(
                    f"Permission denied copying {source_path} to {dest_path}"
                )
            else:
                raise StorageError(
                    f"Error copying file {source_path} to {dest_path}: {e}"
                )

    async def _paginate_async(self, paginator):
        """Convert boto3 paginator to async iterator"""
        for page in paginator:
            yield page


# Register the S3 storage adapter
from .storage_adapters import StorageAdapterFactory

StorageAdapterFactory.register_adapter("s3", S3StorageAdapter)
