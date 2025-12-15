"""CodeDocument data model for processed code files

This module defines the CodeDocument Pydantic model for representing
processed code files with embeddings for semantic search.
"""

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymongo import ASCENDING, DESCENDING, TEXT, IndexModel

from .base import BaseDocument, BaseSerializers


class CodeDocument(BaseDocument):
    """CodeDocument model for processed code files

    Represents a processed code file for semantic search and analysis.
    Includes content, metadata, and optional vector embeddings.
    """

    # Core identification
    id: str = Field(description="Unique document identifier")
    repository_id: UUID = Field(description="Foreign key to Repository")
    file_path: str = Field(description="Path relative to repository root")
    language: str = Field(description="Programming language")

    # Content fields
    content: str = Field(description="Original file content")
    processed_content: str = Field(description="Cleaned content for embedding")

    # Metadata and embeddings
    metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    embedding: Optional[List[float]] = Field(
        default=None, description="Vector embedding (optional, stored separately)"
    )

    # Audit fields
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last update timestamp",
    )

    class Settings:
        name = "code_documents"
        indexes = [
            IndexModel(
                [("repository_id", ASCENDING), ("file_path", ASCENDING)], unique=True
            ),
            IndexModel("repository_id"),
            IndexModel("language"),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("updated_at", DESCENDING)]),
            IndexModel(
                [("processed_content", TEXT)],
                default_language="none",
                language_override="text_search_language",
            ),
        ]

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path is relative to repository root"""
        if not v:
            raise ValueError("File path cannot be empty")

        # Ensure path is relative (doesn't start with / or contain ..)
        if v.startswith("/") or ".." in v:
            raise ValueError(
                "File path must be relative to repository root and cannot contain '..'"
            )

        # Normalize path separators
        return v.replace("\\", "/")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        """Validate programming language"""
        if not v:
            raise ValueError("Language cannot be empty")

        # Common programming languages (extensible list)
        supported_languages = {
            "python",
            "javascript",
            "typescript",
            "java",
            "go",
            "rust",
            "cpp",
            "c",
            "csharp",
            "php",
            "ruby",
            "swift",
            "kotlin",
            "scala",
            "clojure",
            "haskell",
            "erlang",
            "elixir",
            "html",
            "css",
            "scss",
            "less",
            "sql",
            "shell",
            "bash",
            "powershell",
            "dockerfile",
            "yaml",
            "json",
            "xml",
            "markdown",
            "text",
            "binary",
        }

        language_lower = v.lower()
        if language_lower not in supported_languages:
            # Allow any language but normalize to lowercase
            return language_lower

        return language_lower

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate content is not empty for text files"""
        # Allow empty content for binary files or specific cases
        return v

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding vector dimensions"""
        if v is None:
            return v

        # Common embedding dimensions
        valid_dimensions = {128, 256, 384, 512, 768, 1024, 1536, 3072}

        if len(v) not in valid_dimensions:
            # Allow any dimension but warn about common sizes
            pass

        # Validate all values are finite floats
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)) or not (-1.0 <= val <= 1.0):
                raise ValueError(
                    f"Embedding value at index {i} must be a float between -1.0 and 1.0"
                )

        return v

    def update_content(self, content: str, processed_content: str) -> None:
        """Update document content and processed content"""
        self.content = content
        self.processed_content = processed_content
        self.updated_at = datetime.now(timezone.utc)

        # Clear embedding when content changes
        self.embedding = None

    def set_embedding(self, embedding: List[float]) -> None:
        """Set the vector embedding for the document"""
        self.embedding = embedding
        self.updated_at = datetime.now(timezone.utc)

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata field"""
        if not self.metadata:
            self.metadata = {}
        self.metadata[key] = value
        self.updated_at = datetime.now(timezone.utc)

    def get_file_extension(self) -> str:
        """Get file extension from file path"""
        return self.file_path.split(".")[-1].lower() if "." in self.file_path else ""

    def get_file_name(self) -> str:
        """Get file name without path"""
        return self.file_path.split("/")[-1]

    def get_directory_path(self) -> str:
        """Get directory path without file name"""
        parts = self.file_path.split("/")
        return "/".join(parts[:-1]) if len(parts) > 1 else ""

    def is_text_file(self) -> bool:
        """Check if this is a text file (not binary)"""
        return self.language != "binary"

    def get_content_size(self) -> int:
        """Get content size in bytes"""
        return len(self.content.encode("utf-8"))

    def get_processed_content_size(self) -> int:
        """Get processed content size in bytes"""
        return len(self.processed_content.encode("utf-8"))

    def has_embedding(self) -> bool:
        """Check if document has an embedding"""
        return self.embedding is not None and len(self.embedding) > 0

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return len(self.embedding) if self.embedding else 0

    def __str__(self) -> str:
        return f"{self.language}:{self.file_path}"

    def __repr__(self) -> str:
        return f"CodeDocument(id={self.id}, file_path={self.file_path}, language={self.language})"


class CodeDocumentCreate(BaseModel):
    """CodeDocument creation model"""

    repository_id: UUID = Field(description="Repository ID")
    file_path: str = Field(description="File path relative to repository root")
    language: str = Field(description="Programming language")
    content: str = Field(description="File content")
    processed_content: str = Field(description="Processed content for embedding")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="File metadata"
    )


class CodeDocumentUpdate(BaseModel):
    """CodeDocument update model"""

    content: Optional[str] = Field(default=None, description="Updated file content")
    processed_content: Optional[str] = Field(
        default=None, description="Updated processed content"
    )
    language: Optional[str] = Field(
        default=None, description="Updated programming language"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Updated metadata"
    )
    embedding: Optional[List[float]] = Field(
        default=None, description="Updated embedding"
    )


class CodeDocumentResponse(BaseSerializers):
    """CodeDocument response model for API responses"""

    id: str = Field(description="Document identifier")
    repository_id: UUID = Field(description="Repository ID")
    file_path: str = Field(description="File path")
    language: str = Field(description="Programming language")
    metadata: Dict[str, Any] = Field(description="File metadata")
    created_at: datetime = Field(description="Creation timestamp")
    updated_at: datetime = Field(description="Last update timestamp")
    has_embedding: bool = Field(description="Whether document has embedding")

    model_config = ConfigDict()

    @classmethod
    def from_code_document(cls, doc: CodeDocument) -> "CodeDocumentResponse":
        """Create response model from CodeDocument"""
        return cls(
            id=doc.id,
            repository_id=doc.repository_id,
            file_path=doc.file_path,
            language=doc.language,
            metadata=doc.metadata,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            has_embedding=doc.has_embedding(),
        )


class FileList(BaseSerializers):
    """File list response model"""

    files: List[CodeDocumentResponse] = Field(description="List of processed files")
    repository_id: UUID = Field(description="Repository ID")
    total: int = Field(description="Total number of files")
    languages: Dict[str, int] = Field(description="Count of files per language")

    model_config = ConfigDict()


class DocumentSearchResult(BaseModel):
    """Document search result model"""

    document: CodeDocumentResponse = Field(description="Document details")
    score: float = Field(description="Relevance score (0.0 to 1.0)")
    highlights: Optional[List[str]] = Field(
        default=None, description="Highlighted text snippets"
    )

    model_config = ConfigDict()


class DocumentSearchResponse(BaseModel):
    """Document search response model"""

    results: List[DocumentSearchResult] = Field(description="Search results")
    query: str = Field(description="Original search query")
    total_results: int = Field(description="Total number of results")
    search_time: float = Field(description="Search time in seconds")

    model_config = ConfigDict()
