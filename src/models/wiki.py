"""Wiki data models for generated documentation

This module defines the wiki-related Pydantic models including
WikiStructure, WikiPageDetail, and WikiSection based on data-model.md.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pymongo import TEXT, IndexModel

from .base import BaseDocument


class PageImportance(str, Enum):
    """Page importance levels"""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class WikiPageDetail(BaseModel):
    """Individual wiki page with content and metadata

    Represents a single wiki page with its content, relationships,
    and metadata about source files and importance.
    """

    # Core identification
    id: str = Field(description="Unique page identifier")
    title: str = Field(description="Page title")
    description: str = Field(description="Page description")

    # Page metadata
    importance: PageImportance = Field(description="Priority level")
    file_paths: List[str] = Field(
        default_factory=list, description="Source file paths this page covers"
    )
    related_pages: List[str] = Field(
        default_factory=list, description="IDs of related pages"
    )

    # Content
    content: str = Field(default="", description="Generated markdown content")

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate page ID format"""
        if not v or not v.strip():
            raise ValueError("Page ID cannot be empty")

        # Page ID should be URL-friendly
        import re

        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError(
                "Page ID must contain only alphanumeric characters, hyphens, and underscores"
            )

        return v.strip()

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate page title"""
        if not v or not v.strip():
            raise ValueError("Page title cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate page description"""
        if not v or not v.strip():
            raise ValueError("Page description cannot be empty")
        return v.strip()

    @field_validator("related_pages")
    @classmethod
    def validate_related_pages(cls, v: List[str]) -> List[str]:
        """Validate related page IDs"""
        if not v:
            return v

        validated_pages = []
        for page_id in v:
            if not page_id or not page_id.strip():
                continue

            # Same validation as page ID
            import re

            if not re.match(r"^[a-zA-Z0-9\-_]+$", page_id.strip()):
                raise ValueError(f"Related page ID must be valid: {page_id}")

            validated_pages.append(page_id.strip())

        return validated_pages

    def add_file_path(self, file_path: str) -> None:
        """Add a file path to this page"""
        if file_path and file_path not in self.file_paths:
            self.file_paths.append(file_path)

    def add_related_page(self, page_id: str) -> None:
        """Add a related page ID"""
        if page_id and page_id not in self.related_pages:
            self.related_pages.append(page_id)

    def set_content(self, content: str) -> None:
        """Set page content"""
        self.content = content or ""

    def has_content(self) -> bool:
        """Check if page has content"""
        return bool(self.content and self.content.strip())

    def get_word_count(self) -> int:
        """Get approximate word count of content"""
        if not self.content:
            return 0
        return len(self.content.split())

    def __str__(self) -> str:
        return f"{self.title} ({self.importance})"

    def __repr__(self) -> str:
        return f"WikiPageDetail(id={self.id}, title={self.title}, importance={self.importance})"


class WikiSection(BaseModel):
    """Organizational section containing pages

    Represents a section in the wiki structure that contains pages.
    """

    # Core identification
    id: str = Field(description="Unique section identifier")
    title: str = Field(description="Section title")

    # Structure - pages are now embedded as full objects
    pages: List[WikiPageDetail] = Field(
        default_factory=list, description="Pages in this section"
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate section ID format"""
        if not v or not v.strip():
            raise ValueError("Section ID cannot be empty")

        # Section ID should be URL-friendly
        import re

        if not re.match(r"^[a-zA-Z0-9\-_]+$", v):
            raise ValueError(
                "Section ID must contain only alphanumeric characters, hyphens, and underscores"
            )

        return v.strip()

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate section title"""
        if not v or not v.strip():
            raise ValueError("Section title cannot be empty")
        return v.strip()

    def add_page(self, page: WikiPageDetail) -> None:
        """Add a page to this section"""
        if page and page.id not in [p.id for p in self.pages]:
            self.pages.append(page)

    def remove_page(self, page_id: str) -> None:
        """Remove a page from this section by ID"""
        self.pages = [p for p in self.pages if p.id != page_id]

    def get_page(self, page_id: str) -> Optional[WikiPageDetail]:
        """Get a page by ID"""
        for page in self.pages:
            if page.id == page_id:
                return page
        return None

    def has_pages(self) -> bool:
        """Check if section has pages"""
        return len(self.pages) > 0

    def get_total_pages(self) -> int:
        """Get total number of pages in this section"""
        return len(self.pages)

    def __str__(self) -> str:
        return f"{self.title} ({len(self.pages)} pages)"

    def __repr__(self) -> str:
        return f"WikiSection(id={self.id}, title={self.title})"


class WikiStructure(BaseDocument):
    """Complete wiki structure for a repository

    Represents the complete wiki structure including all sections
    with their embedded pages.
    """

    # Core identification - uses UUID pattern like all other collections
    id: UUID = Field(default_factory=uuid4, description="Unique wiki identifier")
    repository_id: UUID = Field(description="Repository identifier")
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")

    # Structure components - pages are now embedded within sections
    sections: List[WikiSection] = Field(
        default_factory=list, description="All wiki sections with embedded pages"
    )

    class Settings:
        name = "wiki_structures"
        indexes = [
            IndexModel("repository_id", unique=True),
            IndexModel([("title", TEXT), ("description", TEXT)]),
        ]

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate wiki title"""
        if not v or not v.strip():
            raise ValueError("Wiki title cannot be empty")
        return v.strip()

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Validate wiki description"""
        if not v or not v.strip():
            raise ValueError("Wiki description cannot be empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_structure_consistency(self) -> "WikiStructure":
        """Validate wiki structure consistency"""
        # Collect all page IDs across all sections
        all_page_ids = set()
        for section in self.sections:
            for page in section.pages:
                if page.id in all_page_ids:
                    raise ValueError(f"Duplicate page ID '{page.id}' found in sections")
                all_page_ids.add(page.id)

        # Validate page related_pages references
        for section in self.sections:
            for page in section.pages:
                for related_page_id in page.related_pages:
                    if related_page_id not in all_page_ids:
                        raise ValueError(
                            f"Related page '{related_page_id}' for page '{page.id}' not found"
                        )

        return self

    def add_section(self, section: WikiSection) -> None:
        """Add a section to the wiki"""
        # Check for duplicate IDs
        existing_ids = {s.id for s in self.sections}
        if section.id in existing_ids:
            raise ValueError(f"Section with ID '{section.id}' already exists")

        self.sections.append(section)

    def get_section(self, section_id: str) -> Optional[WikiSection]:
        """Get a section by ID"""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def get_page(self, page_id: str) -> Optional[WikiPageDetail]:
        """Get a page by ID from any section"""
        for section in self.sections:
            page = section.get_page(page_id)
            if page:
                return page
        return None

    def get_all_pages(self) -> List[WikiPageDetail]:
        """Get all pages from all sections"""
        pages = []
        for section in self.sections:
            pages.extend(section.pages)
        return pages

    def get_total_pages(self) -> int:
        """Get total number of pages across all sections"""
        return sum(section.get_total_pages() for section in self.sections)

    def get_total_sections(self) -> int:
        """Get total number of sections"""
        return len(self.sections)

    def get_pages_by_importance(
        self, importance: PageImportance
    ) -> List[WikiPageDetail]:
        """Get pages filtered by importance level"""
        return [page for page in self.get_all_pages() if page.importance == importance]

    def __str__(self) -> str:
        return f"{self.title} ({self.get_total_pages()} pages, {len(self.sections)} sections)"

    def __repr__(self) -> str:
        return f"WikiStructure(id={self.id}, title={self.title}, pages={self.get_total_pages()}, sections={len(self.sections)})"


# API Models


class WikiPageCreate(BaseModel):
    """Wiki page creation model"""

    id: str = Field(description="Page identifier")
    title: str = Field(description="Page title")
    description: str = Field(description="Page description")
    importance: PageImportance = Field(
        default=PageImportance.MEDIUM, description="Page importance"
    )
    file_paths: Optional[List[str]] = Field(
        default_factory=list, description="Source file paths"
    )
    related_pages: Optional[List[str]] = Field(
        default_factory=list, description="Related page IDs"
    )
    content: Optional[str] = Field(default="", description="Page content")


class WikiSectionCreate(BaseModel):
    """Wiki section creation model"""

    id: str = Field(description="Section identifier")
    title: str = Field(description="Section title")
    pages: Optional[List[WikiPageCreate]] = Field(
        default_factory=list, description="Pages in this section"
    )


class WikiStructureCreate(BaseModel):
    """Wiki structure creation model"""

    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    sections: Optional[List[WikiSectionCreate]] = Field(
        default_factory=list, description="Wiki sections with embedded pages"
    )


class PullRequestRequest(BaseModel):
    """Pull request creation request model"""

    target_branch: Optional[str] = Field(
        default=None, description="Target branch for pull request"
    )
    title: Optional[str] = Field(default=None, description="Custom pull request title")
    description: Optional[str] = Field(
        default=None, description="Custom pull request description"
    )
    force_update: bool = Field(
        default=False, description="Force update even if no changes detected"
    )


class PullRequestResponse(BaseModel):
    """Pull request creation response model"""

    pull_request_url: str = Field(description="URL of created pull request")
    branch_name: str = Field(description="Name of created branch")
    files_changed: List[str] = Field(description="List of files modified")
    commit_sha: str = Field(description="Commit SHA of changes")

    model_config = ConfigDict()
