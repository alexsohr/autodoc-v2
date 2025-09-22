"""Wiki data models for generated documentation

This module defines the wiki-related Pydantic models including
WikiStructure, WikiPageDetail, and WikiSection based on data-model.md.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any
from uuid import UUID
from pydantic import BaseModel, Field, field_validator, model_validator


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
    file_paths: List[str] = Field(default_factory=list, description="Source file paths this page covers")
    related_pages: List[str] = Field(default_factory=list, description="IDs of related pages")
    
    # Content
    content: str = Field(default="", description="Generated markdown content")
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
        use_enum_values = True
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate page ID format"""
        if not v or not v.strip():
            raise ValueError("Page ID cannot be empty")
        
        # Page ID should be URL-friendly
        import re
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError("Page ID must contain only alphanumeric characters, hyphens, and underscores")
        
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
    
    @field_validator("file_paths")
    @classmethod
    def validate_file_paths(cls, v: List[str]) -> List[str]:
        """Validate file paths are relative and valid"""
        if not v:
            return v
        
        validated_paths = []
        for path in v:
            if not path or not path.strip():
                continue
            
            # Normalize path separators and ensure relative
            normalized_path = path.strip().replace('\\', '/')
            if normalized_path.startswith('/') or '..' in normalized_path:
                raise ValueError(f"File path must be relative: {path}")
            
            validated_paths.append(normalized_path)
        
        return validated_paths
    
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
            if not re.match(r'^[a-zA-Z0-9\-_]+$', page_id.strip()):
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
    """Organizational section containing pages and subsections
    
    Represents a section in the wiki structure that can contain
    pages and nested subsections for hierarchical organization.
    """
    
    # Core identification
    id: str = Field(description="Unique section identifier")
    title: str = Field(description="Section title")
    
    # Structure
    pages: List[str] = Field(default_factory=list, description="Page IDs in this section")
    subsections: List[str] = Field(default_factory=list, description="Subsection IDs")
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate section ID format"""
        if not v or not v.strip():
            raise ValueError("Section ID cannot be empty")
        
        # Section ID should be URL-friendly
        import re
        if not re.match(r'^[a-zA-Z0-9\-_]+$', v):
            raise ValueError("Section ID must contain only alphanumeric characters, hyphens, and underscores")
        
        return v.strip()
    
    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate section title"""
        if not v or not v.strip():
            raise ValueError("Section title cannot be empty")
        return v.strip()
    
    @field_validator("pages")
    @classmethod
    def validate_pages(cls, v: List[str]) -> List[str]:
        """Validate page IDs"""
        if not v:
            return v
        
        validated_pages = []
        for page_id in v:
            if not page_id or not page_id.strip():
                continue
            
            import re
            if not re.match(r'^[a-zA-Z0-9\-_]+$', page_id.strip()):
                raise ValueError(f"Page ID must be valid: {page_id}")
            
            validated_pages.append(page_id.strip())
        
        return validated_pages
    
    @field_validator("subsections")
    @classmethod
    def validate_subsections(cls, v: List[str]) -> List[str]:
        """Validate subsection IDs"""
        if not v:
            return v
        
        validated_subsections = []
        for section_id in v:
            if not section_id or not section_id.strip():
                continue
            
            import re
            if not re.match(r'^[a-zA-Z0-9\-_]+$', section_id.strip()):
                raise ValueError(f"Subsection ID must be valid: {section_id}")
            
            validated_subsections.append(section_id.strip())
        
        return validated_subsections
    
    def add_page(self, page_id: str) -> None:
        """Add a page to this section"""
        if page_id and page_id not in self.pages:
            self.pages.append(page_id)
    
    def add_subsection(self, section_id: str) -> None:
        """Add a subsection to this section"""
        if section_id and section_id not in self.subsections:
            self.subsections.append(section_id)
    
    def remove_page(self, page_id: str) -> None:
        """Remove a page from this section"""
        if page_id in self.pages:
            self.pages.remove(page_id)
    
    def remove_subsection(self, section_id: str) -> None:
        """Remove a subsection from this section"""
        if section_id in self.subsections:
            self.subsections.remove(section_id)
    
    def has_pages(self) -> bool:
        """Check if section has pages"""
        return len(self.pages) > 0
    
    def has_subsections(self) -> bool:
        """Check if section has subsections"""
        return len(self.subsections) > 0
    
    def __str__(self) -> str:
        return f"{self.title} ({len(self.pages)} pages, {len(self.subsections)} subsections)"
    
    def __repr__(self) -> str:
        return f"WikiSection(id={self.id}, title={self.title})"


class WikiStructure(BaseModel):
    """Complete wiki structure for a repository
    
    Represents the complete wiki structure including all pages,
    sections, and their hierarchical relationships.
    """
    
    # Core identification
    id: str = Field(description="Unique wiki identifier")
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    
    # Structure components
    pages: List[WikiPageDetail] = Field(default_factory=list, description="All wiki pages")
    sections: List[WikiSection] = Field(default_factory=list, description="All wiki sections")
    root_sections: List[str] = Field(default_factory=list, description="Top-level section IDs")
    
    class Config:
        """Pydantic configuration"""
        validate_assignment = True
    
    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Validate wiki ID format"""
        if not v or not v.strip():
            raise ValueError("Wiki ID cannot be empty")
        return v.strip()
    
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
        # Collect all page and section IDs
        page_ids = {page.id for page in self.pages}
        section_ids = {section.id for section in self.sections}
        
        # Validate root sections exist
        for root_section_id in self.root_sections:
            if root_section_id not in section_ids:
                raise ValueError(f"Root section '{root_section_id}' not found in sections")
        
        # Validate section references
        for section in self.sections:
            # Check page references
            for page_id in section.pages:
                if page_id not in page_ids:
                    raise ValueError(f"Page '{page_id}' referenced in section '{section.id}' not found")
            
            # Check subsection references
            for subsection_id in section.subsections:
                if subsection_id not in section_ids:
                    raise ValueError(f"Subsection '{subsection_id}' referenced in section '{section.id}' not found")
        
        # Validate page related_pages references
        for page in self.pages:
            for related_page_id in page.related_pages:
                if related_page_id not in page_ids:
                    raise ValueError(f"Related page '{related_page_id}' for page '{page.id}' not found")
        
        return self
    
    def add_page(self, page: WikiPageDetail) -> None:
        """Add a page to the wiki"""
        # Check for duplicate IDs
        existing_ids = {p.id for p in self.pages}
        if page.id in existing_ids:
            raise ValueError(f"Page with ID '{page.id}' already exists")
        
        self.pages.append(page)
    
    def add_section(self, section: WikiSection) -> None:
        """Add a section to the wiki"""
        # Check for duplicate IDs
        existing_ids = {s.id for s in self.sections}
        if section.id in existing_ids:
            raise ValueError(f"Section with ID '{section.id}' already exists")
        
        self.sections.append(section)
    
    def get_page(self, page_id: str) -> Optional[WikiPageDetail]:
        """Get a page by ID"""
        for page in self.pages:
            if page.id == page_id:
                return page
        return None
    
    def get_section(self, section_id: str) -> Optional[WikiSection]:
        """Get a section by ID"""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None
    
    def get_pages_in_section(self, section_id: str) -> List[WikiPageDetail]:
        """Get all pages in a specific section"""
        section = self.get_section(section_id)
        if not section:
            return []
        
        return [page for page in self.pages if page.id in section.pages]
    
    def get_subsections(self, section_id: str) -> List[WikiSection]:
        """Get all subsections of a specific section"""
        section = self.get_section(section_id)
        if not section:
            return []
        
        return [s for s in self.sections if s.id in section.subsections]
    
    def get_root_sections(self) -> List[WikiSection]:
        """Get all root sections"""
        return [s for s in self.sections if s.id in self.root_sections]
    
    def get_total_pages(self) -> int:
        """Get total number of pages"""
        return len(self.pages)
    
    def get_total_sections(self) -> int:
        """Get total number of sections"""
        return len(self.sections)
    
    def get_pages_by_importance(self, importance: PageImportance) -> List[WikiPageDetail]:
        """Get pages filtered by importance level"""
        return [page for page in self.pages if page.importance == importance]
    
    def __str__(self) -> str:
        return f"{self.title} ({len(self.pages)} pages, {len(self.sections)} sections)"
    
    def __repr__(self) -> str:
        return f"WikiStructure(id={self.id}, title={self.title}, pages={len(self.pages)}, sections={len(self.sections)})"


# API Models

class WikiPageCreate(BaseModel):
    """Wiki page creation model"""
    id: str = Field(description="Page identifier")
    title: str = Field(description="Page title")
    description: str = Field(description="Page description")
    importance: PageImportance = Field(default=PageImportance.MEDIUM, description="Page importance")
    file_paths: Optional[List[str]] = Field(default_factory=list, description="Source file paths")
    related_pages: Optional[List[str]] = Field(default_factory=list, description="Related page IDs")
    content: Optional[str] = Field(default="", description="Page content")


class WikiSectionCreate(BaseModel):
    """Wiki section creation model"""
    id: str = Field(description="Section identifier")
    title: str = Field(description="Section title")
    pages: Optional[List[str]] = Field(default_factory=list, description="Page IDs")
    subsections: Optional[List[str]] = Field(default_factory=list, description="Subsection IDs")


class WikiStructureCreate(BaseModel):
    """Wiki structure creation model"""
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    pages: Optional[List[WikiPageCreate]] = Field(default_factory=list, description="Wiki pages")
    sections: Optional[List[WikiSectionCreate]] = Field(default_factory=list, description="Wiki sections")
    root_sections: Optional[List[str]] = Field(default_factory=list, description="Root section IDs")


class PullRequestRequest(BaseModel):
    """Pull request creation request model"""
    target_branch: Optional[str] = Field(default=None, description="Target branch for pull request")
    title: Optional[str] = Field(default=None, description="Custom pull request title")
    description: Optional[str] = Field(default=None, description="Custom pull request description")
    force_update: bool = Field(default=False, description="Force update even if no changes detected")


class PullRequestResponse(BaseModel):
    """Pull request creation response model"""
    pull_request_url: str = Field(description="URL of created pull request")
    branch_name: str = Field(description="Name of created branch")
    files_changed: List[str] = Field(description="List of files modified")
    commit_sha: str = Field(description="Commit SHA of changes")
    
    class Config:
        """Pydantic configuration"""
        json_encoders = {
            datetime: lambda dt: dt.isoformat()
        }
