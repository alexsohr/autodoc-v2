"""Repository data model with webhook support

This module defines the Repository Pydantic model based on the data-model.md specification.
Includes webhook configuration fields and validation rules.
"""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, field_serializer, model_validator


class RepositoryProvider(str, Enum):
    """Repository provider types"""

    GITHUB = "github"
    BITBUCKET = "bitbucket"
    GITLAB = "gitlab"


class AccessScope(str, Enum):
    """Repository access scope"""

    PUBLIC = "public"
    PRIVATE = "private"


class AnalysisStatus(str, Enum):
    """Repository analysis status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Repository(BaseModel):
    """Repository model with webhook fields

    Represents a source code repository being analyzed by AutoDoc.
    Includes webhook configuration for automatic documentation updates.
    """

    # Core identification fields
    id: UUID = Field(default_factory=uuid4, description="Unique repository identifier")
    provider: RepositoryProvider = Field(description="Repository provider")
    url: str = Field(description="Repository URL")
    org: str = Field(description="Organization/owner name")
    name: str = Field(description="Repository name")
    default_branch: str = Field(description="Default branch name")

    # Access and status fields
    access_scope: AccessScope = Field(description="Repository visibility")
    last_analyzed: Optional[datetime] = Field(
        default=None, description="Last analysis timestamp"
    )
    analysis_status: AnalysisStatus = Field(
        default=AnalysisStatus.PENDING, description="Current analysis status"
    )
    commit_sha: Optional[str] = Field(
        default=None, description="Last analyzed commit SHA"
    )

    # Webhook configuration fields
    webhook_configured: bool = Field(
        default=False, description="Whether webhook is configured"
    )
    webhook_secret: Optional[str] = Field(
        default=None, description="Secret for validating webhook signatures"
    )
    subscribed_events: List[str] = Field(
        default_factory=list, description="Events that trigger documentation updates"
    )
    last_webhook_event: Optional[datetime] = Field(
        default=None, description="Timestamp of last received webhook event"
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

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )

    @field_serializer('created_at', 'updated_at', 'last_analyzed', 'last_webhook_event')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate repository URL format"""
        if not v:
            raise ValueError("URL cannot be empty")

        # Basic URL validation
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(v):
            raise ValueError("Invalid URL format")

        return v

    @field_validator("default_branch")
    @classmethod
    def validate_default_branch(cls, v: str) -> str:
        """Validate default branch is not empty"""
        if not v or not v.strip():
            raise ValueError("Default branch cannot be empty")
        return v.strip()

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, v: Optional[str]) -> Optional[str]:
        """Validate commit SHA format"""
        if v is None:
            return v

        # Git SHA-1 hash is 40 hexadecimal characters
        if not re.match(r"^[a-f0-9]{40}$", v, re.IGNORECASE):
            raise ValueError("Invalid git commit SHA format")

        return v.lower()

    @field_validator("subscribed_events")
    @classmethod
    def validate_subscribed_events(cls, v: List[str]) -> List[str]:
        """Validate subscribed events for the provider"""
        if not v:
            return v

        # Common webhook events across providers
        valid_events = {
            "push",
            "pull_request",
            "merge",
            "pullrequest:fulfilled",
            "repo:push",
            "repository:push",
            "merge_request",
        }

        for event in v:
            if event not in valid_events:
                # Allow provider-specific events (more flexible validation)
                if not re.match(r"^[a-zA-Z_:]+$", event):
                    raise ValueError(f"Invalid event format: {event}")

        return v

    @model_validator(mode="after")
    def validate_webhook_configuration(self) -> "Repository":
        """Validate webhook configuration consistency"""
        if self.webhook_configured and not self.webhook_secret:
            raise ValueError(
                "Webhook secret must be provided when webhook is configured"
            )

        return self

    @model_validator(mode="after")
    def validate_provider_url_consistency(self) -> "Repository":
        """Validate provider matches URL domain"""
        if not self.url:
            return self

        url_lower = self.url.lower()

        provider_domains = {
            RepositoryProvider.GITHUB: "github.com",
            RepositoryProvider.BITBUCKET: "bitbucket.org",
            RepositoryProvider.GITLAB: "gitlab.com",
        }

        expected_domain = provider_domains.get(self.provider)
        if expected_domain and expected_domain not in url_lower:
            raise ValueError(f"Provider {self.provider} does not match URL domain")

        return self

    @model_validator(mode="after")
    def extract_org_and_name_from_url(self) -> "Repository":
        """Extract organization and repository name from URL"""
        if not self.url:
            return self

        # Extract org and name from URL patterns like:
        # https://github.com/org/repo
        # https://bitbucket.org/org/repo
        # https://gitlab.com/org/repo

        url_patterns = [
            r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"https?://bitbucket\.org/([^/]+)/([^/]+?)(?:\.git)?/?$",
            r"https?://gitlab\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        ]

        for pattern in url_patterns:
            match = re.match(pattern, self.url, re.IGNORECASE)
            if match:
                extracted_org, extracted_name = match.groups()

                # Update org and name if not already set
                if not hasattr(self, "org") or not self.org:
                    self.org = extracted_org
                if not hasattr(self, "name") or not self.name:
                    self.name = extracted_name
                break

        return self

    def update_analysis_status(
        self, status: AnalysisStatus, commit_sha: Optional[str] = None
    ) -> None:
        """Update analysis status and related fields"""
        self.analysis_status = status
        self.updated_at = datetime.now(timezone.utc)

        if status == AnalysisStatus.COMPLETED:
            self.last_analyzed = datetime.now(timezone.utc)
            if commit_sha:
                self.commit_sha = commit_sha

    def configure_webhook(self, secret: str, events: List[str]) -> None:
        """Configure webhook settings"""
        self.webhook_configured = True
        self.webhook_secret = secret
        self.subscribed_events = events
        self.updated_at = datetime.now(timezone.utc)

    def record_webhook_event(self) -> None:
        """Record webhook event timestamp"""
        self.last_webhook_event = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def is_webhook_event_subscribed(self, event_type: str) -> bool:
        """Check if repository is subscribed to a webhook event type"""
        return self.webhook_configured and event_type in self.subscribed_events

    def get_clone_url(self) -> str:
        """Get the clone URL for the repository"""
        if self.url.endswith(".git"):
            return self.url
        return f"{self.url}.git"

    def get_web_url(self) -> str:
        """Get the web URL for the repository"""
        return self.url.rstrip(".git")

    def __str__(self) -> str:
        return f"{self.provider}:{self.org}/{self.name}"

    def __repr__(self) -> str:
        return f"Repository(id={self.id}, provider={self.provider}, org={self.org}, name={self.name})"


class RepositoryCreate(BaseModel):
    """Repository creation model for API requests"""

    url: str = Field(description="Repository URL")
    branch: Optional[str] = Field(
        default=None, description="Target branch (defaults to repository default)"
    )
    provider: Optional[RepositoryProvider] = Field(
        default=None, description="Repository provider (auto-detected if not specified)"
    )


class RepositoryUpdate(BaseModel):
    """Repository update model for API requests"""

    default_branch: Optional[str] = Field(
        default=None, description="Default branch name"
    )
    webhook_configured: Optional[bool] = Field(
        default=None, description="Webhook configuration status"
    )
    webhook_secret: Optional[str] = Field(default=None, description="Webhook secret")
    subscribed_events: Optional[List[str]] = Field(
        default=None, description="Subscribed webhook events"
    )


class RepositoryResponse(Repository):
    """Repository response model for API responses"""
    
    # Override webhook_secret to exclude it from serialization (write-only behavior)
    webhook_secret: Optional[str] = Field(
        default=None, 
        description="Secret for validating webhook signatures",
        exclude=True  # Excludes from serialization, equivalent to write_only in V1
    )


class RepositoryList(BaseModel):
    """Repository list response model"""

    repositories: List[RepositoryResponse] = Field(description="List of repositories")
    total: int = Field(description="Total number of repositories")
    limit: int = Field(description="Number of repositories per page")
    offset: int = Field(description="Offset for pagination")
