"""Chat data models for conversational queries

This module defines chat-related Pydantic models including
ChatSession, Question, Answer, and Citation based on data-model.md.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, field_serializer, model_validator


class SessionStatus(str, Enum):
    """Chat session status"""

    ACTIVE = "active"
    EXPIRED = "expired"


class Citation(BaseModel):
    """Reference to source code supporting an answer

    Provides traceable references to specific code locations
    that support the generated answer.
    """

    # Source location
    file_path: str = Field(description="Path to source file")
    line_start: Optional[int] = Field(default=None, description="Starting line number")
    line_end: Optional[int] = Field(default=None, description="Ending line number")

    # Version and access
    commit_sha: str = Field(description="Commit SHA of referenced code")
    url: str = Field(description="Direct link to source code")

    # Content excerpt
    excerpt: Optional[str] = Field(default=None, description="Relevant code snippet")

    model_config = ConfigDict(
        validate_assignment=True
    )

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        """Validate file path format"""
        if not v or not v.strip():
            raise ValueError("File path cannot be empty")

        # Normalize path separators
        return v.strip().replace("\\", "/")

    @field_validator("line_start", "line_end")
    @classmethod
    def validate_line_numbers(cls, v: Optional[int]) -> Optional[int]:
        """Validate line numbers are positive"""
        if v is not None and v <= 0:
            raise ValueError("Line numbers must be positive")
        return v

    @field_validator("commit_sha")
    @classmethod
    def validate_commit_sha(cls, v: str) -> str:
        """Validate commit SHA format"""
        if not v or not v.strip():
            raise ValueError("Commit SHA cannot be empty")

        # Git SHA-1 hash validation
        import re

        if not re.match(r"^[a-f0-9]{40}$", v.strip(), re.IGNORECASE):
            raise ValueError("Invalid git commit SHA format")

        return v.strip().lower()

    @field_validator("url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format"""
        if not v or not v.strip():
            raise ValueError("URL cannot be empty")

        # Basic URL validation
        import re

        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IP
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(v.strip()):
            raise ValueError("Invalid URL format")

        return v.strip()

    @model_validator(mode="after")
    def validate_line_range(self) -> "Citation":
        """Validate line range consistency"""
        if self.line_start is not None and self.line_end is not None:
            if self.line_start > self.line_end:
                raise ValueError("line_start must be less than or equal to line_end")

        return self

    def get_line_range_str(self) -> Optional[str]:
        """Get line range as string (e.g., 'L15-L30')"""
        if self.line_start is None:
            return None

        if self.line_end is None or self.line_start == self.line_end:
            return f"L{self.line_start}"

        return f"L{self.line_start}-L{self.line_end}"

    def get_file_name(self) -> str:
        """Get file name without path"""
        return self.file_path.split("/")[-1]

    def __str__(self) -> str:
        line_info = self.get_line_range_str()
        if line_info:
            return f"{self.file_path}#{line_info}"
        return self.file_path

    def __repr__(self) -> str:
        return (
            f"Citation(file_path={self.file_path}, commit_sha={self.commit_sha[:8]}...)"
        )


class Answer(BaseModel):
    """AI-generated response to user question

    Contains the generated answer with citations, confidence metrics,
    and performance information.
    """

    # Core identification
    id: UUID = Field(default_factory=uuid4, description="Unique answer identifier")
    question_id: UUID = Field(description="Foreign key to Question")

    # Answer content
    content: str = Field(description="Answer text")
    citations: List[Citation] = Field(
        default_factory=list, description="Source code references"
    )

    # Quality metrics
    confidence_score: float = Field(description="Answer confidence (0.0-1.0)")
    generation_time: float = Field(description="Response generation time (seconds)")

    # Audit field
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Answer timestamp",
    )

    model_config = ConfigDict(
        validate_assignment=True
    )

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate answer content"""
        if not v or not v.strip():
            raise ValueError("Answer content cannot be empty")
        return v.strip()

    @field_validator("confidence_score")
    @classmethod
    def validate_confidence_score(cls, v: float) -> float:
        """Validate confidence score range"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v

    @field_validator("generation_time")
    @classmethod
    def validate_generation_time(cls, v: float) -> float:
        """Validate generation time is positive"""
        if v < 0:
            raise ValueError("Generation time must be non-negative")
        return v

    def add_citation(self, citation: Citation) -> None:
        """Add a citation to the answer"""
        self.citations.append(citation)

    def get_citation_count(self) -> int:
        """Get number of citations"""
        return len(self.citations)

    def get_unique_files(self) -> List[str]:
        """Get unique file paths from citations"""
        return list(set(citation.file_path for citation in self.citations))

    def get_word_count(self) -> int:
        """Get approximate word count of answer"""
        return len(self.content.split())

    def has_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if answer has high confidence"""
        return self.confidence_score >= threshold

    def __str__(self) -> str:
        return f"Answer (confidence: {self.confidence_score:.2f}, {len(self.citations)} citations)"

    def __repr__(self) -> str:
        return f"Answer(id={self.id}, question_id={self.question_id}, confidence={self.confidence_score:.2f})"


class Question(BaseModel):
    """User query about the codebase

    Represents a user's question with context and metadata
    about the semantic search results used.
    """

    # Core identification
    id: UUID = Field(default_factory=uuid4, description="Unique question identifier")
    session_id: UUID = Field(description="Foreign key to ChatSession")

    # Question content
    content: str = Field(description="Question text")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Question timestamp",
    )

    # Context information
    context_nodes: List[str] = Field(
        default_factory=list, description="Relevant analysis node IDs used for context"
    )

    model_config = ConfigDict(
        validate_assignment=True
    )

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate question content"""
        if not v or not v.strip():
            raise ValueError("Question content cannot be empty")
        return v.strip()

    def get_word_count(self) -> int:
        """Get approximate word count of question"""
        return len(self.content.split())

    def has_context(self) -> bool:
        """Check if question has context nodes"""
        return len(self.context_nodes) > 0

    def __str__(self) -> str:
        preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"Question: {preview}"

    def __repr__(self) -> str:
        return f"Question(id={self.id}, session_id={self.session_id})"


class ChatSession(BaseModel):
    """Conversational query session for a repository

    Manages a conversation session with state tracking
    and message counting.
    """

    # Core identification
    id: UUID = Field(default_factory=uuid4, description="Session identifier")
    repository_id: UUID = Field(description="Foreign key to Repository")

    # Session state
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Session creation timestamp",
    )
    last_activity: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last activity timestamp",
    )
    status: SessionStatus = Field(
        default=SessionStatus.ACTIVE, description="Session status"
    )

    # Statistics
    message_count: int = Field(
        default=0, description="Number of questions in this session"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )

    @field_serializer('created_at', 'last_activity')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)

    @field_validator("message_count")
    @classmethod
    def validate_message_count(cls, v: int) -> int:
        """Validate message count is non-negative"""
        if v < 0:
            raise ValueError("Message count must be non-negative")
        return v

    def update_activity(self) -> None:
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)

    def increment_message_count(self) -> None:
        """Increment message count and update activity"""
        self.message_count += 1
        self.update_activity()

    def expire(self) -> None:
        """Mark session as expired"""
        self.status = SessionStatus.EXPIRED
        self.update_activity()

    def is_active(self) -> bool:
        """Check if session is active"""
        return self.status == SessionStatus.ACTIVE

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return self.status == SessionStatus.EXPIRED

    def get_session_duration(self) -> float:
        """Get session duration in seconds"""
        return (self.last_activity - self.created_at).total_seconds()

    def __str__(self) -> str:
        return f"ChatSession ({self.status}, {self.message_count} messages)"

    def __repr__(self) -> str:
        return f"ChatSession(id={self.id}, repository_id={self.repository_id}, status={self.status})"


# API Models


class QuestionRequest(BaseModel):
    """Question request model for API"""

    content: str = Field(description="The question to ask about the codebase")
    context_hint: Optional[str] = Field(
        default=None,
        description="Optional hint about what part of codebase to focus on",
    )

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """Validate question content"""
        if not v or not v.strip():
            raise ValueError("Question content cannot be empty")
        return v.strip()


class QuestionResponse(BaseModel):
    """Question response model for API"""

    id: UUID = Field(description="Question ID")
    session_id: UUID = Field(description="Session ID")
    content: str = Field(description="Question content")
    timestamp: datetime = Field(description="Question timestamp")
    context_files: List[str] = Field(
        description="File paths used for context via semantic search"
    )

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id', 'session_id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)


class AnswerResponse(BaseModel):
    """Answer response model for API"""

    id: UUID = Field(description="Answer ID")
    question_id: UUID = Field(description="Question ID")
    content: str = Field(description="The generated answer")
    citations: List[Citation] = Field(description="Source code citations")
    confidence_score: float = Field(description="Answer confidence score")
    generation_time: float = Field(description="Response generation time in seconds")
    timestamp: datetime = Field(description="Answer timestamp")

    model_config = ConfigDict()

    @field_serializer('timestamp')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id', 'question_id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)


class QuestionAnswer(BaseModel):
    """Combined question and answer model for API responses"""

    question: QuestionResponse = Field(description="Question details")
    answer: AnswerResponse = Field(description="Answer details")


class ChatSessionResponse(BaseModel):
    """Chat session response model for API"""

    id: UUID = Field(description="Session ID")
    repository_id: UUID = Field(description="Repository ID")
    created_at: datetime = Field(description="Session creation timestamp")
    last_activity: datetime = Field(description="Last activity timestamp")
    status: SessionStatus = Field(description="Session status")
    message_count: int = Field(description="Number of questions in this session")

    model_config = ConfigDict(
        use_enum_values=True
    )

    @field_serializer('created_at', 'last_activity')
    def serialize_datetime(self, value: datetime) -> str:
        """Serialize datetime to ISO format"""
        return value.isoformat()
    
    @field_serializer('id', 'repository_id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)


class ChatSessionList(BaseModel):
    """Chat session list response model"""

    sessions: List[ChatSessionResponse] = Field(description="List of chat sessions")
    total: int = Field(description="Total number of sessions")

    model_config = ConfigDict()

    @field_serializer('sessions')
    def serialize_sessions(self, value: List[ChatSessionResponse]) -> List[Dict[str, Any]]:
        """Serialize sessions ensuring UUIDs are properly handled"""
        return [session.model_dump() for session in value]


class ConversationHistory(BaseModel):
    """Conversation history response model"""

    session_id: UUID = Field(description="Session ID")
    questions_and_answers: List[QuestionAnswer] = Field(
        description="Q&A pairs in chronological order"
    )
    total: int = Field(description="Total number of Q&A pairs")
    has_more: bool = Field(description="Whether there are more results")

    model_config = ConfigDict()

    @field_serializer('session_id')
    def serialize_uuid(self, value: UUID) -> str:
        """Serialize UUID to string"""
        return str(value)
