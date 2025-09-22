"""Chat API routes

This module implements the chat API endpoints for conversational
repository queries based on the chat_api.yaml contract specification.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from ...models.chat import (
    ChatSessionList,
    ChatSessionResponse,
    ConversationHistory,
    QuestionAnswer,
    QuestionRequest,
    SessionStatus,
)
from ...services.auth_service import User
from ...services.chat_service import chat_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repositories", tags=["chat"])


# Dependency for getting current user (simplified for now)
async def get_current_user(token: str = Depends(lambda: "mock-token")) -> User:
    """Get current authenticated user"""
    # For now, return a mock user since auth middleware isn't implemented yet
    from uuid import uuid4

    return User(
        id=uuid4(),
        username="admin",
        email="admin@autodoc.dev",
        full_name="Admin User",
        is_admin=True,
    )


@router.post(
    "/{repository_id}/chat/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_chat_session(
    repository_id: UUID, current_user: User = Depends(get_current_user)
):
    """Create chat session"""
    try:
        # Create session using service
        result = await chat_service.create_chat_session(repository_id)

        if result["status"] != "success":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found",
                        "message": result["error"],
                    },
                )
            elif error_type == "RepositoryNotAnalyzed":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not analyzed",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to create chat session",
                        "message": result["error"],
                    },
                )

        # Convert to response model
        session_data = result["session"]
        session_data["id"] = UUID(session_data["id"])
        session_data["repository_id"] = UUID(session_data["repository_id"])

        return ChatSessionResponse(**session_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create chat session endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to create chat session",
            },
        )


@router.get("/{repository_id}/chat/sessions", response_model=ChatSessionList)
async def list_chat_sessions(
    repository_id: UUID,
    status_filter: Optional[str] = Query(
        "active", description="Filter by session status"
    ),
    limit: int = Query(50, ge=1, le=100, description="Number of sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    current_user: User = Depends(get_current_user),
):
    """List chat sessions"""
    try:
        # Validate status filter
        if status_filter and status_filter not in ["active", "expired"]:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "Invalid status filter",
                    "message": "Status must be 'active' or 'expired'",
                },
            )

        # List sessions using service
        result = await chat_service.list_chat_sessions(
            repository_id=repository_id,
            status_filter=status_filter,
            limit=limit,
            offset=offset,
        )

        if result["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to list chat sessions",
                    "message": result["error"],
                },
            )

        # Convert to response format
        sessions = []
        for session_data in result["sessions"]:
            session_data["id"] = UUID(session_data["id"])
            session_data["repository_id"] = UUID(session_data["repository_id"])
            sessions.append(ChatSessionResponse(**session_data))

        return ChatSessionList(sessions=sessions, total=result["total"])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List chat sessions endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to list chat sessions",
            },
        )


@router.get(
    "/{repository_id}/chat/sessions/{session_id}", response_model=ChatSessionResponse
)
async def get_chat_session(
    repository_id: UUID,
    session_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """Get chat session details"""
    try:
        # Get session using service
        result = await chat_service.get_chat_session(repository_id, session_id)

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Session not found", "message": result["error"]},
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get chat session",
                        "message": result["error"],
                    },
                )

        # Convert to response model
        session_data = result["session"]
        session_data["id"] = UUID(session_data["id"])
        session_data["repository_id"] = UUID(session_data["repository_id"])

        return ChatSessionResponse(**session_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get chat session endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get chat session",
            },
        )


@router.delete(
    "/{repository_id}/chat/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_chat_session(
    repository_id: UUID,
    session_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """End chat session"""
    try:
        # Delete session using service
        result = await chat_service.delete_chat_session(repository_id, session_id)

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Session not found", "message": result["error"]},
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to delete chat session",
                        "message": result["error"],
                    },
                )

        return None  # 204 No Content

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete chat session endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to delete chat session",
            },
        )


@router.post(
    "/{repository_id}/chat/sessions/{session_id}/questions",
    response_model=QuestionAnswer,
    status_code=status.HTTP_201_CREATED,
)
async def ask_question(
    repository_id: UUID,
    session_id: UUID,
    question_request: QuestionRequest,
    current_user: User = Depends(get_current_user),
):
    """Submit a question about the repository codebase"""
    try:
        # Validate question content
        if not question_request.content or not question_request.content.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid question format",
                    "message": "Question content cannot be empty",
                },
            )

        # Ask question using service
        result = await chat_service.ask_question(
            repository_id=repository_id,
            session_id=session_id,
            question_request=question_request,
        )

        if result["status"] != "success":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Session not found", "message": result["error"]},
                )
            elif error_type == "SessionInactive":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Session is not active",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to process question",
                        "message": result["error"],
                    },
                )

        # Convert to response model
        question_data = result["question"]
        answer_data = result["answer"]

        # Convert UUIDs
        question_data["id"] = UUID(question_data["id"])
        question_data["session_id"] = UUID(question_data["session_id"])
        answer_data["id"] = UUID(answer_data["id"])
        answer_data["question_id"] = UUID(answer_data["question_id"])

        return QuestionAnswer(question=question_data, answer=answer_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ask question endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to process question",
            },
        )


@router.get("/{repository_id}/chat/sessions/{session_id}/stream")
async def stream_chat_responses(
    repository_id: UUID,
    session_id: UUID,
    current_user: User = Depends(get_current_user),
):
    """Server-sent events stream for real-time chat responses"""
    try:
        # This would implement SSE streaming
        # For now, return a basic response

        async def generate_stream():
            """Generate SSE stream"""
            yield 'data: {"message": "Streaming endpoint ready"}\n\n'
            yield 'data: {"status": "connected"}\n\n'

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    except Exception as e:
        logger.error(f"Stream chat responses endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to setup streaming",
            },
        )


@router.get(
    "/{repository_id}/chat/sessions/{session_id}/history",
    response_model=ConversationHistory,
)
async def get_conversation_history(
    repository_id: UUID,
    session_id: UUID,
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    before: Optional[str] = Query(
        None, description="Get messages before this timestamp"
    ),
    current_user: User = Depends(get_current_user),
):
    """Get conversation history"""
    try:
        # Parse before timestamp if provided
        before_datetime = None
        if before:
            from datetime import datetime

            try:
                before_datetime = datetime.fromisoformat(before.replace("Z", "+00:00"))
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={
                        "error": "Invalid timestamp format",
                        "message": "Use ISO 8601 format for 'before' parameter",
                    },
                )

        # Get conversation history using service
        result = await chat_service.get_conversation_history(
            repository_id=repository_id,
            session_id=session_id,
            limit=limit,
            before=before_datetime,
        )

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Session not found", "message": result["error"]},
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get conversation history",
                        "message": result["error"],
                    },
                )

        # Convert to response model
        return ConversationHistory(
            session_id=session_id,
            questions_and_answers=result["questions_and_answers"],
            total=result["total"],
            has_more=result["has_more"],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation history endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get conversation history",
            },
        )
