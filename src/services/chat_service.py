"""Chat service for conversational repository queries

This module provides chat services including session management,
question answering, and conversational AI capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional
from uuid import UUID, uuid4

from ..agents.workflow import WorkflowType, workflow_orchestrator
from ..models.chat import (
    Answer,
    AnswerResponse,
    ChatSession,
    ChatSessionResponse,
    Citation,
    Question,
    QuestionAnswer,
    QuestionRequest,
    QuestionResponse,
    SessionStatus,
)
from ..repository.repository_repository import RepositoryRepository
from ..tools.context_tool import context_tool
from ..tools.llm_tool import llm_tool
from ..utils.config_loader import get_settings

logger = logging.getLogger(__name__)


class ChatService:
    """Chat service for conversational repository queries

    Provides comprehensive chat capabilities including session management,
    question answering with RAG, and conversation history tracking.
    """

    def __init__(self, chat_session_repo=None, question_repo=None, answer_repo=None):
        """Initialize chat service with repository dependency injection"""
        self.settings = get_settings()
        self.session_timeout_hours = 24  # Sessions expire after 24 hours
        self.max_context_documents = 10
        self.min_confidence_threshold = 0.3
        # Repositories will be injected or lazily loaded
        self._chat_session_repo = chat_session_repo
        self._question_repo = question_repo
        self._answer_repo = answer_repo

    async def _get_chat_session_repo(self):
        """Get chat session repository instance (lazy loading)"""
        if self._chat_session_repo is None:
            from ..repository.chat_session_repository import ChatSessionRepository
            from ..models.chat import ChatSession
            
            self._chat_session_repo = ChatSessionRepository(ChatSession)
        return self._chat_session_repo

    async def _get_question_repo(self):
        """Get question repository instance (lazy loading)"""
        if self._question_repo is None:
            from ..repository.question_repository import QuestionRepository
            from ..models.chat import Question
            
            self._question_repo = QuestionRepository(Question)
        return self._question_repo

    async def _get_answer_repo(self):
        """Get answer repository instance (lazy loading)"""
        if self._answer_repo is None:
            from ..repository.answer_repository import AnswerRepository
            from ..models.chat import Answer
            
            self._answer_repo = AnswerRepository(Answer)
        return self._answer_repo

    async def create_chat_session(self, repository_id: UUID) -> Dict[str, Any]:
        """Create a new chat session for repository

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with session creation result
        """
        try:
            # Check if repository exists and is analyzed
            from ..models.repository import Repository
            repo_repository = RepositoryRepository(Repository)
            repository = await repo_repository.find_one({"id": repository_id})

            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Check if repository has processed documents
            from ..repository.code_document_repository import CodeDocumentRepository
            from ..models.code_document import CodeDocument
            code_document_repo = CodeDocumentRepository(CodeDocument)
            doc_count = await code_document_repo.count(
                {"repository_id": str(repository_id)}
            )
            if doc_count == 0:
                return {
                    "status": "error",
                    "error": "Repository not analyzed yet. Please run analysis first.",
                    "error_type": "RepositoryNotAnalyzed",
                }

            # Create chat session
            chat_session = ChatSession(repository_id=repository_id)

            # Store in database
            chat_session_repo = await self._get_chat_session_repo()
            await chat_session_repo.insert(chat_session)

            return {
                "status": "success",
                "session": ChatSessionResponse(
                    id=chat_session.id,
                    repository_id=repository_id,
                    created_at=chat_session.created_at,
                    last_activity=chat_session.last_activity,
                    status=chat_session.status,
                    message_count=chat_session.message_count,
                ).model_dump(),
            }

        except Exception as e:
            logger.error(f"Chat session creation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def get_chat_session(
        self, repository_id: UUID, session_id: UUID
    ) -> Dict[str, Any]:
        """Get chat session details

        Args:
            repository_id: Repository UUID
            session_id: Session UUID

        Returns:
            Dictionary with session details
        """
        try:
            chat_session_repo = await self._get_chat_session_repo()

            # Get session
            session = await chat_session_repo.find_one(
                {"id": str(session_id), "repository_id": str(repository_id)}
            )
            session_data = chat_session_repo.serialize(session) if session else None

            if not session_data:
                return {
                    "status": "error",
                    "error": "Session not found",
                    "error_type": "NotFound",
                }

            # Check if session is expired
            session_data["id"] = UUID(session_data["id"])
            session_data["repository_id"] = UUID(session_data["repository_id"])
            chat_session = ChatSession(**session_data)

            if self._is_session_expired(chat_session):
                # Mark as expired
                await self._expire_session(session_id)
                chat_session.status = SessionStatus.EXPIRED

            return {
                "status": "success",
                "session": ChatSessionResponse(
                    id=chat_session.id,
                    repository_id=chat_session.repository_id,
                    created_at=chat_session.created_at,
                    last_activity=chat_session.last_activity,
                    status=chat_session.status,
                    message_count=chat_session.message_count,
                ).model_dump(),
            }

        except Exception as e:
            logger.error(f"Get chat session failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": str(session_id),
            }

    async def list_chat_sessions(
        self,
        repository_id: UUID,
        status_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List chat sessions for repository

        Args:
            repository_id: Repository UUID
            status_filter: Optional status filter
            limit: Maximum sessions to return
            offset: Number of sessions to skip

        Returns:
            Dictionary with session list
        """
        try:
            chat_session_repo = await self._get_chat_session_repo()

            # Build query
            query = {"repository_id": str(repository_id)}
            if status_filter:
                query["status"] = status_filter

            # Get sessions
            sessions = await chat_session_repo.find_many(
                query,
                limit=limit,
                offset=offset,
                sort=[("last_activity", -1)],
            )
            sessions_data = chat_session_repo.serialize_many(sessions)

            # Convert to response format
            sessions = []
            for session_data in sessions_data:
                session_data["id"] = UUID(session_data["id"])
                session_data["repository_id"] = UUID(session_data["repository_id"])
                chat_session = ChatSession(**session_data)

                # Check expiration
                if self._is_session_expired(chat_session):
                    await self._expire_session(chat_session.id)
                    chat_session.status = SessionStatus.EXPIRED

                sessions.append(
                    ChatSessionResponse(
                        id=chat_session.id,
                        repository_id=chat_session.repository_id,
                        created_at=chat_session.created_at,
                        last_activity=chat_session.last_activity,
                        status=chat_session.status,
                        message_count=chat_session.message_count,
                    )
                )

            # Get total count
            total_count = await chat_session_repo.count(query)

            return {
                "status": "success",
                "sessions": [session.model_dump() for session in sessions],
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "status_filter": status_filter,
            }

        except Exception as e:
            logger.error(f"List chat sessions failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "sessions": [],
            }

    async def delete_chat_session(
        self, repository_id: UUID, session_id: UUID
    ) -> Dict[str, Any]:
        """Delete chat session and all associated data

        Args:
            repository_id: Repository UUID
            session_id: Session UUID

        Returns:
            Dictionary with deletion result
        """
        try:
            chat_session_repo = await self._get_chat_session_repo()
            question_repo = await self._get_question_repo()
            answer_repo = await self._get_answer_repo()

            # Check if session exists
            session = await chat_session_repo.find_one(
                {"id": str(session_id), "repository_id": str(repository_id)}
            )
            session_data = chat_session_repo.serialize(session) if session else None

            if not session_data:
                return {
                    "status": "error",
                    "error": "Session not found",
                    "error_type": "NotFound",
                }

            # Delete session and associated Q&A
            # TODO: Implement proper transaction handling in data access layer

            # Delete questions and answers
            questions = await question_repo.find_many(
                {"session_id": str(session_id)}
            )
            for question in questions:
                question_data = question_repo.serialize(question)
                await answer_repo.delete_one(
                    {"question_id": question_data["id"]}
                )

            await question_repo.delete_many({"session_id": str(session_id)})

            # Delete session
            await chat_session_repo.delete_one({"id": str(session_id)})

            return {"status": "success", "message": "Session deleted successfully"}

        except Exception as e:
            logger.error(f"Delete chat session failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": str(session_id),
            }

    async def ask_question(
        self, repository_id: UUID, session_id: UUID, question_request: QuestionRequest
    ) -> Dict[str, Any]:
        """Ask a question in a chat session

        Args:
            repository_id: Repository UUID
            session_id: Session UUID
            question_request: Question data

        Returns:
            Dictionary with question and answer
        """
        try:
            # Validate session
            session_result = await self.get_chat_session(repository_id, session_id)
            if session_result["status"] != "success":
                return session_result

            session_data = session_result["session"]
            if session_data["status"] != SessionStatus.ACTIVE.value:
                return {
                    "status": "error",
                    "error": "Session is not active",
                    "error_type": "SessionInactive",
                }

            # Create question
            question = Question(session_id=session_id, content=question_request.content)

            # Retrieve relevant context
            context_result = await context_tool._arun(
                "hybrid_search",
                query=question_request.content,
                repository_id=str(repository_id),
                k=self.max_context_documents,
            )

            if context_result["status"] != "success":
                logger.warning(
                    f"Context retrieval failed: {context_result.get('error')}"
                )
                context_documents = []
            else:
                context_documents = context_result.get("contexts", [])

            # Update question with context files
            question.context_nodes = [
                ctx.get("file_path", "") for ctx in context_documents
            ]

            # Generate answer using LLM
            answer_result = await llm_tool._arun(
                "answer_question",
                question=question_request.content,
                context_documents=context_documents,
            )

            if answer_result["status"] != "success":
                return {
                    "status": "error",
                    "error": f"Answer generation failed: {answer_result.get('error')}",
                    "error_type": "AnswerGenerationFailed",
                }

            # Create answer with citations
            citations = []
            for citation_data in answer_result.get("citations", []):
                citation = Citation(
                    file_path=citation_data["file_path"],
                    line_start=citation_data.get("line_start"),
                    line_end=citation_data.get("line_end"),
                    commit_sha=citation_data.get("commit_sha", "unknown"),
                    url=citation_data.get("url", ""),
                    excerpt=citation_data.get("excerpt"),
                )
                citations.append(citation)

            answer = Answer(
                question_id=question.id,
                content=answer_result["generated_text"],
                citations=citations,
                confidence_score=answer_result.get("confidence_score", 0.8),
                generation_time=answer_result.get("generation_time", 0.0),
            )

            # Store question and answer
            question_repo = await self._get_question_repo()
            answer_repo = await self._get_answer_repo()
            chat_session_repo = await self._get_chat_session_repo()

            # Store question
            await question_repo.insert(question)

            # Store answer
            await answer_repo.insert(answer)

            # Update session activity
            await chat_session_repo.update_one(
                {"id": str(session_id)},
                {
                    "last_activity": datetime.now(timezone.utc),
                    "message_count": session_data["message_count"] + 1,
                },
            )

            # Create response
            question_response = QuestionResponse(
                id=question.id,
                session_id=session_id,
                content=question.content,
                timestamp=question.timestamp,
                context_files=question.context_nodes,
            )

            answer_response = AnswerResponse(
                id=answer.id,
                question_id=question.id,
                content=answer.content,
                citations=citations,
                confidence_score=answer.confidence_score,
                generation_time=answer.generation_time,
                timestamp=answer.timestamp,
            )

            return {
                "status": "success",
                "question": question_response.model_dump(),
                "answer": answer_response.model_dump(),
            }

        except Exception as e:
            logger.error(f"Ask question failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": str(session_id),
            }

    async def get_conversation_history(
        self,
        repository_id: UUID,
        session_id: UUID,
        limit: int = 50,
        before: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get conversation history for session

        Args:
            repository_id: Repository UUID
            session_id: Session UUID
            limit: Maximum Q&A pairs to return
            before: Get messages before this timestamp

        Returns:
            Dictionary with conversation history
        """
        try:
            # Validate session
            session_result = await self.get_chat_session(repository_id, session_id)
            if session_result["status"] != "success":
                return session_result

            question_repo = await self._get_question_repo()
            answer_repo = await self._get_answer_repo()

            # Build query for questions
            question_query = {"session_id": str(session_id)}
            if before:
                question_query["timestamp"] = {"$lt": before}

            # Get questions
            questions = await question_repo.find_many(
                question_query,
                limit=limit,
                sort=[("timestamp", -1)],  # Most recent first
            )
            questions_data = question_repo.serialize_many(questions)

            # Get answers for questions
            question_answer_pairs = []
            for question_data in questions_data:
                question_data["id"] = UUID(question_data["id"])
                question_data["session_id"] = UUID(question_data["session_id"])
                question = Question(**question_data)

                # Get corresponding answer
                answer = await answer_repo.find_one(
                    {"question_id": str(question.id)}
                )
                answer_data = answer_repo.serialize(answer) if answer else None

                if answer_data:
                    answer_data["id"] = UUID(answer_data["id"])
                    answer_data["question_id"] = UUID(answer_data["question_id"])
                    answer = Answer(**answer_data)

                    # Create response objects
                    question_response = QuestionResponse(
                        id=question.id,
                        session_id=session_id,
                        content=question.content,
                        timestamp=question.timestamp,
                        context_files=question.context_nodes,
                    )

                    answer_response = AnswerResponse(
                        id=answer.id,
                        question_id=question.id,
                        content=answer.content,
                        citations=answer.citations,
                        confidence_score=answer.confidence_score,
                        generation_time=answer.generation_time,
                        timestamp=answer.timestamp,
                    )

                    question_answer_pairs.append(
                        QuestionAnswer(
                            question=question_response, answer=answer_response
                        )
                    )

            # Check if there are more messages
            has_more = len(questions_data) == limit

            return {
                "status": "success",
                "session_id": str(session_id),
                "questions_and_answers": [
                    qa.model_dump() for qa in question_answer_pairs
                ],
                "total": len(question_answer_pairs),
                "has_more": has_more,
                "limit": limit,
            }

        except Exception as e:
            logger.error(f"Get conversation history failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "session_id": str(session_id),
            }

    async def stream_chat_response(
        self, repository_id: UUID, session_id: UUID, question: str
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream chat response for real-time interaction

        Args:
            repository_id: Repository UUID
            session_id: Session UUID
            question: User question

        Yields:
            Dictionary with streaming response chunks
        """
        try:
            # Validate session
            session_result = await self.get_chat_session(repository_id, session_id)
            if session_result["status"] != "success":
                yield {
                    "status": "error",
                    "error": session_result.get("error", "Session validation failed"),
                    "finished": True,
                }
                return

            # Retrieve context
            yield {
                "status": "processing",
                "step": "retrieving_context",
                "message": "Searching for relevant context...",
                "finished": False,
            }

            context_result = await context_tool._arun(
                "hybrid_search",
                query=question,
                repository_id=str(repository_id),
                k=self.max_context_documents,
            )

            if context_result["status"] != "success":
                context_documents = []
            else:
                context_documents = context_result.get("contexts", [])

            yield {
                "status": "processing",
                "step": "generating_response",
                "message": f"Found {len(context_documents)} relevant documents. Generating answer...",
                "finished": False,
            }

            # Stream answer generation
            async for chunk in llm_tool._arun(
                "stream",
                prompt=f"Question: {question}\n\nContext: {context_documents[:3]}",  # Limit context for streaming
            ):
                yield {
                    "status": "streaming",
                    "chunk": chunk.get("chunk", ""),
                    "finished": chunk.get("finished", False),
                    "total_content": chunk.get("total_content", ""),
                }

                if chunk.get("finished"):
                    break

            # Final completion message
            yield {
                "status": "completed",
                "message": "Response generated successfully",
                "finished": True,
            }

        except Exception as e:
            logger.error(f"Stream chat response failed: {e}")
            yield {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "finished": True,
            }

    async def expire_old_sessions(self) -> Dict[str, Any]:
        """Expire old inactive sessions

        Returns:
            Dictionary with expiration results
        """
        try:
            chat_session_repo = await self._get_chat_session_repo()

            # Calculate cutoff time
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=self.session_timeout_hours
            )

            # Find expired sessions
            expired_sessions = await chat_session_repo.find_many(
                {
                    "status": SessionStatus.ACTIVE.value,
                    "last_activity": {"$lt": cutoff_time},
                }
            )

            # Expire sessions
            expired_count = 0
            for session in expired_sessions:
                session_data = chat_session_repo.serialize(session)
                await chat_session_repo.update_one(
                    {"id": session_data["id"]},
                    {"status": SessionStatus.EXPIRED.value},
                )
                expired_count += 1

            return {
                "status": "success",
                "expired_sessions": expired_count,
                "cutoff_time": cutoff_time.isoformat(),
            }

        except Exception as e:
            logger.error(f"Session expiration failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    def _is_session_expired(self, session: ChatSession) -> bool:
        """Check if session is expired

        Args:
            session: Chat session

        Returns:
            True if session is expired
        """
        try:
            if session.status == SessionStatus.EXPIRED:
                return True

            # Check timeout
            cutoff_time = datetime.now(timezone.utc) - timedelta(
                hours=self.session_timeout_hours
            )
            return session.last_activity < cutoff_time

        except Exception:
            return False

    async def _expire_session(self, session_id: UUID) -> None:
        """Mark session as expired

        Args:
            session_id: Session UUID
        """
        try:
            chat_session_repo = await self._get_chat_session_repo()
            await chat_session_repo.update_one(
                {"id": str(session_id)},
                {"status": SessionStatus.EXPIRED.value},
            )
        except Exception as e:
            logger.error(f"Failed to expire session {session_id}: {e}")

    async def get_chat_statistics(
        self, repository_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get chat usage statistics

        Args:
            repository_id: Optional repository filter

        Returns:
            Dictionary with chat statistics
        """
        try:
            chat_session_repo = await self._get_chat_session_repo()
            question_repo = await self._get_question_repo()

            # Build base query
            base_query = {}
            if repository_id:
                base_query["repository_id"] = str(repository_id)

            # Get session statistics
            total_sessions = await chat_session_repo.count(base_query)
            active_sessions = await chat_session_repo.count(
                {**base_query, "status": SessionStatus.ACTIVE.value}
            )

            # Get question statistics
            if repository_id:
                # Get questions for this repository's sessions
                sessions = await chat_session_repo.find_many(base_query)
                session_ids = [chat_session_repo.serialize(session)["id"] for session in sessions]

                question_query = {"session_id": {"$in": session_ids}}
            else:
                question_query = {}

            total_questions = await question_repo.count(question_query)

            # Get recent activity
            recent_sessions_docs = await chat_session_repo.find_many(
                base_query, limit=5, sort=[("last_activity", -1)]
            )
            recent_sessions = chat_session_repo.serialize_many(recent_sessions_docs)

            return {
                "status": "success",
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "expired_sessions": total_sessions - active_sessions,
                "total_questions": total_questions,
                "average_questions_per_session": (
                    total_questions / total_sessions if total_sessions > 0 else 0
                ),
                "recent_activity": [
                    {
                        "session_id": session["id"],
                        "repository_id": session["repository_id"],
                        "message_count": session["message_count"],
                        "last_activity": session["last_activity"],
                    }
                    for session in recent_sessions
                ],
                "repository_id": str(repository_id) if repository_id else None,
            }

        except Exception as e:
            logger.error(f"Get chat statistics failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# Global chat service instance
chat_service = ChatService()
