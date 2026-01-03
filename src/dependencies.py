"""Centralized dependency injection for AutoDoc v2.

Provides factory functions for repositories and services following FastAPI's
dependency injection pattern. Dependencies are request-scoped by default.

Usage:
    # In API routes with FastAPI Depends()
    from fastapi import Depends
    from typing import Annotated
    from src.dependencies import get_repository_service

    @router.post("/")
    async def create_repository(
        data: RepositoryCreate,
        service: Annotated[RepositoryService, Depends(get_repository_service)]
    ):
        return await service.create_repository(data)

Note: FastAPI automatically caches dependencies per request, so you get a fresh
instance for each request while still being efficient within the same request.
"""

from typing import Annotated

from fastapi import Depends

from .agents.document_agent import DocumentProcessingAgent
from .agents.wiki_agent import WikiGenerationAgent
from .agents.workflow import WorkflowOrchestrator
from .services.mcp_filesystem_client import MCPFilesystemClient, get_mcp_filesystem_client
from .models.chat import Answer, ChatSession, Question
from .models.code_document import CodeDocument
from .models.repository import Repository
from .models.wiki import WikiStructure
from .repository.answer_repository import AnswerRepository
from .repository.chat_session_repository import ChatSessionRepository
from .repository.code_document_repository import CodeDocumentRepository
from .repository.question_repository import QuestionRepository
from .repository.repository_repository import RepositoryRepository
from .repository.user_repository import UserRepository
from .repository.wiki_structure_repository import WikiStructureRepository
from .services.auth_service import AuthenticationService
from .services.chat_service import ChatService
from .services.document_service import DocumentProcessingService
from .services.repository_service import RepositoryService
from .services.wiki_service import WikiGenerationService
from .tools.context_tool import ContextTool
from .tools.embedding_tool import EmbeddingTool
from .tools.llm_tool import LLMTool
from .tools.repository_tool import RepositoryTool


# =============================================================================
# REPOSITORY FACTORIES
# =============================================================================


def get_code_document_repo() -> CodeDocumentRepository:
    """Get CodeDocumentRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return CodeDocumentRepository(CodeDocument)


def get_repository_repo() -> RepositoryRepository:
    """Get RepositoryRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return RepositoryRepository(Repository)


def get_wiki_structure_repo() -> WikiStructureRepository:
    """Get WikiStructureRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return WikiStructureRepository(WikiStructure)


def get_chat_session_repo() -> ChatSessionRepository:
    """Get ChatSessionRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return ChatSessionRepository(ChatSession)


def get_question_repo() -> QuestionRepository:
    """Get QuestionRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return QuestionRepository(Question)


def get_answer_repo() -> AnswerRepository:
    """Get AnswerRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return AnswerRepository(Answer)


def get_user_repo() -> UserRepository:
    """Get UserRepository instance.
    
    FastAPI will cache this per request automatically.
    """
    return UserRepository()


# =============================================================================
# TOOL FACTORIES
# =============================================================================


def get_repository_tool() -> RepositoryTool:
    """Get RepositoryTool instance.
    
    FastAPI will cache this per request automatically.
    """
    return RepositoryTool()


def get_embedding_tool(
    code_document_repo: Annotated[CodeDocumentRepository, Depends(get_code_document_repo)]
) -> EmbeddingTool:
    """Get EmbeddingTool instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return EmbeddingTool(code_document_repo=code_document_repo)


def get_context_tool(
    embedding_tool: Annotated[EmbeddingTool, Depends(get_embedding_tool)],
    code_document_repo: Annotated[CodeDocumentRepository, Depends(get_code_document_repo)]
) -> ContextTool:
    """Get ContextTool instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return ContextTool(
        embedding_tool=embedding_tool,
        code_document_repo=code_document_repo
    )


def get_mcp_client() -> MCPFilesystemClient:
    """Get the global MCPFilesystemClient instance.

    Returns the singleton MCP client that was initialized at application startup.
    The client may or may not be initialized depending on settings.
    """
    return get_mcp_filesystem_client()


def get_document_agent(
    repository_tool: Annotated[RepositoryTool, Depends(get_repository_tool)],
    repository_repo: Annotated[RepositoryRepository, Depends(get_repository_repo)],
) -> DocumentProcessingAgent:
    """Get DocumentProcessingAgent instance with injected dependencies.

    FastAPI will cache this per request automatically.
    """
    return DocumentProcessingAgent(
        repository_tool=repository_tool,
        repository_repo=repository_repo,
    )


def get_document_service(
    code_document_repo: Annotated[CodeDocumentRepository, Depends(get_code_document_repo)],
    document_agent: Annotated[DocumentProcessingAgent, Depends(get_document_agent)],
    context_tool: Annotated[ContextTool, Depends(get_context_tool)],
    embedding_tool: Annotated[EmbeddingTool, Depends(get_embedding_tool)]
) -> DocumentProcessingService:
    """Get DocumentProcessingService instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return DocumentProcessingService(
        code_document_repo=code_document_repo,
        document_agent=document_agent,
        context_tool=context_tool,
        embedding_tool=embedding_tool
    )


def get_llm_tool() -> LLMTool:
    """Get LLMTool instance.

    FastAPI will cache this per request automatically.
    """
    return LLMTool()


# =============================================================================
# AGENT FACTORIES
# =============================================================================


def get_wiki_agent(
    wiki_structure_repo: Annotated[WikiStructureRepository, Depends(get_wiki_structure_repo)],
    repository_repo: Annotated[RepositoryRepository, Depends(get_repository_repo)],
) -> WikiGenerationAgent:
    """Get WikiGenerationAgent instance with injected dependencies.

    FastAPI will cache this per request automatically.
    """
    return WikiGenerationAgent(
        wiki_structure_repo=wiki_structure_repo,
        repository_repo=repository_repo,
    )


# =============================================================================
# SERVICE FACTORIES
# =============================================================================

def get_wiki_service(
    wiki_structure_repo: Annotated[WikiStructureRepository, Depends(get_wiki_structure_repo)],
    code_document_repo: Annotated[CodeDocumentRepository, Depends(get_code_document_repo)],
    wiki_agent: Annotated[WikiGenerationAgent, Depends(get_wiki_agent)],
    context_tool: Annotated[ContextTool, Depends(get_context_tool)],
    llm_tool: Annotated[LLMTool, Depends(get_llm_tool)]
) -> WikiGenerationService:
    """Get WikiGenerationService instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return WikiGenerationService(
        wiki_structure_repo=wiki_structure_repo,
        code_document_repo=code_document_repo,
        wiki_agent=wiki_agent,
        context_tool=context_tool,
        llm_tool=llm_tool
    )


def get_workflow_orchestrator(
    context_tool: Annotated[ContextTool, Depends(get_context_tool)],
    embedding_tool: Annotated[EmbeddingTool, Depends(get_embedding_tool)],
    llm_tool: Annotated[LLMTool, Depends(get_llm_tool)],
    repository_tool: Annotated[RepositoryTool, Depends(get_repository_tool)],
    repository_repo: Annotated[RepositoryRepository, Depends(get_repository_repo)],
    code_document_repo: Annotated[CodeDocumentRepository, Depends(get_code_document_repo)],
    wiki_structure_repo: Annotated[WikiStructureRepository, Depends(get_wiki_structure_repo)],
    document_agent: Annotated[DocumentProcessingAgent, Depends(get_document_agent)],
    wiki_agent: Annotated[WikiGenerationAgent, Depends(get_wiki_agent)],
) -> WorkflowOrchestrator:
    """Get WorkflowOrchestrator instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return WorkflowOrchestrator(
        context_tool=context_tool,
        embedding_tool=embedding_tool,
        llm_tool=llm_tool,
        repository_tool=repository_tool,
        repository_repo=repository_repo,
        code_document_repo=code_document_repo,
        wiki_structure_repo=wiki_structure_repo,
        document_agent=document_agent,
        wiki_agent=wiki_agent,
    )


def get_repository_service(
    repository_repo: Annotated[RepositoryRepository, Depends(get_repository_repo)],
    code_document_repo: Annotated[CodeDocumentRepository, Depends(get_code_document_repo)],
    workflow_orchestrator: Annotated[WorkflowOrchestrator, Depends(get_workflow_orchestrator)]
) -> RepositoryService:
    """Get RepositoryService instance with injected dependencies.

    FastAPI will cache this per request automatically.
    """
    return RepositoryService(
        repository_repo=repository_repo,
        code_document_repo=code_document_repo,
        workflow_orchestrator=workflow_orchestrator,
    )


def get_chat_service(
    chat_session_repo: Annotated[ChatSessionRepository, Depends(get_chat_session_repo)],
    question_repo: Annotated[QuestionRepository, Depends(get_question_repo)],
    answer_repo: Annotated[AnswerRepository, Depends(get_answer_repo)],
    workflow_orchestrator: Annotated[WorkflowOrchestrator, Depends(get_workflow_orchestrator)],
    context_tool: Annotated[ContextTool, Depends(get_context_tool)],
    llm_tool: Annotated[LLMTool, Depends(get_llm_tool)]
) -> ChatService:
    """Get ChatService instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return ChatService(
        chat_session_repo=chat_session_repo,
        question_repo=question_repo,
        answer_repo=answer_repo,
        workflow_orchestrator=workflow_orchestrator,
        context_tool=context_tool,
        llm_tool=llm_tool
    )


def get_auth_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repo)]
) -> AuthenticationService:
    """Get AuthenticationService instance with injected dependencies.
    
    FastAPI will cache this per request automatically.
    """
    return AuthenticationService(user_repository=user_repo)
