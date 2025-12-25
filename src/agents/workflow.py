"""LangGraph workflow orchestration

This module implements the main workflow orchestration that coordinates
document processing, wiki generation, and other AI-powered workflows.
"""

import asyncio
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..models.repository import AnalysisStatus, Repository
from ..repository.code_document_repository import CodeDocumentRepository
from ..repository.repository_repository import RepositoryRepository
from ..repository.wiki_structure_repository import WikiStructureRepository
from ..tools.context_tool import ContextTool
from ..tools.embedding_tool import EmbeddingTool
from ..tools.llm_tool import LLMTool
from ..tools.repository_tool import RepositoryTool

# Import TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.document_agent import DocumentProcessingAgent
    from ..agents.wiki_agent import WikiGenerationAgent

logger = logging.getLogger(__name__)


class WorkflowType(str, Enum):
    """Types of workflows supported"""

    FULL_ANALYSIS = "full_analysis"
    DOCUMENT_PROCESSING = "document_processing"
    WIKI_GENERATION = "wiki_generation"
    INCREMENTAL_UPDATE = "incremental_update"
    CHAT_RESPONSE = "chat_response"


class WorkflowState(TypedDict):
    """Main workflow state"""

    workflow_type: str
    repository_id: str
    repository_url: Optional[str]
    branch: Optional[str]
    force_update: bool
    current_stage: str
    stages_completed: List[str]
    error_message: Optional[str]
    progress: float
    start_time: str
    results: Dict[str, Any]
    messages: List[BaseMessage]


class WorkflowOrchestrator:
    """Main workflow orchestrator for AutoDoc v2

    Coordinates all AI-powered workflows including repository analysis,
    document processing, wiki generation, and chat responses.
    """

    def __init__(
        self,
        context_tool: ContextTool,
        embedding_tool: EmbeddingTool,
        llm_tool: LLMTool,
        repository_tool: RepositoryTool,
        repository_repo: RepositoryRepository,
        code_document_repo: CodeDocumentRepository,
        wiki_structure_repo: WikiStructureRepository,
        document_agent: "DocumentProcessingAgent",
        wiki_agent: "WikiGenerationAgent",
    ):
        """Initialize workflow orchestrator with dependency injection.
        
        Args:
            context_tool: ContextTool instance (injected via DI).
            embedding_tool: EmbeddingTool instance (injected via DI).
            llm_tool: LLMTool instance (injected via DI).
            repository_tool: RepositoryTool instance (injected via DI).
            repository_repo: RepositoryRepository instance (injected via DI).
            code_document_repo: CodeDocumentRepository instance (injected via DI).
            wiki_structure_repo: WikiStructureRepository instance (injected via DI).
            document_agent: DocumentProcessingAgent instance (injected via DI).
            wiki_agent: WikiGenerationAgent instance (injected via DI).
        """
        self.memory = MemorySaver()
        
        # Tool dependencies
        self._context_tool = context_tool
        self._embedding_tool = embedding_tool
        self._llm_tool = llm_tool
        self._repository_tool = repository_tool
        
        # Repository dependencies
        self._repository_repo = repository_repo
        self._code_document_repo = code_document_repo
        self._wiki_structure_repo = wiki_structure_repo
        
        # Agent dependencies
        self._document_agent = document_agent
        self._wiki_agent = wiki_agent
        
        self.workflows = self._create_workflows()

        # Workflow stage definitions
        self.workflow_stages = {
            WorkflowType.FULL_ANALYSIS: [
                "validate_repository",
                "process_documents",
                "generate_wiki",
                "finalize",
            ],
            WorkflowType.DOCUMENT_PROCESSING: [
                "validate_repository",
                "process_documents",
                "finalize",
            ],
            WorkflowType.WIKI_GENERATION: [
                "validate_repository",
                "generate_wiki",
                "finalize",
            ],
            WorkflowType.INCREMENTAL_UPDATE: [
                "detect_changes",
                "process_changed_documents",
                "update_wiki",
                "finalize",
            ],
            WorkflowType.CHAT_RESPONSE: [
                "retrieve_context",
                "generate_response",
                "finalize",
            ],
        }

    def _create_workflows(self) -> Dict[str, StateGraph]:
        """Create workflow graphs for different workflow types

        Returns:
            Dictionary of compiled workflow graphs
        """
        workflows = {}

        # Full analysis workflow
        workflows[WorkflowType.FULL_ANALYSIS] = self._create_full_analysis_workflow()

        # Document processing workflow
        workflows[WorkflowType.DOCUMENT_PROCESSING] = (
            self._create_document_processing_workflow()
        )

        # Wiki generation workflow
        workflows[WorkflowType.WIKI_GENERATION] = (
            self._create_wiki_generation_workflow()
        )

        # Incremental update workflow
        workflows[WorkflowType.INCREMENTAL_UPDATE] = (
            self._create_incremental_update_workflow()
        )

        # Chat response workflow
        workflows[WorkflowType.CHAT_RESPONSE] = self._create_chat_response_workflow()

        return workflows

    def _create_full_analysis_workflow(self) -> StateGraph:
        """Create full repository analysis workflow"""
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("validate_repository", self._validate_repository_node)
        workflow.add_node("process_documents", self._process_documents_node)
        workflow.add_node("generate_wiki", self._generate_wiki_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define edges
        workflow.add_edge(START, "validate_repository")
        workflow.add_edge("validate_repository", "process_documents")
        workflow.add_edge("process_documents", "generate_wiki")
        workflow.add_edge("generate_wiki", "finalize")
        workflow.add_edge("finalize", END)
        workflow.add_edge("handle_error", END)

        app = workflow.compile(checkpointer=self.memory).with_config({"run_name": "workflow.full_analysis_workflow"})
        logger.debug(f"Full analysis workflow:\n {app.get_graph().draw_mermaid()}")
        return app

    def _create_document_processing_workflow(self) -> StateGraph:
        """Create document processing only workflow"""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("validate_repository", self._validate_repository_node)
        workflow.add_node("process_documents", self._process_documents_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.add_edge(START, "validate_repository")
        workflow.add_edge("validate_repository", "process_documents")
        workflow.add_edge("process_documents", "finalize")
        workflow.add_edge("finalize", END)
        workflow.add_edge("handle_error", END)

        app = workflow.compile(checkpointer=self.memory).with_config({"run_name": "workflow.document_processing_workflow"})
        logger.debug(
            f"Document processing workflow:\n {app.get_graph().draw_mermaid()}"
        )
        return app

    def _create_wiki_generation_workflow(self) -> StateGraph:
        """Create wiki generation only workflow"""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("validate_repository", self._validate_repository_node)
        workflow.add_node("generate_wiki", self._generate_wiki_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.add_edge(START, "validate_repository")
        workflow.add_edge("validate_repository", "generate_wiki")
        workflow.add_edge("generate_wiki", "finalize")
        workflow.add_edge("finalize", END)
        workflow.add_edge("handle_error", END)

        app = workflow.compile(checkpointer=self.memory).with_config({"run_name": "workflow.wiki_generation_workflow"})
        logger.debug(f"Wiki generation workflow:\n {app.get_graph().draw_mermaid()}")
        return app

    def _create_incremental_update_workflow(self) -> StateGraph:
        """Create incremental update workflow"""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("detect_changes", self._detect_changes_node)
        workflow.add_node(
            "process_changed_documents", self._process_changed_documents_node
        )
        workflow.add_node("update_wiki", self._update_wiki_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.add_edge(START, "detect_changes")
        workflow.add_edge("detect_changes", "process_changed_documents")
        workflow.add_edge("process_changed_documents", "update_wiki")
        workflow.add_edge("update_wiki", "finalize")
        workflow.add_edge("finalize", END)
        workflow.add_edge("handle_error", END)

        app = workflow.compile(checkpointer=self.memory).with_config({"run_name": "workflow.incremental_update_workflow"})
        logger.debug(f"Incremental update workflow:\n {app.get_graph().draw_mermaid()}")
        return app

    def _create_chat_response_workflow(self) -> StateGraph:
        """Create chat response workflow"""
        workflow = StateGraph(WorkflowState)

        workflow.add_node("retrieve_context", self._retrieve_context_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("finalize", self._finalize_node)
        workflow.add_node("handle_error", self._handle_error_node)

        workflow.add_edge(START, "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", "finalize")
        workflow.add_edge("finalize", END)
        workflow.add_edge("handle_error", END)

        app = workflow.compile(checkpointer=self.memory).with_config({"run_name": "workflow.chat_response_workflow"})
        logger.debug(f"Chat response workflow:\n {app.get_graph().draw_mermaid()}")
        return app

    async def execute_workflow(
        self,
        workflow_type: WorkflowType,
        repository_id: str,
        repository_url: Optional[str] = None,
        branch: Optional[str] = None,
        force_update: bool = False,
        additional_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a specific workflow

        Args:
            workflow_type: Type of workflow to execute
            repository_id: Repository identifier
            repository_url: Repository URL (required for new repositories)
            branch: Specific branch to process
            force_update: Force update even if already processed
            additional_params: Additional workflow parameters

        Returns:
            Dictionary with workflow execution results
        """
        try:
            # Get workflow
            workflow = self.workflows.get(workflow_type)
            if not workflow:
                raise ValueError(f"Unknown workflow type: {workflow_type}")

            # Initialize state
            initial_state: WorkflowState = {
                "workflow_type": workflow_type.value,
                "repository_id": repository_id,
                "repository_url": repository_url,
                "branch": branch,
                "force_update": force_update,
                "current_stage": "starting",
                "stages_completed": [],
                "error_message": None,
                "progress": 0.0,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "results": additional_params or {},
                "messages": [
                    HumanMessage(
                        content=f"Execute {workflow_type.value} workflow for repository {repository_id}"
                    )
                ],
            }

            # Execute workflow with checkpointing
            config = {
                "configurable": {"thread_id": f"{workflow_type.value}_{repository_id}"}
            }
            result = await workflow.ainvoke(initial_state, config=config)

            return {
                "status": "completed" if not result.get("error_message") else "failed",
                "workflow_type": workflow_type.value,
                "repository_id": repository_id,
                "stages_completed": result.get("stages_completed", []),
                "progress": result.get("progress", 0),
                "results": result.get("results", {}),
                "error_message": result.get("error_message"),
                "execution_time": self._calculate_execution_time(
                    result.get("start_time", "")
                ),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "status": "failed",
                "workflow_type": workflow_type.value,
                "repository_id": repository_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    # Workflow nodes

    async def _validate_repository_node(self, state: WorkflowState) -> WorkflowState:
        """Validate repository node"""
        try:
            state["current_stage"] = "validate_repository"
            state["progress"] = 5.0

            # Get repository from database
            repo_repository = self._repository_repo
            repository = await repo_repository.get(UUID(state["repository_id"]))

            if not repository:
                state["error_message"] = "Repository not found"
                return state

            # Check if repository is accessible
            if not repository.url:
                state["error_message"] = "Repository URL not available"
                return state

            state["repository_url"] = repository.url
            state["stages_completed"].append("validate_repository")
            state["progress"] = 10.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Repository validated: {repository.org}/{repository.name}"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Repository validation failed: {str(e)}"
            return state

    async def _process_documents_node(self, state: WorkflowState) -> WorkflowState:
        """Process documents node"""
        try:
            state["current_stage"] = "process_documents"
            state["progress"] = 20.0

            # Check if documents already processed (unless force update)
            if not state["force_update"]:
                doc_count = await self._code_document_repo.count(
                    {"repository_id": state["repository_id"]}
                )

                if doc_count > 0:
                    state["stages_completed"].append("process_documents")
                    state["progress"] = 60.0
                    state["results"]["documents_processed"] = doc_count

                    state["messages"].append(
                        AIMessage(
                            content=f"Using existing {doc_count} processed documents"
                        )
                    )

                    return state

            # Process documents using document agent
            processing_result = await self._document_agent.process_repository(
                repository_id=state["repository_id"],
                repository_url=state["repository_url"],
                branch=state["branch"],
            )

            if processing_result["status"] != "completed":
                state["error_message"] = (
                    f"Document processing failed: {processing_result.get('error_message', 'Unknown error')}"
                )
                return state

            state["stages_completed"].append("process_documents")
            state["progress"] = 60.0
            state["results"]["document_processing"] = processing_result

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Processed {processing_result['processed_files']} documents with {processing_result['embeddings_generated']} embeddings"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Document processing node failed: {str(e)}"
            return state

    async def _generate_wiki_node(self, state: WorkflowState) -> WorkflowState:
        """Generate wiki node"""
        try:
            state["current_stage"] = "generate_wiki"
            state["progress"] = 70.0

            # Check if wiki already exists (unless force update)
            if not state["force_update"]:
                existing_wiki = await self._wiki_structure_repo.find_one(
                    {"repository_id": state["repository_id"]}
                )

                if existing_wiki:
                    state["stages_completed"].append("generate_wiki")
                    state["progress"] = 90.0
                    state["results"]["wiki_generation"] = {
                        "status": "exists",
                        "wiki_id": str(existing_wiki.id),
                    }

                    state["messages"].append(
                        AIMessage(content="Using existing wiki structure")
                    )

                    return state

            # Generate wiki using wiki agent
            wiki_result = await self._wiki_agent.generate_wiki(
                repository_id=state["repository_id"],
                force_regenerate=state["force_update"],
            )

            if wiki_result["status"] not in ["completed", "exists"]:
                state["error_message"] = (
                    f"Wiki generation failed: {wiki_result.get('error_message', 'Unknown error')}"
                )
                return state

            state["stages_completed"].append("generate_wiki")
            state["progress"] = 90.0
            state["results"]["wiki_generation"] = wiki_result

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Generated wiki with {wiki_result.get('pages_generated', 0)} pages"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Wiki generation node failed: {str(e)}"
            return state

    async def _detect_changes_node(self, state: WorkflowState) -> WorkflowState:
        """Detect changes node for incremental updates"""
        try:
            state["current_stage"] = "detect_changes"
            state["progress"] = 10.0

            # This would implement change detection logic
            # For now, assume changes detected
            state["results"]["changes_detected"] = True
            state["results"]["changed_files"] = []

            state["stages_completed"].append("detect_changes")
            state["progress"] = 30.0

            return state

        except Exception as e:
            state["error_message"] = f"Change detection failed: {str(e)}"
            return state

    async def _process_changed_documents_node(
        self, state: WorkflowState
    ) -> WorkflowState:
        """Process changed documents node"""
        try:
            state["current_stage"] = "process_changed_documents"
            state["progress"] = 40.0

            # Process only changed documents
            # Implementation would be similar to full document processing
            # but only for changed files

            state["stages_completed"].append("process_changed_documents")
            state["progress"] = 70.0

            return state

        except Exception as e:
            state["error_message"] = f"Changed document processing failed: {str(e)}"
            return state

    async def _update_wiki_node(self, state: WorkflowState) -> WorkflowState:
        """Update wiki node for incremental updates"""
        try:
            state["current_stage"] = "update_wiki"
            state["progress"] = 80.0

            # Update wiki based on changed documents
            # Implementation would update existing wiki structure

            state["stages_completed"].append("update_wiki")
            state["progress"] = 90.0

            return state

        except Exception as e:
            state["error_message"] = f"Wiki update failed: {str(e)}"
            return state

    async def _retrieve_context_node(self, state: WorkflowState) -> WorkflowState:
        """Retrieve context node for chat responses"""
        try:
            state["current_stage"] = "retrieve_context"
            state["progress"] = 20.0

            question = state["results"].get("question", "")
            if not question:
                state["error_message"] = "No question provided for context retrieval"
                return state

            # Retrieve relevant context
            context_result = await self._context_tool._arun(
                "hybrid_search",
                query=question,
                repository_id=state["repository_id"],
                k=10,
            )

            if context_result["status"] != "success":
                state["error_message"] = (
                    f"Context retrieval failed: {context_result.get('error', 'Unknown error')}"
                )
                return state

            state["results"]["context"] = context_result["contexts"]
            state["stages_completed"].append("retrieve_context")
            state["progress"] = 50.0

            return state

        except Exception as e:
            state["error_message"] = f"Context retrieval failed: {str(e)}"
            return state

    async def _generate_response_node(self, state: WorkflowState) -> WorkflowState:
        """Generate response node for chat"""
        try:
            state["current_stage"] = "generate_response"
            state["progress"] = 70.0

            question = state["results"].get("question", "")
            context_documents = state["results"].get("context", [])

            # Generate answer using LLM tool
            answer_result = await self._llm_tool._arun(
                "answer_question",
                question=question,
                context_documents=context_documents,
            )

            if answer_result["status"] != "success":
                state["error_message"] = (
                    f"Response generation failed: {answer_result.get('error', 'Unknown error')}"
                )
                return state

            state["results"]["answer"] = answer_result
            state["stages_completed"].append("generate_response")
            state["progress"] = 90.0

            return state

        except Exception as e:
            state["error_message"] = f"Response generation failed: {str(e)}"
            return state

    async def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow execution"""
        try:
            state["current_stage"] = "finalize"
            state["progress"] = 100.0

            # Calculate execution time
            start_time = datetime.fromisoformat(state["start_time"])
            end_time = datetime.now(timezone.utc)
            execution_time = (end_time - start_time).total_seconds()

            state["results"]["execution_time"] = execution_time
            state["results"]["completed_at"] = end_time.isoformat()

            # Add completion message
            state["messages"].append(
                AIMessage(
                    content=f"Workflow {state['workflow_type']} completed successfully in {execution_time:.2f} seconds"
                )
            )

            return state

        except Exception as e:
            logger.error(f"Workflow finalization failed: {e}")
            state["error_message"] = f"Finalization failed: {str(e)}"
            return state

    async def _handle_error_node(self, state: WorkflowState) -> WorkflowState:
        """Handle error node"""
        try:
            # Log error details
            logger.error(
                f"Workflow {state['workflow_type']} failed for repository {state['repository_id']}: {state.get('error_message')}"
            )

            # Add error message
            state["messages"].append(
                AIMessage(
                    content=f"Workflow failed at stage {state['current_stage']}: {state.get('error_message', 'Unknown error')}"
                )
            )

            # Update repository status if needed
            if state["workflow_type"] in [
                WorkflowType.FULL_ANALYSIS.value,
                WorkflowType.DOCUMENT_PROCESSING.value,
            ]:
                await self._update_repository_status(
                    state["repository_id"],
                    AnalysisStatus.FAILED,
                    state.get("error_message"),
                )

            return state

        except Exception as e:
            logger.error(f"Error handling failed: {e}")
            return state

    # Helper methods

    def _calculate_execution_time(self, start_time_str: str) -> float:
        """Calculate execution time from start time string

        Args:
            start_time_str: Start time ISO string

        Returns:
            Execution time in seconds
        """
        try:
            start_time = datetime.fromisoformat(start_time_str)
            end_time = datetime.now(timezone.utc)
            return (end_time - start_time).total_seconds()
        except Exception:
            return 0.0

    async def _update_repository_status(
        self,
        repository_id: str,
        status: AnalysisStatus,
        error_message: Optional[str] = None,
    ) -> None:
        """Update repository analysis status

        Args:
            repository_id: Repository ID
            status: New analysis status
            error_message: Optional error message
        """
        try:

            updates = {
                "analysis_status": status.value,
                "updated_at": datetime.now(timezone.utc),
            }

            if status == AnalysisStatus.COMPLETED:
                updates["last_analyzed"] = datetime.now(timezone.utc)

            if error_message:
                updates["error_message"] = error_message

            await self._repository_repo.update(UUID(repository_id), updates)

        except Exception as e:
            logger.error(f"Failed to update repository status: {e}")

    async def get_workflow_status(
        self, workflow_type: WorkflowType, repository_id: str
    ) -> Dict[str, Any]:
        """Get current workflow status

        Args:
            workflow_type: Workflow type
            repository_id: Repository ID

        Returns:
            Dictionary with workflow status
        """
        try:
            # Check workflow state from memory
            config = {
                "configurable": {"thread_id": f"{workflow_type.value}_{repository_id}"}
            }

            # This would retrieve state from checkpointer if workflow is running
            # For now, return basic status

            repository = await self._repository_repo.get(UUID(repository_id))

            if not repository:
                return {
                    "status": "repository_not_found",
                    "error": "Repository not found",
                }

            # Determine status based on workflow type
            if workflow_type == WorkflowType.FULL_ANALYSIS:

                doc_count = await self._code_document_repo.count(
                    {"repository_id": repository_id}
                )
                wiki_exists = await self._wiki_structure_repo.find_one(
                    {"repository_id": repository_id}
                )

                if doc_count > 0 and wiki_exists:
                    status = "completed"
                elif doc_count > 0:
                    status = "documents_processed"
                else:
                    status = "not_started"

            elif workflow_type == WorkflowType.DOCUMENT_PROCESSING:
                doc_count = await self._code_document_repo.count(
                    {"repository_id": repository_id}
                )
                status = "completed" if doc_count > 0 else "not_started"

            elif workflow_type == WorkflowType.WIKI_GENERATION:
                wiki_exists = await self._wiki_structure_repo.find_one(
                    {"repository_id": repository_id}
                )
                status = "completed" if wiki_exists else "not_started"

            else:
                status = "unknown"

            return {
                "status": status,
                "workflow_type": workflow_type.value,
                "repository_id": repository_id,
                "analysis_status": repository.analysis_status.value if repository.analysis_status else "unknown",
                "last_analyzed": repository.last_analyzed,
            }

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_type": workflow_type.value,
                "repository_id": repository_id,
            }

    async def list_available_workflows(self) -> List[Dict[str, Any]]:
        """List all available workflow types

        Returns:
            List of available workflows with descriptions
        """
        workflows = []

        for workflow_type in WorkflowType:
            stages = self.workflow_stages.get(workflow_type, [])

            workflows.append(
                {
                    "type": workflow_type.value,
                    "description": self._get_workflow_description(workflow_type),
                    "stages": stages,
                    "stage_count": len(stages),
                }
            )

        return workflows

    def _get_workflow_description(self, workflow_type: WorkflowType) -> str:
        """Get description for workflow type

        Args:
            workflow_type: Workflow type

        Returns:
            Workflow description
        """
        descriptions = {
            WorkflowType.FULL_ANALYSIS: "Complete repository analysis including document processing and wiki generation",
            WorkflowType.DOCUMENT_PROCESSING: "Process repository documents and generate embeddings for semantic search",
            WorkflowType.WIKI_GENERATION: "Generate comprehensive wiki documentation from processed repository",
            WorkflowType.INCREMENTAL_UPDATE: "Update documentation based on repository changes",
            WorkflowType.CHAT_RESPONSE: "Generate contextual responses to user questions about the repository",
        }

        return descriptions.get(workflow_type, "Unknown workflow")
