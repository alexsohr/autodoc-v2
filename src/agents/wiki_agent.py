"""Wiki generation agent for LangGraph workflows

This module implements the wiki generation agent that creates comprehensive
documentation wikis from analyzed repositories. It orchestrates the wiki_workflow
which uses React agents for structure extraction and page generation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..models.wiki import PageImportance, WikiPageDetail, WikiSection, WikiStructure
from .wiki_workflow import wiki_workflow, WikiWorkflowState
from ..repository.repository_repository import RepositoryRepository
from ..repository.wiki_structure_repository import WikiStructureRepository

logger = structlog.get_logger(__name__)

# Maximum concurrent page generation workers to prevent API rate limiting
MAX_PAGE_GENERATION_CONCURRENCY = 5


class WikiGenerationState(TypedDict):
    """State for wiki generation workflow.

    Tracks the progress and data through the wiki generation pipeline.
    """

    repository_id: str
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]  # Complete wiki with page contents
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
    clone_path: Optional[str]  # Path to cloned repository for file access
    force_regenerate: bool  # Whether to force regeneration even if wiki exists
    

class WikiGenerationAgent:
    """LangGraph agent for wiki generation workflows.

    Orchestrates the complete wiki generation pipeline using the wiki_workflow
    which leverages React agents for structure extraction and page generation.
    """

    def __init__(
        self,
        wiki_structure_repo: WikiStructureRepository,
        repository_repo: RepositoryRepository,
    ):
        """Initialize wiki generation agent with dependency injection.

        Args:
            wiki_structure_repo: WikiStructureRepository instance (injected via DI).
            repository_repo: RepositoryRepository instance (injected via DI).
        """
        self._wiki_structure_repo = wiki_structure_repo
        self._repository_repo = repository_repo

        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the wiki generation workflow.

        Workflow:
            START -> analyze_repository -> generate_wiki -> store_wiki -> END

        The generate_wiki node invokes the wiki_workflow which uses React agents
        for structure extraction and sequential page generation.

        Returns:
            Compiled LangGraph StateGraph
        """
        # Create workflow graph
        workflow = StateGraph(WikiGenerationState)

        # Add nodes
        workflow.add_node("analyze_repository", self._analyze_repository_node)
        workflow.add_node("generate_wiki", self._generate_wiki_node)
        workflow.add_node("store_wiki", self._store_wiki_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define simple linear workflow
        workflow.add_edge(START, "analyze_repository")
        workflow.add_edge("analyze_repository", "generate_wiki")
        workflow.add_edge("generate_wiki", "store_wiki")
        workflow.add_edge("store_wiki", END)

        # Error handling
        workflow.add_edge("handle_error", END)

        app = workflow.compile().with_config(
            {"run_name": "wiki_agent.wiki_generation_workflow"}
        )
        logger.debug("Wiki generation workflow created")
        return app

    async def generate_wiki(
        self,
        repository_id: str,
        file_tree: str = "",
        readme_content: str = "",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Generate complete wiki for repository.

        Args:
            repository_id: Repository identifier
            file_tree: ASCII file tree structure from document processing
            readme_content: Formatted documentation files content
            force_regenerate: Force regeneration even if wiki exists

        Returns:
            Dictionary with wiki generation results
        """
        try:
            # Initialize state - wiki existence check is now a workflow node
            initial_state: WikiGenerationState = {
                "repository_id": repository_id,
                "file_tree": file_tree,
                "readme_content": readme_content,
                "wiki_structure": None,
                "current_step": "starting",
                "error_message": None,
                "progress": 0.0,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "messages": [
                    HumanMessage(
                        content=f"Generate wiki for repository: {repository_id}"
                    )
                ],
                "clone_path": None,  # Will be set in _generate_wiki_node
                "force_regenerate": force_regenerate            }

            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)

            # Defensive check - workflow should never return None
            if result is None:
                logger.error(
                    "Workflow returned None unexpectedly",
                    repository_id=repository_id,
                )
                return {
                    "status": "failed",
                    "repository_id": repository_id,
                    "error_message": "Workflow returned None - check LangGraph logs",
                }

            # Count pages with content
            wiki_structure = result.get("wiki_structure", {})
            pages = wiki_structure.get("pages", []) if wiki_structure else []
            pages_with_content = len([p for p in pages if p.get("content")])

            return {
                "status": "completed" if not result.get("error_message") else "failed",
                "repository_id": repository_id,
                "wiki_structure": wiki_structure,
                "pages_generated": pages_with_content,
                "error_message": result.get("error_message"),
            }

        except Exception as e:
            logger.error(f"Wiki generation failed for repository {repository_id}: {e}")
            return {
                "status": "failed",
                "repository_id": repository_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    async def _analyze_repository_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Analyze repository for wiki generation - validates pre-populated values."""
        logger.info(
            "Node: analyze_repository - START",
            repository_id=state.get("repository_id"),
        )
        try:
            state["current_step"] = "analyzing_repository"
            state["progress"] = 10.0

            # Validate we have the required data
            if not state["file_tree"]:
                state["error_message"] = "No file tree provided"
                return state

            if not state["readme_content"]:
                logger.warning("No readme content provided, wiki may be less detailed")

            # Verify repository exists
            repository = await self._repository_repo.find_one(
                {"_id": UUID(state["repository_id"])}
            )
            if not repository:
                state["error_message"] = "Repository not found"
                return state

            state["progress"] = 20.0

            # Count files in tree for logging
            file_count = state["file_tree"].count("├") + state["file_tree"].count("└")
            state["messages"].append(
                AIMessage(content=f"Analyzed repository with ~{file_count} files")
            )

            logger.info(
                "Node: analyze_repository - END",
                repository_id=state.get("repository_id"),
            )
            return state

        except Exception as e:
            logger.error("Node: analyze_repository - FAILED", error=str(e))
            state["error_message"] = f"Repository analysis failed: {str(e)}"
            return state

    async def _generate_wiki_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Generate wiki using the new workflow.

        Invokes the LangGraph wiki_workflow which handles:
        - Structure extraction
        - Sequential page generation
        - Finalization and storage
        """
        from pathlib import Path

        logger.info(
            "Node: generate_wiki - START", repository_id=state.get("repository_id")
        )
        try:
            state["current_step"] = "generating_wiki"
            state["progress"] = 30.0

            # Fetch repository from database to get clone_path
            repository = await self._repository_repo.find_one(
                {"_id": UUID(state["repository_id"])}
            )
            if not repository:
                state["error_message"] = "Repository not found"
                return state

            # Validate clone_path exists
            if not repository.clone_path:
                state["error_message"] = "Repository not cloned - no local path available"
                return state

            clone_path = Path(repository.clone_path)
            if not clone_path.exists():
                state["error_message"] = f"Clone path does not exist: {clone_path}"
                return state

            # Set clone_path in state
            state["clone_path"] = str(clone_path)

            # Prepare state for new workflow (WikiWorkflowState is a TypedDict)
            workflow_state: WikiWorkflowState = {
                "repository_id": state["repository_id"],
                "clone_path": str(clone_path),
                "file_tree": state["file_tree"],
                "readme_content": state["readme_content"],
                "structure": None,
                "pages": [],
                "error": None,
                "current_step": "init",
                "force_regenerate": state.get("force_regenerate", False),
            }

            logger.info(
                "Running wiki_workflow",
                repository_id=state["repository_id"],
                clone_path=str(clone_path),
                max_concurrency=MAX_PAGE_GENERATION_CONCURRENCY,
            )

            # Invoke workflow with concurrency limit to prevent API rate limiting
            config = {"configurable": {"max_concurrency": MAX_PAGE_GENERATION_CONCURRENCY}}
            result = await wiki_workflow.ainvoke(workflow_state, config=config)

            if result.get("error"):
                state["error_message"] = result["error"]
                return state

            # Extract structure and pages from workflow result
            structure = result.get("structure")
            pages = result.get("pages", [])

            if not structure:
                state["error_message"] = "Wiki workflow failed to generate structure"
                return state

            # Convert WikiStructure to dict format expected by _store_wiki_node
            # The workflow already stored to DB, but we populate wiki_structure
            # for compatibility with the existing flow
            pages_data = []
            for section in structure.sections:
                for page in section.pages:
                    # Find page content from generated pages list
                    page_content = ""
                    for gen_page in pages:
                        if gen_page.id == page.id:
                            page_content = gen_page.content or ""
                            break

                    pages_data.append({
                        "id": page.id,
                        "slug": page.id,
                        "title": page.title,
                        "description": page.description,
                        "importance": page.importance.value if hasattr(page.importance, 'value') else str(page.importance),
                        "file_paths": page.file_paths,
                        "related_pages": page.related_pages if hasattr(page, 'related_pages') else [],
                        "section": section.title,
                        "content": page_content,
                    })

            wiki_structure_dict = {
                "title": structure.title,
                "description": structure.description,
                "pages": pages_data,
            }

            # Store result in state for store_wiki node
            state["wiki_structure"] = wiki_structure_dict
            state["progress"] = 90.0

            # Count pages with content
            pages_with_content = [p for p in pages_data if p.get("content")]

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=(
                        f"Generated wiki with {len(pages_data)} pages "
                        f"({len(pages_with_content)} with content) using wiki_workflow"
                    )
                )
            )

            logger.info(
                "Node: generate_wiki - END",
                repository_id=state.get("repository_id"),
                page_count=len(pages_data),
                pages_with_content=len(pages_with_content),
            )
            return state

        except Exception as e:
            logger.error(
                "Node: generate_wiki - FAILED",
                error=str(e),
                error_type=type(e).__name__,
            )
            state["error_message"] = f"Wiki generation failed: {str(e)}"
            return state

    async def _store_wiki_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Store generated wiki in database."""
        try:
            state["current_step"] = "storing_wiki"
            state["progress"] = 95.0

            if not state["wiki_structure"]:
                state["error_message"] = "No wiki structure to store"
                return state

            wiki_data = state["wiki_structure"]
            pages_to_store = wiki_data.get("pages", [])

            if not pages_to_store:
                state["error_message"] = "No wiki pages to store"
                return state

            # Build a map of page_id/slug -> WikiPageDetail
            pages_map = {}
            for page_data in pages_to_store:
                page_id = page_data.get("id") or page_data.get("slug")
                if not page_id:
                    logger.warning(
                        "Page missing id/slug, skipping",
                        page_title=page_data.get("title"),
                    )
                    continue

                # Handle importance - may be string or PageImportance enum
                importance_value = page_data.get("importance", "medium")
                if isinstance(importance_value, str):
                    try:
                        importance = PageImportance(importance_value)
                    except ValueError:
                        importance = PageImportance.MEDIUM
                else:
                    importance = importance_value

                page = WikiPageDetail(
                    id=page_id,
                    title=page_data["title"],
                    description=page_data.get("description", ""),
                    importance=importance,
                    file_paths=page_data.get("file_paths", []),
                    related_pages=page_data.get("related_pages", []),
                    content=page_data.get("content", ""),
                )
                pages_map[page.id] = page

            # Build sections from pages
            from collections import defaultdict

            section_pages_map = defaultdict(list)
            section_order = []

            for page_data in pages_to_store:
                section_name = page_data.get("section", "General")
                page_id = page_data.get("id") or page_data.get("slug")

                if page_id and page_id in pages_map:
                    if section_name not in section_pages_map:
                        section_order.append(section_name)
                    section_pages_map[section_name].append(pages_map[page_id])

            # Create WikiSection objects
            wiki_sections = []
            for section_name in section_order:
                section_id = section_name.lower().replace(" ", "-")
                section = WikiSection(
                    id=section_id,
                    title=section_name,
                    pages=section_pages_map[section_name],
                )
                wiki_sections.append(section)

            logger.info(
                "Built sections from pages",
                num_sections=len(wiki_sections),
                total_pages=sum(len(s.pages) for s in wiki_sections),
            )

            # Create complete wiki structure (id auto-generates as UUID)
            wiki_structure = WikiStructure(
                repository_id=UUID(state["repository_id"]),
                title=wiki_data["title"],
                description=wiki_data["description"],
                sections=wiki_sections,
            )

            # NOTE: wiki_workflow.finalize_node already saved to DB via upsert()
            # We skip redundant save here to avoid duplicate documents
            # This node now only builds the structure for state compatibility

            state["progress"] = 100.0

            # Add success message
            total_pages = wiki_structure.get_total_pages()
            state["messages"].append(
                AIMessage(
                    content=f"Successfully stored wiki with {total_pages} pages in {len(wiki_sections)} sections"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Wiki storage failed: {str(e)}"
            return state

    async def _handle_error_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Handle error node."""
        try:
            logger.error(
                f"Wiki generation failed for repository {state['repository_id']}: "
                f"{state.get('error_message')}"
            )

            state["messages"].append(
                AIMessage(
                    content=f"Wiki generation failed: {state.get('error_message', 'Unknown error')}"
                )
            )

            state["current_step"] = "error_handling"

            return state

        except Exception as e:
            logger.error(f"Error handling node failed: {e}")
            return state

    async def get_wiki_generation_status(self, repository_id: str) -> Dict[str, Any]:
        """Get wiki generation status for repository.

        Args:
            repository_id: Repository ID

        Returns:
            Dictionary with wiki generation status
        """
        try:
            wiki = await self._wiki_structure_repo.find_one(
                {"repository_id": UUID(repository_id)}
            )

            if wiki:
                # Count pages with content
                pages_with_content = sum(
                    1
                    for page in wiki.pages
                    if page.content and page.content.strip()
                )

                return {
                    "status": "completed",
                    "wiki_exists": True,
                    "total_pages": len(wiki.pages),
                    "pages_with_content": pages_with_content,
                    "wiki_title": wiki.title,
                    "last_updated": wiki.updated_at,
                }
            else:
                return {
                    "status": "not_generated",
                    "wiki_exists": False,
                    "message": "Wiki has not been generated for this repository",
                }

        except Exception as e:
            logger.error(f"Failed to get wiki status: {e}")
            return {"status": "error", "error": str(e), "repository_id": repository_id}
