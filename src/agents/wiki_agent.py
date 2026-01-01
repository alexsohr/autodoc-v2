"""Wiki generation agent for LangGraph workflows

This module implements the wiki generation agent that creates comprehensive
documentation wikis from analyzed repositories using the prompts from
wiki-generation-prompts.md.
"""

import asyncio
import json
import operator
import os
import re
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from uuid import UUID

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from ..models.repository import Repository
from ..models.wiki import PageImportance, WikiPageDetail, WikiSection, WikiStructure
from ..repository.code_document_repository import CodeDocumentRepository
from ..repository.repository_repository import RepositoryRepository
from ..repository.wiki_structure_repository import WikiStructureRepository
from ..tools.context_tool import ContextTool
from ..tools.llm_tool import LLMTool

logger = structlog.get_logger(__name__)


class WikiGenerationState(TypedDict):
    """State for wiki generation workflow"""

    repository_id: str
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]
    # CRITICAL: Use Annotated with operator.add for parallel worker result aggregation
    generated_pages: Annotated[List[Dict[str, Any]], operator.add]
    current_page: Optional[str]
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
    clone_path: Optional[str]  # Path to cloned repository for file access


class PageWorkerState(TypedDict):
    """State for individual page generation worker.

    CRITICAL: This state MUST include `generated_pages` with the SAME
    Annotated reducer as WikiGenerationState. This is required for
    LangGraph to properly aggregate results from parallel workers.

    The worker receives this state via Send() and writes its result
    to generated_pages, which gets merged into the main state.
    """

    page_info: Dict[str, Any]  # Page definition from wiki_structure
    clone_path: Optional[str]  # For reading files from disk
    # MUST match WikiGenerationState for aggregation!
    generated_pages: Annotated[List[Dict[str, Any]], operator.add]


def fan_out_to_page_workers(state: WikiGenerationState) -> list[Send]:
    """Create parallel page generation tasks using LangGraph Send.

    Each Send creates a separate page_worker execution with its own state.
    The worker state includes `generated_pages` with the same reducer as
    the main state, allowing LangGraph to aggregate results automatically.

    Args:
        state: Current workflow state with wiki_structure and clone_path

    Returns:
        List of Send objects, one per page to generate
    """
    if not state.get("wiki_structure"):
        logger.warning("No wiki_structure found, cannot fan out to page workers")
        return []

    pages = state["wiki_structure"].get("pages", [])
    if not pages:
        logger.warning("No pages in wiki_structure")
        return []

    clone_path = state.get("clone_path")
    if not clone_path:
        logger.error("clone_path not set in state, cannot read files for page generation")
        return []

    sends = []
    for page in pages:
        # Each Send payload becomes the worker's input state
        # CRITICAL: Must include generated_pages for aggregation to work
        sends.append(
            Send("page_worker", {
                "page_info": page,
                "clone_path": clone_path,
                "generated_pages": [],  # Will be populated by worker, then aggregated
            })
        )

    logger.info("Fanning out to parallel page workers", num_pages=len(sends))
    return sends


async def page_worker_node(state: PageWorkerState) -> Dict[str, Any]:
    """Generate content for a single wiki page by reading files from disk.

    This node runs in parallel for each page via LangGraph's Send API.
    It reads files using clone_path + file_paths, then generates content.

    IMPORTANT: Returns {"generated_pages": [page_result]} which gets
    aggregated with other workers via the operator.add reducer.

    Args:
        state: PageWorkerState with page_info, clone_path, and generated_pages

    Returns:
        Dict with 'generated_pages' list containing the page with generated content
    """
    from ..utils.config_loader import get_settings

    page_info = state["page_info"]
    clone_path = state.get("clone_path")
    file_paths = page_info.get("file_paths", [])

    # Validate page_info has required fields
    required_keys = ["title", "section", "description", "slug"]
    missing_keys = [k for k in required_keys if k not in page_info]
    if missing_keys:
        logger.warning(
            "page_info missing required fields",
            page_title=page_info.get("title", "UNKNOWN"),
            missing_keys=missing_keys
        )
        return {"generated_pages": []}

    logger.info("Page worker starting", page_title=page_info["title"], num_files=len(file_paths))

    # Read file contents from disk
    file_contents: Dict[str, str] = {}

    if clone_path and file_paths:
        for file_path in file_paths[:10]:  # Limit to 10 files per page
            full_path = os.path.join(clone_path, file_path)
            try:
                if os.path.exists(full_path) and os.path.isfile(full_path):
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        # Limit file size to prevent token overflow
                        file_contents[file_path] = content[:15000]
                        logger.debug("Read file", file_path=file_path, chars=len(content))
                else:
                    logger.warning("File not found or not a file", full_path=full_path)
            except Exception as e:
                logger.warning("Failed to read file", file_path=file_path, error=str(e))

    if not file_contents:
        logger.warning("No file contents available for page", page_title=page_info["title"])
        # Return page without content - will be skipped or show placeholder
        return {"generated_pages": []}

    # Build prompt with file contents
    files_markdown = "\n\n".join([
        f"### File: {path}\n```\n{content[:8000]}\n```"
        for path, content in file_contents.items()
    ])

    prompt = f"""Generate comprehensive wiki documentation for this page.

## Page Information
- **Title:** {page_info['title']}
- **Section:** {page_info['section']}
- **Description:** {page_info['description']}

## Source Files
{files_markdown}

## Requirements
1. Write clear, professional technical documentation in Markdown
2. Include code examples extracted from the source files
3. Explain the purpose and usage of each component
4. Use proper headings, lists, and code blocks
5. Be comprehensive but concise
6. Do NOT include a title heading (it will be added automatically)

Generate the page content now:
"""

    try:
        settings = get_settings()
        llm = ChatOpenAI(model=settings.openai_model, temperature=0)
        response = await llm.ainvoke([HumanMessage(content=prompt)])

        # Build result page
        page_result = {
            "title": page_info["title"],
            "slug": page_info["slug"],
            "section": page_info["section"],
            "description": page_info["description"],
            "file_paths": page_info.get("file_paths", []),
            "content": response.content,
        }

        logger.info("Generated content for page", page_title=page_info["title"], content_chars=len(response.content))

        # Return in format for aggregation via operator.add
        return {"generated_pages": [page_result]}

    except Exception as e:
        logger.error("Failed to generate page", page_title=page_info["title"], error=str(e))
        return {"generated_pages": []}


async def aggregate_pages_node(state: WikiGenerationState) -> Dict[str, Any]:
    """Aggregate results from all parallel page workers.

    NOTE: LangGraph automatically aggregates generated_pages from all workers
    via the operator.add reducer BEFORE this node runs. This node receives
    the already-merged results.

    This node:
    1. Logs the aggregation results
    2. Updates progress and step status
    3. Prepares state for the store_wiki node

    Args:
        state: WikiGenerationState with aggregated generated_pages

    Returns:
        Updated state dict with progress and step updates
    """
    generated_pages = state.get("generated_pages", [])

    logger.info("Aggregated pages from parallel workers", num_pages=len(generated_pages))

    return {
        "current_step": "pages_generated",
        "progress": 90.0,
    }


class WikiGenerationAgent:
    """LangGraph agent for wiki generation workflows

    Orchestrates the complete wiki generation pipeline from repository
    analysis to structured documentation creation with Mermaid diagrams.
    """

    def __init__(self, context_tool: ContextTool, llm_tool: LLMTool,
                 wiki_structure_repo: WikiStructureRepository, repository_repo: RepositoryRepository, code_document_repo: CodeDocumentRepository):
        """Initialize wiki generation agent with dependency injection.
        
        Args:
            context_tool: ContextTool instance (injected via DI).
            llm_tool: LLMTool instance (injected via DI).
            wiki_structure_repo: WikiStructureRepository instance (injected via DI).
            repository_repo: RepositoryRepository instance (injected via DI).
            code_document_repo: CodeDocumentRepository instance (injected via DI).
        """
        self._context_tool = context_tool
        self._llm_tool = llm_tool
        self._wiki_structure_repo = wiki_structure_repo
        self._repository_repo = repository_repo
        self._code_document_repo = code_document_repo
        
        self.workflow = self._create_workflow()

        # Load prompts from wiki-generation-prompts.md
        self.structure_prompt_template = self._load_structure_prompt()
        self.page_prompt_template = self._load_page_prompt()

    def _create_workflow(self) -> StateGraph:
        """Create the wiki generation workflow graph

        Returns:
            LangGraph StateGraph for wiki generation
        """
        # Create workflow graph
        workflow = StateGraph(WikiGenerationState)

        # Add nodes
        workflow.add_node("analyze_repository", self._analyze_repository_node)
        workflow.add_node("generate_structure", self._generate_structure_node)
        workflow.add_node("generate_pages", self._generate_pages_node)
        workflow.add_node("store_wiki", self._store_wiki_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Define workflow edges
        workflow.add_edge(START, "analyze_repository")

        # Sequential processing flow
        workflow.add_edge("analyze_repository", "generate_structure")
        workflow.add_edge("generate_structure", "generate_pages")
        workflow.add_edge("generate_pages", "store_wiki")
        workflow.add_edge("store_wiki", END)

        # Error handling
        workflow.add_edge("handle_error", END)

        app = workflow.compile().with_config({"run_name": "wiki_agent.wiki_generation_workflow"})
        logger.debug(f"Wiki generation workflow:\n {app.get_graph().draw_mermaid()}")
        return app

    def _load_structure_prompt(self) -> str:
        """Load wiki structure generation prompt"""
        return """Analyze this GitHub repository {owner}/{repo} and create a wiki structure for it.

The complete file tree of the project:

<file_tree> {file_tree} </file_tree>

The README file of the project:

<readme> {readme_content} </readme>

I want to create a wiki for this repository. Determine the most logical structure for a wiki based on the repository's content.

IMPORTANT: The wiki content will be generated in 'English' language.

When designing the wiki structure, include pages that would benefit from visual diagrams, such as:

- Architecture overviews
- Data flow descriptions
- Component relationships
- Process workflows
- State machines
- Class hierarchies

Create a structured wiki with the following main sections:

- Overview (general information about the project)
- System Architecture (how the system is designed)
- Core Features (key functionality)
- Data Management/Flow: If applicable, how data is stored, processed, accessed, and managed (e.g., database schema, data pipelines, state management).
- Frontend Components (UI elements, if applicable.)
- Backend Systems (server-side components)
- Model Integration (AI model connections)
- Deployment/Infrastructure (how to deploy, what's the infrastructure like)
- Extensibility and Customization: If the project architecture supports it, explain how to extend or customize its functionality (e.g., plugins, theming, custom modules, hooks).

Each section should contain relevant pages. For example, the "Frontend Components" section might include pages for "Home Page", "Repository Wiki Page", "Ask Component", etc.

Analyze the repository structure and create a comprehensive wiki organization with:

1. Create 8-12 pages that would make a comprehensive wiki for this repository
2. Each page should focus on a specific aspect of the codebase (e.g., architecture, key features, setup)
3. The file_paths should be actual files from the repository that would be used to generate that page
4. Organize pages into logical sections for easy navigation
5. Ensure proper cross-references between related pages

The response will be automatically structured according to the required schema."""

    def _load_page_prompt(self) -> str:
        """Load wiki page generation prompt"""
        return """You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about a specific feature, system, or module within a given software project.

You will be given:
1. The "{page_title}" for the page you need to create.
2. A list of "RELEVANT_SOURCE_FILES" from the project that you MUST use as the sole basis for the content. You MUST use AT LEAST 5 relevant source files for comprehensive coverage - if fewer are provided, search for additional related files in the codebase.

CRITICAL STARTING INSTRUCTION:
The very first thing on the page MUST be a <details> block listing ALL the RELEVANT_SOURCE_FILES you used to generate the content. There MUST be AT LEAST 5 source files listed - if fewer were provided, you MUST find additional related files to include.
Format it exactly like this:
<details>
<summary>Relevant source files</summary>

Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the <details> block. JUST START with the <details> block.
The following files were used as context for generating this wiki page:

{file_list_markdown}
<!-- Add additional relevant files if fewer than 5 were provided -->
</details>

Immediately after the <details> block, the main title of the page should be a H1 Markdown heading: # {page_title}.

Based ONLY on the content of the RELEVANT_SOURCE_FILES:

1.  **Introduction:** Start with a concise introduction (1-2 paragraphs) explaining the purpose, scope, and high-level overview of "{page_title}" within the context of the overall project. If relevant, and if information is available in the provided files, link to other potential wiki pages using the format `[Link Text](#page-anchor-or-id)`.

2.  **Detailed Sections:** Break down "{page_title}" into logical sections using H2 (`##`) and H3 (`###`) Markdown headings. For each section:
    *   Explain the architecture, components, data flow, or logic relevant to the section's focus, as evidenced in the source files.
    *   Identify key functions, classes, data structures, API endpoints, or configuration elements pertinent to that section.

3.  **Mermaid Diagrams:**
    *   EXTENSIVELY use Mermaid diagrams (e.g., `flowchart TD`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `graph TD`) to visually represent architectures, flows, relationships, and schemas found in the source files.
    *   Ensure diagrams are accurate and directly derived from information in the RELEVANT_SOURCE_FILES.
    *   Provide a brief explanation before or after each diagram to give context.
    *   CRITICAL: All diagrams MUST follow strict vertical orientation:
       - Use "graph TD" (top-down) directive for flow diagrams
       - NEVER use "graph LR" (left-right)
       - Maximum node width should be 3-4 words
       - For sequence diagrams:
         - Start with "sequenceDiagram" directive on its own line
         - Define ALL participants at the beginning
         - Use descriptive but concise participant names
         - Use the correct arrow types:
           - ->> for request/asynchronous messages
           - -->> for response messages
           - -x for failed messages
         - Include activation boxes using +/- notation
         - Add notes for clarification using "Note over" or "Note right of"

4.  **Tables:**
    *   Use Markdown tables to summarize information such as:
        *   Key features or components and their descriptions.
        *   API endpoint parameters, types, and descriptions.
        *   Configuration options, their types, and default values.
        *   Data model fields, types, constraints, and descriptions.

5.  **Code Snippets:**
    *   Include short, relevant code snippets (e.g., Python, Java, JavaScript, SQL, JSON, YAML) directly from the RELEVANT_SOURCE_FILES to illustrate key implementation details, data structures, or configurations.
    *   Ensure snippets are well-formatted within Markdown code blocks with appropriate language identifiers.

6.  **Source Citations (EXTREMELY IMPORTANT):**
    *   For EVERY piece of significant information, explanation, diagram, table entry, or code snippet, you MUST cite the specific source file(s) and relevant line numbers from which the information was derived.
    *   Place citations at the end of the paragraph, under the diagram/table, or after the code snippet.
    *   Use the exact format: `Sources: [filename.ext:start_line-end_line]()` for a range, or `Sources: [filename.ext:line_number]()` for a single line. Multiple files can be cited: `Sources: [file1.ext:1-10](), [file2.ext:5](), [dir/file3.ext]()` (if the whole file is relevant and line numbers are not applicable or too broad).
    *   If an entire section is overwhelmingly based on one or two files, you can cite them under the section heading in addition to more specific citations within the section.
    *   IMPORTANT: You MUST cite AT LEAST 5 different source files throughout the wiki page to ensure comprehensive coverage.

7.  **Technical Accuracy:** All information must be derived SOLELY from the RELEVANT_SOURCE_FILES. Do not infer, invent, or use external knowledge about similar systems or common practices unless it's directly supported by the provided code. If information is not present in the provided files, do not include it or explicitly state its absence if crucial to the topic.

8.  **Clarity and Conciseness:** Use clear, professional, and concise technical language suitable for other developers working on or learning about the project. Avoid unnecessary jargon, but use correct technical terms where appropriate.

9.  **Conclusion/Summary:** End with a brief summary paragraph if appropriate for "{page_title}", reiterating the key aspects covered and their significance within the project.

IMPORTANT: Generate the content in 'English' language.

Remember:
- Ground every claim in the provided source files.
- Prioritize accuracy and direct representation of the code's functionality and structure.
- Structure the document logically for easy understanding by other developers."""

    async def generate_wiki(
        self,
        repository_id: str,
        file_tree: str = "",
        readme_content: str = "",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Generate complete wiki for repository

        Args:
            repository_id: Repository identifier
            file_tree: ASCII file tree structure from document processing
            readme_content: Formatted documentation files content
            force_regenerate: Force regeneration even if wiki exists

        Returns:
            Dictionary with wiki generation results
        """
        try:
            # Check if wiki already exists
            if not force_regenerate:
                existing_wiki = await self._wiki_structure_repo.find_one(
                    {"repository_id": repository_id}
                )
                if existing_wiki:
                    return {
                        "status": "exists",
                        "message": "Wiki already exists for this repository",
                        "wiki_id": str(existing_wiki.id),
                    }

            # Initialize state
            initial_state: WikiGenerationState = {
                "repository_id": repository_id,
                "file_tree": file_tree,
                "readme_content": readme_content,
                "wiki_structure": None,
                "generated_pages": [],
                "current_page": None,
                "current_step": "starting",
                "error_message": None,
                "progress": 0.0,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "messages": [
                    HumanMessage(
                        content=f"Generate wiki for repository: {repository_id}"
                    )
                ],
                "clone_path": None,  # Will be set in _generate_structure_node
            }

            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)

            return {
                "status": "completed" if not result.get("error_message") else "failed",
                "repository_id": repository_id,
                "wiki_structure": result.get("wiki_structure"),
                "pages_generated": len(result.get("generated_pages", [])),
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
        """Analyze repository for wiki generation - uses pre-populated values"""
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

            return state

        except Exception as e:
            state["error_message"] = f"Repository analysis failed: {str(e)}"
            return state

    async def _generate_structure_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Generate wiki structure using Deep Agent for repository exploration."""
        from pathlib import Path

        from src.agents.deep_structure_agent import run_structure_agent
        from src.utils.config_loader import get_settings

        try:
            state["current_step"] = "generating_structure"
            state["progress"] = 30.0

            # Fetch repository from database
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

            # IMPORTANT: Set clone_path in state for page workers to access files
            state["clone_path"] = str(clone_path)

            owner = repository.org or "unknown"
            repo_name = repository.name or "unknown"

            # Get model from settings
            settings = get_settings()
            model = settings.openai_model

            # Run the Deep Agent to explore and generate structure
            logger.info(
                "Running Deep Agent for wiki structure",
                repository_id=state["repository_id"],
                clone_path=str(clone_path),
            )

            wiki_structure = await run_structure_agent(
                clone_path=str(clone_path),
                owner=owner,
                repo=repo_name,
                file_tree=state["file_tree"],
                readme_content=state["readme_content"],
                timeout=300.0,  # 5 minute timeout
                model=model,
            )

            if not wiki_structure:
                state["error_message"] = "Deep Agent failed to generate wiki structure"
                return state

            if not wiki_structure.get("pages"):
                state["error_message"] = "Deep Agent produced empty wiki structure"
                return state

            state["wiki_structure"] = wiki_structure
            state["progress"] = 50.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Generated wiki structure with {len(wiki_structure.get('pages', []))} pages using Deep Agent exploration"
                )
            )

            return state

        except Exception as e:
            logger.error(f"Structure generation failed: {e}")
            state["error_message"] = f"Structure generation failed: {str(e)}"
            return state

    async def _generate_pages_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Generate individual wiki pages"""
        try:
            state["current_step"] = "generating_pages"
            state["progress"] = 60.0

            if not state["wiki_structure"]:
                state["error_message"] = (
                    "No wiki structure available for page generation"
                )
                return state

            pages = state["wiki_structure"].get("pages", [])
            generated_pages = []

            # Generate each page
            for i, page_info in enumerate(pages):
                try:
                    state["current_page"] = page_info["id"]

                    # Generate page content
                    page_content = await self._generate_page_content(
                        page_info, state["repository_id"]
                    )

                    if page_content:
                        page_info["content"] = page_content
                        generated_pages.append(page_info)

                    # Update progress
                    page_progress = 60.0 + (i / len(pages)) * 30.0
                    state["progress"] = page_progress

                except Exception as e:
                    logger.warning(f"Failed to generate page {page_info['id']}: {e}")
                    continue

            state["generated_pages"] = generated_pages
            state["progress"] = 90.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Generated content for {len(generated_pages)} wiki pages"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Page generation failed: {str(e)}"
            return state

    async def _store_wiki_node(self, state: WikiGenerationState) -> WikiGenerationState:
        """Store generated wiki in database"""
        try:
            state["current_step"] = "storing_wiki"
            state["progress"] = 95.0

            if not state["wiki_structure"] or not state["generated_pages"]:
                state["error_message"] = "No wiki content to store"
                return state

            # Create WikiStructure object
            wiki_data = state["wiki_structure"]

            # Build a map of page_id -> WikiPageDetail from generated pages
            pages_map = {}
            for page_data in state["generated_pages"]:
                page = WikiPageDetail(
                    id=page_data["id"],
                    title=page_data["title"],
                    description=page_data["description"],
                    importance=PageImportance(page_data["importance"]),
                    file_paths=page_data.get("file_paths", []),
                    related_pages=page_data.get("related_pages", []),
                    content=page_data.get("content", ""),
                )
                pages_map[page.id] = page

            # Convert sections to WikiSection objects with embedded pages
            wiki_sections = []
            for section_data in wiki_data.get("sections", []):
                # Get page objects for this section
                section_pages = []
                for page_info in section_data.get("pages", []):
                    # page_info can be a dict (from structured output) or a string ID
                    if isinstance(page_info, dict):
                        page_id = page_info.get("id")
                    else:
                        page_id = page_info
                    
                    if page_id and page_id in pages_map:
                        section_pages.append(pages_map[page_id])

                section = WikiSection(
                    id=section_data["id"],
                    title=section_data["title"],
                    pages=section_pages,
                )
                wiki_sections.append(section)

            # Create complete wiki structure (pages are now in sections)
            wiki_structure = WikiStructure(
                id=f"wiki_{state['repository_id']}",
                repository_id=state["repository_id"],
                title=wiki_data["title"],
                description=wiki_data["description"],
                sections=wiki_sections,
            )

            # Delete existing wiki if force regenerating
            await self._wiki_structure_repo.delete_one(
                {"repository_id": state["repository_id"]}
            )

            # Store new wiki
            await self._wiki_structure_repo.insert(wiki_structure)

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
        """Handle error node"""
        try:
            # Log error details
            logger.error(
                f"Wiki generation failed for repository {state['repository_id']}: {state.get('error_message')}"
            )

            # Add error message
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

    async def _generate_page_content(
        self, page_info: Dict[str, Any], repository_id: str
    ) -> Optional[str]:
        """Generate content for a specific wiki page

        Args:
            page_info: Page information from wiki structure
            repository_id: Repository ID

        Returns:
            Generated page content or None
        """
        try:
            # Get relevant source files for this page
            relevant_files = await self._get_relevant_source_files(
                page_info.get("file_paths", []), repository_id
            )

            if len(relevant_files) < 3:
                # Find additional relevant files using semantic search
                additional_files = await self._find_additional_relevant_files(
                    page_info["title"],
                    page_info["description"],
                    repository_id,
                    exclude_files=[f["file_path"] for f in relevant_files],
                )
                relevant_files.extend(additional_files)

            if not relevant_files:
                logger.warning(f"No relevant files found for page {page_info['id']}")
                return None

            # Prepare file list markdown
            file_list_markdown = "\n".join(
                [
                    f"- [{file_info['file_path']}]({file_info['file_path']}) - {file_info['language']} ({file_info['size']} bytes)"
                    for file_info in relevant_files[:10]  # Limit to 10 files
                ]
            )

            # Prepare page generation prompt
            page_prompt = self.page_prompt_template.format(
                page_title=page_info["title"], file_list_markdown=file_list_markdown
            )

            # Add source file contents to prompt
            source_contents = []
            for file_info in relevant_files[
                :5
            ]:  # Limit to 5 files for token efficiency
                content = file_info.get("content", "")[:2000]  # Truncate long files
                source_contents.append(
                    f"File: {file_info['file_path']}\n"
                    f"Language: {file_info['language']}\n"
                    f"Content:\n{content}\n"
                )

            full_prompt = (
                f"{page_prompt}\n\nRELEVANT_SOURCE_FILES:\n\n"
                + "\n---\n".join(source_contents)
            )

            # Generate page content
            generation_result = await self._llm_tool._arun(
                "generate",
                prompt=full_prompt,
                system_message="You are an expert technical writer. Generate comprehensive, accurate wiki pages based solely on the provided source files.",
                max_tokens=6000,
            )

            if generation_result["status"] == "success":
                return generation_result["generated_text"]
            else:
                logger.error(
                    f"Page content generation failed: {generation_result.get('error')}"
                )
                return None

        except Exception as e:
            logger.error(f"Page content generation failed for {page_info['id']}: {e}")
            return None

    async def _get_relevant_source_files(
        self, file_paths: List[str], repository_id: str
    ) -> List[Dict[str, Any]]:
        """Get relevant source files from database

        Args:
            file_paths: List of file paths
            repository_id: Repository ID

        Returns:
            List of file information dictionaries
        """
        try:
            relevant_files = []

            for file_path in file_paths:
                doc = await self._code_document_repo.find_one(
                    {"repository_id": repository_id, "file_path": file_path}
                )

                if doc:
                    relevant_files.append(
                        {
                            "file_path": doc.file_path,
                            "language": doc.language,
                            "content": doc.content,
                            "size": len(doc.content),
                        }
                    )

            return relevant_files

        except Exception as e:
            logger.error(f"Failed to get relevant source files: {e}")
            return []

    async def _find_additional_relevant_files(
        self,
        page_title: str,
        page_description: str,
        repository_id: str,
        exclude_files: List[str],
        max_additional: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find additional relevant files using semantic search

        Args:
            page_title: Page title
            page_description: Page description
            repository_id: Repository ID
            exclude_files: Files to exclude from search
            max_additional: Maximum additional files to find

        Returns:
            List of additional relevant files
        """
        try:
            # Create search query from page title and description
            search_query = f"{page_title} {page_description}"

            # Perform context search
            search_result = await self._context_tool._arun(
                "search",
                query=search_query,
                repository_id=repository_id,
                k=max_additional * 2,  # Get more candidates
            )

            if search_result["status"] != "success":
                return []

            # Filter out excluded files and convert to expected format
            additional_files = []
            for result in search_result["results"]:
                file_path = result["file_path"]
                if file_path not in exclude_files:
                    # Get full document content
                    doc = await self._code_document_repo.find_one(
                        {"repository_id": repository_id, "file_path": file_path}
                    )

                    if doc:
                        additional_files.append(
                            {
                                "file_path": doc.file_path,
                                "language": doc.language,
                                "content": doc.content,
                                "size": len(doc.content),
                                "relevance_score": result["similarity_score"],
                            }
                        )

                        if len(additional_files) >= max_additional:
                            break

            return additional_files

        except Exception as e:
            logger.warning(f"Additional file search failed: {e}")
            return []

    async def get_wiki_generation_status(self, repository_id: str) -> Dict[str, Any]:
        """Get wiki generation status for repository

        Args:
            repository_id: Repository ID

        Returns:
            Dictionary with wiki generation status
        """
        try:
            # Check if wiki exists
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
