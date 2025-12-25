"""Wiki generation agent for LangGraph workflows

This module implements the wiki generation agent that creates comprehensive
documentation wikis from analyzed repositories using the prompts from
wiki-generation-prompts.md.
"""

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from ..models.repository import Repository
from ..models.wiki import PageImportance, WikiPageDetail, WikiSection, WikiStructure
from ..repository.code_document_repository import CodeDocumentRepository
from ..repository.repository_repository import RepositoryRepository
from ..repository.wiki_structure_repository import WikiStructureRepository
from ..tools.context_tool import ContextTool
from ..tools.llm_tool import LLMTool

logger = logging.getLogger(__name__)


# Pydantic models for structured output
class WikiPageSchema(BaseModel):
    """Schema for wiki page structure"""

    id: str = Field(description="Unique page identifier (URL-friendly)")
    title: str = Field(description="Page title")
    description: str = Field(description="Brief description of what this page covers")
    importance: str = Field(description="Page importance: high, medium, or low")
    file_paths: List[str] = Field(
        description="List of relevant file paths from the repository"
    )
    related_pages: List[str] = Field(
        default_factory=list, description="List of related page IDs"
    )


class WikiSectionSchema(BaseModel):
    """Schema for wiki section structure"""

    id: str = Field(description="Unique section identifier (URL-friendly)")
    title: str = Field(description="Section title")
    pages: List[str] = Field(
        default_factory=list, description="List of page IDs in this section"
    )
    subsections: List[str] = Field(
        default_factory=list, description="List of subsection IDs"
    )


class WikiStructureSchema(BaseModel):
    """Schema for complete wiki structure"""

    title: str = Field(description="Overall title for the wiki")
    description: str = Field(description="Brief description of the repository")
    sections: List[WikiSectionSchema] = Field(description="List of wiki sections")
    pages: List[WikiPageSchema] = Field(description="List of wiki pages")
    root_sections: List[str] = Field(description="List of top-level section IDs")


class WikiGenerationState(TypedDict):
    """State for wiki generation workflow"""

    repository_id: str
    repository_info: Dict[str, Any]
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]
    generated_pages: List[Dict[str, Any]]
    current_page: Optional[str]
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]


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
        self, repository_id: str, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """Generate complete wiki for repository

        Args:
            repository_id: Repository identifier
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
                "repository_info": {},
                "file_tree": "",
                "readme_content": "",
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
        """Analyze repository for wiki generation"""
        try:
            state["current_step"] = "analyzing_repository"
            state["progress"] = 10.0

            repository = await self._repository_repo.find_one(
                {"id": UUID(state["repository_id"])}
            )
            if not repository:
                state["error_message"] = "Repository not found"
                return state

            # Convert repository model to dict for state
            state["repository_info"] = repository.model_dump(mode="python")

            # Get file tree from processed documents
            documents = await self._code_document_repo.find_many(
                {"repository_id": UUID(state["repository_id"])},
                limit=1000
            )

            if not documents:
                state["error_message"] = "No processed documents found for repository"
                return state

            # Build file tree
            file_paths = [doc.file_path for doc in documents]
            state["file_tree"] = self._build_file_tree(file_paths)

            # Find and read README - convert models to dicts
            documents_dicts = [doc.model_dump(mode="python") for doc in documents]
            readme_content = await self._find_readme_content(documents_dicts)
            state["readme_content"] = readme_content

            state["progress"] = 20.0

            # Add success message
            state["messages"].append(
                AIMessage(content=f"Analyzed repository with {len(documents)} files")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Repository analysis failed: {str(e)}"
            return state

    async def _generate_structure_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Generate wiki structure using LLM"""
        try:
            state["current_step"] = "generating_structure"
            state["progress"] = 30.0

            # Extract repository info
            repo_info = state["repository_info"]
            owner = repo_info.get("org", "unknown")
            repo_name = repo_info.get("name", "unknown")

            # Prepare structure generation prompt
            structure_prompt = self.structure_prompt_template.format(
                owner=owner,
                repo=repo_name,
                file_tree=state["file_tree"],
                readme_content=state["readme_content"],
            )

            # Generate structure using LLM with structured output
            wiki_structure = await self._generate_structured_wiki_structure(
                structure_prompt, state["repository_info"]
            )

            if not wiki_structure:
                state["error_message"] = "Failed to parse generated wiki structure"
                return state

            state["wiki_structure"] = wiki_structure
            state["progress"] = 50.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Generated wiki structure with {len(wiki_structure.get('pages', []))} pages"
                )
            )

            return state

        except Exception as e:
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

            # Convert pages to WikiPageDetail objects
            wiki_pages = []
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
                wiki_pages.append(page)

            # Convert sections to WikiSection objects
            wiki_sections = []
            for section_data in wiki_data.get("sections", []):
                section = WikiSection(
                    id=section_data["id"],
                    title=section_data["title"],
                    pages=section_data.get("pages", []),
                    subsections=section_data.get("subsections", []),
                )
                wiki_sections.append(section)

            # Create complete wiki structure
            wiki_structure = WikiStructure(
                id=f"wiki_{state['repository_id']}",
                repository_id=state["repository_id"],
                title=wiki_data["title"],
                description=wiki_data["description"],
                pages=wiki_pages,
                sections=wiki_sections,
                root_sections=wiki_data.get("root_sections", []),
            )

            # Delete existing wiki if force regenerating
            await self._wiki_structure_repo.delete_one(
                {"repository_id": state["repository_id"]}
            )

            # Store new wiki
            await self._wiki_structure_repo.insert(wiki_structure)

            state["progress"] = 100.0

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Successfully stored wiki with {len(wiki_pages)} pages"
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

    def _build_file_tree(self, file_paths: List[str]) -> str:
        """Build file tree representation from file paths

        Args:
            file_paths: List of file paths

        Returns:
            String representation of file tree
        """
        try:
            # Group files by directory
            tree_structure = {}

            for file_path in sorted(file_paths):
                parts = file_path.split("/")
                current = tree_structure

                # Build nested structure
                for part in parts[:-1]:  # Directories
                    if part not in current:
                        current[part] = {}
                    current = current[part]

                # Add file
                if parts:
                    filename = parts[-1]
                    current[filename] = None  # Files are leaf nodes

            # Convert to string representation
            return self._format_tree_structure(tree_structure)

        except Exception as e:
            logger.warning(f"File tree generation failed: {e}")
            return "\n".join(sorted(file_paths))

    def _format_tree_structure(
        self, structure: Dict[str, Any], indent: str = ""
    ) -> str:
        """Format tree structure as string

        Args:
            structure: Nested dictionary representing tree
            indent: Current indentation level

        Returns:
            Formatted tree string
        """
        lines = []

        for name, content in sorted(structure.items()):
            if content is None:  # File
                lines.append(f"{indent}├── {name}")
            else:  # Directory
                lines.append(f"{indent}├── {name}/")
                if content:  # Non-empty directory
                    sub_lines = self._format_tree_structure(content, indent + "│   ")
                    lines.append(sub_lines)

        return "\n".join(lines)

    async def _find_readme_content(self, documents: List[Dict[str, Any]]) -> str:
        """Find and extract README content

        Args:
            documents: List of document dictionaries

        Returns:
            README content or empty string
        """
        try:
            # Look for README files
            readme_patterns = ["readme.md", "readme.txt", "readme.rst", "readme"]

            for doc in documents:
                file_path = doc["file_path"].lower()
                filename = file_path.split("/")[-1]

                if filename in readme_patterns or filename.startswith("readme"):
                    return doc.get("content", "")[:5000]  # Limit README size

            return "No README file found in repository."

        except Exception as e:
            logger.warning(f"README extraction failed: {e}")
            return "Could not extract README content."

    async def _generate_structured_wiki_structure(
        self, structure_prompt: str, repository_info: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate wiki structure using LLM with structured output

        Args:
            structure_prompt: Prompt for structure generation
            repository_info: Repository information

        Returns:
            Structured wiki data or None
        """
        try:
            # Generate structured wiki structure using LLM tool
            generation_result = await self._llm_tool._arun(
                "generate_structured",
                prompt=structure_prompt,
                schema=WikiStructureSchema,
                system_message="You are an expert technical writer creating wiki structures for software projects.",
            )

            if generation_result["status"] != "success":
                logger.error(
                    f"Structured generation failed: {generation_result.get('error')}"
                )
                return None

            wiki_structure = generation_result["structured_output"]

            # Ensure root_sections is populated
            if not wiki_structure.get("root_sections"):
                # Find sections that are not subsections of others
                all_subsection_ids = set()
                for section in wiki_structure.get("sections", []):
                    all_subsection_ids.update(section.get("subsections", []))

                root_sections = [
                    section["id"]
                    for section in wiki_structure.get("sections", [])
                    if section["id"] not in all_subsection_ids
                ]
                wiki_structure["root_sections"] = root_sections

            return wiki_structure

        except Exception as e:
            logger.error(f"Structured wiki generation failed: {e}")
            return None

    async def _generate_pages_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Generate individual wiki pages"""
        try:
            state["current_step"] = "generating_pages"

            if not state["wiki_structure"]:
                state["error_message"] = "No wiki structure available"
                return state

            pages = state["wiki_structure"]["pages"]
            generated_pages = []

            # Generate each page
            for i, page_info in enumerate(pages):
                try:
                    page_content = await self._generate_page_content(
                        page_info, state["repository_id"]
                    )

                    if page_content:
                        page_info["content"] = page_content
                        generated_pages.append(page_info)

                        logger.info(f"Generated content for page: {page_info['id']}")

                    # Update progress
                    page_progress = 50.0 + (i / len(pages)) * 40.0
                    state["progress"] = page_progress

                except Exception as e:
                    logger.warning(f"Failed to generate page {page_info['id']}: {e}")
                    continue

            state["generated_pages"] = generated_pages
            state["progress"] = 90.0

            return state

        except Exception as e:
            state["error_message"] = f"Page generation failed: {str(e)}"
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
