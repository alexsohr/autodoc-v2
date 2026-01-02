"""
Wiki generation workflow using LangGraph sequential pattern.

This module replaces the autonomous deep agent approach with a predictable
workflow that:
1. Extracts wiki structure (deterministic)
2. Generates pages sequentially
3. Finalizes wiki storage
"""

import operator
import yaml
from pathlib import Path
from typing import Annotated, Optional, List, TypedDict, Dict, Any
from uuid import uuid4, UUID

import structlog
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance
from src.repository.wiki_structure_repository import WikiStructureRepository
from src.agents.wiki_react_agents import create_structure_agent, create_page_agent

logger = structlog.get_logger(__name__)


def _load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent.parent / "prompts" / "wiki_prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPTS = _load_prompts()


# =============================================================================
# Pydantic Schemas for LLM Structured Output
# =============================================================================


class LLMPageSchema(BaseModel):
    """Schema for LLM to generate page details."""

    id: str = Field(description="URL-friendly page identifier (lowercase, hyphens)")
    title: str = Field(description="Page title")
    description: str = Field(description="Brief description of what this page covers")
    importance: str = Field(
        description="Page importance: 'high', 'medium', or 'low'",
        default="medium"
    )
    file_paths: List[str] = Field(
        default_factory=list,
        description="Relevant source file paths"
    )


class LLMSectionSchema(BaseModel):
    """Schema for LLM to generate section details."""

    id: str = Field(description="URL-friendly section identifier")
    title: str = Field(description="Section title")
    pages: List[LLMPageSchema] = Field(
        default_factory=list,
        description="Pages in this section"
    )


class LLMWikiStructureSchema(BaseModel):
    """Schema for LLM to generate wiki structure."""

    title: str = Field(description="Wiki title for this project")
    description: str = Field(description="One-paragraph wiki description")
    sections: List[LLMSectionSchema] = Field(
        default_factory=list,
        description="Wiki sections with their pages"
    )


class WikiWorkflowState(TypedDict):
    """State for the wiki generation workflow.

    Attributes:
        repository_id: UUID of the repository being documented
        clone_path: Local filesystem path to cloned repository
        file_tree: String representation of repository file structure
        readme_content: Content of repository README file
        structure: Extracted wiki structure (sections and pages)
        pages: List of pages with generated content (reducer: append)
        error: Error message if workflow fails
        current_step: Current workflow step for observability
    """
    repository_id: str
    clone_path: str
    file_tree: str
    readme_content: str
    structure: Optional[WikiStructure]
    pages: Annotated[List[WikiPageDetail], operator.add]
    error: Optional[str]
    current_step: str


# =============================================================================
# Workflow Nodes
# =============================================================================


def _convert_llm_structure_to_wiki_structure(
    llm_structure: Dict[str, Any],
    repository_id: str,
) -> WikiStructure:
    """Convert LLM-generated structure to WikiStructure model.

    Args:
        llm_structure: Dictionary from LLM structured output
        repository_id: Repository UUID string

    Returns:
        WikiStructure model instance
    """
    sections = []
    for section_data in llm_structure.get("sections", []):
        pages = []
        for page_data in section_data.get("pages", []):
            # Convert importance string to enum
            importance_str = page_data.get("importance", "medium").lower()
            try:
                importance = PageImportance(importance_str)
            except ValueError:
                importance = PageImportance.MEDIUM

            page = WikiPageDetail(
                id=page_data["id"],
                title=page_data["title"],
                description=page_data["description"],
                importance=importance,
                file_paths=page_data.get("file_paths", []),
            )
            pages.append(page)

        section = WikiSection(
            id=section_data["id"],
            title=section_data["title"],
            pages=pages,
        )
        sections.append(section)

    # Generate a unique wiki ID
    wiki_id = f"wiki-{uuid4().hex[:8]}"

    return WikiStructure(
        id=wiki_id,
        repository_id=repository_id,
        title=llm_structure["title"],
        description=llm_structure["description"],
        sections=sections,
    )


async def extract_structure_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Extract wiki structure using React agent with MCP tools.

    The React agent can explore the codebase using filesystem tools
    and returns a structured WikiStructure.

    Args:
        state: Current workflow state with file_tree and readme_content

    Returns:
        Dict with 'structure' and updated 'current_step'
    """
    clone_path = state.get("clone_path", "")
    file_tree = state.get("file_tree", "")
    readme_content = state.get("readme_content", "")

    # Create React agent with MCP tools
    agent = await create_structure_agent(
        clone_path=clone_path,
        file_tree=file_tree,
        readme_content=readme_content,
    )

    # Build user message for the agent
    user_message = f"""Analyze this repository and create a comprehensive wiki structure.

## Repository Context
- Clone Path: {clone_path}

## File Tree
```
{file_tree}
```

## README
```
{readme_content}
```

Explore the codebase using the filesystem tools to understand:
1. Project architecture and structure
2. Key modules and their purposes
3. Important files and their relationships

Then design a wiki with 8-12 pages covering:
- Overview and Getting Started
- Architecture and core concepts
- Key features and functionality
- API reference (if applicable)
- Development and deployment guides

Use the filesystem tools to examine actual source files before finalizing the structure.
"""

    try:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_message}]
        })

        # Extract structured response
        structured_output = result.get("structured_response")
        if not structured_output:
            return {
                "error": "Agent did not return structured output",
                "current_step": "error",
            }

        # Convert to WikiStructure
        if isinstance(structured_output, dict):
            structure = _convert_llm_structure_to_wiki_structure(
                structured_output,
                state["repository_id"],
            )
        else:
            structure = _convert_llm_structure_to_wiki_structure(
                structured_output.model_dump(),
                state["repository_id"],
            )

        return {
            "structure": structure,
            "current_step": "structure_extracted",
        }

    except Exception as e:
        logger.error("Structure extraction failed", error=str(e))
        return {
            "error": f"Structure extraction failed: {str(e)}",
            "current_step": "error",
        }


async def generate_pages_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Generate content for all wiki pages sequentially using React agents.

    Creates a React agent with MCP filesystem tools and invokes it
    for each page in sequence. The agent can read source files to
    generate accurate, source-grounded documentation.

    Args:
        state: Current state with extracted structure

    Returns:
        Dict with 'pages' list and updated 'current_step'
    """
    # Check for existing errors or missing structure
    if state.get("error"):
        return {
            "error": state.get("error"),
            "current_step": "error",
        }

    if not state.get("structure"):
        return {
            "error": "No structure available",
            "current_step": "error",
        }

    structure = state["structure"]
    clone_path = state["clone_path"]

    # Create React agent with MCP tools (reused for all pages)
    agent = await create_page_agent(clone_path=clone_path)

    generated_pages = []
    all_pages = structure.get_all_pages()

    logger.info(
        "Starting sequential page generation",
        total_pages=len(all_pages),
        clone_path=clone_path,
    )

    for idx, page in enumerate(all_pages):
        logger.info(
            "Generating page",
            page_id=page.id,
            page_title=page.title,
            progress=f"{idx + 1}/{len(all_pages)}",
        )

        # Build user message for this page
        file_list = "\n".join(f"- {clone_path}/{fp}" for fp in page.file_paths) if page.file_paths else "No specific files assigned"

        user_message = f"""Generate comprehensive documentation for this wiki page.

## Page Details
- Title: {page.title}
- Description: {page.description}
- Importance: {page.importance.value if hasattr(page.importance, 'value') else page.importance}

## Files to Read (MUST READ ALL COMPLETELY)
{file_list}

## CRITICAL Instructions

1. **READ ALL FILES COMPLETELY** - Use read_file WITHOUT the head parameter
   - You need full implementation details, not just headers
   - Read each file in the list above in its entirety

2. **Understand the implementation** - After reading:
   - Data structures and their purposes
   - Function logic and data flow
   - Error handling patterns
   - Integration with other components

3. **Generate comprehensive documentation** including:
   - Clear explanation with implementation details
   - Mermaid diagrams (use graph TD, never LR)
   - Actual code snippets from source files
   - Source citations: Sources: [filename:line-range]()

4. **Start output with:** # {page.title}

Remember: All file paths are absolute. Read files COMPLETELY, not just headers!
"""

        try:
            result = await agent.ainvoke({
                "messages": [{"role": "user", "content": user_message}]
            })

            # Extract content from last message
            messages = result.get("messages", [])
            content = ""
            if messages:
                last_message = messages[-1]
                content = last_message.content if hasattr(last_message, 'content') else str(last_message)

            page_with_content = page.model_copy(update={"content": content})

        except Exception as e:
            logger.error(
                "Page generation failed",
                page_id=page.id,
                error=str(e),
            )
            page_with_content = page.model_copy(update={
                "content": f"*Error generating content: {str(e)}*"
            })

        generated_pages.append(page_with_content)

    logger.info(
        "Page generation complete",
        total_pages=len(generated_pages),
        pages_with_content=len([p for p in generated_pages if p.content]),
    )

    return {
        "pages": generated_pages,
        "current_step": "pages_generated",
    }


async def finalize_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Finalize wiki by combining pages and storing to database.

    This is the fan-in step. It takes all generated pages, updates
    the structure with content, and persists to MongoDB.

    Args:
        state: State with structure and generated pages

    Returns:
        Dict with updated current_step
    """
    if state.get("error"):
        return {"current_step": "error"}

    structure = state.get("structure")
    pages = state.get("pages", [])

    if not structure:
        return {
            "error": "No structure available for finalization",
            "current_step": "error",
        }

    # Create page lookup by ID
    page_content_map = {p.id: p.content for p in pages if p.content}

    # Update structure sections with generated content
    updated_sections = []
    for section in structure.sections:
        updated_pages = []
        for page in section.pages:
            content = page_content_map.get(page.id)
            if content:
                updated_page = page.model_copy(update={"content": content})
            else:
                updated_page = page
            updated_pages.append(updated_page)

        updated_section = section.model_copy(update={"pages": updated_pages})
        updated_sections.append(updated_section)

    # Create final wiki structure
    final_wiki = WikiStructure(
        id=structure.id,
        repository_id=state["repository_id"],
        title=structure.title,
        description=structure.description,
        sections=updated_sections,
    )

    # Store to database using repository layer
    try:
        wiki_repo = WikiStructureRepository(WikiStructure)
        await wiki_repo.upsert(
            repository_id=UUID(state["repository_id"]),
            wiki=final_wiki
        )
    except Exception as e:
        return {
            "error": f"Failed to save wiki: {str(e)}",
            "current_step": "error",
        }

    return {
        "current_step": "completed",
    }


# =============================================================================
# Workflow Assembly
# =============================================================================


def create_wiki_workflow():
    """Create and compile the wiki generation workflow.

    The workflow follows a sequential pattern:
    1. extract_structure: Analyze repo and create wiki structure
    2. generate_pages: Generate content for all pages sequentially
    3. finalize: Combine results and store to database

    Returns:
        Compiled LangGraph workflow
    """
    builder = StateGraph(WikiWorkflowState)

    # Add nodes
    builder.add_node("extract_structure", extract_structure_node)
    builder.add_node("generate_pages", generate_pages_node)
    builder.add_node("finalize", finalize_node)

    # Add edges - simple sequential flow
    builder.add_edge(START, "extract_structure")
    builder.add_edge("extract_structure", "generate_pages")
    builder.add_edge("generate_pages", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create singleton workflow instance
wiki_workflow = create_wiki_workflow()
