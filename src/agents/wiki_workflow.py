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
from src.services.wiki_memory_service import WikiMemoryService

logger = structlog.get_logger(__name__)

# Maximum iterations for structure extraction agent to prevent infinite loops
# Per LangGraph: recursion_limit = 2 * max_iterations + 1
MAX_STRUCTURE_ITERATIONS = 15
STRUCTURE_RECURSION_LIMIT = 2 * MAX_STRUCTURE_ITERATIONS + 1  # = 31


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
        force_regenerate: Flag to purge memories and regenerate from scratch
    """
    repository_id: str
    clone_path: str
    file_tree: str
    readme_content: str
    structure: Optional[WikiStructure]
    pages: Annotated[List[WikiPageDetail], operator.add]
    error: Optional[str]
    current_step: str
    force_regenerate: bool


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

    # id auto-generates as UUID via default_factory
    return WikiStructure(
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
    readme_content = state.get("readme_content", "")
    file_tree = state.get("file_tree", "")
    repository_id = state.get("repository_id")

    # Purge memories if force regenerate is requested
    if state.get("force_regenerate", False) and repository_id:
        try:
            repo_uuid = UUID(repository_id) if isinstance(repository_id, str) else repository_id
            result = await WikiMemoryService.purge_for_repository(repo_uuid)
            logger.info(
                "Purged wiki memories for force regenerate",
                repository_id=repository_id,
                deleted_count=result.get("deleted_count", 0),
            )
        except Exception as e:
            logger.warning(
                "Failed to purge memories for force regenerate",
                repository_id=repository_id,
                error=str(e),
            )

    # Create React agent with MCP tools
    agent = await create_structure_agent()

    # Build user message for the agent
    user_message = f"""Analyze this repository and propose a wiki structure that fully explains the code and architecture for a junior developer.

Repository root (clone path): {clone_path}

Directory tree:
```{file_tree}```

README:
```{readme_content}```
"""

    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]}
        )

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
    agent = await create_page_agent()

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
        user_message = PROMPTS.get("page_generation_full", {}).get("user_prompt", "").format(
            page_title=page.title,
            page_description=page.description,
            importance=page.importance.value if hasattr(page.importance, 'value') else page.importance,
            seed_paths_list=file_list,
            clone_path=clone_path,
            repo_name=Path(clone_path).name if clone_path else "",
            repo_description=structure.description
        )
        
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


async def aggregate_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Aggregate generated page content back into the wiki structure.

    Merges the content from generated pages back into the corresponding
    WikiPageDetail objects in the structure.

    Args:
        state: Current state with structure and generated pages

    Returns:
        Dict with updated 'structure' and 'current_step'
    """
    if state.get("error"):
        return {
            "error": state.get("error"),
            "current_step": "error",
        }

    structure = state.get("structure")
    pages = state.get("pages", [])

    if not structure:
        return {
            "error": "No structure to aggregate into",
            "current_step": "error",
        }

    # Build lookup of page content by id
    content_by_id = {page.id: page.content for page in pages if page.content}

    logger.info(
        "Aggregating page content",
        total_pages=len(pages),
        pages_with_content=len(content_by_id),
    )

    # Update structure with generated content
    updated_sections = []
    for section in structure.sections:
        updated_pages = []
        for page in section.pages:
            content = content_by_id.get(page.id, page.content)
            updated_page = page.model_copy(update={"content": content})
            updated_pages.append(updated_page)

        updated_section = section.model_copy(update={"pages": updated_pages})
        updated_sections.append(updated_section)

    updated_structure = structure.model_copy(update={"sections": updated_sections})

    # Count pages with content
    all_pages = updated_structure.get_all_pages()
    pages_with_content = [p for p in all_pages if p.content]

    logger.info(
        "Aggregation complete",
        total_pages=len(all_pages),
        pages_with_content=len(pages_with_content),
    )

    return {
        "structure": updated_structure,
        "current_step": "aggregated",
    }


async def finalize_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Finalize wiki by saving to database.

    The structure should already have content merged from the aggregate step.
    This node simply persists the final wiki to MongoDB.

    Args:
        state: State with aggregated structure (content already merged)

    Returns:
        Dict with updated current_step
    """
    if state.get("error"):
        return {"current_step": "error"}

    structure = state.get("structure")

    if not structure:
        return {
            "error": "No structure available for finalization",
            "current_step": "error",
        }

    # The structure already has content merged by aggregate_node
    # Just save it directly
    try:
        wiki_repo = WikiStructureRepository(WikiStructure)
        await wiki_repo.upsert(
            repository_id=UUID(state["repository_id"]),
            wiki=structure
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
    3. aggregate: Merge page content back into structure
    4. finalize: Save to database

    Returns:
        Compiled LangGraph workflow
    """
    builder = StateGraph(WikiWorkflowState)

    # Add nodes
    builder.add_node("extract_structure", extract_structure_node)
    builder.add_node("generate_pages", generate_pages_node)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("finalize", finalize_node)

    # Add edges - simple sequential flow
    builder.add_edge(START, "extract_structure")
    builder.add_edge("extract_structure", "generate_pages")
    builder.add_edge("generate_pages", "aggregate")
    builder.add_edge("aggregate", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


# Create singleton workflow instance
wiki_workflow = create_wiki_workflow()
