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
from uuid import uuid4

from pydantic import BaseModel, Field

from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance
from src.tools.llm_tool import LLMTool
from src.utils.config_loader import get_settings


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
    """Extract wiki structure from repository analysis.

    Uses LLM with structured output to generate a WikiStructure
    based on the repository's file tree and README content.

    Args:
        state: Current workflow state with file_tree and readme_content

    Returns:
        Dict with 'structure' and updated 'current_step'
    """
    settings = get_settings()
    llm_tool = LLMTool()

    # Build prompt from template
    system_prompt = PROMPTS["structure_agent"]["system_prompt"]
    user_prompt = f"""Analyze this repository and create a wiki structure.

## File Tree
```
{state["file_tree"]}
```

## README Content
```
{state["readme_content"]}
```

Create a comprehensive wiki structure with sections and pages.
Design a wiki with 8-12 pages covering:
- Overview and Getting Started
- Architecture and core concepts
- Key features and functionality
- API reference (if applicable)
- Development and deployment guides

For each page, provide:
- id: URL-friendly identifier (lowercase, hyphens only)
- title: Descriptive page title
- description: What this page covers
- importance: 'high', 'medium', or 'low'
- file_paths: List of relevant source files"""

    result = await llm_tool.generate_structured(
        prompt=user_prompt,
        schema=LLMWikiStructureSchema,
        system_message=system_prompt,
    )

    if result["status"] == "error":
        return {
            "error": f"Structure extraction failed: {result.get('error', 'Unknown error')}",
            "current_step": "error",
        }

    # Parse structured output back to WikiStructure
    structure_data = result["structured_output"]
    if isinstance(structure_data, dict):
        structure = _convert_llm_structure_to_wiki_structure(
            structure_data,
            state["repository_id"],
        )
    else:
        # If it's already a Pydantic model, convert to dict first
        structure = _convert_llm_structure_to_wiki_structure(
            structure_data.model_dump() if hasattr(structure_data, "model_dump") else structure_data,
            state["repository_id"],
        )

    return {
        "structure": structure,
        "current_step": "structure_extracted",
    }
