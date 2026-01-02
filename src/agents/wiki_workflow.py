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
    llm_tool = LLMTool()

    # Extract owner/repo from clone_path if possible
    # clone_path typically looks like: /path/to/repos/owner/repo or similar
    clone_path = state.get("clone_path", "")
    path_parts = Path(clone_path).parts if clone_path else []
    # Try to get last two parts as owner/repo, fallback to empty strings
    owner = path_parts[-2] if len(path_parts) >= 2 else ""
    repo = path_parts[-1] if len(path_parts) >= 1 else ""

    # Build prompt from template with variable substitution
    system_prompt_template = PROMPTS["structure_agent"]["system_prompt"]
    # The exploration_instructions is not needed for structured output extraction,
    # since we provide the file_tree and readme in the user prompt
    system_prompt = system_prompt_template.format(
        owner=owner,
        repo=repo,
        file_tree=state.get("file_tree", ""),
        readme_content=state.get("readme_content", ""),
        clone_path=clone_path,
        exploration_instructions="",  # Not needed for structure extraction
    )
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


async def generate_pages_node(state: WikiWorkflowState) -> Dict[str, Any]:
    """Generate content for all wiki pages sequentially.

    Iterates through all pages in the structure and generates
    markdown content for each one using the LLM.

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
    file_tree = state["file_tree"]

    llm_tool = LLMTool()
    system_prompt = PROMPTS["page_generation_full"]["system_prompt"]

    generated_pages = []
    all_pages = structure.get_all_pages()

    for page in all_pages:
        # Read relevant files if specified
        file_contents = ""
        if page.file_paths:
            for file_path in page.file_paths[:5]:  # Limit to 5 files
                full_path = Path(clone_path) / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        content = full_path.read_text(encoding="utf-8", errors="ignore")
                        file_contents += f"\n\n### File: {file_path}\n```\n{content[:5000]}\n```"
                    except Exception:
                        pass

        user_prompt = f"""Generate comprehensive documentation for this wiki page.

## Page Details
- Title: {page.title}
- Description: {page.description}
- Importance: {page.importance.value if hasattr(page.importance, 'value') else page.importance}

## Repository File Tree
```
{file_tree}
```

## Relevant Source Files
{file_contents if file_contents else "No specific files referenced."}

Generate the markdown content for this page."""

        # IMPORTANT: Use generate_text, not generate
        result = await llm_tool.generate_text(
            prompt=user_prompt,
            system_message=system_prompt,
        )

        if result["status"] == "error":
            page_with_content = page.model_copy(update={
                "content": f"*Error generating content: {result.get('error', 'Unknown')}*"
            })
        else:
            # IMPORTANT: Use "generated_text" not "content"
            page_with_content = page.model_copy(update={
                "content": result["generated_text"]
            })

        generated_pages.append(page_with_content)

    return {
        "pages": generated_pages,
        "current_step": "pages_generated",
    }
