"""
React agent factories for wiki generation.

Creates React agents with MCP filesystem tools for:
- Structure extraction (explores codebase, designs wiki structure)
- Page generation (reads source files, generates documentation)

These agents replace the direct LLMTool.generate_structured() approach
by using LangGraph's create_react_agent with MCP filesystem tools bound,
giving the agents actual filesystem access.
"""

from pathlib import Path
from typing import Any, List, Optional

import structlog
import yaml
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from pydantic import BaseModel, Field

from src.services.mcp_filesystem_client import MCPFilesystemClient
from src.tools.llm_tool import LLMTool

logger = structlog.get_logger(__name__)


def _load_prompts() -> dict:
    """Load prompts from YAML file."""
    prompts_path = Path(__file__).parent.parent / "prompts" / "wiki_prompts.yaml"
    with open(prompts_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


PROMPTS = _load_prompts()


# =============================================================================
# Tool and LLM Accessors
# =============================================================================


async def get_mcp_tools(names: Optional[List[str]] = None) -> List[Any]:
    """Get MCP filesystem tools.

    Returns:
        List of LangChain-compatible tools from MCP client.
    """
    mcp_client = MCPFilesystemClient.get_instance()
    if not mcp_client.is_initialized:
        await mcp_client.initialize()
    return mcp_client.get_tools(names)


def get_llm() -> BaseChatModel:
    """Get configured LLM for agents.

    Returns:
        Configured LangChain chat model.
    """
    llm_tool = LLMTool()
    return llm_tool._get_llm_provider()


# =============================================================================
# Pydantic Schemas for Structured Output
# =============================================================================


class LLMPageSchema(BaseModel):
    """Schema for LLM to generate page details."""

    id: str = Field(description="URL-friendly page identifier (lowercase, hyphens)")
    title: str = Field(description="Page title")
    description: str = Field(description="Brief description of what this page covers")
    importance: str = Field(
        description="Page importance: 'high', 'medium', or 'low'", default="medium"
    )
    file_paths: List[str] = Field(
        default_factory=list, description="Relevant source file paths"
    )


class LLMSectionSchema(BaseModel):
    """Schema for LLM to generate section details."""

    id: str = Field(description="URL-friendly section identifier")
    title: str = Field(description="Section title")
    description: str = Field(description="Brief section description")
    order: int = Field(description="Display order (1-based)")
    pages: List[LLMPageSchema] = Field(
        default_factory=list, description="Pages in this section"
    )


class LLMWikiStructureSchema(BaseModel):
    """Schema for complete wiki structure from LLM."""

    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    sections: List[LLMSectionSchema] = Field(
        default_factory=list, description="Wiki sections"
    )


# =============================================================================
# React Agent Factories
# =============================================================================


async def create_structure_agent() -> Any:
    """Create a React agent for wiki structure extraction.

    The agent can explore the codebase using MCP filesystem tools
    and returns a structured WikiStructure.

    Args:
        clone_path: Path to cloned repository
        file_tree: Pre-computed file tree string
        readme_content: README content

    Returns:
        Compiled React agent graph with ainvoke() method
    """

    tools = await get_mcp_tools(names=["read_text_file", "list_directory_with_sizes", "read_multiple_files"])
    llm = get_llm()

    # Build system prompt with context
    system_prompt = PROMPTS.get("structure_agent", {}).get("system_prompt", "")
    
    # Create React agent with MCP tools and structured output
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        response_format=LLMWikiStructureSchema,
    )

    logger.info(
        "Created structure agent",
        num_tools=len(tools),
    )

    return agent


async def create_page_agent(
    clone_path: str = "",
) -> Any:
    """Create a React agent for page content generation.

    The agent has FULL MCP filesystem tool access to read complete
    source files and generate detailed documentation with citations.

    Unlike the structure agent (context-efficient reading), this agent
    should read ENTIRE files to understand implementation details.

    Args:
        clone_path: Path to cloned repository

    Returns:
        Compiled React agent graph with ainvoke() method
    """
    tools = await get_mcp_tools(names=["read_text_file", "read_multiple_files"])
    llm = get_llm()

    # Use page_generation_full with MCP tool instructions
    page_prompt_template = PROMPTS.get("page_generation_full", {}).get(
        "system_prompt", ""
    )
    
    page_prompt = page_prompt_template.format(
        page_title="{page_title}",  # Placeholder for invocation
        page_description="{page_description}",
        file_hints_str="{file_hints_str}",
        clone_path=clone_path,
        repo_name=Path(clone_path).name if clone_path else "",
        repo_description="{repo_description}"
    )

    # Create React agent with MCP tools (no structured output - returns markdown)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=page_prompt,
    )

    logger.info(
        "Created page agent",
        clone_path=clone_path,
        num_tools=len(tools),
    )

    return agent
