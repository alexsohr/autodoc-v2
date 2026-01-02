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
from typing import Any, List

import structlog
import yaml
from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent
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


async def get_mcp_tools() -> List[Any]:
    """Get MCP filesystem tools.

    Returns:
        List of LangChain-compatible tools from MCP client.
    """
    mcp_client = MCPFilesystemClient.get_instance()
    if not mcp_client.is_initialized:
        await mcp_client.initialize()
    return mcp_client.get_tools()


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


async def create_structure_agent(
    clone_path: str = "",
    file_tree: str = "",
    readme_content: str = "",
) -> Any:
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
    tools = await get_mcp_tools()
    llm = get_llm()

    # Extract owner/repo from clone_path if possible
    path_parts = Path(clone_path).parts if clone_path else []
    owner = path_parts[-2] if len(path_parts) >= 2 else ""
    repo = path_parts[-1] if len(path_parts) >= 1 else ""

    # Get exploration instructions for MCP tools
    exploration_instructions = PROMPTS.get("structure_agent", {}).get(
        "exploration_mcp", ""
    )
    if exploration_instructions:
        exploration_instructions = exploration_instructions.format(clone_path=clone_path)

    # Build system prompt with context
    system_prompt_template = PROMPTS.get("structure_agent", {}).get("system_prompt", "")
    system_prompt = system_prompt_template.format(
        owner=owner,
        repo=repo,
        file_tree=file_tree,
        readme_content=readme_content,
        clone_path=clone_path,
        exploration_instructions=exploration_instructions,
    )

    # Create React agent with MCP tools and structured output
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        response_format=LLMWikiStructureSchema,
    )

    logger.info(
        "Created structure agent",
        clone_path=clone_path,
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
    tools = await get_mcp_tools()
    llm = get_llm()

    # Try to get page_generation_react prompt, fall back to page_generation_full
    page_prompt = PROMPTS.get("page_generation_react", {}).get("system_prompt", "")

    if not page_prompt:
        # Use page_generation_full with MCP tool instructions
        page_prompt_template = PROMPTS.get("page_generation_full", {}).get(
            "system_prompt", ""
        )
        tool_instructions = PROMPTS.get("page_generation_full", {}).get(
            "tool_instructions_mcp", ""
        )
        if tool_instructions:
            tool_instructions = tool_instructions.format(clone_path=clone_path)

        if page_prompt_template:
            # Fill in what we can; page-specific fields will be filled at invocation
            page_prompt = page_prompt_template.format(
                page_title="{page_title}",  # Placeholder for invocation
                page_description="{page_description}",
                file_hints_str="{file_hints_str}",
                clone_path=clone_path,
                repo_name=Path(clone_path).name if clone_path else "",
                repo_description="{repo_description}",
                tool_instructions=tool_instructions,
            )

    # Final fallback to a comprehensive default prompt
    if not page_prompt:
        page_prompt = f"""You are an expert technical writer generating wiki documentation.

## Available Filesystem Tools

You have FULL access to filesystem tools:
- `read_text_file(path)`: Read COMPLETE file content (no head limit!)
- `list_directory(path)`: List directory contents
- `search_files(path, pattern)`: Search for files

All paths must be absolute, starting with: {clone_path}

## CRITICAL: Full File Reading Strategy

For documentation, you MUST read files COMPLETELY:

**DO:**
- Read the ENTIRE content of files assigned to this page
- Understand implementation details, data flow, logic
- Extract actual code snippets for examples
- Note specific line numbers for citations
- Read related files if needed for full context

**DON'T:**
- Use head parameter (you need full content)
- Skip reading files
- Guess at implementation details
- Cite lines without reading them

## Documentation Requirements

Generate comprehensive markdown including:
- Clear explanations of functionality
- Mermaid diagrams for architecture (use graph TD, never LR)
- Code snippets from actual source files
- Source citations: `Sources: [filename:line-range]()`
"""

    # Create React agent with MCP tools (no structured output - returns markdown)
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=page_prompt,
    )

    logger.info(
        "Created page agent",
        clone_path=clone_path,
        num_tools=len(tools),
    )

    return agent
