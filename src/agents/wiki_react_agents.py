"""
React agent factories for wiki generation.

Creates React agents with MCP filesystem tools for:
- Structure extraction (explores codebase, designs wiki structure)
- Page generation (reads source files, generates documentation)

These agents replace the direct LLMTool.generate_structured() approach
by using LangGraph's create_react_agent with MCP filesystem tools bound,
giving the agents actual filesystem access.
"""

from functools import wraps
from pathlib import Path
from typing import Any, List, Optional

from deepagents.graph import PatchToolCallsMiddleware, SummarizationMiddleware, TodoListMiddleware
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
import structlog
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from langchain.agents import create_agent
from pydantic import BaseModel, Field

from src.services.mcp_filesystem_client import MCPFilesystemClient
from src.tools.llm_tool import LLMTool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


def _wrap_read_text_file_tool(tool: Any) -> Any:
    """Wrap read_text_file tool to fix mutually exclusive head/tail parameters.

    If both head and tail are specified, this wrapper keeps only head
    (since head is more commonly used for initial file inspection).

    Args:
        tool: The original MCP tool

    Returns:
        Wrapped tool with parameter validation
    """
    if tool.name != "read_text_file":
        return tool

    original_func = tool.coroutine

    @wraps(original_func)
    async def wrapped_func(*args, **kwargs):
        # Check if both head and tail are specified
        if kwargs.get("head") is not None and kwargs.get("tail") is not None:
            logger.warning(
                "read_text_file called with both head and tail - removing tail",
                head=kwargs.get("head"),
                tail=kwargs.get("tail"),
                path=kwargs.get("path"),
            )
            # Keep head, remove tail (head is more commonly used for initial inspection)
            del kwargs["tail"]

        return await original_func(*args, **kwargs)

    # Create a new StructuredTool with the wrapped function
    # Preserve all original tool properties
    return StructuredTool(
        name=tool.name,
        description=tool.description + "\n\nNOTE: head and tail are mutually exclusive - do NOT use both.",
        func=None,  # Sync function not needed
        coroutine=wrapped_func,
        args_schema=tool.args_schema,
        return_direct=getattr(tool, "return_direct", False),
    )


def _wrap_mcp_tools(tools: List[Any]) -> List[Any]:
    """Wrap MCP tools to add parameter validation.

    Args:
        tools: List of MCP tools

    Returns:
        List of wrapped tools with validation
    """
    return [_wrap_read_text_file_tool(tool) for tool in tools]


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
    """Get MCP filesystem tools with parameter validation wrappers.

    Returns:
        List of LangChain-compatible tools from MCP client, wrapped to
        validate parameters (e.g., prevent head+tail being used together).
    """
    mcp_client = MCPFilesystemClient.get_instance()
    if not mcp_client.is_initialized:
        await mcp_client.initialize()
    tools = mcp_client.get_tools(names)
    # Wrap tools to add parameter validation
    return _wrap_mcp_tools(tools)


def get_llm(provider: Optional[str] = None) -> BaseChatModel:
    """Get configured LLM for agents.

    Args:
        provider: Optional provider name ("openai", "gemini", etc.)
                  If None, uses the first available provider.

    Returns:
        Configured LangChain chat model.
    """
    llm_tool = LLMTool()
    return llm_tool._get_llm_provider(provider=provider)


# =============================================================================
# Pydantic Schemas for Structured Output
# =============================================================================


class LLMPageSchema(BaseModel):
    """Schema for LLM to generate page details and content.

    Used in two contexts:
    1. Structure agent: generates metadata (id, title, description, importance, file_paths)
    2. Page agent: generates content (content field) - metadata fields are echoed back
    """

    id: str = Field(description="URL-friendly page identifier (lowercase, hyphens)")
    title: str = Field(description="Page title")
    description: str = Field(description="Brief description of what this page covers")
    importance: str = Field(
        description="Page importance: 'high', 'medium', or 'low'", default="medium"
    )
    file_paths: List[str] = Field(
        default_factory=list, description="Relevant source file paths"
    )
    content: str = Field(
        default="",
        description="The complete wiki page documentation in Markdown format, "
        "including all sections, headers, code snippets, mermaid diagrams, and citations"
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

    The agent explores the codebase using MCP filesystem tools and returns
    a structured wiki structure using LangGraph's native response_format.

    Uses LangGraph's native response_format parameter to automatically
    extract structured output from the agent's message history after
    tool usage completes. The result is available in 'structured_response'.

    Returns:
        Compiled agent with ainvoke() that returns result with 'structured_response'
        containing an LLMWikiStructureSchema.
    """
    settings = get_settings()

    tools = await get_mcp_tools(names=["read_text_file", "read_multiple_files"])

    # Use Gemini 2.5 Pro
    llm = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model="gemini-2.5-pro",
        temperature=0,
    )

    # Build system prompt
    system_prompt = PROMPTS.get("structure_agent", {}).get("system_prompt", "")

    # Create React agent with MCP tools and native structured output
    # response_format adds a separate step at the end of the agent loop
    # that extracts the structured response from message history
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        response_format=LLMWikiStructureSchema,  # Native structured output extraction
        middleware=[
            # WikiMemoryMiddleware("structure_agent"),  # Memory persistence first
            TodoListMiddleware(),
            SummarizationMiddleware(model="gpt-4o-mini"),
            PatchToolCallsMiddleware(),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        ]
    )

    logger.info(
        "Created structure agent with native structured output",
        num_tools=len(tools),
        model="gemini-2.5-pro",
    )

    return agent

async def create_page_agent() -> Any:
    """Create a React agent for page content generation.

    The agent has FULL MCP filesystem tool access to read complete
    source files and generate detailed documentation with citations.

    Unlike the structure agent (context-efficient reading), this agent
    should read ENTIRE files to understand implementation details.

    Uses LangGraph's native response_format parameter to automatically
    extract structured output from the agent's message history after
    tool usage completes. The result is available in 'structured_response'.

    Returns:
        Compiled agent with ainvoke() that returns result with 'structured_response'
        containing an LLMPageSchema with the generated markdown content.
    """
    tools = await get_mcp_tools(names=["list_directory_with_sizes", "read_text_file", "read_multiple_files"])
    settings = get_settings()
    # Use Gemini 2.5 Pro
    llm = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model="gemini-2.5-pro",
        temperature=0,
    )

    # Use page_generation_full with MCP tool instructions
    page_prompt = PROMPTS.get("page_generation_full", {}).get(
        "system_prompt", ""
    )

    # Create React agent with MCP tools and native structured output
    # response_format adds a separate step at the end of the agent loop
    # that extracts the structured response from message history
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=page_prompt,
        response_format=LLMPageSchema,  # Native structured output extraction
        middleware=[
            # WikiMemoryMiddleware("page_agent"),  # Memory persistence first
            SummarizationMiddleware(model="gpt-4o-mini"),
            PatchToolCallsMiddleware(),
            ModelRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            ),
            ToolRetryMiddleware(
                max_retries=3,
                backoff_factor=2.0,
                initial_delay=1.0,
            )
        ]
    )

    logger.info(
        "Created page agent with native structured output",
        num_tools=len(tools),
    )

    return agent
