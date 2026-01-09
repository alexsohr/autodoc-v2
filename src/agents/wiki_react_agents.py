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
from typing import Any, Callable, List, Optional

from deepagents.graph import AnthropicPromptCachingMiddleware, PatchToolCallsMiddleware, SummarizationMiddleware, TodoListMiddleware
from langchain.agents.middleware import ModelRetryMiddleware, ToolRetryMiddleware
import structlog
import yaml
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from langchain.agents import create_agent
from deepagents import SubAgentMiddleware, create_deep_agent
from deepagents.backends import FilesystemBackend
from pydantic import BaseModel, Field

from src.services.mcp_filesystem_client import MCPFilesystemClient
from src.tools.llm_tool import LLMTool
from langchain_google_genai import ChatGoogleGenerativeAI
from src.utils.config_loader import get_settings
from src.agents.middleware import WikiMemoryMiddleware

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


class StructuredAgentWrapper:
    """Wrapper that adds structured output extraction after agent exploration.
    
    This implements a two-stage approach to work around Gemini's limitation
    of not supporting tools + structured output simultaneously:
    
    1. Stage 1: Agent explores codebase using tools (no structured output)
    2. Stage 2: Separate LLM call extracts structured output from exploration results
    """
    
    def __init__(self, agent: Any, llm: Any, schema: type):
        """Initialize wrapper.
        
        Args:
            agent: The exploration agent (without structured output constraint)
            llm: LLM instance for structured output extraction
            schema: Pydantic schema for structured output
        """
        self.agent = agent
        self.llm = llm
        self.schema = schema
        self._structured_llm = llm.with_structured_output(schema)
    
    async def ainvoke(self, inputs: dict, config: Optional[dict] = None) -> dict:
        """Run agent and extract structured output.
        
        Args:
            inputs: Input dict with 'messages' key
            config: Optional LangGraph config
            
        Returns:
            Dict with 'structured_response' key containing the schema
        """
        # Stage 1: Run exploration agent
        logger.info("Stage 1: Running exploration agent with tools")
        result = await self.agent.ainvoke(inputs, config=config)
        
        # Extract final text from agent response
        final_text = self._extract_final_response(result)
        logger.info(
            "Stage 1 complete, extracting structured output",
            response_length=len(final_text),
        )
        
        # Stage 2: Extract structured output
        logger.info("Stage 2: Extracting structured output")
        extraction_prompt = f"""Based on the following wiki structure analysis, extract the structured wiki format.

Analysis:
{final_text}

Extract the wiki structure with title, description, and sections (each section has id, title, description, importance, and pages)."""

        structured_result = await self._structured_llm.ainvoke(extraction_prompt)
        
        logger.info(
            "Stage 2 complete",
            result_type=type(structured_result).__name__,
        )
        
        # Return in expected format for extract_structure_node
        return {"structured_response": structured_result}
    
    def _extract_final_response(self, result: dict) -> str:
        """Extract the final text response from agent result.
        
        Args:
            result: Agent result dict with 'messages' key
            
        Returns:
            Final text content from the last AI message
        """
        messages = result.get("messages", [])
        
        # Find the last AI message with content
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # Handle content blocks
                    text_parts = []
                    for block in msg.content:
                        if isinstance(block, str):
                            text_parts.append(block)
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    if text_parts:
                        return "\n".join(text_parts)

        return ""


class PageAgentWrapper:
    """Wrapper that extracts wiki page content from agent messages.

    The page agent uses tools (including memory tools) which add messages
    to the conversation. This wrapper intelligently extracts the actual
    wiki content from the message history, ignoring tool acknowledgments.

    This implements Option B from the wiki workflow improvement proposal:
    State-based output with dedicated generated_content field.
    """

    # Minimum character threshold for wiki content
    MIN_CONTENT_LENGTH = 500

    def __init__(self, agent: Any):
        """Initialize wrapper.

        Args:
            agent: The page generation agent (compiled LangGraph)
        """
        self.agent = agent

    async def ainvoke(self, inputs: dict, config: Optional[dict] = None) -> dict:
        """Run agent and extract wiki content.

        Args:
            inputs: Input dict with 'messages' key
            config: Optional LangGraph config

        Returns:
            Dict with original result plus 'generated_content' key
        """
        import re

        # Run the page agent
        result = await self.agent.ainvoke(inputs, config=config)

        # Extract wiki content from messages
        content = self._extract_wiki_content(result)

        # Add generated_content to result
        result["generated_content"] = content

        logger.info(
            "PageAgentWrapper extracted content",
            content_length=len(content) if content else 0,
            message_count=len(result.get("messages", [])),
        )

        return result

    def _extract_wiki_content(self, result: dict) -> str:
        """Extract wiki page content from agent messages.

        Uses intelligent heuristics to find the actual wiki content:
        1. Filters to AI messages only
        2. Excludes tool acknowledgments
        3. Requires minimum content length
        4. Selects longest qualifying message

        Args:
            result: Agent result dict with 'messages' key

        Returns:
            Extracted wiki content string, or empty string if not found
        """
        messages = result.get("messages", [])
        candidates = []

        for msg in messages:
            # Skip non-AI messages
            if not self._is_ai_message(msg):
                continue

            # Get message content
            content = self._get_message_content(msg)
            if not content:
                continue

            # Skip tool acknowledgments
            if self._is_tool_acknowledgment(content):
                continue

            # Skip messages that are too short to be wiki content
            if len(content) < self.MIN_CONTENT_LENGTH:
                continue

            # Skip messages with pending tool calls (not final content)
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                continue

            candidates.append(content)

        if not candidates:
            # Fallback: try to find any substantial AI message
            logger.warning(
                "No wiki content candidates found, falling back to last AI message",
                message_count=len(messages),
            )
            return self._fallback_extract(messages)

        # Return the longest candidate (most comprehensive wiki content)
        return max(candidates, key=len)

    def _is_ai_message(self, msg: Any) -> bool:
        """Check if message is an AI message."""
        # Check message type name
        msg_type = type(msg).__name__
        if msg_type in ("AIMessage", "AIMessageChunk"):
            return True

        # Check for role attribute
        if hasattr(msg, "role") and msg.role == "assistant":
            return True

        # Check for type attribute
        if hasattr(msg, "type") and msg.type == "ai":
            return True

        return False

    def _get_message_content(self, msg: Any) -> str:
        """Extract text content from a message."""
        if not hasattr(msg, "content"):
            return ""

        content = msg.content

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Handle content blocks
            text_parts = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "\n".join(text_parts)

        return str(content) if content else ""

    def _is_tool_acknowledgment(self, content: str) -> bool:
        """Check if content is a tool acknowledgment message.

        Detects memory tool responses that should not be treated as wiki content.
        """
        import re

        content_lower = content.lower().strip()

        # Patterns that indicate tool acknowledgments (not wiki content)
        acknowledgment_patterns = [
            r"^memory stored",
            r"^found \d+ relevant memories",
            r"^found \d+ file-related memories",
            r"^no relevant memories found",
            r"^no memories found",
            r"^error: repository_id",
            r"^failed to store memory",
            r"^failed to recall memories",
            r"^failed to get file memories",
            r"^the documentation has been successfully",
            r"^i have successfully stored",
            r"^i've stored the memory",
        ]

        for pattern in acknowledgment_patterns:
            if re.search(pattern, content_lower):
                return True

        return False

    def _fallback_extract(self, messages: list) -> str:
        """Fallback extraction: find last AI message with any content.

        Used when no candidates pass the strict criteria.
        """
        for msg in reversed(messages):
            if self._is_ai_message(msg):
                content = self._get_message_content(msg)
                # Relaxed threshold for fallback
                if content and len(content) > 100:
                    # Still skip obvious tool acknowledgments
                    if not self._is_tool_acknowledgment(content):
                        return content

        return ""


# =============================================================================
# React Agent Factories
# =============================================================================


async def create_structure_agent() -> StructuredAgentWrapper:
    """Create a React agent for wiki structure extraction.

    Uses a two-stage approach to work around Gemini's limitation:
    1. Agent explores codebase with tools (no structured output constraint)
    2. Separate call extracts structured output from exploration results

    Args:
        clone_path: Path to cloned repository

    Returns:
        StructuredAgentWrapper with ainvoke() that returns LLMWikiStructureSchema
    """

    
    settings = get_settings()

    tools = await get_mcp_tools(names=["read_text_file", "read_multiple_files"])
    
    # Use Gemini 2.5 Flash with thinking
    llm = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model="gemini-2.5-pro",
        temperature=0,
    )

    # Build system prompt - instruct agent to output structured wiki info
    base_prompt = PROMPTS.get("structure_agent", {}).get("system_prompt", "")
    system_prompt = f"""{base_prompt}

IMPORTANT: After exploring the codebase, provide your final wiki structure in a clear format with:
- Wiki title and description
- Sections (each with id, title, description, importance level)
- Pages within each section (each with id, title, description, importance, source file paths)

This information will be extracted into a structured format."""

    # Stage 1 agent: Exploration with tools, NO response_format
    exploration_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
        # NO response_format - Gemini can't do tools + structured output together
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
        "Created structure agent with two-stage approach",
        num_tools=len(tools),
        model="gemini-2.5-flash",
    )

    # Wrap with structured output extraction (Stage 2)
    return StructuredAgentWrapper(
        agent=exploration_agent,
        llm=llm,
        schema=LLMWikiStructureSchema,
    )

async def create_page_agent() -> PageAgentWrapper:
    """Create a React agent for page content generation.

    The agent has FULL MCP filesystem tool access to read complete
    source files and generate detailed documentation with citations.

    Unlike the structure agent (context-efficient reading), this agent
    should read ENTIRE files to understand implementation details.

    The agent is wrapped with PageAgentWrapper to intelligently extract
    wiki content from messages, handling memory tool acknowledgments.

    Returns:
        PageAgentWrapper with ainvoke() that returns result with 'generated_content'
    """
    tools = await get_mcp_tools(names=["list_directory_with_sizes", "read_text_file", "read_multiple_files"])
    settings = get_settings()
        # Use Gemini 2.5 Flash with thinking
    llm = ChatGoogleGenerativeAI(
        google_api_key=settings.google_api_key,
        model="gemini-2.5-pro",
        temperature=0,
    )

    # Use page_generation_full with MCP tool instructions
    page_prompt = PROMPTS.get("page_generation_full", {}).get(
        "system_prompt", ""
    )

    # Create React agent with MCP tools (no structured output - returns markdown)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=page_prompt,
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
        "Created page agent with content extraction wrapper",
        num_tools=len(tools),
    )

    # Wrap with content extraction
    return PageAgentWrapper(agent)
