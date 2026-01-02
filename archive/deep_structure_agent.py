"""Deep Agent for wiki structure generation.

This module provides a Deep Agent that autonomously explores a cloned repository
to generate an accurate wiki structure.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.tools import tool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class WikiPageInput(BaseModel):
    """Schema for a wiki page in the structure"""
    title: str = Field(description="Page title")
    slug: str = Field(description="URL-friendly page identifier")
    section: str = Field(description="Section this page belongs to")
    file_paths: List[str] = Field(description="Source files relevant to this page")
    description: str = Field(description="Brief description of page content")


class FinalizeWikiStructureInput(BaseModel):
    """Schema for finalizing the wiki structure"""
    title: str = Field(description="Wiki title")
    description: str = Field(description="Wiki description")
    pages: List[Dict[str, Any]] = Field(
        description="List of wiki pages with title, slug, section, file_paths, description"
    )


def create_finalize_tool(capture_dict: Dict[str, Any]) -> Any:
    """Create the finalize_wiki_structure tool with closure for capturing output.

    Args:
        capture_dict: Dictionary that will be mutated to store the captured structure

    Returns:
        A langchain tool that captures the wiki structure
    """
    @tool(args_schema=FinalizeWikiStructureInput)
    def finalize_wiki_structure(
        title: str,
        description: str,
        pages: List[Dict[str, Any]]
    ) -> str:
        """Finalize the wiki structure after exploring the repository.

        Call this tool when you have finished analyzing the repository and are
        ready to submit the final wiki structure. Include 8-12 pages covering
        the key aspects of the codebase.
        """
        capture_dict["title"] = title
        capture_dict["description"] = description
        capture_dict["pages"] = pages
        logger.info(
            f"Wiki structure finalized, title={title}, page_count={len(pages)}"
        )
        return f"Wiki structure captured successfully with {len(pages)} pages."

    return finalize_wiki_structure


def get_structure_prompt(
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    clone_path: Optional[str] = None,
    use_mcp_tools: bool = False,
) -> str:
    """Generate the system prompt for the structure agent.

    Args:
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README file content
        clone_path: Absolute path to the cloned repository (for MCP tools)
        use_mcp_tools: Whether MCP filesystem tools are available

    Returns:
        Formatted system prompt
    """
    if use_mcp_tools and clone_path:
        exploration_instructions = f"""## Exploration Strategy
The repository is located at: {clone_path}

The file tree above already shows the complete directory structure - use it to identify files to read.

### Context-Efficient Reading (IMPORTANT)
To minimize context usage and improve efficiency, follow this reading strategy:

1. **Read File Headers First (50 lines)**
   Use `read_text_file` with `head=50` to read only the first 50 lines:
   ```
   read_text_file(path="{clone_path}/src/main.py", head=50)
   ```
   The first 50 lines typically contain imports, docstrings, and class/function signatures -
   enough to understand the file's purpose without loading full content.

2. **Read More Only When Needed**
   If 50 lines aren't enough to understand a file:
   - Use `head=100` or `head=150` for larger files
   - Use `tail=50` to see the end of a file (exports, main block)
   - Use `grep` to find specific patterns in the file
   - Only read full files for small config files (< 50 lines anyway)

3. **Use search_files for Discovery**
   Find related files by pattern:
   ```
   search_files(path="{clone_path}", pattern="**/*controller*.py")
   search_files(path="{clone_path}", pattern="**/test_*.py")
   ```

4. **Use read_multiple_files Efficiently**
   When reading multiple small files (like configs), batch them:
   ```
   read_multiple_files(paths=["{clone_path}/package.json", "{clone_path}/tsconfig.json"])
   ```

### What to Read
Focus on understanding the codebase architecture:
- Config files: package.json, pyproject.toml, Cargo.toml, setup.py (read in full - usually small)
- Entry points: main.py, index.ts, App.tsx (use head=50 first)
- Core modules: understand structure before reading details

### Exploration Workflow
1. Start with config files (usually small, read in full)
2. Read entry points with head=50 to understand structure
3. Use search_files to discover related components
4. Read additional files only as needed for page decisions

IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files.
Do NOT read full files unless absolutely necessary - prefer head=50 for initial exploration."""
    else:
        exploration_instructions = """## Exploration Strategy
The file tree above already shows the complete directory structure - use it to identify files to read.

1. Use `read_file` to read key files and understand the codebase:
   - Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
   - Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
   - Core modules and their purposes
2. Use `glob` to find specific file patterns if needed (e.g., `**/*.py`)
3. Use `grep` to search for patterns like class definitions, API routes, exports"""

    return f"""You are an expert technical writer analyzing a repository to design a wiki structure.

## Repository
- Owner: {owner}
- Name: {repo}

## Initial File Tree
{file_tree}

## README Content
{readme_content}

## Your Task
Explore this repository to understand its architecture, then design a comprehensive wiki structure.

{exploration_instructions}

## Output Requirements
When you have sufficient understanding, call `finalize_wiki_structure` with:
- **title**: A descriptive wiki title for this project
- **description**: A one-paragraph description of the wiki
- **pages**: A list of 8-12 pages, each with:
  - title: Page title
  - slug: URL-friendly identifier (lowercase, hyphens)
  - section: One of "Overview", "Architecture", "Features", "API", "Deployment", "Development"
  - file_paths: List of relevant source files
  - description: What this page covers

Focus on what would help a new developer understand and work with this codebase.
"""


def create_structure_agent(
    clone_path: str,
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    model: Optional[str] = None,
    mcp_tools: Optional[List[Any]] = None,
) -> Any:
    """Create a Deep Agent configured for wiki structure generation.

    Args:
        clone_path: Path to the cloned repository
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README content
        model: Optional model override (default: uses deepagents default)
        mcp_tools: Optional MCP filesystem tools to use instead of built-in FilesystemBackend

    Returns:
        Configured Deep Agent
    """
    import os

    # This will be populated by the finalize tool
    captured_structure: Dict[str, Any] = {}

    # Create the finalize tool with capture closure
    finalize_tool = create_finalize_tool(captured_structure)

    # Generate system prompt with MCP-aware instructions
    system_prompt = get_structure_prompt(
        owner, repo, file_tree, readme_content,
        clone_path=clone_path if mcp_tools else None,
        use_mcp_tools=mcp_tools is not None
    )

    # Build agent kwargs
    agent_kwargs = {
        "system_prompt": system_prompt,
    }

    # Use MCP tools if provided, otherwise fall back to FilesystemBackend
    if mcp_tools:
        logger.info(f"Using MCP filesystem tools ({len(mcp_tools)} tools) for repository access")
        # Combine MCP tools with our finalize tool
        agent_kwargs["tools"] = list(mcp_tools) + [finalize_tool]
        # Don't use FilesystemBackend - MCP tools handle filesystem access
    else:
        logger.warning(
            "MCP filesystem not available, using FilesystemBackend. "
            "Note: Windows absolute paths may not work correctly."
        )
        backend = FilesystemBackend(root_dir=clone_path)
        agent_kwargs["backend"] = backend
        agent_kwargs["tools"] = [finalize_tool]

    if model:
        from langchain.chat_models import init_chat_model

        # Use init_chat_model with explicit provider:model format
        # Add openai: prefix if no provider specified (e.g., gpt-4o-mini -> openai:gpt-4o-mini)
        if ":" not in model:
            model_string = f"openai:{model}"
            provider = "openai"
        else:
            model_string = model
            provider = model.split(":")[0]

        # Build kwargs for init_chat_model
        model_kwargs = {"temperature": 0}

        # Explicitly pass API key if available in environment (belt and suspenders)
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            model_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
            logger.debug("Using OPENAI_API_KEY from environment")
        elif provider == "google" and os.environ.get("GOOGLE_API_KEY"):
            model_kwargs["api_key"] = os.environ["GOOGLE_API_KEY"]
            logger.debug("Using GOOGLE_API_KEY from environment")
        elif provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            model_kwargs["api_key"] = os.environ["ANTHROPIC_API_KEY"]
            logger.debug("Using ANTHROPIC_API_KEY from environment")

        logger.info(f"Initializing Deep Agent with model: {model_string}")
        agent_kwargs["model"] = init_chat_model(model_string, **model_kwargs)

    agent = create_deep_agent(**agent_kwargs)

    # Attach the capture dict to the agent for retrieval
    agent._structure_capture = captured_structure

    return agent


async def run_structure_agent(
    clone_path: str,
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str,
    timeout: float = 300.0,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the structure agent and return the captured wiki structure.

    Args:
        clone_path: Path to the cloned repository
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README content
        timeout: Maximum execution time in seconds
        model: Optional model override

    Returns:
        Captured wiki structure dict or None if failed
    """
    from langsmith import traceable

    from src.services.mcp_filesystem_client import get_mcp_filesystem_client

    # Try to get MCP filesystem tools
    mcp_tools = None
    try:
        mcp_client = get_mcp_filesystem_client()
        if mcp_client.is_initialized:
            # Get the cached tools from the MCP client
            mcp_tools = list(mcp_client._tools.values())
            logger.info(f"Retrieved {len(mcp_tools)} MCP filesystem tools")
        else:
            logger.warning("MCP filesystem client not initialized")
    except Exception as e:
        logger.warning(f"Could not get MCP filesystem tools: {e}")

    agent = create_structure_agent(
        clone_path=clone_path,
        owner=owner,
        repo=repo,
        file_tree=file_tree,
        readme_content=readme_content,
        model=model,
        mcp_tools=mcp_tools,
    )

    # Build the user message with clone path for MCP tools
    if mcp_tools:
        user_message = (
            f"Analyze this repository and create a comprehensive wiki structure. "
            f"The repository is located at: {clone_path}\n\n"
            f"The file tree is already provided above.\n\n"
            f"CONTEXT-EFFICIENT EXPLORATION:\n"
            f"- Use `read_text_file` with `head=50` to read only file headers first\n"
            f"- Example: read_text_file(path='{clone_path}/src/main.py', head=50)\n"
            f"- Only read more if 50 lines aren't enough to understand the file's purpose\n"
            f"- Use `search_files` to discover related files by pattern\n\n"
            f"Explore efficiently, then call finalize_wiki_structure with your findings."
        )
    else:
        user_message = (
            "Analyze this repository and create a comprehensive wiki structure. "
            "The file tree is already provided. Read key files to understand the codebase, "
            "then call finalize_wiki_structure with your findings."
        )

    # Wrap the ainvoke call with traceable for LangSmith visibility
    @traceable(name=f"deep_agent_{owner}_{repo}", run_type="chain")
    async def _invoke_agent():
        return await agent.ainvoke(
            {
                "messages": [{
                    "role": "user",
                    "content": user_message
                }]
            },
            config={"run_name": f"structure_agent_{owner}_{repo}"}
        )

    try:
        logger.info(f"Starting Deep Agent for {owner}/{repo}, timeout={timeout}s, model={model}, mcp_tools={mcp_tools is not None}")

        result = await asyncio.wait_for(
            _invoke_agent(),
            timeout=timeout
        )

        logger.info(f"Deep Agent completed, result keys: {list(result.keys()) if result else 'None'}")

        # Retrieve captured structure
        structure = getattr(agent, "_structure_capture", {})

        logger.info(f"Captured structure keys: {list(structure.keys()) if structure else 'empty'}")

        if structure and structure.get("pages"):
            logger.info(f"Wiki structure generated with {len(structure['pages'])} pages")
            return structure

        logger.warning(f"Agent did not produce wiki structure, captured: {structure}")
        return None

    except asyncio.TimeoutError:
        logger.error(f"Structure agent timed out after {timeout}s")
        return None
    except Exception as e:
        logger.exception(f"Structure agent failed: {e}")
        return None
