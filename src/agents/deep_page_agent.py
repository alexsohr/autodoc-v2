"""Deep agent for generating wiki page content with autonomous exploration."""

import asyncio
import os
from typing import Any, Dict, List, Optional

import structlog
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class PageSection(BaseModel):
    """A section within the wiki page."""

    heading: str = Field(description="Section heading (H2 or H3)")
    content: str = Field(description="Markdown content for this section")


class PageContent(BaseModel):
    """Structured output for a wiki page."""

    title: str = Field(description="Page title")
    content: str = Field(description="Full markdown content of the page")
    source_files: List[str] = Field(
        default_factory=list,
        description="Source files used to generate this page (minimum 5)",
    )


class FinalizePageInput(BaseModel):
    """Input schema for the finalize_page tool."""

    title: str = Field(description="Page title")
    content: str = Field(
        description="Full markdown content including details block, headings, diagrams, tables, and citations"
    )
    source_files: List[str] = Field(description="List of source files used (minimum 5)")


def create_page_finalize_tool(captured_content: Dict[str, Any]) -> StructuredTool:
    """Create finalize tool that captures page content.

    Args:
        captured_content: Dict that will be populated with the finalized content

    Returns:
        StructuredTool for the agent to call when done
    """

    def finalize_page(title: str, content: str, source_files: List[str]) -> str:
        """Submit the final page content. Call this when documentation is complete."""
        if len(source_files) < 5:
            return (
                f"Error: Must cite at least 5 source files. You provided {len(source_files)}. "
                f"Explore more files and try again."
            )

        captured_content["title"] = title
        captured_content["content"] = content
        captured_content["source_files"] = source_files

        logger.info(
            "Page content finalized",
            title=title,
            content_length=len(content),
            num_source_files=len(source_files),
        )
        return "Page content finalized successfully."

    return StructuredTool.from_function(
        func=finalize_page,
        name="finalize_page",
        description=(
            "Submit the final wiki page content. Call this when you have completed "
            "the documentation with at least 5 source files cited."
        ),
        args_schema=FinalizePageInput,
    )


def get_page_prompt(
    page_title: str,
    page_description: str,
    file_hints: List[str],
    clone_path: str,
    repo_name: str,
    repo_description: str,
    use_mcp_tools: bool = True,
) -> str:
    """Generate the system prompt for the page agent.

    Args:
        page_title: Title of the wiki page to generate
        page_description: Description of what the page should cover
        file_hints: Initial file paths that are likely relevant
        clone_path: Path to the cloned repository
        repo_name: Name of the repository
        repo_description: Description of the repository
        use_mcp_tools: Whether MCP filesystem tools are available

    Returns:
        System prompt string
    """
    file_hints_str = (
        "\n".join(f"- {f}" for f in file_hints)
        if file_hints
        else "- No specific files provided - you must explore to find relevant ones"
    )

    tool_instructions = ""
    if use_mcp_tools:
        tool_instructions = f"""
## Available Filesystem Tools

You have access to the following tools to explore the codebase:
- `read_text_file(path, head=N)`: Read file contents. Use head=50 to efficiently read only the first 50 lines.
- `search_files(path, pattern)`: Search for files matching a pattern.
- `list_directory(path)`: List directory contents.
- `directory_tree(path)`: Get directory tree structure.

All paths must be absolute, starting with: {clone_path}

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
   - Only read full files for small config files (< 50 lines anyway)

3. **Use search_files for Discovery**
   Find related files by pattern:
   ```
   search_files(path="{clone_path}", pattern="**/*controller*.py")
   search_files(path="{clone_path}", pattern="**/test_*.py")
   ```

IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files.
"""

    return f'''You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about a specific feature, system, or module within a given software project.

## Repository Context
- **Repository:** {repo_name}
- **Description:** {repo_description}
- **Clone Path:** {clone_path}

## Your Assignment
You will be given:
1. The "[WIKI_PAGE_TOPIC]" for the page you need to create: **{page_title}**
2. Page Description: {page_description}
3. A list of "[RELEVANT_SOURCE_FILES]" as starting hints - you MUST explore beyond these to find at least 5 relevant files.

## Starting File Hints
{file_hints_str}
{tool_instructions}
## CRITICAL STARTING INSTRUCTION

The very first thing on the page MUST be a `<details>` block listing ALL the `[RELEVANT_SOURCE_FILES]` you used to generate the content. There MUST be AT LEAST 5 source files listed - if fewer were provided, you MUST find additional related files to include.

Format it exactly like this:
```
<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- path/to/file1.py
- path/to/file2.py
- path/to/file3.py
- path/to/file4.py
- path/to/file5.py
</details>
```

Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the `<details>` block. JUST START with the `<details>` block.

Immediately after the `<details>` block, the main title of the page should be a H1 Markdown heading: `# {page_title}`.

## Content Requirements

Based ONLY on the content of the `[RELEVANT_SOURCE_FILES]`:

1. **Introduction:** Start with a concise introduction (1-2 paragraphs) explaining the purpose, scope, and high-level overview of "{page_title}" within the context of the overall project. If relevant, link to other potential wiki pages using the format `[Link Text](/wiki/page-slug)`.

2. **Detailed Sections:** Break down "{page_title}" into logical sections using H2 (`##`) and H3 (`###`) Markdown headings. For each section:
   - Explain the architecture, components, data flow, or logic relevant to the section's focus
   - Identify key functions, classes, data structures, API endpoints, or configuration elements

3. **Mermaid Diagrams:**
   - EXTENSIVELY use Mermaid diagrams (`flowchart TD`, `sequenceDiagram`, `classDiagram`, `erDiagram`, `graph TD`) to visually represent architectures, flows, relationships, and schemas
   - Ensure diagrams are accurate and directly derived from the source files
   - Provide a brief explanation before or after each diagram
   - CRITICAL diagram rules:
     * Use "graph TD" (top-down) directive - NEVER use "graph LR" (left-right)
     * NEVER use parentheses or slashes in node text
     * Maximum node width: 3-4 words
     * For sequence diagrams: define ALL participants first, use correct arrow types (->> for request, -->> for response, -x for failed)

4. **Tables:** Use Markdown tables to summarize:
   - Key features or components and their descriptions
   - API endpoint parameters, types, and descriptions
   - Configuration options, their types, and default values
   - Data model fields, types, constraints, and descriptions

5. **Code Snippets:** Include short, relevant code snippets directly from the source files with appropriate language identifiers.

6. **Source Citations (EXTREMELY IMPORTANT):**
   - For EVERY piece of significant information, you MUST cite the specific source file(s) and relevant line numbers
   - Place citations at the end of paragraphs, under diagrams/tables, or after code snippets
   - Use the exact format: `Sources: [filename.ext:start_line-end_line]()` for ranges, `Sources: [filename.ext:line_number]()` for single lines
   - Multiple files: `Sources: [file1.ext:1-10](), [file2.ext:5]()`
   - You MUST cite AT LEAST 5 different source files throughout the wiki page

7. **Technical Accuracy:** All information must be derived SOLELY from the source files. Do not infer, invent, or use external knowledge. If information is not present, do not include it.

8. **Clarity and Conciseness:** Use clear, professional technical language suitable for developers.

9. **Conclusion/Summary:** End with a brief summary paragraph reiterating key aspects covered.

## When Complete

Call `finalize_page` with:
- title: "{page_title}"
- content: The complete markdown content (starting with the <details> block)
- source_files: List of all source files you referenced (minimum 5)

IMPORTANT: Generate the content in English language. Ground every claim in the provided source files.
'''


def create_page_agent(
    clone_path: str,
    page_title: str,
    page_description: str,
    file_hints: List[str],
    repo_name: str,
    repo_description: str,
    model: Optional[str] = None,
    mcp_tools: Optional[List[Any]] = None,
) -> Any:
    """Create a Deep Agent configured for wiki page generation.

    Args:
        clone_path: Path to the cloned repository
        page_title: Title of the page to generate
        page_description: Description of page content
        file_hints: Initial file paths as starting points
        repo_name: Repository name
        repo_description: Repository description
        model: Optional model override (default: uses deepagents default)
        mcp_tools: Optional MCP filesystem tools

    Returns:
        Configured Deep Agent
    """
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    # This will be populated by the finalize tool
    captured_content: Dict[str, Any] = {}

    # Create the finalize tool with capture closure
    finalize_tool = create_page_finalize_tool(captured_content)

    # Generate system prompt
    system_prompt = get_page_prompt(
        page_title=page_title,
        page_description=page_description,
        file_hints=file_hints,
        clone_path=clone_path,
        repo_name=repo_name,
        repo_description=repo_description,
        use_mcp_tools=mcp_tools is not None,
    )

    # Build agent kwargs
    agent_kwargs = {
        "system_prompt": system_prompt,
    }

    # Use MCP tools if provided, otherwise fall back to FilesystemBackend
    if mcp_tools:
        logger.info(
            "Using MCP filesystem tools for page generation",
            num_tools=len(mcp_tools),
            page_title=page_title,
        )
        agent_kwargs["tools"] = list(mcp_tools) + [finalize_tool]
    else:
        logger.warning(
            "MCP filesystem not available, using FilesystemBackend. "
            "Note: Windows absolute paths may not work correctly.",
            page_title=page_title,
        )
        backend = FilesystemBackend(root_dir=clone_path)
        agent_kwargs["backend"] = backend
        agent_kwargs["tools"] = [finalize_tool]

    if model:
        from langchain.chat_models import init_chat_model

        # Add openai: prefix if no provider specified
        if ":" not in model:
            model_string = f"openai:{model}"
            provider = "openai"
        else:
            model_string = model
            provider = model.split(":")[0]

        model_kwargs = {"temperature": 0}

        # Pass API key from environment
        if provider == "openai" and os.environ.get("OPENAI_API_KEY"):
            model_kwargs["api_key"] = os.environ["OPENAI_API_KEY"]
        elif provider == "google" and os.environ.get("GOOGLE_API_KEY"):
            model_kwargs["api_key"] = os.environ["GOOGLE_API_KEY"]
        elif provider == "anthropic" and os.environ.get("ANTHROPIC_API_KEY"):
            model_kwargs["api_key"] = os.environ["ANTHROPIC_API_KEY"]

        logger.info("Initializing page agent with model", model=model_string, page_title=page_title)
        agent_kwargs["model"] = init_chat_model(model_string, **model_kwargs)

    agent = create_deep_agent(**agent_kwargs)

    # Attach the capture dict to the agent for retrieval
    agent._page_capture = captured_content

    return agent


async def run_page_agent(
    clone_path: str,
    page_title: str,
    page_description: str,
    file_hints: List[str],
    repo_name: str,
    repo_description: str,
    timeout: float = 120.0,
    model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Run the page agent and return the captured page content.

    Args:
        clone_path: Path to the cloned repository
        page_title: Title of the page to generate
        page_description: Description of page content
        file_hints: Initial file paths as starting points
        repo_name: Repository name
        repo_description: Repository description
        timeout: Maximum execution time in seconds (default 120s for pages)
        model: Optional model override

    Returns:
        Captured page content dict or None if failed
    """
    from langsmith import traceable

    from src.services.mcp_filesystem_client import get_mcp_filesystem_client

    # Try to get MCP filesystem tools
    mcp_tools = None
    try:
        mcp_client = get_mcp_filesystem_client()
        if mcp_client.is_initialized:
            mcp_tools = list(mcp_client._tools.values())
            logger.info(
                "Retrieved MCP filesystem tools for page agent",
                num_tools=len(mcp_tools),
                page_title=page_title,
            )
        else:
            logger.warning("MCP filesystem client not initialized for page agent")
    except Exception as e:
        logger.warning(
            "Could not get MCP filesystem tools for page agent",
            error=str(e),
            page_title=page_title,
        )

    agent = create_page_agent(
        clone_path=clone_path,
        page_title=page_title,
        page_description=page_description,
        file_hints=file_hints,
        repo_name=repo_name,
        repo_description=repo_description,
        model=model,
        mcp_tools=mcp_tools,
    )

    # Build the user message with clone path for MCP tools
    if mcp_tools:
        user_message = (
            f"Generate comprehensive wiki documentation for: {page_title}\n\n"
            f"The repository is located at: {clone_path}\n\n"
            f"CONTEXT-EFFICIENT EXPLORATION:\n"
            f"- Use `read_text_file` with `head=50` to read only file headers first\n"
            f"- Example: read_text_file(path='{clone_path}/src/main.py', head=50)\n"
            f"- Only read more if 50 lines aren't enough to understand the file's purpose\n"
            f"- Use `search_files` to discover related files by pattern\n\n"
            f"You MUST use at least 5 source files and cite them with line numbers. "
            f"When done, call finalize_page with your complete documentation."
        )
    else:
        user_message = (
            f"Generate comprehensive wiki documentation for: {page_title}\n\n"
            f"Start by exploring the hinted files, then search for related code. "
            f"You MUST use at least 5 source files and cite them with line numbers. "
            f"When done, call finalize_page with your complete documentation."
        )

    # Wrap with traceable for LangSmith visibility
    safe_title = page_title.replace(" ", "_").replace("/", "_")[:30]

    @traceable(name=f"page_agent_{safe_title}", run_type="chain")
    async def _invoke_agent():
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
            config={"run_name": f"page_agent_{safe_title}"},
        )

    try:
        logger.info(
            "Starting page agent",
            page_title=page_title,
            timeout=timeout,
            model=model,
            has_mcp_tools=mcp_tools is not None,
            num_file_hints=len(file_hints),
        )

        result = await asyncio.wait_for(_invoke_agent(), timeout=timeout)

        logger.info(
            "Page agent completed",
            page_title=page_title,
            result_keys=list(result.keys()) if result else None,
        )

        # Retrieve captured content
        content = getattr(agent, "_page_capture", {})

        if content and content.get("content"):
            logger.info(
                "Page content generated",
                page_title=page_title,
                content_length=len(content.get("content", "")),
                num_source_files=len(content.get("source_files", [])),
            )
            return content

        logger.warning(
            "Page agent did not produce content",
            page_title=page_title,
            captured=content,
        )
        return None

    except asyncio.TimeoutError:
        logger.error("Page agent timed out", page_title=page_title, timeout=timeout)
        return None
    except Exception as e:
        logger.exception("Page agent failed", page_title=page_title, error=str(e))
        return None
