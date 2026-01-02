"""Shared tools and utilities for wiki page generation.

This module contains the finalization tool used by
both the unified wiki agent and its page-generator subagent.
"""

from typing import Any, Dict, List

import structlog
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ============================================================================
# Page Finalization Tool
# ============================================================================


class FinalizePageInput(BaseModel):
    """Input schema for the finalize_page tool."""

    title: str = Field(description="Page title - REQUIRED")
    content: str = Field(
        description="Full markdown content including details block, headings, diagrams, tables, and citations - REQUIRED"
    )
    source_files: List[str] = Field(
        description="List of source files used (minimum 5) - REQUIRED, must provide at least 5 file paths"
    )


def create_page_finalize_tool(captured_content: Dict[str, Any]) -> StructuredTool:
    """Create finalize tool that captures page content.

    Args:
        captured_content: Dict that will be populated with the finalized content

    Returns:
        StructuredTool for the agent to call when done
    """

    def finalize_page(title: str, content: str, source_files: List[str]) -> str:
        """Submit the final page content. Call this when documentation is complete.

        REQUIRED PARAMETERS:
        - title: str - The page title
        - content: str - Complete markdown documentation
        - source_files: List[str] - At least 5 source file paths you referenced

        Example call:
        finalize_page(
            title="Getting Started",
            content="<details>...",
            source_files=["src/main.py", "src/config.py", "src/utils.py", "README.md", "setup.py"]
        )
        """
        # Validate source_files is provided and has enough entries
        if not source_files:
            return (
                "ERROR: The 'source_files' parameter is REQUIRED but was empty or not provided.\n\n"
                "You MUST call finalize_page with ALL THREE parameters:\n"
                "  1. title: str - The page title\n"
                "  2. content: str - Your complete markdown documentation\n"
                "  3. source_files: List[str] - A list of at least 5 source files you referenced\n\n"
                "Example:\n"
                "finalize_page(\n"
                '    title="Page Title",\n'
                '    content="<details>\\n<summary>Relevant source files</summary>...",\n'
                '    source_files=["path/to/file1.py", "path/to/file2.py", "path/to/file3.py", '
                '"path/to/file4.py", "path/to/file5.py"]\n'
                ")\n\n"
                "Please call finalize_page again with the source_files parameter included."
            )

        if len(source_files) < 5:
            return (
                f"ERROR: Must cite at least 5 source files. You provided {len(source_files)}: {source_files}\n\n"
                f"Explore more files in the repository and call finalize_page again with at least 5 source files.\n"
                f"The source_files parameter must be a list of file paths you referenced in your documentation."
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
            "Submit the final wiki page content. REQUIRES ALL THREE PARAMETERS:\n"
            "- title (str): Page title\n"
            "- content (str): Complete markdown documentation\n"
            "- source_files (List[str]): At least 5 source file paths you referenced\n\n"
            "Example: finalize_page(title='...', content='...', source_files=['file1.py', 'file2.py', ...])"
        ),
        args_schema=FinalizePageInput,
    )


def get_page_system_prompt(
    page_title: str,
    page_description: str,
    file_hints: List[str],
    clone_path: str,
    repo_name: str,
    repo_description: str,
    use_mcp_tools: bool = True,
) -> str:
    """Generate the system prompt for the page subagent.

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
   read_text_file(path="{clone_path}\\src\\main.py", head=50)
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

IMPORTANT: Always use absolute paths starting with "{clone_path}\\" when accessing files.
"""

    return f'''You are an expert technical writer and software architect.
Your task is to generate a comprehensive and accurate technical wiki page in Markdown format about a specific feature, system, or module within a given software project.

## CRITICAL: MANDATORY TOOL CALL

You MUST complete this task by calling the `finalize_page` tool at the end. Your work is NOT complete until you call this tool.
The task will FAIL if you do not call `finalize_page` with:
- title: The page title
- content: Complete markdown documentation
- source_files: List of at least 5 source files you referenced

DO NOT just output text. You MUST call the `finalize_page` tool to submit your work.

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

Remember, do not provide any acknowledgements, disclaimers, apologies, or any other preface before the `<details>` block. JUST START with the `<details>` block.

The main title of the page should be a H1 Markdown heading: `# {page_title}`.

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

## FINAL STEP - MANDATORY

When you have finished writing the documentation, you MUST call the `finalize_page` tool:

```
finalize_page(
    title="{page_title}",
    content="<your complete markdown content starting with <details> block>",
    source_files=["file1.py", "file2.py", "file3.py", "file4.py", "file5.py"]
)
```

Requirements for the tool call:
- title: Must be exactly "{page_title}"
- content: The complete markdown documentation (starting with the <details> block)
- source_files: List of at least 5 source file paths you referenced

YOUR TASK IS NOT COMPLETE UNTIL YOU CALL `finalize_page`. Do not just output text - you MUST submit via the tool.

IMPORTANT: Generate the content in English language. Ground every claim in the provided source files.
'''
