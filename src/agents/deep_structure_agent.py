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


def create_finalize_tool(capture_dict: Dict[str, Any]):
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
            "Wiki structure finalized",
            title=title,
            page_count=len(pages)
        )
        return f"Wiki structure captured successfully with {len(pages)} pages."

    return finalize_wiki_structure


def get_structure_prompt(
    owner: str,
    repo: str,
    file_tree: str,
    readme_content: str
) -> str:
    """Generate the system prompt for the structure agent.

    Args:
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README file content

    Returns:
        Formatted system prompt
    """
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

## Exploration Strategy
1. Start by examining the file tree and README provided above
2. Use `ls` to explore directory contents in detail
3. Use `glob` to find specific file patterns (e.g., `**/*.py`, `**/test_*.py`)
4. Read key files to understand the codebase:
   - Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
   - Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
   - Core modules and their purposes
5. Use `grep` to find patterns like class definitions, API routes, exports
6. Use `write_todos` to track your exploration progress

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
) -> Any:
    """Create a Deep Agent configured for wiki structure generation.

    Args:
        clone_path: Path to the cloned repository
        owner: Repository owner/organization
        repo: Repository name
        file_tree: ASCII file tree representation
        readme_content: README content
        model: Optional model override (default: uses deepagents default)

    Returns:
        Configured Deep Agent
    """
    # This will be populated by the finalize tool
    captured_structure: Dict[str, Any] = {}

    # Create the finalize tool with capture closure
    finalize_tool = create_finalize_tool(captured_structure)

    # Create backend pointing to cloned repo
    backend = FilesystemBackend(root_dir=clone_path)

    # Generate system prompt
    system_prompt = get_structure_prompt(owner, repo, file_tree, readme_content)

    # Create the agent
    agent_kwargs = {
        "backend": backend,
        "tools": [finalize_tool],
        "system_prompt": system_prompt,
    }

    if model:
        from langchain_openai import ChatOpenAI
        agent_kwargs["model"] = ChatOpenAI(model=model, temperature=0)

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
    agent = create_structure_agent(
        clone_path=clone_path,
        owner=owner,
        repo=repo,
        file_tree=file_tree,
        readme_content=readme_content,
        model=model,
    )

    try:
        await asyncio.wait_for(
            agent.ainvoke({
                "messages": [{
                    "role": "user",
                    "content": "Analyze this repository and create a comprehensive wiki structure. "
                              "Explore the codebase thoroughly, then call finalize_wiki_structure with your findings."
                }]
            }),
            timeout=timeout
        )

        # Retrieve captured structure
        structure = getattr(agent, "_structure_capture", {})

        if structure and structure.get("pages"):
            return structure

        logger.warning("Agent did not produce wiki structure")
        return None

    except asyncio.TimeoutError:
        logger.error("Structure agent timed out", timeout=timeout)
        return None
    except Exception as e:
        logger.error("Structure agent failed", error=str(e))
        return None
