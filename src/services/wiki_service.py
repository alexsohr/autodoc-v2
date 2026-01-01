"""Wiki generation service for documentation creation

This module provides wiki generation services including structure creation,
page content generation, and documentation management.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..agents.wiki_agent import WikiGenerationAgent
from ..models.repository import Repository
from ..models.wiki import PageImportance, WikiPageDetail, WikiSection, WikiStructure
from ..repository.code_document_repository import CodeDocumentRepository
from ..repository.wiki_structure_repository import WikiStructureRepository
from ..tools.context_tool import ContextTool
from ..tools.llm_tool import LLMTool
from ..utils.config_loader import get_settings

logger = logging.getLogger(__name__)


class WikiGenerationService:
    """Wiki generation service for repository documentation

    Provides comprehensive wiki generation including structure creation,
    content generation, and documentation management.
    """

    def __init__(
        self,
        wiki_structure_repo: WikiStructureRepository,
        code_document_repo: CodeDocumentRepository,
        wiki_agent: WikiGenerationAgent,
        context_tool: ContextTool,
        llm_tool: LLMTool,
    ):
        """Initialize wiki generation service with dependency injection.
        
        Args:
            wiki_structure_repo: WikiStructureRepository instance (injected via DI).
            code_document_repo: CodeDocumentRepository instance (injected via DI).
            wiki_agent: WikiGenerationAgent instance (injected via DI).
            context_tool: ContextTool instance (injected via DI).
            llm_tool: LLMTool instance (injected via DI).
        """
        self.settings = get_settings()
        self._wiki_structure_repo = wiki_structure_repo
        self._code_document_repo = code_document_repo
        self._wiki_agent = wiki_agent
        self._context_tool = context_tool
        self._llm_tool = llm_tool

    async def generate_wiki(
        self, repository_id: UUID, force_regenerate: bool = False
    ) -> Dict[str, Any]:
        """Generate complete wiki for repository

        Args:
            repository_id: Repository UUID
            force_regenerate: Force regeneration even if wiki exists

        Returns:
            Dictionary with wiki generation results
        """
        try:
            # Check if repository exists and is analyzed
            from ..repository.repository_repository import RepositoryRepository
            from ..models.repository import Repository
            repo_repository = RepositoryRepository(Repository)
            repository = await repo_repository.find_one({"id": repository_id})

            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Check if repository is analyzed
            doc_count = await self._code_document_repo.count({"repository_id": str(repository_id)})
            if doc_count == 0:
                return {
                    "status": "error",
                    "error": "Repository not analyzed yet. Please run document analysis first.",
                    "error_type": "RepositoryNotAnalyzed",
                }

            # Generate wiki using wiki agent
            generation_result = await self._wiki_agent.generate_wiki(
                repository_id=str(repository_id), force_regenerate=force_regenerate
            )

            return {
                "status": generation_result["status"],
                "repository_id": str(repository_id),
                "wiki_structure": generation_result.get("wiki_structure"),
                "pages_generated": generation_result.get("pages_generated", 0),
                "force_regenerate": force_regenerate,
                "error_message": generation_result.get("error_message"),
                "generation_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Wiki generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def get_wiki_structure(
        self,
        repository_id: UUID,
        include_content: bool = False,
        section_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get wiki structure for repository

        Args:
            repository_id: Repository UUID
            include_content: Include page content in response
            section_filter: Optional section filter

        Returns:
            Dictionary with wiki structure
        """
        try:
            # Get wiki structure
            wiki = await self._wiki_structure_repo.find_one(
                {"repository_id": str(repository_id)}
            )
            wiki_data = self._wiki_structure_repo.serialize(wiki) if wiki else None

            if not wiki_data:
                return {
                    "status": "error",
                    "error": "Wiki not found for this repository",
                    "error_type": "WikiNotFound",
                }

            # Convert to WikiStructure model
            wiki_data["id"] = wiki_data.get("id", f"wiki_{repository_id}")

            # Convert sections to WikiSection objects with embedded pages
            sections = []
            for section_data in wiki_data.get("sections", []):
                # Convert pages within section
                section_pages = []
                for page_data in section_data.get("pages", []):
                    if not include_content:
                        page_data["content"] = ""  # Exclude content if not requested
                    page = WikiPageDetail(**page_data)
                    section_pages.append(page)

                section = WikiSection(
                    id=section_data["id"],
                    title=section_data["title"],
                    pages=section_pages,
                )
                sections.append(section)

            # Create complete wiki structure
            wiki_structure = WikiStructure(
                id=wiki_data["id"],
                repository_id=repository_id,
                title=wiki_data["title"],
                description=wiki_data["description"],
                sections=sections,
            )

            # Apply section filter if specified
            if section_filter:
                wiki_structure = self._filter_wiki_by_section(
                    wiki_structure, section_filter
                )

            return {
                "status": "success",
                "wiki_structure": wiki_structure.model_dump(),
                "repository_id": str(repository_id),
                "include_content": include_content,
                "section_filter": section_filter,
            }

        except Exception as e:
            logger.error(f"Get wiki structure failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def get_wiki_page(
        self, repository_id: UUID, page_id: str, format: str = "json"
    ) -> Dict[str, Any]:
        """Get specific wiki page

        Args:
            repository_id: Repository UUID
            page_id: Page identifier
            format: Response format (json or markdown)

        Returns:
            Dictionary with page data
        """
        try:
            # Get wiki structure
            wiki = await self._wiki_structure_repo.find_one(
                {"repository_id": str(repository_id)}
            )
            wiki_data = self._wiki_structure_repo.serialize(wiki) if wiki else None

            if not wiki_data:
                return {
                    "status": "error",
                    "error": "Wiki not found for this repository",
                    "error_type": "WikiNotFound",
                }

            # Find specific page within sections
            page_data = None
            for section in wiki_data.get("sections", []):
                for page in section.get("pages", []):
                    if page["id"] == page_id:
                        page_data = page
                        break
                if page_data:
                    break

            if not page_data:
                return {
                    "status": "error",
                    "error": f"Page '{page_id}' not found",
                    "error_type": "PageNotFound",
                }

            # Convert to WikiPageDetail
            wiki_page = WikiPageDetail(**page_data)

            if format == "markdown":
                return {
                    "status": "success",
                    "content": wiki_page.content,
                    "content_type": "text/markdown",
                    "page_id": page_id,
                }
            else:
                return {
                    "status": "success",
                    "page": wiki_page.model_dump(),
                    "page_id": page_id,
                    "repository_id": str(repository_id),
                }

        except Exception as e:
            logger.error(f"Get wiki page failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "page_id": page_id,
            }

    async def update_wiki_page_content(
        self, repository_id: UUID, page_id: str, new_content: str
    ) -> Dict[str, Any]:
        """Update wiki page content

        Args:
            repository_id: Repository UUID
            page_id: Page identifier
            new_content: New page content

        Returns:
            Dictionary with update result
        """
        try:
            # Update page content in wiki structure
            wiki = await self._wiki_structure_repo.find_one(
                {"repository_id": str(repository_id)}
            )
            wiki_data = self._wiki_structure_repo.serialize(wiki) if wiki else None

            if not wiki_data:
                return {
                    "status": "error",
                    "error": "Wiki not found",
                    "error_type": "WikiNotFound",
                }

            # Find and update page within sections
            sections = wiki_data.get("sections", [])
            page_found = False

            for section in sections:
                for page in section.get("pages", []):
                    if page["id"] == page_id:
                        page["content"] = new_content
                        page_found = True
                        break
                if page_found:
                    break

            if not page_found:
                return {
                    "status": "error",
                    "error": f"Page '{page_id}' not found",
                    "error_type": "PageNotFound",
                }

            # Update wiki structure in database
            wiki_data["sections"] = sections
            wiki_data["updated_at"] = datetime.now(timezone.utc)

            success = await wiki_structure_repo.update_one(
                "wiki_structures",
                {"repository_id": str(repository_id)},
                {"sections": sections, "updated_at": wiki_data["updated_at"]},
            )

            if success:
                return {
                    "status": "success",
                    "page_id": page_id,
                    "message": "Page content updated successfully",
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to update page content",
                    "error_type": "UpdateFailed",
                }

        except Exception as e:
            logger.error(f"Update wiki page failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "page_id": page_id,
            }

    async def create_documentation_pull_request(
        self,
        repository_id: UUID,
        target_branch: Optional[str] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        force_update: bool = False,
    ) -> Dict[str, Any]:
        """Create pull request with updated documentation

        Args:
            repository_id: Repository UUID
            target_branch: Target branch for PR
            title: Custom PR title
            description: Custom PR description
            force_update: Force update even if no changes

        Returns:
            Dictionary with PR creation result
        """
        try:
            # Get repository
            from ..repository.repository_repository import RepositoryRepository
            from ..models.repository import Repository
            repo_repository = RepositoryRepository(Repository)
            repository = await repo_repository.find_one({"id": repository_id})
            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Get wiki structure
            wiki = await self._wiki_structure_repo.find_one(
                {"repository_id": str(repository_id)}
            )
            wiki_data = self._wiki_structure_repo.serialize(wiki) if wiki else None
            if not wiki_data:
                return {
                    "status": "error",
                    "error": "Wiki not found for this repository",
                    "error_type": "WikiNotFound",
                }

            # Generate documentation files
            doc_files = await self._generate_documentation_files(wiki_data)

            if not doc_files and not force_update:
                return {
                    "status": "error",
                    "error": "No documentation changes to commit",
                    "error_type": "NoChanges",
                }

            # Create PR (this would integrate with Git provider APIs)
            pr_result = await self._create_git_pull_request(
                repository=repository,
                doc_files=doc_files,
                target_branch=target_branch or repository.default_branch,
                title=title,
                description=description,
            )

            return pr_result

        except Exception as e:
            logger.error(f"Documentation PR creation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    def _filter_wiki_by_section(
        self, wiki_structure: WikiStructure, section_id: str
    ) -> WikiStructure:
        """Filter wiki structure by specific section

        Args:
            wiki_structure: Complete wiki structure
            section_id: Section ID to filter by

        Returns:
            Filtered wiki structure
        """
        try:
            # Find the requested section
            target_section = wiki_structure.get_section(section_id)

            if not target_section:
                return wiki_structure  # Return original if section not found

            # Create filtered structure with only the target section
            filtered_structure = WikiStructure(
                id=wiki_structure.id,
                repository_id=wiki_structure.repository_id,
                title=wiki_structure.title,
                description=wiki_structure.description,
                sections=[target_section],
            )

            return filtered_structure

        except Exception as e:
            logger.error(f"Wiki filtering failed: {e}")
            return wiki_structure

    async def _generate_documentation_files(
        self, wiki_data: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate documentation files from wiki structure

        Args:
            wiki_data: Wiki structure data

        Returns:
            List of documentation files
        """
        try:
            doc_files = []

            # Generate main README
            readme_content = await self._generate_main_readme(wiki_data)
            if readme_content:
                doc_files.append({"path": "docs/README.md", "content": readme_content})

            # Generate individual page files from sections
            for section in wiki_data.get("sections", []):
                for page_data in section.get("pages", []):
                    if page_data.get("content"):
                        file_path = f"docs/{page_data['id']}.md"
                        doc_files.append(
                            {"path": file_path, "content": page_data["content"]}
                        )

            # Generate navigation index
            nav_content = self._generate_navigation_index(wiki_data)
            if nav_content:
                doc_files.append({"path": "docs/NAVIGATION.md", "content": nav_content})

            return doc_files

        except Exception as e:
            logger.error(f"Documentation file generation failed: {e}")
            return []

    async def _generate_main_readme(self, wiki_data: Dict[str, Any]) -> Optional[str]:
        """Generate main README file from wiki structure

        Args:
            wiki_data: Wiki structure data

        Returns:
            README content or None
        """
        try:
            # Create README template
            readme_content = f"""# {wiki_data.get('title', 'Repository Documentation')}

{wiki_data.get('description', 'Generated documentation for this repository.')}

## Documentation Structure

This documentation is organized into the following sections:

"""

            # Add section links (pages are now embedded in sections)
            for section_data in wiki_data.get("sections", []):
                section_title = section_data.get("title", section_data["id"])
                readme_content += f"### {section_title}\n\n"

                for page_data in section_data.get("pages", []):
                    page_id = page_data.get("id") if isinstance(page_data, dict) else page_data
                    page_title = page_data.get("title", page_id) if isinstance(page_data, dict) else page_id
                    page_desc = page_data.get("description", "") if isinstance(page_data, dict) else ""
                    readme_content += (
                        f"- [{page_title}](docs/{page_id}.md) - {page_desc}\n"
                    )

                readme_content += "\n"

            # Add generation timestamp
            readme_content += f"\n---\n*Documentation generated on {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC*\n"

            return readme_content

        except Exception as e:
            logger.error(f"Main README generation failed: {e}")
            return None

    def _generate_navigation_index(self, wiki_data: Dict[str, Any]) -> Optional[str]:
        """Generate navigation index file

        Args:
            wiki_data: Wiki structure data

        Returns:
            Navigation content or None
        """
        try:
            nav_content = f"""# Documentation Navigation

## {wiki_data.get('title', 'Repository Documentation')}

{wiki_data.get('description', '')}

## Table of Contents

"""

            # Generate navigation for all sections
            for section_data in wiki_data.get("sections", []):
                nav_content += self._generate_section_nav(section_data, level=1)

            return nav_content

        except Exception as e:
            logger.error(f"Navigation index generation failed: {e}")
            return None

    def _generate_section_nav(
        self, section_data: Dict[str, Any], level: int = 1
    ) -> str:
        """Generate navigation for a section

        Args:
            section_data: Section data with embedded pages
            level: Nesting level

        Returns:
            Navigation content for section
        """
        indent = "  " * (level - 1)
        nav_content = f"{indent}- **{section_data.get('title', section_data['id'])}**\n"

        # Add pages (now embedded in section)
        for page_data in section_data.get("pages", []):
            page_id = page_data.get("id") if isinstance(page_data, dict) else page_data
            page_title = page_data.get("title", page_id) if isinstance(page_data, dict) else page_id
            nav_content += f"{indent}  - [{page_title}](docs/{page_id}.md)\n"

        return nav_content

    async def _create_git_pull_request(
        self,
        repository: Repository,
        doc_files: List[Dict[str, str]],
        target_branch: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create Git pull request with documentation updates

        Args:
            repository: Repository object
            doc_files: List of documentation files
            target_branch: Target branch
            title: PR title
            description: PR description

        Returns:
            Dictionary with PR creation result
        """
        try:
            # For now, return a mock PR response
            # In a real implementation, this would integrate with GitHub/Bitbucket/GitLab APIs

            branch_name = f"autodoc/update-documentation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            pr_title = (
                title
                or f"📚 Update documentation for {repository.org}/{repository.name}"
            )
            pr_description = (
                description
                or f"""## AutoDoc Documentation Update

This pull request contains updated documentation generated by AutoDoc v2.

### Changes:
- Updated wiki structure and content
- Generated {len(doc_files)} documentation files
- Synchronized with latest codebase changes

### Files Changed:
{chr(10).join(f'- {file["path"]}' for file in doc_files)}

---
*This PR was automatically generated by AutoDoc v2*
"""
            )

            # Mock PR URL (would be real in production)
            pr_url = f"https://{repository.provider.value}.com/{repository.org}/{repository.name}/pull/123"

            return {
                "status": "success",
                "pull_request_url": pr_url,
                "branch_name": branch_name,
                "files_changed": [file["path"] for file in doc_files],
                "commit_sha": "abc123def456",  # Mock commit SHA
                "title": pr_title,
                "description": pr_description,
            }

        except Exception as e:
            logger.error(f"Git PR creation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def get_wiki_generation_status(self, repository_id: UUID) -> Dict[str, Any]:
        """Get wiki generation status

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with wiki generation status
        """
        try:
            # Use wiki agent to get status
            status_result = await self._wiki_agent.get_wiki_generation_status(
                str(repository_id)
            )

            # Add repository ID to result
            status_result["repository_id"] = str(repository_id)

            return status_result

        except Exception as e:
            logger.error(f"Get wiki generation status failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def regenerate_wiki_page(
        self,
        repository_id: UUID,
        page_id: str,
        additional_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Regenerate content for a specific wiki page

        Args:
            repository_id: Repository UUID
            page_id: Page identifier
            additional_context: Additional context for generation

        Returns:
            Dictionary with regeneration result
        """
        try:
            # Get wiki structure
            wiki = await self._wiki_structure_repo.find_one(
                {"repository_id": str(repository_id)}
            )
            wiki_data = self._wiki_structure_repo.serialize(wiki) if wiki else None
            if not wiki_data:
                return {
                    "status": "error",
                    "error": "Wiki not found",
                    "error_type": "WikiNotFound",
                }

            # Find page within sections
            page_data = None
            for section in wiki_data.get("sections", []):
                for page in section.get("pages", []):
                    if page["id"] == page_id:
                        page_data = page
                        break
                if page_data:
                    break

            if not page_data:
                return {
                    "status": "error",
                    "error": f"Page '{page_id}' not found",
                    "error_type": "PageNotFound",
                }

            # Regenerate page content using wiki agent
            new_content = await self._wiki_agent._generate_page_content(
                page_data, str(repository_id)
            )

            if not new_content:
                return {
                    "status": "error",
                    "error": "Failed to generate page content",
                    "error_type": "GenerationFailed",
                }

            # Update page content
            update_result = await self.update_wiki_page_content(
                repository_id=repository_id, page_id=page_id, new_content=new_content
            )

            return {
                "status": update_result["status"],
                "page_id": page_id,
                "content_regenerated": update_result["status"] == "success",
                "new_content_length": len(new_content),
                "error_message": update_result.get("error"),
                "regeneration_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Wiki page regeneration failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "page_id": page_id,
            }


# Deprecated: Module-level singleton removed
# Use get_wiki_service() from src.dependencies with FastAPI's Depends() instead
# from ..dependencies import get_wiki_service
# wiki_service = get_wiki_service()  # REMOVED - use dependency injection
