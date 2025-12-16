"""Wiki generation endpoints

This module implements the documentation/wiki API endpoints
based on the documentation_api.yaml contract specification.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import Response

from ...dependencies import get_document_service, get_wiki_service
from ...models.code_document import FileList
from ...models.wiki import PullRequestRequest, WikiPageDetail, WikiStructure
from ...models.user import User
from ...services.document_service import DocumentProcessingService
from ...services.wiki_service import WikiGenerationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repositories", tags=["wiki"])


# Dependency for getting current user (simplified for now)
async def get_current_user(token: str = Depends(lambda: "mock-token")) -> User:
    """Get current authenticated user"""
    # For now, return a mock user since auth middleware isn't implemented yet
    from uuid import uuid4

    return User(
        id=uuid4(),
        username="admin",
        email="admin@autodoc.dev",
        full_name="Admin User",
        is_admin=True,
    )


@router.get("/{repository_id}/wiki")
async def get_wiki_structure(
    repository_id: UUID,
    include_content: bool = Query(
        False, description="Include page content in response"
    ),
    section_id: Optional[str] = Query(None, description="Filter to specific section"),
    current_user: User = Depends(get_current_user),
    service: WikiGenerationService = Depends(get_wiki_service),
):
    """Get repository wiki structure"""
    try:
        # Get wiki structure using service
        result = await service.get_wiki_structure(
            repository_id=repository_id,
            include_content=include_content,
            section_filter=section_id,
        )

        if result["status"] != "success":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "WikiNotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Wiki not found",
                        "message": "No wiki available for this repository. Please generate wiki first.",
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get wiki structure",
                        "message": result["error"],
                    },
                )

        return result["wiki_structure"]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get wiki structure endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get wiki structure",
            },
        )


@router.get("/{repository_id}/wiki/pages/{page_id}")
async def get_wiki_page(
    repository_id: UUID,
    page_id: str,
    format: str = Query("json", regex="^(json|markdown)$"),
    current_user: User = Depends(get_current_user),
    service: WikiGenerationService = Depends(get_wiki_service),
):
    """Get specific wiki page"""
    try:
        # Get wiki page using service
        result = await service.get_wiki_page(
            repository_id=repository_id, page_id=page_id, format=format
        )

        if result["status"] != "success":
            error_type = result.get("error_type", "UnknownError")

            if error_type in ["WikiNotFound", "PageNotFound"]:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "Page not found", "message": result["error"]},
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get wiki page",
                        "message": result["error"],
                    },
                )

        if format == "markdown":
            return Response(content=result["content"], media_type="text/markdown")
        else:
            return result["page"]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get wiki page endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get wiki page",
            },
        )


@router.post("/{repository_id}/pull-request", status_code=status.HTTP_201_CREATED)
async def create_documentation_pr(
    repository_id: UUID,
    pr_request: Optional[PullRequestRequest] = None,
    current_user: User = Depends(get_current_user),
    service: WikiGenerationService = Depends(get_wiki_service),
):
    """Create documentation pull request"""
    try:
        # Parse PR request
        target_branch = None
        title = None
        description = None
        force_update = False

        if pr_request:
            target_branch = pr_request.target_branch
            title = pr_request.title
            description = pr_request.description
            force_update = pr_request.force_update

        # Create PR using service
        result = await service.create_documentation_pull_request(
            repository_id=repository_id,
            target_branch=target_branch,
            title=title,
            description=description,
            force_update=force_update,
        )

        if result["status"] != "success":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found",
                        "message": result["error"],
                    },
                )
            elif error_type == "WikiNotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Wiki not found",
                        "message": "No wiki available for this repository. Please generate wiki first.",
                    },
                )
            elif error_type == "NoChanges":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error": "No changes to commit",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to create pull request",
                        "message": result["error"],
                    },
                )

        return {
            "pull_request_url": result["pull_request_url"],
            "branch_name": result["branch_name"],
            "files_changed": result["files_changed"],
            "commit_sha": result["commit_sha"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create documentation PR endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to create documentation pull request",
            },
        )


@router.get("/{repository_id}/files")
async def get_repository_files(
    repository_id: UUID,
    language: Optional[str] = Query(None, description="Filter by programming language"),
    path_pattern: Optional[str] = Query(
        None, description="Filter by file path pattern"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Number of files to return"),
    offset: int = Query(0, ge=0, description="Number of files to skip"),
    current_user: User = Depends(get_current_user),
    service: DocumentProcessingService = Depends(get_document_service),
):
    """Get repository file list"""
    try:
        # Get repository files using service
        result = await service.get_repository_documents(
            repository_id=repository_id,
            language_filter=language,
            path_pattern=path_pattern,
            limit=limit,
            offset=offset,
        )

        if result["status"] != "success":
            if "not found" in result.get("error", "").lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found or not processed",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get repository files",
                        "message": result["error"],
                    },
                )

        return {
            "files": result["files"],
            "repository_id": result["repository_id"],
            "total": result["total"],
            "languages": result["languages"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get repository files endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get repository files",
            },
        )
