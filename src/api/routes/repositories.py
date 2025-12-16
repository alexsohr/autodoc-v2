"""Repository API routes

This module implements the repository management API endpoints
based on the repository_api.yaml contract specification.
"""

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ...dependencies import get_repository_service
from ...models.config import LLMConfig, StorageConfig
from ...models.repository import (
    AnalysisStatus,
    Repository,
    RepositoryCreate,
    RepositoryList,
    RepositoryProvider,
    RepositoryResponse,
    RepositoryUpdate,
)
from ...models.user import User
from ...services.repository_service import RepositoryService
from ...utils.config_loader import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/repositories", tags=["repositories"])


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


@router.post(
    "/",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new repository",
    description="Register a code repository for analysis and documentation generation. The repository will be automatically analyzed and indexed for chat queries and wiki generation.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "github_public": {
                            "summary": "GitHub Public Repository",
                            "description": "Register a public GitHub repository",
                            "value": {
                                "url": "https://github.com/fastapi/fastapi",
                                "branch": "main",
                            },
                        },
                        "github_private": {
                            "summary": "GitHub Private Repository",
                            "description": "Register a private GitHub repository with specific provider",
                            "value": {
                                "url": "https://github.com/myorg/private-repo",
                                "provider": "github",
                                "branch": "develop",
                            },
                        },
                        "gitlab_repo": {
                            "summary": "GitLab Repository",
                            "description": "Register a GitLab repository",
                            "value": {
                                "url": "https://gitlab.com/myorg/my-project",
                                "provider": "gitlab",
                            },
                        },
                    }
                }
            }
        },
        "responses": {
            "201": {
                "description": "Repository successfully registered",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "provider": "github",
                            "url": "https://github.com/fastapi/fastapi",
                            "org": "fastapi",
                            "name": "fastapi",
                            "default_branch": "main",
                            "access_scope": "public",
                            "analysis_status": "pending",
                            "webhook_configured": False,
                            "subscribed_events": [],
                            "created_at": "2024-01-01T12:00:00Z",
                            "updated_at": "2024-01-01T12:00:00Z",
                        }
                    }
                },
            }
        },
    },
)
async def create_repository(
    repository_data: RepositoryCreate,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Register and analyze a repository"""
    try:
        # Create repository using service
        result = await service.create_repository(repository_data)

        if result["status"] != "success":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "DuplicateRepository":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error": "Repository already exists",
                        "message": result["error"],
                    },
                )
            elif error_type in [
                "InvalidURL",
                "ProviderDetectionFailed",
                "URLParsingFailed",
            ]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Invalid repository URL or parameters",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Repository creation failed",
                        "message": result["error"],
                    },
                )

        # Convert to response model
        repository_dict = result["repository"]
        repository_dict["id"] = UUID(repository_dict["id"])
        repository = Repository(**repository_dict)

        return RepositoryResponse(**repository.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Repository creation endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Repository creation failed",
            },
        )


@router.get(
    "/",
    response_model=RepositoryList,
    summary="List repositories",
    description="Get a paginated list of registered repositories with optional filtering by status and provider.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "List of repositories",
                "content": {
                    "application/json": {
                        "example": {
                            "repositories": [
                                {
                                    "id": "550e8400-e29b-41d4-a716-446655440000",
                                    "provider": "github",
                                    "url": "https://github.com/fastapi/fastapi",
                                    "org": "fastapi",
                                    "name": "fastapi",
                                    "default_branch": "main",
                                    "access_scope": "public",
                                    "analysis_status": "completed",
                                    "commit_sha": "abc123def456789...",
                                    "last_analyzed": "2024-01-01T12:30:00Z",
                                    "webhook_configured": True,
                                    "subscribed_events": ["push", "pull_request"],
                                    "created_at": "2024-01-01T12:00:00Z",
                                    "updated_at": "2024-01-01T12:30:00Z",
                                }
                            ],
                            "total": 1,
                            "limit": 50,
                            "offset": 0,
                        }
                    }
                },
            }
        }
    },
)
async def list_repositories(
    limit: int = Query(
        50, ge=1, le=100, description="Number of repositories to return"
    ),
    offset: int = Query(0, ge=0, description="Number of repositories to skip"),
    status: Optional[AnalysisStatus] = Query(
        None, description="Filter by analysis status"
    ),
    provider: Optional[RepositoryProvider] = Query(
        None, description="Filter by repository provider"
    ),
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """List repositories with pagination and filtering"""
    try:
        # Get repositories using service
        result = await service.list_repositories(
            limit=limit,
            offset=offset,
            status_filter=status.value if status else None,
            provider_filter=provider.value if provider else None,
        )

        if result["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Failed to list repositories",
                    "message": result["error"],
                },
            )

        # Convert to response format
        repositories = []
        for repo_dict in result["repositories"]:
            repo_dict["id"] = UUID(repo_dict["id"])
            repository = Repository(**repo_dict)
            repositories.append(RepositoryResponse(**repository.model_dump()))

        return RepositoryList(
            repositories=repositories, total=result["total"], limit=limit, offset=offset
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List repositories endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to list repositories",
            },
        )


@router.get(
    "/{repository_id}",
    response_model=RepositoryResponse,
    summary="Get repository details",
    description="Retrieve detailed information about a specific repository including analysis status and webhook configuration.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Repository details",
                "content": {
                    "application/json": {
                        "example": {
                            "id": "550e8400-e29b-41d4-a716-446655440000",
                            "provider": "github",
                            "url": "https://github.com/fastapi/fastapi",
                            "org": "fastapi",
                            "name": "fastapi",
                            "default_branch": "main",
                            "access_scope": "public",
                            "analysis_status": "completed",
                            "commit_sha": "abc123def456789012345678901234567890abcd",
                            "last_analyzed": "2024-01-01T12:30:00Z",
                            "webhook_configured": True,
                            "subscribed_events": ["push", "pull_request"],
                            "last_webhook_event": "2024-01-01T13:00:00Z",
                            "created_at": "2024-01-01T12:00:00Z",
                            "updated_at": "2024-01-01T12:30:00Z",
                        }
                    }
                },
            },
            "404": {
                "description": "Repository not found",
                "content": {
                    "application/json": {
                        "example": {
                            "error": "Repository not found",
                            "message": "Repository with ID 550e8400-e29b-41d4-a716-446655440000 does not exist",
                        }
                    }
                },
            },
        }
    },
)
async def get_repository(
    repository_id: UUID,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Get repository details"""
    try:
        # Get repository using service
        result = await service.get_repository(repository_id)

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get repository",
                        "message": result["error"],
                    },
                )

        # Convert to response model
        repository_dict = result["repository"]
        repository_dict["id"] = UUID(repository_dict["id"])
        repository = Repository(**repository_dict)

        return RepositoryResponse(**repository.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get repository endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get repository",
            },
        )


@router.delete("/{repository_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_repository(
    repository_id: UUID,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Remove repository and all associated data"""
    try:
        # Check permissions (admin only)
        if not current_user.is_admin:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "Insufficient permissions",
                    "message": "Admin access required",
                },
            )

        # Delete repository using service
        result = await service.delete_repository(repository_id)

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to delete repository",
                        "message": result["error"],
                    },
                )

        return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete repository endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to delete repository",
            },
        )


@router.post(
    "/{repository_id}/analyze",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger repository analysis",
    description="Start or restart analysis of a repository. This will analyze the codebase, generate embeddings, and prepare it for chat queries and wiki generation.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "basic_analysis": {
                            "summary": "Basic Analysis",
                            "description": "Trigger analysis with default settings",
                            "value": {},
                        },
                        "force_analysis": {
                            "summary": "Force Re-analysis",
                            "description": "Force re-analysis even if already completed",
                            "value": {"force": True},
                        },
                        "specific_branch": {
                            "summary": "Analyze Specific Branch",
                            "description": "Analyze a specific branch instead of default",
                            "value": {"branch": "develop", "force": False},
                        },
                    }
                }
            }
        },
        "responses": {
            "202": {
                "description": "Analysis started successfully",
                "content": {
                    "application/json": {
                        "example": {
                            "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                            "status": "processing",
                            "progress": 0,
                            "current_step": "Analysis started",
                            "estimated_completion": "2024-01-01T12:45:00Z",
                            "message": "Analysis started successfully",
                        }
                    }
                },
            }
        },
    },
)
async def trigger_repository_analysis(
    repository_id: UUID,
    analysis_request: Optional[dict] = None,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Trigger repository analysis"""
    try:
        # Parse analysis request
        force = False
        branch = None

        if analysis_request:
            force = analysis_request.get("force", False)
            branch = analysis_request.get("branch")

        # Trigger analysis using service
        result = await service.trigger_analysis(
            repository_id=repository_id, force=force, branch=branch
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
            elif error_type == "AnalysisInProgress":
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "error": "Analysis already in progress",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to trigger analysis",
                        "message": result["error"],
                    },
                )

        return {
            "repository_id": str(repository_id),
            "status": "processing",
            "progress": 0,
            "current_step": "Analysis started",
            "estimated_completion": result.get("estimated_completion"),
            "message": result.get("message", "Analysis started successfully"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trigger analysis endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to trigger analysis",
            },
        )


@router.get(
    "/{repository_id}/status",
    summary="Get analysis status",
    description="Check the current analysis status of a repository including progress, current step, and any error messages.",
    openapi_extra={
        "responses": {
            "200": {
                "description": "Analysis status information",
                "content": {
                    "application/json": {
                        "examples": {
                            "completed": {
                                "summary": "Analysis Completed",
                                "value": {
                                    "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                                    "status": "completed",
                                    "progress": 100,
                                    "current_step": "Analysis completed",
                                    "last_analyzed": "2024-01-01T12:30:00Z",
                                    "commit_sha": "abc123def456789012345678901234567890abcd",
                                    "documents_processed": 245,
                                    "embeddings_generated": 1250,
                                    "error_message": None,
                                },
                            },
                            "processing": {
                                "summary": "Analysis In Progress",
                                "value": {
                                    "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                                    "status": "processing",
                                    "progress": 45,
                                    "current_step": "Generating embeddings",
                                    "last_analyzed": None,
                                    "commit_sha": None,
                                    "documents_processed": 110,
                                    "embeddings_generated": 560,
                                    "error_message": None,
                                },
                            },
                            "failed": {
                                "summary": "Analysis Failed",
                                "value": {
                                    "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                                    "status": "failed",
                                    "progress": 25,
                                    "current_step": "Repository cloning",
                                    "last_analyzed": None,
                                    "commit_sha": None,
                                    "documents_processed": 0,
                                    "embeddings_generated": 0,
                                    "error_message": "Failed to clone repository: Permission denied",
                                },
                            },
                        }
                    }
                },
            }
        }
    },
)
async def get_analysis_status(
    repository_id: UUID,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Get analysis status for repository"""
    try:
        # Get analysis status using service
        result = await service.get_analysis_status(repository_id)

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get analysis status",
                        "message": result["error"],
                    },
                )

        return {
            "repository_id": result["repository_id"],
            "status": result["analysis_status"],
            "progress": result["progress"],
            "current_step": result["current_step"],
            "last_analyzed": result["last_analyzed"],
            "commit_sha": result["commit_sha"],
            "documents_processed": result["documents_processed"],
            "embeddings_generated": result["embeddings_generated"],
            "error_message": result["error_message"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get analysis status endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get analysis status",
            },
        )


@router.put("/{repository_id}/webhook")
async def configure_repository_webhook(
    repository_id: UUID,
    webhook_config: dict,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Configure repository webhook settings"""
    try:
        # Extract webhook configuration
        webhook_secret = webhook_config.get("webhook_secret")
        subscribed_events = webhook_config.get("subscribed_events", [])

        if not webhook_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Missing webhook secret",
                    "message": "webhook_secret is required",
                },
            )

        # Configure webhook using service
        result = await service.configure_webhook(
            repository_id=repository_id,
            webhook_secret=webhook_secret,
            subscribed_events=subscribed_events,
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
            elif error_type in ["InvalidWebhookSecret", "InvalidEvents"]:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Invalid webhook configuration",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Webhook configuration failed",
                        "message": result["error"],
                    },
                )

        return {
            "webhook_configured": result["webhook_configured"],
            "webhook_secret": result["webhook_secret"],
            "subscribed_events": result["subscribed_events"],
            "setup_instructions": result["setup_instructions"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configure webhook endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to configure webhook",
            },
        )


@router.get("/{repository_id}/webhook")
async def get_repository_webhook_config(
    repository_id: UUID,
    current_user: User = Depends(get_current_user),
    service: RepositoryService = Depends(get_repository_service),
):
    """Get repository webhook configuration and setup instructions"""
    try:
        # Get webhook config using service
        result = await service.get_webhook_config(repository_id)

        if result["status"] != "success":
            if result.get("error_type") == "NotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Failed to get webhook configuration",
                        "message": result["error"],
                    },
                )

        return {
            "webhook_configured": result["webhook_configured"],
            "webhook_secret": result["webhook_secret"],
            "subscribed_events": result["subscribed_events"],
            "setup_instructions": result["setup_instructions"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get webhook config endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to get webhook configuration",
            },
        )
