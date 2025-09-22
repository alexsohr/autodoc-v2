"""Webhook API routes

This module implements webhook endpoints for GitHub, Bitbucket, and other
Git providers based on the repository_api.yaml contract specification.
"""

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import JSONResponse

from ...services.repository_service import repository_service
from ...utils.config_loader import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])


@router.post(
    "/github",
    summary="GitHub webhook endpoint",
    description="Receive webhook events from GitHub to trigger automatic documentation updates when repository changes occur.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "push_event": {
                            "summary": "Push Event",
                            "description": "GitHub push event webhook payload",
                            "value": {
                                "ref": "refs/heads/main",
                                "repository": {
                                    "id": 123456789,
                                    "full_name": "myorg/myrepo",
                                    "html_url": "https://github.com/myorg/myrepo",
                                    "clone_url": "https://github.com/myorg/myrepo.git",
                                    "default_branch": "main"
                                },
                                "commits": [
                                    {
                                        "id": "abc123def456789012345678901234567890abcd",
                                        "message": "Update README.md",
                                        "author": {
                                            "name": "John Doe",
                                            "email": "john@example.com"
                                        }
                                    }
                                ]
                            }
                        },
                        "pull_request_event": {
                            "summary": "Pull Request Event",
                            "description": "GitHub pull request webhook payload",
                            "value": {
                                "action": "opened",
                                "pull_request": {
                                    "id": 123456,
                                    "number": 42,
                                    "title": "Add new feature",
                                    "state": "open"
                                },
                                "repository": {
                                    "full_name": "myorg/myrepo",
                                    "html_url": "https://github.com/myorg/myrepo"
                                }
                            }
                        }
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "Webhook processed successfully",
                "content": {
                    "application/json": {
                        "examples": {
                            "processed": {
                                "summary": "Event Processed",
                                "value": {
                                    "status": "processed",
                                    "message": "Repository analysis triggered",
                                    "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                                    "event_type": "push",
                                    "processing_time": 0.5
                                }
                            },
                            "ignored": {
                                "summary": "Event Ignored",
                                "value": {
                                    "status": "ignored",
                                    "message": "Event type not subscribed",
                                    "repository_id": "550e8400-e29b-41d4-a716-446655440000",
                                    "event_type": "issues",
                                    "processing_time": 0.1
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def github_webhook(
    request: Request,
    x_github_event: str = Header(..., alias="X-GitHub-Event", description="GitHub event type (e.g., push, pull_request)"),
    x_hub_signature_256: str = Header(..., alias="X-Hub-Signature-256", description="GitHub webhook signature for validation"),
    x_github_delivery: str = Header(..., alias="X-GitHub-Delivery", description="Unique delivery ID for this webhook"),
):
    """GitHub webhook endpoint"""
    try:
        # Get request body
        body = await request.body()

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid JSON payload",
                    "message": "Request body must be valid JSON",
                },
            )

        # Extract repository URL from payload
        repository_url = None
        if "repository" in payload:
            repo_info = payload["repository"]
            if "clone_url" in repo_info:
                repository_url = repo_info["clone_url"]
            elif "html_url" in repo_info:
                repository_url = repo_info["html_url"]
            elif "full_name" in repo_info:
                repository_url = f"https://github.com/{repo_info['full_name']}"

        if not repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid webhook payload",
                    "message": "Could not extract repository URL from payload",
                },
            )

        # Process webhook using service
        processing_start = logger.info(
            f"Processing GitHub webhook: {x_github_event} for {repository_url}"
        )

        result = await repository_service.process_webhook_event(
            repository_url=repository_url,
            event_type=x_github_event,
            payload=payload,
            signature=x_hub_signature_256,
        )

        # Map service result to HTTP response
        if result["status"] == "processed":
            return {
                "status": "processed",
                "message": result["message"],
                "repository_id": result.get("repository_id"),
                "event_type": x_github_event,
                "processing_time": 0.5,  # Mock processing time
            }

        elif result["status"] == "ignored":
            return {
                "status": "ignored",
                "message": result["message"],
                "repository_id": result.get("repository_id"),
                "event_type": x_github_event,
                "processing_time": 0.1,
            }

        elif result["status"] == "error":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "RepositoryNotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found in AutoDoc system",
                        "message": result["error"],
                    },
                )
            elif error_type == "InvalidSignature":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "error": "Invalid webhook signature",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Webhook processing failed",
                        "message": result["error"],
                    },
                )

        else:
            # Unknown status
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Unknown processing result",
                    "message": f"Unexpected status: {result['status']}",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GitHub webhook endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to process GitHub webhook",
            },
        )


@router.post("/bitbucket")
async def bitbucket_webhook(
    request: Request,
    x_event_key: str = Header(..., alias="X-Event-Key"),
    x_hook_uuid: str = Header(..., alias="X-Hook-UUID"),
):
    """Bitbucket webhook endpoint"""
    try:
        # Get request body
        body = await request.body()

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid JSON payload",
                    "message": "Request body must be valid JSON",
                },
            )

        # Extract repository URL from payload
        repository_url = None
        if "repository" in payload:
            repo_info = payload["repository"]
            if "links" in repo_info and "clone" in repo_info["links"]:
                clone_links = repo_info["links"]["clone"]
                for link in clone_links:
                    if link.get("name") == "https":
                        repository_url = link["href"]
                        break
            elif "full_name" in repo_info:
                repository_url = f"https://bitbucket.org/{repo_info['full_name']}"

        if not repository_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "error": "Invalid webhook payload",
                    "message": "Could not extract repository URL from payload",
                },
            )

        # Process webhook using service
        logger.info(f"Processing Bitbucket webhook: {x_event_key} for {repository_url}")

        result = await repository_service.process_webhook_event(
            repository_url=repository_url,
            event_type=x_event_key,
            payload=payload,
            signature=None,  # Bitbucket doesn't use signature validation by default
        )

        # Map service result to HTTP response
        if result["status"] == "processed":
            return {
                "status": "processed",
                "message": result["message"],
                "repository_id": result.get("repository_id"),
                "event_type": x_event_key,
                "processing_time": 0.5,  # Mock processing time
            }

        elif result["status"] == "ignored":
            return {
                "status": "ignored",
                "message": result["message"],
                "repository_id": result.get("repository_id"),
                "event_type": x_event_key,
                "processing_time": 0.1,
            }

        elif result["status"] == "error":
            error_type = result.get("error_type", "UnknownError")

            if error_type == "RepositoryNotFound":
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "Repository not found in AutoDoc system",
                        "message": result["error"],
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Webhook processing failed",
                        "message": result["error"],
                    },
                )

        else:
            # Unknown status
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "error": "Unknown processing result",
                    "message": f"Unexpected status: {result['status']}",
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bitbucket webhook endpoint failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error",
                "message": "Failed to process Bitbucket webhook",
            },
        )
