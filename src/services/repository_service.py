"""Repository service for CRUD operations and webhook configuration

This module provides comprehensive repository management services including
CRUD operations, webhook configuration, and analysis workflow coordination.
"""

import asyncio
import structlog
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from uuid import UUID, uuid4

from ..agents.workflow import WorkflowOrchestrator, WorkflowType
from ..models.repository import (
    AccessScope,
    AnalysisStatus,
    Repository,
    RepositoryCreate,
    RepositoryProvider,
    RepositoryUpdate,
)
from ..repository.code_document_repository import CodeDocumentRepository
from ..repository.repository_repository import RepositoryRepository
from ..utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


class RepositoryService:
    """Repository service for CRUD operations and workflow management

    Provides comprehensive repository management including creation, analysis,
    webhook configuration, and status tracking.
    """

    def __init__(
        self,
        repository_repo: RepositoryRepository,
        code_document_repo: CodeDocumentRepository,
        workflow_orchestrator: Optional[WorkflowOrchestrator] = None,
    ):
        """Initialize repository service with repository dependency injection."""
        self.settings = get_settings()
        self.supported_providers = {
            "github.com": RepositoryProvider.GITHUB,
            "bitbucket.org": RepositoryProvider.BITBUCKET,
            "gitlab.com": RepositoryProvider.GITLAB,
        }
        self._repository_repo = repository_repo
        self._code_document_repo = code_document_repo
        self._workflow_orchestrator = workflow_orchestrator

    async def create_repository(
        self, repository_data: RepositoryCreate
    ) -> Dict[str, Any]:
        """Create a new repository

        Args:
            repository_data: Repository creation data

        Returns:
            Dictionary with creation result
        """
        try:
            # Validate repository URL
            validation_result = self._validate_repository_url(repository_data.url)
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "error": validation_result["error"],
                    "error_type": "InvalidURL",
                }

            # Auto-detect provider if not specified
            provider = repository_data.provider
            if not provider:
                provider = self._detect_provider_from_url(repository_data.url)
                if not provider:
                    return {
                        "status": "error",
                        "error": "Could not detect repository provider",
                        "error_type": "ProviderDetectionFailed",
                    }

            # Extract org and name from URL
            org, name = self._extract_org_and_name(repository_data.url)
            if not org or not name:
                return {
                    "status": "error",
                    "error": "Could not extract organization and repository name from URL",
                    "error_type": "URLParsingFailed",
                }

            # Check if repository already exists
            existing_repo = await self._repository_repo.get_by_url(repository_data.url)
            if existing_repo:
                return {
                    "status": "error",
                    "error": "Repository already exists",
                    "error_type": "DuplicateRepository",
                }

            # Determine access scope (simplified - assume public for now)
            access_scope = AccessScope.PUBLIC

            # Create repository
            repository = Repository(
                provider=provider,
                url=repository_data.url,
                org=org,
                name=name,
                default_branch=repository_data.branch or "main",
                access_scope=access_scope,
                analysis_status=AnalysisStatus.PENDING,
            )

            # Store in database using repository pattern
            created_repo = await self._repository_repo.create(repository)

            # Note: Analysis is NOT auto-triggered on creation.
            # Use the /analyze endpoint to explicitly start analysis.

            return {
                "status": "success",
                "repository": created_repo.model_dump(),
                "message": "Repository created successfully",
            }

        except Exception as e:
            logger.error(f"Repository creation failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def get_repository(self, repository_id: UUID) -> Dict[str, Any]:
        """Get repository by ID

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with repository data or error
        """
        try:
            repository = await self._repository_repo.get(repository_id)

            if repository:
                return {"status": "success", "repository": repository.model_dump()}
            else:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

        except Exception as e:
            logger.error(f"Get repository failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def list_repositories(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
        provider_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List repositories with pagination and filtering

        Args:
            limit: Maximum number of repositories to return
            offset: Number of repositories to skip
            status_filter: Optional analysis status filter
            provider_filter: Optional provider filter

        Returns:
            Dictionary with repository list and metadata
        """
        try:
            # Get repositories using repository pattern
            repositories, total_count = await self._repository_repo.list_paginated(
                limit=limit, offset=offset, status_filter=status_filter
            )

            return {
                "status": "success",
                "repositories": [repo.model_dump() for repo in repositories],
                "total": total_count,
                "limit": limit,
                "offset": offset,
                "filters": {"status": status_filter, "provider": provider_filter},
            }

        except Exception as e:
            logger.error(f"List repositories failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repositories": [],
            }

    async def update_repository(
        self, repository_id: UUID, updates: RepositoryUpdate
    ) -> Dict[str, Any]:
        """Update repository

        Args:
            repository_id: Repository UUID
            updates: Repository update data

        Returns:
            Dictionary with update result
        """
        try:
            # Convert updates to dict and remove None values
            update_dict = {
                k: v for k, v in updates.model_dump().items() if v is not None
            }

            if not update_dict:
                return {
                    "status": "error",
                    "error": "No updates provided",
                    "error_type": "InvalidInput",
                }

            # Update repository using repository pattern
            success = await self._repository_repo.update(repository_id, update_dict)

            if success:
                # Get updated repository
                updated_repo = await self._repository_repo.get(repository_id)
                return {
                    "status": "success",
                    "repository": updated_repo.model_dump() if updated_repo else None,
                    "message": "Repository updated successfully",
                }
            else:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

        except Exception as e:
            logger.error(f"Repository update failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def delete_repository(self, repository_id: UUID) -> Dict[str, Any]:
        """Delete repository and all associated data

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with deletion result
        """
        try:
            # Check if repository exists
            repository = await self._repository_repo.get(repository_id)
            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Delete repository and all related data
            success = await self._repository_repo.delete(repository_id)

            if success:
                return {
                    "status": "success",
                    "message": f"Repository {repository.org}/{repository.name} deleted successfully",
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to delete repository",
                    "error_type": "DeletionFailed",
                }

        except Exception as e:
            logger.error(f"Repository deletion failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def trigger_analysis(
        self, repository_id: UUID, force: bool = False, branch: Optional[str] = None
    ) -> Dict[str, Any]:
        """Trigger repository analysis

        Args:
            repository_id: Repository UUID
            force: Force re-analysis even if already completed
            branch: Specific branch to analyze

        Returns:
            Dictionary with analysis trigger result
        """
        try:
            repository = await self._repository_repo.get(repository_id)

            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Check if analysis is already in progress
            # Note: analysis_status may be string or enum depending on use_enum_values setting
            current_status = (
                repository.analysis_status.value
                if hasattr(repository.analysis_status, "value")
                else repository.analysis_status
            )

            if current_status == AnalysisStatus.PROCESSING.value and not force:
                return {
                    "status": "error",
                    "error": "Analysis already in progress",
                    "error_type": "AnalysisInProgress",
                }

            # Check if analysis is completed and force is not set
            if current_status == AnalysisStatus.COMPLETED.value and not force:
                return {
                    "status": "success",
                    "message": "Repository already analyzed",
                    "analysis_status": current_status,
                    "last_analyzed": (
                        repository.last_analyzed.isoformat()
                        if repository.last_analyzed
                        else None
                    ),
                }

            # Update status to processing
            await self._repository_repo.update(
                repository_id,
                {
                    "analysis_status": AnalysisStatus.PROCESSING.value,
                    "updated_at": datetime.now(timezone.utc),
                },
            )

            # Start analysis workflow
            workflow_result = await self._workflow_orchestrator.execute_workflow(
                workflow_type=WorkflowType.FULL_ANALYSIS,
                repository_id=str(repository_id),
                repository_url=repository.url,
                branch=branch or repository.default_branch,
                force_update=force,
            )

            return {
                "status": "success",
                "message": "Analysis started",
                "analysis_status": "processing",
                "workflow_id": workflow_result.get("workflow_id"),
                "estimated_completion": self._estimate_completion_time(),
            }

        except Exception as e:
            logger.error(f"Analysis trigger failed: {e}")

            # Reset status on error
            try:
                await self._repository_repo.update(
                    repository_id,
                    {
                        "analysis_status": AnalysisStatus.FAILED.value,
                        "error_message": str(e),
                    },
                )
            except Exception:
                pass

            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def get_analysis_status(self, repository_id: UUID) -> Dict[str, Any]:
        """Get repository analysis status

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with analysis status
        """
        try:
            repository = await self._repository_repo.get(repository_id)

            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            doc_count = await self._code_document_repo.count({"repository_id": str(repository_id)})
            embedding_count = await self._code_document_repo.count(
                {"repository_id": str(repository_id), "embedding": {"$exists": True}}
            )

            progress = 0.0
            if repository.analysis_status == AnalysisStatus.COMPLETED:
                progress = 100.0
            elif repository.analysis_status == AnalysisStatus.PROCESSING:
                if doc_count > 0:
                    progress = min(90.0, (embedding_count / doc_count) * 80.0 + 10.0)
                else:
                    progress = 20.0
            elif repository.analysis_status == AnalysisStatus.FAILED:
                progress = 0.0

            # Handle both enum and string types for analysis_status
            # (Repository model uses use_enum_values=True, so values may be strings)
            status_value = getattr(repository.analysis_status, 'value', repository.analysis_status)

            return {
                "status": "success",
                "repository_id": str(repository_id),
                "analysis_status": status_value,
                "progress": progress,
                "current_step": self._get_current_step(repository.analysis_status),
                "last_analyzed": (
                    repository.last_analyzed.isoformat()
                    if repository.last_analyzed
                    else None
                ),
                "commit_sha": repository.commit_sha,
                "documents_processed": doc_count,
                "embeddings_generated": embedding_count,
                "error_message": getattr(repository, "error_message", None),
            }

        except Exception as e:
            logger.error(f"Get analysis status failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def configure_webhook(
        self, repository_id: UUID, webhook_secret: str, subscribed_events: List[str]
    ) -> Dict[str, Any]:
        """Configure webhook settings for repository

        Args:
            repository_id: Repository UUID
            webhook_secret: Secret for webhook validation
            subscribed_events: List of events to subscribe to

        Returns:
            Dictionary with webhook configuration result
        """
        try:
            # Validate webhook secret
            if not webhook_secret or len(webhook_secret) < 16:
                return {
                    "status": "error",
                    "error": "Webhook secret must be at least 16 characters",
                    "error_type": "InvalidWebhookSecret",
                }

            # Validate events
            valid_events = self._get_valid_events_for_provider()
            invalid_events = [
                event for event in subscribed_events if event not in valid_events
            ]
            if invalid_events:
                return {
                    "status": "error",
                    "error": f"Invalid events: {invalid_events}",
                    "error_type": "InvalidEvents",
                }

            # Update repository webhook configuration
            updates = {
                "webhook_configured": True,
                "webhook_secret": webhook_secret,
                "subscribed_events": subscribed_events,
                "updated_at": datetime.now(timezone.utc),
            }

            success = await self._repository_repo.update(repository_id, updates)

            if not success:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Get updated repository for response
            repository = await self._repository_repo.get(repository_id)

            # Generate setup instructions
            setup_instructions = self._generate_webhook_setup_instructions(repository)

            return {
                "status": "success",
                "webhook_configured": True,
                "webhook_secret": webhook_secret,
                "subscribed_events": subscribed_events,
                "setup_instructions": setup_instructions,
                "message": "Webhook configured successfully",
            }

        except Exception as e:
            logger.error(f"Webhook configuration failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def get_webhook_config(self, repository_id: UUID) -> Dict[str, Any]:
        """Get webhook configuration for repository

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with webhook configuration
        """
        try:
            repository = await self._repository_repo.get(repository_id)

            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found",
                    "error_type": "NotFound",
                }

            # Generate setup instructions
            setup_instructions = self._generate_webhook_setup_instructions(repository)

            return {
                "status": "success",
                "webhook_configured": repository.webhook_configured,
                "webhook_secret": (
                    repository.webhook_secret if repository.webhook_configured else None
                ),
                "subscribed_events": repository.subscribed_events,
                "setup_instructions": setup_instructions,
            }

        except Exception as e:
            logger.error(f"Get webhook config failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def find_repository_by_url(self, repository_url: str) -> Optional[Repository]:
        """Find repository by URL

        Args:
            repository_url: Repository URL

        Returns:
            Repository object or None
        """
        try:
            return await self._repository_repo.get_by_url(repository_url)
        except Exception as e:
            logger.error(f"Find repository by URL failed: {e}")
            return None

    async def process_webhook_event(
        self,
        repository_url: str,
        event_type: str,
        payload: Dict[str, Any],
        signature: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process webhook event for repository

        Args:
            repository_url: Repository URL from webhook
            event_type: Type of webhook event
            payload: Webhook payload
            signature: Webhook signature for validation

        Returns:
            Dictionary with processing result
        """
        try:
            # Find repository
            repository = await self.find_repository_by_url(repository_url)
            if not repository:
                return {
                    "status": "error",
                    "error": "Repository not found in AutoDoc system",
                    "error_type": "RepositoryNotFound",
                }

            # Check if webhook is configured
            if not repository.webhook_configured:
                return {
                    "status": "ignored",
                    "message": "Webhook not configured for this repository",
                    "repository_id": str(repository.id),
                }

            # Validate signature if provided
            if signature and repository.webhook_secret:
                if not self._validate_webhook_signature(
                    payload, signature, repository.webhook_secret
                ):
                    return {
                        "status": "error",
                        "error": "Invalid webhook signature",
                        "error_type": "InvalidSignature",
                    }

            # Check if event is subscribed
            if not repository.is_webhook_event_subscribed(event_type):
                return {
                    "status": "ignored",
                    "message": f"Event {event_type} not subscribed",
                    "repository_id": str(repository.id),
                    "event_type": event_type,
                }

            # Update webhook event timestamp
            await self._repository_repo.update(
                repository.id, {"last_webhook_event": datetime.now(timezone.utc)}
            )

            # Trigger incremental analysis
            workflow_result = await self._workflow_orchestrator.execute_workflow(
                workflow_type=WorkflowType.INCREMENTAL_UPDATE,
                repository_id=str(repository.id),
                repository_url=repository.url,
                additional_params={
                    "webhook_event": event_type,
                    "webhook_payload": payload,
                },
            )

            return {
                "status": "processed",
                "message": "Webhook event processed successfully",
                "repository_id": str(repository.id),
                "event_type": event_type,
                "workflow_status": workflow_result.get("status"),
                "processing_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Webhook event processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "event_type": event_type,
            }

    def _validate_repository_url(self, url: str) -> Dict[str, Any]:
        """Validate repository URL format

        Args:
            url: Repository URL

        Returns:
            Dictionary with validation result
        """
        try:
            if not url or not url.strip():
                return {"valid": False, "error": "URL cannot be empty"}

            # Parse URL
            parsed = urlparse(url)

            if not parsed.scheme or not parsed.netloc:
                return {"valid": False, "error": "Invalid URL format"}

            # Check if it's a supported git hosting provider
            if parsed.netloc not in self.supported_providers:
                return {
                    "valid": False,
                    "error": f"Unsupported provider: {parsed.netloc}",
                }

            # Check URL pattern
            path_parts = parsed.path.strip("/").split("/")
            if len(path_parts) < 2:
                return {
                    "valid": False,
                    "error": "URL must include organization and repository name",
                }

            return {"valid": True}

        except Exception as e:
            return {"valid": False, "error": f"URL validation error: {str(e)}"}

    def _detect_provider_from_url(self, url: str) -> Optional[RepositoryProvider]:
        """Detect repository provider from URL

        Args:
            url: Repository URL

        Returns:
            Repository provider or None
        """
        try:
            parsed = urlparse(url)
            return self.supported_providers.get(parsed.netloc)
        except Exception:
            return None

    def _extract_org_and_name(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract organization and repository name from URL

        Args:
            url: Repository URL

        Returns:
            Tuple of (organization, repository_name) or (None, None)
        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.strip("/").split("/")

            if len(path_parts) >= 2:
                org = path_parts[0]
                repo_name = path_parts[1].replace(".git", "")
                return org, repo_name

            return None, None

        except Exception:
            return None, None

    def _get_valid_events_for_provider(self) -> List[str]:
        """Get valid webhook events for all providers

        Returns:
            List of valid webhook events
        """
        return [
            "push",
            "pull_request",
            "merge",
            "pullrequest:fulfilled",
            "repo:push",
            "repository:push",
            "merge_request",
        ]

    def _generate_webhook_setup_instructions(
        self, repository: Repository
    ) -> Dict[str, Any]:
        """Generate webhook setup instructions for repository

        Args:
            repository: Repository object

        Returns:
            Dictionary with setup instructions
        """
        base_url = "https://your-autodoc-instance.com"  # TODO: Get from config

        instructions = {}

        if repository.provider == RepositoryProvider.GITHUB:
            instructions["github"] = {
                "webhook_url": f"{base_url}/webhooks/github",
                "content_type": "application/json",
                "events": ["push", "pull_request"],
                "instructions": "Go to Settings > Webhooks > Add webhook in your GitHub repository",
            }

        elif repository.provider == RepositoryProvider.BITBUCKET:
            instructions["bitbucket"] = {
                "webhook_url": f"{base_url}/webhooks/bitbucket",
                "events": ["repo:push", "pullrequest:fulfilled"],
                "instructions": "Go to Settings > Webhooks > Add webhook in your Bitbucket repository",
            }

        elif repository.provider == RepositoryProvider.GITLAB:
            instructions["gitlab"] = {
                "webhook_url": f"{base_url}/webhooks/gitlab",
                "events": ["push", "merge_request"],
                "instructions": "Go to Settings > Webhooks > Add webhook in your GitLab repository",
            }

        return instructions

    def _validate_webhook_signature(
        self, payload: Dict[str, Any], signature: str, secret: str
    ) -> bool:
        """Validate webhook signature

        Args:
            payload: Webhook payload
            signature: Webhook signature
            secret: Webhook secret

        Returns:
            True if signature is valid
        """
        try:
            import hashlib
            import hmac
            import json

            # Convert payload to JSON string
            payload_str = json.dumps(payload, separators=(",", ":"))
            payload_bytes = payload_str.encode("utf-8")

            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode("utf-8"), payload_bytes, hashlib.sha256
            ).hexdigest()

            # GitHub format: sha256=<signature>
            if signature.startswith("sha256="):
                provided_signature = signature[7:]
            else:
                provided_signature = signature

            # Compare signatures
            return hmac.compare_digest(expected_signature, provided_signature)

        except Exception as e:
            logger.error(f"Webhook signature validation failed: {e}")
            return False

    def _get_current_step(self, status: AnalysisStatus) -> str:
        """Get current step description from analysis status

        Args:
            status: Analysis status

        Returns:
            Current step description
        """
        step_descriptions = {
            AnalysisStatus.PENDING: "Waiting to start analysis",
            AnalysisStatus.PROCESSING: "Analyzing repository content",
            AnalysisStatus.COMPLETED: "Analysis completed successfully",
            AnalysisStatus.FAILED: "Analysis failed",
        }

        return step_descriptions.get(status, "Unknown status")

    def _estimate_completion_time(self) -> str:
        """Estimate analysis completion time

        Returns:
            ISO timestamp of estimated completion
        """
        # Simple estimation: 15 minutes from now
        estimated_time = datetime.now(timezone.utc) + timedelta(minutes=15)
        return estimated_time.isoformat()

    async def _trigger_analysis(
        self, repository_id: str, repository_url: str, branch: Optional[str]
    ) -> None:
        """Trigger analysis workflow (async background task)

        Args:
            repository_id: Repository ID
            repository_url: Repository URL
            branch: Branch to analyze
        """
        try:
            await self._workflow_orchestrator.execute_workflow(
                workflow_type=WorkflowType.FULL_ANALYSIS,
                repository_id=repository_id,
                repository_url=repository_url,
                branch=branch,
            )
        except Exception as e:
            logger.error(f"Background analysis failed: {e}")

    async def get_repository_statistics(self) -> Dict[str, Any]:
        """Get repository statistics

        Returns:
            Dictionary with repository statistics
        """
        try:
            stats = await self._repository_repo.get_statistics()

            recent_activity = [
                {
                    "id": repo.id,
                    "name": f"{repo.org}/{repo.name}",
                    "status": repo.analysis_status,
                    "updated_at": repo.updated_at,
                }
                for repo in stats["recent"]
            ]

            return {
                "status": "success",
                "total_repositories": stats["total"],
                "status_breakdown": stats["by_status"],
                "provider_breakdown": stats["by_provider"],
                "recent_activity": recent_activity,
            }

        except Exception as e:
            logger.error(f"Get repository statistics failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

