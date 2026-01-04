"""Webhook signature validation utilities

This module provides secure webhook signature validation for different
Git providers including GitHub, Bitbucket, and GitLab.
"""

import hashlib
import hmac
import json
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class WebhookProvider(str, Enum):
    """Supported webhook providers"""

    GITHUB = "github"
    BITBUCKET = "bitbucket"
    GITLAB = "gitlab"


class WebhookValidator:
    """Webhook signature validator for secure webhook processing

    Provides signature validation methods for different Git providers
    to ensure webhook authenticity and prevent tampering.
    """

    def __init__(self):
        """Initialize webhook validator"""
        self.supported_algorithms = {"sha1", "sha256", "sha512"}

    def validate_github_signature(
        self, payload: Union[bytes, str, Dict[str, Any]], signature: str, secret: str
    ) -> bool:
        """Validate GitHub webhook signature

        GitHub uses HMAC-SHA256 with the format: sha256=<signature>

        Args:
            payload: Webhook payload (bytes, string, or dict)
            signature: GitHub signature header (X-Hub-Signature-256)
            secret: Webhook secret configured in GitHub

        Returns:
            True if signature is valid, False otherwise
        """
        try:
            # Convert payload to bytes
            if isinstance(payload, dict):
                payload_bytes = json.dumps(payload, separators=(",", ":")).encode(
                    "utf-8"
                )
            elif isinstance(payload, str):
                payload_bytes = payload.encode("utf-8")
            else:
                payload_bytes = payload

            # Extract signature from header
            if not signature.startswith("sha256="):
                logger.warning("GitHub signature does not start with 'sha256='")
                return False

            provided_signature = signature[7:]  # Remove 'sha256=' prefix

            # Calculate expected signature
            expected_signature = hmac.new(
                secret.encode("utf-8"), payload_bytes, hashlib.sha256
            ).hexdigest()

            # Use constant-time comparison to prevent timing attacks
            return hmac.compare_digest(expected_signature, provided_signature)

        except Exception as e:
            logger.error(f"GitHub signature validation failed: {e}")
            return False

    def validate_bitbucket_signature(
        self,
        payload: Union[bytes, str, Dict[str, Any]],
        signature: Optional[str],
        secret: Optional[str],
    ) -> bool:
        """Validate Bitbucket webhook signature

        Bitbucket webhook signatures are optional and use different formats
        depending on configuration. This method handles the most common cases.

        Args:
            payload: Webhook payload
            signature: Bitbucket signature header (if present)
            secret: Webhook secret (if configured)

        Returns:
            True if signature is valid or not required, False if invalid
        """
        try:
            # If no signature or secret provided, assume validation passed
            # (Bitbucket webhooks can be configured without signatures)
            if not signature or not secret:
                logger.info(
                    "Bitbucket webhook validation skipped (no signature/secret)"
                )
                return True

            # Convert payload to bytes
            if isinstance(payload, dict):
                payload_bytes = json.dumps(payload, separators=(",", ":")).encode(
                    "utf-8"
                )
            elif isinstance(payload, str):
                payload_bytes = payload.encode("utf-8")
            else:
                payload_bytes = payload

            # Bitbucket may use different signature formats
            # Try common formats

            # Format 1: Plain HMAC-SHA256
            try:
                expected_signature = hmac.new(
                    secret.encode("utf-8"), payload_bytes, hashlib.sha256
                ).hexdigest()

                if hmac.compare_digest(expected_signature, signature):
                    return True
            except Exception:
                pass

            # Format 2: With sha256= prefix
            if signature.startswith("sha256="):
                provided_signature = signature[7:]
                expected_signature = hmac.new(
                    secret.encode("utf-8"), payload_bytes, hashlib.sha256
                ).hexdigest()

                if hmac.compare_digest(expected_signature, provided_signature):
                    return True

            logger.warning("Bitbucket signature validation failed")
            return False

        except Exception as e:
            logger.error(f"Bitbucket signature validation error: {e}")
            return False

    def validate_gitlab_signature(
        self, payload: Union[bytes, str, Dict[str, Any]], signature: str, secret: str
    ) -> bool:
        """Validate GitLab webhook signature

        GitLab uses a simple token-based validation via X-Gitlab-Token header.

        Args:
            payload: Webhook payload (not used for GitLab)
            signature: GitLab token header (X-Gitlab-Token)
            secret: Webhook secret token configured in GitLab

        Returns:
            True if token matches, False otherwise
        """
        try:
            # GitLab uses simple token comparison
            return hmac.compare_digest(secret, signature)

        except Exception as e:
            logger.error(f"GitLab signature validation failed: {e}")
            return False

    def validate_webhook_signature(
        self,
        provider: WebhookProvider,
        payload: Union[bytes, str, Dict[str, Any]],
        signature: str,
        secret: str,
        additional_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Validate webhook signature for any supported provider

        Args:
            provider: Webhook provider
            payload: Webhook payload
            signature: Signature header value
            secret: Webhook secret
            additional_headers: Additional headers for validation

        Returns:
            Dictionary with validation result
        """
        try:
            if provider == WebhookProvider.GITHUB:
                is_valid = self.validate_github_signature(payload, signature, secret)
            elif provider == WebhookProvider.BITBUCKET:
                is_valid = self.validate_bitbucket_signature(payload, signature, secret)
            elif provider == WebhookProvider.GITLAB:
                is_valid = self.validate_gitlab_signature(payload, signature, secret)
            else:
                return {
                    "valid": False,
                    "error": f"Unsupported provider: {provider}",
                    "error_type": "UnsupportedProvider",
                }

            return {
                "valid": is_valid,
                "provider": provider.value,
                "validation_method": f"{provider.value}_signature",
            }

        except Exception as e:
            logger.error(f"Webhook signature validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "provider": provider.value,
            }

    def extract_signature_from_headers(
        self, headers: Dict[str, str], provider: WebhookProvider
    ) -> Optional[str]:
        """Extract signature from webhook headers

        Args:
            headers: Request headers
            provider: Webhook provider

        Returns:
            Signature string or None if not found
        """
        try:
            # Normalize header keys to lowercase
            normalized_headers = {k.lower(): v for k, v in headers.items()}

            if provider == WebhookProvider.GITHUB:
                # GitHub uses X-Hub-Signature-256
                return normalized_headers.get("x-hub-signature-256")

            elif provider == WebhookProvider.BITBUCKET:
                # Bitbucket may use various signature headers
                # Check common header names
                signature_headers = [
                    "x-hub-signature",
                    "x-bitbucket-signature",
                    "x-signature",
                ]

                for header in signature_headers:
                    if header in normalized_headers:
                        return normalized_headers[header]

                return None

            elif provider == WebhookProvider.GITLAB:
                # GitLab uses X-Gitlab-Token
                return normalized_headers.get("x-gitlab-token")

            else:
                logger.warning(f"Unknown provider for signature extraction: {provider}")
                return None

        except Exception as e:
            logger.error(f"Signature extraction failed: {e}")
            return None

    def get_event_type_from_headers(
        self, headers: Dict[str, str], provider: WebhookProvider
    ) -> Optional[str]:
        """Extract event type from webhook headers

        Args:
            headers: Request headers
            provider: Webhook provider

        Returns:
            Event type string or None if not found
        """
        try:
            # Normalize header keys to lowercase
            normalized_headers = {k.lower(): v for k, v in headers.items()}

            if provider == WebhookProvider.GITHUB:
                return normalized_headers.get("x-github-event")

            elif provider == WebhookProvider.BITBUCKET:
                return normalized_headers.get("x-event-key")

            elif provider == WebhookProvider.GITLAB:
                return normalized_headers.get("x-gitlab-event")

            else:
                logger.warning(
                    f"Unknown provider for event type extraction: {provider}"
                )
                return None

        except Exception as e:
            logger.error(f"Event type extraction failed: {e}")
            return None

    def validate_webhook_headers(
        self, headers: Dict[str, str], provider: WebhookProvider
    ) -> Dict[str, Any]:
        """Validate required webhook headers for provider

        Args:
            headers: Request headers
            provider: Webhook provider

        Returns:
            Dictionary with validation result
        """
        try:
            normalized_headers = {k.lower(): v for k, v in headers.items()}
            missing_headers = []

            if provider == WebhookProvider.GITHUB:
                required_headers = ["x-github-event", "x-github-delivery"]
                for header in required_headers:
                    if header not in normalized_headers:
                        missing_headers.append(header)

            elif provider == WebhookProvider.BITBUCKET:
                required_headers = ["x-event-key", "x-hook-uuid"]
                for header in required_headers:
                    if header not in normalized_headers:
                        missing_headers.append(header)

            elif provider == WebhookProvider.GITLAB:
                required_headers = ["x-gitlab-event"]
                for header in required_headers:
                    if header not in normalized_headers:
                        missing_headers.append(header)

            if missing_headers:
                return {
                    "valid": False,
                    "error": f"Missing required headers: {missing_headers}",
                    "error_type": "MissingHeaders",
                    "missing_headers": missing_headers,
                }

            return {"valid": True, "provider": provider.value}

        except Exception as e:
            logger.error(f"Header validation failed: {e}")
            return {"valid": False, "error": str(e), "error_type": type(e).__name__}

    def detect_provider_from_headers(
        self, headers: Dict[str, str]
    ) -> Optional[WebhookProvider]:
        """Detect webhook provider from headers

        Args:
            headers: Request headers

        Returns:
            Detected provider or None
        """
        try:
            normalized_headers = {k.lower(): v for k, v in headers.items()}

            # Check for GitHub headers
            if "x-github-event" in normalized_headers:
                return WebhookProvider.GITHUB

            # Check for Bitbucket headers
            if "x-event-key" in normalized_headers:
                return WebhookProvider.BITBUCKET

            # Check for GitLab headers
            if "x-gitlab-event" in normalized_headers:
                return WebhookProvider.GITLAB

            return None

        except Exception as e:
            logger.error(f"Provider detection failed: {e}")
            return None

    def get_supported_events(self, provider: WebhookProvider) -> List[str]:
        """Get list of supported events for provider

        Args:
            provider: Webhook provider

        Returns:
            List of supported event types
        """
        event_mappings = {
            WebhookProvider.GITHUB: [
                "push",
                "pull_request",
                "pull_request_review",
                "issues",
                "issue_comment",
                "release",
                "create",
                "delete",
            ],
            WebhookProvider.BITBUCKET: [
                "repo:push",
                "repo:fork",
                "repo:updated",
                "pullrequest:created",
                "pullrequest:updated",
                "pullrequest:approved",
                "pullrequest:fulfilled",
                "pullrequest:rejected",
            ],
            WebhookProvider.GITLAB: [
                "push",
                "merge_request",
                "wiki_page",
                "deployment",
                "job",
                "pipeline",
                "release",
            ],
        }

        return event_mappings.get(provider, [])


# Global webhook validator instance
webhook_validator = WebhookValidator()
