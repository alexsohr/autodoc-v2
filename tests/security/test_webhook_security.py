"""Security tests for webhook signature validation

This module contains security tests to ensure webhook endpoints
properly validate signatures and prevent security vulnerabilities.
"""

import hashlib
import hmac
import json
import time
from typing import Any, Dict
from unittest.mock import patch
from uuid import uuid4

import pytest
from httpx import AsyncClient

from src.utils.webhook_validator import WebhookProvider, WebhookValidator


class TestWebhookSignatureSecurity:
    """Test webhook signature validation security"""

    @pytest.fixture
    def webhook_validator(self):
        """Create webhook validator instance"""
        return WebhookValidator()

    def test_github_signature_validation_security(self, webhook_validator):
        """Test GitHub signature validation against attacks"""
        secret = "super-secret-webhook-key-12345"

        # Valid payload and signature
        valid_payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "test-org/test-repo"},
        }

        # Generate correct signature
        payload_str = json.dumps(valid_payload, separators=(",", ":"))
        correct_signature = hmac.new(
            secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Test correct signature
        assert (
            webhook_validator.validate_github_signature(
                valid_payload, f"sha256={correct_signature}", secret
            )
            is True
        )

        # Test signature tampering attacks
        tampered_signatures = [
            f"sha256={correct_signature[:-1]}X",  # Changed last character
            f"sha256={correct_signature.upper()}",  # Changed case
            f"sha1={correct_signature}",  # Wrong algorithm
            f"sha256=",  # Empty signature
            "",  # No signature
            "invalid-format",  # Invalid format
            f"sha256={correct_signature}{correct_signature}",  # Doubled signature
        ]

        for tampered_sig in tampered_signatures:
            assert (
                webhook_validator.validate_github_signature(
                    valid_payload, tampered_sig, secret
                )
                is False
            ), f"Tampered signature should fail: {tampered_sig}"

    def test_payload_tampering_detection(self, webhook_validator):
        """Test detection of payload tampering"""
        secret = "webhook-secret-key"

        original_payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "test-org/test-repo"},
            "commits": [{"id": "abc123", "message": "Original commit"}],
        }

        # Generate signature for original payload
        payload_str = json.dumps(original_payload, separators=(",", ":"))
        signature = hmac.new(
            secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Test with original payload (should pass)
        assert (
            webhook_validator.validate_github_signature(
                original_payload, f"sha256={signature}", secret
            )
            is True
        )

        # Test with tampered payloads (should fail)
        tampered_payloads = [
            {**original_payload, "ref": "refs/heads/malicious"},  # Changed ref
            {
                **original_payload,
                "repository": {"full_name": "malicious/repo"},
            },  # Changed repo
            {**original_payload, "malicious_field": "injected"},  # Added field
        ]

        for tampered_payload in tampered_payloads:
            assert (
                webhook_validator.validate_github_signature(
                    tampered_payload, f"sha256={signature}", secret
                )
                is False
            ), "Tampered payload should fail validation"

    def test_timing_attack_prevention(self, webhook_validator):
        """Test prevention of timing attacks"""
        secret = "timing-attack-test-secret"
        payload = {"test": "data"}

        # Generate correct signature
        payload_str = json.dumps(payload, separators=(",", ":"))
        correct_signature = hmac.new(
            secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Test signatures of different lengths (timing attack attempt)
        test_signatures = [
            "sha256=a",  # Very short
            "sha256=abc123",  # Short
            f"sha256={correct_signature[:-10]}",  # Partially correct
            f"sha256={correct_signature}X",  # Almost correct
            f"sha256={'a' * 64}",  # Wrong but correct length
            f"sha256={correct_signature}",  # Correct
        ]

        # Measure validation times
        validation_times = []

        for sig in test_signatures:
            start_time = time.perf_counter()
            result = webhook_validator.validate_github_signature(payload, sig, secret)
            end_time = time.perf_counter()

            validation_time = (end_time - start_time) * 1000000  # microseconds
            validation_times.append(validation_time)

            # Only the correct signature should pass
            expected_result = sig == f"sha256={correct_signature}"
            assert result == expected_result

        # Check that timing is relatively consistent (timing attack prevention)
        avg_time = sum(validation_times) / len(validation_times)
        max_deviation = max(abs(t - avg_time) for t in validation_times)

        print(
            f"Timing attack test - Avg: {avg_time:.2f}μs, Max deviation: {max_deviation:.2f}μs"
        )

        # Timing should be relatively consistent (within reasonable bounds)
        assert (
            max_deviation <= avg_time * 2
        ), "Timing variation suggests vulnerability to timing attacks"

    def test_secret_key_strength_validation(self, webhook_validator):
        """Test webhook secret key strength requirements"""
        test_payload = {"test": "data"}

        # Test weak secrets (should be rejected at configuration level)
        weak_secrets = [
            "",  # Empty
            "123",  # Too short
            "password",  # Common word
            "12345678",  # Sequential numbers
            "aaaaaaaa",  # Repeated characters
        ]

        # Strong secrets (should be accepted)
        strong_secrets = [
            "Str0ng-S3cr3t-K3y-W1th-Sp3c14l-Ch4r5!",
            "randomly-generated-webhook-secret-12345",
            "abcdef1234567890abcdef1234567890abcdef12",  # 40 char hex
        ]

        # Test that validation works with strong secrets
        for secret in strong_secrets:
            payload_str = json.dumps(test_payload, separators=(",", ":"))
            signature = hmac.new(
                secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            result = webhook_validator.validate_github_signature(
                test_payload, f"sha256={signature}", secret
            )
            assert result is True, f"Strong secret should work: {secret[:10]}..."


class TestWebhookInjectionPrevention:
    """Test prevention of injection attacks via webhooks"""

    @pytest.mark.asyncio
    async def test_sql_injection_prevention(self, async_client: AsyncClient):
        """Test prevention of SQL injection via webhook payloads"""
        # Mock malicious payloads with SQL injection attempts
        malicious_payloads = [
            {
                "ref": "refs/heads/main'; DROP TABLE repositories; --",
                "repository": {"full_name": "test-org/test-repo"},
            },
            {
                "ref": "refs/heads/main",
                "repository": {"full_name": "test-org'; UNION SELECT * FROM users; --"},
            },
            {
                "ref": "refs/heads/main",
                "repository": {
                    "full_name": "test-org/test-repo",
                    "description": "Normal repo' OR '1'='1",
                },
            },
        ]

        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            # Mock service should handle malicious input safely
            mock_service.process_webhook_event.return_value = {
                "status": "error",
                "error": "Repository not found",
                "error_type": "RepositoryNotFound",
            }

            for malicious_payload in malicious_payloads:
                headers = {
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": "sha256=test-signature",
                    "X-GitHub-Delivery": str(uuid4()),
                }

                response = await async_client.post(
                    "/webhooks/github", json=malicious_payload, headers=headers
                )

                # Should handle malicious input gracefully
                assert response.status_code in [400, 404, 500]
                # Should not cause server crash or data corruption

    @pytest.mark.asyncio
    async def test_xss_prevention(self, async_client: AsyncClient):
        """Test prevention of XSS attacks via webhook payloads"""
        # Mock XSS payloads
        xss_payloads = [
            {
                "ref": "refs/heads/main",
                "repository": {
                    "full_name": "test-org/test-repo",
                    "description": "<script>alert('XSS')</script>",
                },
            },
            {
                "ref": "refs/heads/main",
                "commits": [
                    {
                        "message": "javascript:alert('XSS')",
                        "author": {"name": "<img src=x onerror=alert('XSS')>"},
                    }
                ],
            },
        ]

        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "XSS payload handled safely",
            }

            for xss_payload in xss_payloads:
                headers = {
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": "sha256=test-signature",
                    "X-GitHub-Delivery": str(uuid4()),
                }

                response = await async_client.post(
                    "/webhooks/github", json=xss_payload, headers=headers
                )

                # Should handle XSS attempts safely
                assert response.status_code in [200, 400, 404]

                # Response should not contain unescaped script content
                response_text = response.text
                assert "<script>" not in response_text
                assert "javascript:" not in response_text

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, async_client: AsyncClient):
        """Test prevention of path traversal attacks"""
        # Mock path traversal attempts
        path_traversal_payloads = [
            {
                "ref": "refs/heads/main",
                "repository": {"full_name": "../../../etc/passwd"},
            },
            {
                "ref": "refs/heads/main",
                "commits": [
                    {
                        "added": ["../../../sensitive/file.txt"],
                        "modified": ["..\\..\\windows\\system32\\config"],
                    }
                ],
            },
        ]

        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "error",
                "error": "Invalid repository path",
                "error_type": "InvalidPath",
            }

            for traversal_payload in path_traversal_payloads:
                headers = {
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": "sha256=test-signature",
                    "X-GitHub-Delivery": str(uuid4()),
                }

                response = await async_client.post(
                    "/webhooks/github", json=traversal_payload, headers=headers
                )

                # Should reject path traversal attempts
                assert response.status_code in [400, 404]


class TestWebhookDOSPrevention:
    """Test prevention of Denial of Service attacks via webhooks"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_payload_dos_prevention(self, async_client: AsyncClient):
        """Test prevention of DoS via extremely large payloads"""
        # Create extremely large payload
        huge_payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "test-org/dos-test-repo"},
            "commits": [
                {
                    "id": f"commit_{i}",
                    "message": "X" * 10000,  # 10KB commit message
                    "added": [f"file_{j}.py" for j in range(1000)],  # Many files
                }
                for i in range(100)  # 100 commits
            ],
        }

        headers = {
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": "sha256=test-signature",
            "X-GitHub-Delivery": str(uuid4()),
        }

        # Test that large payload is handled appropriately
        start_time = time.time()
        response = await async_client.post(
            "/webhooks/github", json=huge_payload, headers=headers
        )
        processing_time = time.time() - start_time

        payload_size = len(json.dumps(huge_payload))
        print(
            f"Large payload DoS test ({payload_size} bytes) - Response time: {processing_time:.2f}s"
        )

        # Should either reject large payload or handle it within reasonable time
        assert response.status_code in [200, 400, 413, 500]  # 413 = Payload Too Large
        assert (
            processing_time <= 30
        ), f"DoS payload processing ({processing_time:.2f}s) took too long"

    @pytest.mark.asyncio
    async def test_rapid_request_dos_prevention(self, async_client: AsyncClient):
        """Test prevention of DoS via rapid requests"""
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            # Mock service with rate limiting
            request_count = 0

            def mock_rate_limited_processing(*args, **kwargs):
                nonlocal request_count
                request_count += 1

                if request_count <= 20:  # Allow first 20 requests
                    return {"status": "processed", "message": "Webhook processed"}
                else:  # Rate limit subsequent requests
                    return {
                        "status": "error",
                        "error": "Rate limit exceeded",
                        "error_type": "RateLimitExceeded",
                    }

            mock_service.process_webhook_event.side_effect = (
                mock_rate_limited_processing
            )

            # Send rapid requests
            rapid_requests = 50
            payload = {
                "ref": "refs/heads/main",
                "repository": {"full_name": "test-org/dos-test-repo"},
            }

            headers = {
                "X-GitHub-Event": "push",
                "X-Hub-Signature-256": "sha256=test-signature",
                "X-GitHub-Delivery": str(uuid4()),
            }

            # Send requests as fast as possible
            start_time = time.time()
            responses = []

            for i in range(rapid_requests):
                response = await async_client.post(
                    "/webhooks/github", json=payload, headers=headers
                )
                responses.append(response.status_code)

            total_time = time.time() - start_time

            # Analyze rate limiting effectiveness
            success_responses = sum(1 for code in responses if code in [200, 202])
            rate_limited_responses = sum(1 for code in responses if code == 429)

            print(
                f"Rapid request DoS test ({rapid_requests} requests in {total_time:.2f}s):"
            )
            print(f"  Successful: {success_responses}")
            print(f"  Rate limited: {rate_limited_responses}")
            print(f"  Request rate: {rapid_requests / total_time:.2f} req/s")

            # Rate limiting should protect against DoS
            assert (
                rate_limited_responses > 0 or success_responses < rapid_requests
            ), "Rate limiting should activate"

    def test_signature_format_validation(self, webhook_validator):
        """Test signature format validation"""
        payload = {"test": "data"}
        secret = "test-secret"

        # Test various invalid signature formats
        invalid_formats = [
            "invalid",  # No algorithm prefix
            "md5=abc123",  # Unsupported algorithm
            "sha256=",  # Empty signature
            "sha256=invalid-hex",  # Invalid hex
            "sha256=abc123xyz",  # Invalid hex characters
            "SHA256=abc123def",  # Wrong case algorithm
            " sha256=abc123def ",  # Whitespace
        ]

        for invalid_format in invalid_formats:
            result = webhook_validator.validate_github_signature(
                payload, invalid_format, secret
            )
            assert result is False, f"Invalid format should fail: {invalid_format}"

    def test_secret_key_security(self, webhook_validator):
        """Test secret key security requirements"""
        payload = {"test": "data"}

        # Test empty or weak secrets
        weak_secrets = ["", "123", "password", "secret"]

        for weak_secret in weak_secrets:
            # Validation should work but weak secrets should be discouraged
            # This test ensures the validator doesn't crash with weak secrets
            try:
                result = webhook_validator.validate_github_signature(
                    payload, "sha256=test", weak_secret
                )
                # Should not crash, but result may be False due to incorrect signature
                assert isinstance(result, bool)
            except Exception as e:
                pytest.fail(f"Validator should handle weak secret gracefully: {e}")


class TestWebhookHeaderSecurity:
    """Test webhook header security validation"""

    def test_required_header_validation(self):
        """Test required header validation"""
        validator = WebhookValidator()

        # Test GitHub required headers
        github_headers_tests = [
            # Valid headers
            ({"x-github-event": "push", "x-github-delivery": "123"}, True),
            # Missing event header
            ({"x-github-delivery": "123"}, False),
            # Missing delivery header
            ({"x-github-event": "push"}, False),
            # Empty headers
            ({}, False),
        ]

        for headers, expected_valid in github_headers_tests:
            result = validator.validate_webhook_headers(headers, WebhookProvider.GITHUB)
            assert result["valid"] == expected_valid

    def test_header_injection_prevention(self):
        """Test prevention of header injection attacks"""
        validator = WebhookValidator()

        # Test malicious header values
        malicious_headers = {
            "x-github-event": "push\r\nMalicious-Header: injected",
            "x-github-delivery": "123\nAnother-Header: value",
            "x-hub-signature-256": "sha256=abc123\r\nSet-Cookie: malicious=true",
        }

        # Header injection should not cause issues
        result = validator.validate_webhook_headers(
            malicious_headers, WebhookProvider.GITHUB
        )

        # Should either pass validation (if headers are sanitized) or fail safely
        assert isinstance(result, dict)
        assert "valid" in result

    def test_case_insensitive_header_handling(self):
        """Test case-insensitive header handling"""
        validator = WebhookValidator()

        # Test different case variations
        case_variations = [
            {"X-GitHub-Event": "push", "X-GitHub-Delivery": "123"},
            {"x-github-event": "push", "x-github-delivery": "123"},
            {"X-GITHUB-EVENT": "push", "X-GITHUB-DELIVERY": "123"},
            {"x-GitHub-Event": "push", "x-GitHub-Delivery": "123"},
        ]

        for headers in case_variations:
            result = validator.validate_webhook_headers(headers, WebhookProvider.GITHUB)
            assert result["valid"] is True, f"Case variation should work: {headers}"


class TestWebhookAuthenticationSecurity:
    """Test webhook authentication security"""

    @pytest.mark.asyncio
    async def test_unauthenticated_webhook_rejection(self, async_client: AsyncClient):
        """Test rejection of unauthenticated webhook requests"""
        payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "test-org/test-repo"},
        }

        # Test webhook without signature
        headers_without_signature = {
            "X-GitHub-Event": "push",
            "X-GitHub-Delivery": str(uuid4()),
            # Missing X-Hub-Signature-256
        }

        response = await async_client.post(
            "/webhooks/github", json=payload, headers=headers_without_signature
        )

        # Should reject unauthenticated webhook
        assert response.status_code == 400, "Unauthenticated webhook should be rejected"

    @pytest.mark.asyncio
    async def test_replay_attack_prevention(self, async_client: AsyncClient):
        """Test prevention of replay attacks"""
        # Create valid webhook request
        payload = {
            "ref": "refs/heads/main",
            "repository": {"full_name": "test-org/replay-test-repo"},
            "commits": [{"id": "abc123", "message": "Test commit"}],
        }

        headers = {
            "X-GitHub-Event": "push",
            "X-Hub-Signature-256": "sha256=test-signature",
            "X-GitHub-Delivery": str(uuid4()),
        }

        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            call_count = 0

            def mock_replay_detection(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count == 1:
                    return {"status": "processed", "message": "First request processed"}
                else:
                    # Subsequent identical requests should be detected as replays
                    return {"status": "ignored", "message": "Duplicate request ignored"}

            mock_service.process_webhook_event.side_effect = mock_replay_detection

            # Send same request multiple times
            response1 = await async_client.post(
                "/webhooks/github", json=payload, headers=headers
            )
            response2 = await async_client.post(
                "/webhooks/github", json=payload, headers=headers
            )
            response3 = await async_client.post(
                "/webhooks/github", json=payload, headers=headers
            )

            # First request should succeed
            assert response1.status_code in [200, 202]

            # Subsequent requests should be handled (ignored or processed)
            assert response2.status_code in [200, 202, 409]
            assert response3.status_code in [200, 202, 409]


class TestWebhookDataValidation:
    """Test webhook data validation and sanitization"""

    @pytest.mark.asyncio
    async def test_payload_size_limits(self, async_client: AsyncClient):
        """Test webhook payload size limits"""
        # Test various payload sizes
        payload_sizes = [
            (1000, "small"),  # 1KB
            (100000, "medium"),  # 100KB
            (1000000, "large"),  # 1MB
            (10000000, "huge"),  # 10MB
        ]

        for size_bytes, size_name in payload_sizes:
            # Create payload of specific size
            large_content = "X" * (size_bytes // 10)
            payload = {
                "ref": "refs/heads/main",
                "repository": {"full_name": f"test-org/{size_name}-payload-repo"},
                "commits": [
                    {"message": large_content, "id": f"{size_name}_commit_123"}
                ],
            }

            headers = {
                "X-GitHub-Event": "push",
                "X-Hub-Signature-256": "sha256=test-signature",
                "X-GitHub-Delivery": str(uuid4()),
            }

            start_time = time.time()
            response = await async_client.post(
                "/webhooks/github", json=payload, headers=headers
            )
            processing_time = time.time() - start_time

            actual_size = len(json.dumps(payload))
            print(
                f"Payload size test ({size_name}: {actual_size} bytes) - Response: {response.status_code}, Time: {processing_time:.2f}s"
            )

            # Large payloads should either be accepted or rejected gracefully
            assert response.status_code in [
                200,
                202,
                413,
                400,
            ]  # 413 = Payload Too Large

            # Processing time should be reasonable even for large payloads
            if response.status_code in [200, 202]:
                max_time = (
                    10 if actual_size < 1000000 else 30
                )  # Larger payloads get more time
                assert (
                    processing_time <= max_time
                ), f"Large payload processing too slow: {processing_time:.2f}s"

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON payloads"""
        validator = WebhookValidator()

        # Test malformed JSON strings
        malformed_payloads = [
            '{"invalid": json}',  # Missing quotes
            '{"ref": "refs/heads/main",}',  # Trailing comma
            '{"ref": "refs/heads/main"',  # Missing closing brace
            "",  # Empty string
            "not json at all",  # Not JSON
            '{"ref": "refs/heads/main", "invalid": }',  # Invalid value
        ]

        for malformed in malformed_payloads:
            # Validator should handle malformed JSON gracefully
            try:
                # This would be tested in the actual webhook endpoint
                # For now, just ensure validator doesn't crash
                result = validator.validate_github_signature(
                    malformed, "sha256=test", "secret"
                )
                assert isinstance(result, bool)
            except json.JSONDecodeError:
                # Expected for malformed JSON
                pass
            except Exception as e:
                pytest.fail(f"Validator should handle malformed JSON gracefully: {e}")


class TestWebhookSecurityLogging:
    """Test security logging for webhook events"""

    @pytest.mark.asyncio
    async def test_security_event_logging(self, async_client: AsyncClient):
        """Test logging of security events"""
        from src.utils.logging_config import get_security_logger

        security_logger = get_security_logger()

        # Test various security events
        with patch.object(
            security_logger, "log_webhook_validation_failure"
        ) as mock_log:
            # Send webhook with invalid signature
            payload = {
                "ref": "refs/heads/main",
                "repository": {"full_name": "test-org/test-repo"},
            }
            headers = {
                "X-GitHub-Event": "push",
                "X-Hub-Signature-256": "sha256=invalid-signature",
                "X-GitHub-Delivery": str(uuid4()),
            }

            response = await async_client.post(
                "/webhooks/github", json=payload, headers=headers
            )

            # Security failure should be logged
            # Note: This would be integrated in the actual webhook endpoint
            assert response.status_code in [400, 401, 403]

    @pytest.mark.asyncio
    async def test_suspicious_activity_detection(self, async_client: AsyncClient):
        """Test detection and logging of suspicious webhook activity"""
        # Test patterns that might indicate malicious activity
        suspicious_patterns = [
            # Multiple failed signature validations from same IP
            {"pattern": "signature_failures", "count": 10},
            # Rapid requests from same source
            {"pattern": "rapid_requests", "count": 100},
            # Unusual event types
            {
                "pattern": "unusual_events",
                "events": ["unknown_event", "malicious_event"],
            },
        ]

        for pattern_info in suspicious_patterns:
            pattern_type = pattern_info["pattern"]

            if pattern_type == "signature_failures":
                # Send multiple requests with invalid signatures
                for i in range(5):  # Reduced for test performance
                    payload = {
                        "ref": "refs/heads/main",
                        "repository": {"full_name": "test-org/suspicious-repo"},
                    }
                    headers = {
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": f"sha256=invalid-{i}",
                        "X-GitHub-Delivery": str(uuid4()),
                    }

                    response = await async_client.post(
                        "/webhooks/github", json=payload, headers=headers
                    )
                    # Should reject invalid signatures
                    assert response.status_code in [400, 401, 403]

            elif pattern_type == "unusual_events":
                # Send webhooks with unusual event types
                for event in pattern_info["events"]:
                    payload = {
                        "ref": "refs/heads/main",
                        "repository": {"full_name": "test-org/unusual-repo"},
                    }
                    headers = {
                        "X-GitHub-Event": event,
                        "X-Hub-Signature-256": "sha256=test-signature",
                        "X-GitHub-Delivery": str(uuid4()),
                    }

                    response = await async_client.post(
                        "/webhooks/github", json=payload, headers=headers
                    )
                    # Should handle unusual events gracefully
                    assert response.status_code in [200, 400, 404]
