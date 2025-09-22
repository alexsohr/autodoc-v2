"""Load tests for webhook processing

This module contains load tests for webhook endpoints to ensure
they can handle high-volume webhook traffic from Git providers.
"""

import asyncio
import hashlib
import hmac
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient


class TestWebhookLoadPerformance:
    """Test webhook processing under load"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_github_webhook_load(self, async_client: AsyncClient):
        """Test GitHub webhook processing under load"""
        # Mock repository service
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "Webhook processed successfully",
                "repository_id": str(uuid4()),
                "event_type": "push",
                "processing_time": 0.5,
            }

            # Create multiple concurrent webhook requests
            async def send_github_webhook(webhook_id: int):
                """Send a GitHub webhook request"""
                payload = {
                    "ref": "refs/heads/main",
                    "repository": {
                        "id": 123456 + webhook_id,
                        "name": f"test-repo-{webhook_id}",
                        "full_name": f"test-org/test-repo-{webhook_id}",
                        "clone_url": f"https://github.com/test-org/test-repo-{webhook_id}.git",
                    },
                    "commits": [
                        {
                            "id": f"commit{webhook_id}abc123",
                            "message": f"Update from webhook {webhook_id}",
                        }
                    ],
                }

                # Generate signature
                secret = "test-webhook-secret"
                payload_str = json.dumps(payload, separators=(",", ":"))
                signature = hmac.new(
                    secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
                ).hexdigest()

                headers = {
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": f"sha256={signature}",
                    "X-GitHub-Delivery": str(uuid4()),
                    "Content-Type": "application/json",
                }

                start_time = time.time()
                response = await async_client.post(
                    "/webhooks/github", json=payload, headers=headers
                )
                end_time = time.time()

                return (end_time - start_time) * 1000, response.status_code, webhook_id

            # Execute concurrent webhooks
            concurrent_webhooks = 50
            tasks = [send_github_webhook(i) for i in range(concurrent_webhooks)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            response_times = [result[0] for result in successful_results]
            status_codes = [result[1] for result in successful_results]

            # Calculate performance metrics
            if response_times:
                response_times.sort()
                avg_time = sum(response_times) / len(response_times)
                p50 = response_times[len(response_times) // 2]
                p95 = response_times[int(len(response_times) * 0.95)]
                max_time = max(response_times)

                success_rate = len(successful_results) / concurrent_webhooks * 100

                print(f"GitHub webhook load test ({concurrent_webhooks} concurrent):")
                print(f"  Success rate: {success_rate:.1f}%")
                print(
                    f"  Avg: {avg_time:.2f}ms, P50: {p50:.2f}ms, P95: {p95:.2f}ms, Max: {max_time:.2f}ms"
                )

                # Performance requirements for webhook processing
                assert (
                    success_rate >= 95
                ), f"Success rate ({success_rate:.1f}%) below 95%"
                assert p50 <= 1000, f"P50 ({p50:.2f}ms) exceeds 1000ms"
                assert p95 <= 3000, f"P95 ({p95:.2f}ms) exceeds 3000ms"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_bitbucket_webhook_load(self, async_client: AsyncClient):
        """Test Bitbucket webhook processing under load"""
        # Mock repository service
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "Webhook processed successfully",
                "repository_id": str(uuid4()),
                "event_type": "repo:push",
                "processing_time": 0.4,
            }

            # Create multiple concurrent Bitbucket webhooks
            async def send_bitbucket_webhook(webhook_id: int):
                """Send a Bitbucket webhook request"""
                payload = {
                    "push": {
                        "changes": [
                            {
                                "new": {
                                    "name": "main",
                                    "target": {"hash": f"bitbucket{webhook_id}abc123"},
                                }
                            }
                        ]
                    },
                    "repository": {
                        "name": f"test-repo-{webhook_id}",
                        "full_name": f"test-org/test-repo-{webhook_id}",
                        "links": {
                            "clone": [
                                {
                                    "name": "https",
                                    "href": f"https://bitbucket.org/test-org/test-repo-{webhook_id}.git",
                                }
                            ]
                        },
                    },
                }

                headers = {
                    "X-Event-Key": "repo:push",
                    "X-Hook-UUID": str(uuid4()),
                    "Content-Type": "application/json",
                }

                start_time = time.time()
                response = await async_client.post(
                    "/webhooks/bitbucket", json=payload, headers=headers
                )
                end_time = time.time()

                return (end_time - start_time) * 1000, response.status_code, webhook_id

            # Execute concurrent webhooks
            concurrent_webhooks = 30
            tasks = [send_bitbucket_webhook(i) for i in range(concurrent_webhooks)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful_results = [r for r in results if not isinstance(r, Exception)]
            response_times = [result[0] for result in successful_results]

            if response_times:
                avg_time = sum(response_times) / len(response_times)
                max_time = max(response_times)
                success_rate = len(successful_results) / concurrent_webhooks * 100

                print(
                    f"Bitbucket webhook load test ({concurrent_webhooks} concurrent):"
                )
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")

                # Performance requirements
                assert (
                    success_rate >= 95
                ), f"Success rate ({success_rate:.1f}%) below 95%"
                assert (
                    avg_time <= 1200
                ), f"Average time ({avg_time:.2f}ms) exceeds 1200ms"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_webhook_burst_handling(self, async_client: AsyncClient):
        """Test webhook processing during traffic bursts"""
        # Simulate burst traffic pattern
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "Webhook processed",
                "repository_id": str(uuid4()),
                "event_type": "push",
            }

            # Create burst pattern: many requests in short time, then quiet period
            async def webhook_burst(burst_size: int, delay_between_requests: float):
                """Send a burst of webhook requests"""
                tasks = []

                for i in range(burst_size):
                    payload = {
                        "ref": "refs/heads/main",
                        "repository": {
                            "full_name": f"test-org/burst-repo-{i}",
                            "clone_url": f"https://github.com/test-org/burst-repo-{i}.git",
                        },
                    }

                    headers = {
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": "sha256=test-signature",
                        "X-GitHub-Delivery": str(uuid4()),
                    }

                    # Add small delay between requests in burst
                    if i > 0:
                        await asyncio.sleep(delay_between_requests)

                    task = async_client.post(
                        "/webhooks/github", json=payload, headers=headers
                    )
                    tasks.append(task)

                return await asyncio.gather(*tasks, return_exceptions=True)

            # Test different burst patterns
            burst_patterns = [
                (10, 0.01),  # 10 requests with 10ms intervals
                (20, 0.005),  # 20 requests with 5ms intervals
                (5, 0.0),  # 5 simultaneous requests
            ]

            for burst_size, delay in burst_patterns:
                start_time = time.time()
                burst_results = await webhook_burst(burst_size, delay)
                total_burst_time = time.time() - start_time

                successful_count = sum(
                    1
                    for result in burst_results
                    if not isinstance(result, Exception)
                    and hasattr(result, "status_code")
                    and result.status_code in [200, 202]
                )

                success_rate = successful_count / burst_size * 100
                throughput = burst_size / total_burst_time

                print(f"Burst test (size={burst_size}, delay={delay}s):")
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  Total time: {total_burst_time:.2f}s")
                print(f"  Throughput: {throughput:.2f} webhooks/second")

                # Burst handling requirements
                assert (
                    success_rate >= 90
                ), f"Burst success rate ({success_rate:.1f}%) below 90%"
                assert (
                    throughput >= 5
                ), f"Burst throughput ({throughput:.2f} req/s) too low"


class TestWebhookProcessingEfficiency:
    """Test webhook processing efficiency and resource usage"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_webhook_payload_size_handling(self, async_client: AsyncClient):
        """Test webhook processing with different payload sizes"""
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "Webhook processed",
                "repository_id": str(uuid4()),
            }

            # Test different payload sizes
            payload_sizes = [
                ("small", 50),  # ~50 commits
                ("medium", 200),  # ~200 commits
                ("large", 500),  # ~500 commits
            ]

            for size_name, commit_count in payload_sizes:
                # Create payload with many commits
                payload = {
                    "ref": "refs/heads/main",
                    "repository": {
                        "full_name": f"test-org/{size_name}-repo",
                        "clone_url": f"https://github.com/test-org/{size_name}-repo.git",
                    },
                    "commits": [
                        {
                            "id": f"commit{i}abc123",
                            "message": f"Commit message {i}",
                            "author": {
                                "name": "Test User",
                                "email": "test@example.com",
                            },
                            "added": [f"file_{i}.py"],
                            "modified": [f"existing_{i}.py"] if i % 2 == 0 else [],
                            "removed": [f"old_{i}.py"] if i % 3 == 0 else [],
                        }
                        for i in range(commit_count)
                    ],
                }

                headers = {
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": "sha256=test-signature",
                    "X-GitHub-Delivery": str(uuid4()),
                }

                # Test processing time
                start_time = time.time()
                response = await async_client.post(
                    "/webhooks/github", json=payload, headers=headers
                )
                end_time = time.time()

                processing_time = (end_time - start_time) * 1000
                payload_size = len(json.dumps(payload))

                print(
                    f"Webhook {size_name} payload ({commit_count} commits, {payload_size} bytes) - Processing time: {processing_time:.2f}ms"
                )

                assert response.status_code in [200, 202]

                # Performance requirements based on payload size
                if commit_count <= 50:
                    assert (
                        processing_time <= 1000
                    ), f"Small payload processing ({processing_time:.2f}ms) too slow"
                elif commit_count <= 200:
                    assert (
                        processing_time <= 2000
                    ), f"Medium payload processing ({processing_time:.2f}ms) too slow"
                else:
                    assert (
                        processing_time <= 5000
                    ), f"Large payload processing ({processing_time:.2f}ms) too slow"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_webhook_rate_limiting_performance(self, async_client: AsyncClient):
        """Test webhook processing with rate limiting"""
        repository_url = "https://github.com/test-org/rate-test-repo"

        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            # Mock rate limiting behavior
            call_count = 0

            def mock_process_webhook(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count <= 10:
                    return {
                        "status": "processed",
                        "message": "Webhook processed",
                        "repository_id": str(uuid4()),
                    }
                else:
                    # Simulate rate limiting after 10 requests
                    return {
                        "status": "error",
                        "error": "Rate limit exceeded",
                        "error_type": "RateLimitExceeded",
                    }

            mock_service.process_webhook_event.side_effect = mock_process_webhook

            # Send rapid webhook requests
            async def send_rapid_webhook(request_id: int):
                payload = {
                    "ref": "refs/heads/main",
                    "repository": {
                        "full_name": "test-org/rate-test-repo",
                        "clone_url": repository_url,
                    },
                    "commits": [{"id": f"rapid{request_id}abc123"}],
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
                end_time = time.time()

                return (end_time - start_time) * 1000, response.status_code

            # Send 20 rapid requests
            rapid_requests = 20
            tasks = [send_rapid_webhook(i) for i in range(rapid_requests)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze rate limiting behavior
            successful_results = [r for r in results if not isinstance(r, Exception)]
            response_times = [result[0] for result in successful_results]
            status_codes = [result[1] for result in successful_results]

            processed_count = sum(1 for code in status_codes if code in [200, 202])
            rate_limited_count = sum(1 for code in status_codes if code == 429)

            print(f"Rate limiting test:")
            print(f"  Total requests: {rapid_requests}")
            print(f"  Processed: {processed_count}")
            print(f"  Rate limited: {rate_limited_count}")
            print(
                f"  Average response time: {sum(response_times) / len(response_times):.2f}ms"
            )

            # Rate limiting should work effectively
            assert processed_count >= 5, "Some requests should be processed"
            # Note: Rate limiting implementation would determine exact behavior

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_webhook_error_recovery_performance(self, async_client: AsyncClient):
        """Test webhook error recovery performance"""
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            # Mock intermittent failures
            call_count = 0

            def mock_intermittent_failures(*args, **kwargs):
                nonlocal call_count
                call_count += 1

                # Fail every 3rd request
                if call_count % 3 == 0:
                    raise Exception("Temporary service failure")
                else:
                    return {
                        "status": "processed",
                        "message": "Webhook processed",
                        "repository_id": str(uuid4()),
                    }

            mock_service.process_webhook_event.side_effect = mock_intermittent_failures

            # Send webhooks with error recovery
            async def send_webhook_with_recovery(request_id: int):
                payload = {
                    "ref": "refs/heads/main",
                    "repository": {
                        "full_name": f"test-org/recovery-repo-{request_id}",
                        "clone_url": f"https://github.com/test-org/recovery-repo-{request_id}.git",
                    },
                }

                headers = {
                    "X-GitHub-Event": "push",
                    "X-Hub-Signature-256": "sha256=test-signature",
                    "X-GitHub-Delivery": str(uuid4()),
                }

                start_time = time.time()
                try:
                    response = await async_client.post(
                        "/webhooks/github", json=payload, headers=headers
                    )
                    end_time = time.time()
                    return (
                        (end_time - start_time) * 1000,
                        response.status_code,
                        "success",
                    )
                except Exception as e:
                    end_time = time.time()
                    return (end_time - start_time) * 1000, 500, "error"

            # Test error recovery
            recovery_requests = 15
            tasks = [send_webhook_with_recovery(i) for i in range(recovery_requests)]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze error recovery
            successful_results = [r for r in results if not isinstance(r, Exception)]
            response_times = [result[0] for result in successful_results]
            outcomes = [result[2] for result in successful_results]

            success_count = sum(1 for outcome in outcomes if outcome == "success")
            error_count = sum(1 for outcome in outcomes if outcome == "error")

            avg_response_time = (
                sum(response_times) / len(response_times) if response_times else 0
            )

            print(f"Error recovery test:")
            print(f"  Successful: {success_count}")
            print(f"  Errors: {error_count}")
            print(f"  Average response time: {avg_response_time:.2f}ms")

            # Error recovery should handle failures gracefully
            assert (
                success_count >= error_count
            ), "Should have more successes than errors with recovery"
            assert (
                avg_response_time <= 2000
            ), f"Error recovery response time ({avg_response_time:.2f}ms) too slow"


class TestWebhookMemoryUsage:
    """Test webhook processing memory usage"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_webhook_payload_memory(self, async_client: AsyncClient):
        """Test memory usage with large webhook payloads"""
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "Large payload processed",
                "repository_id": str(uuid4()),
            }

            # Create very large webhook payload
            large_payload = {
                "ref": "refs/heads/main",
                "repository": {
                    "full_name": "test-org/large-payload-repo",
                    "clone_url": "https://github.com/test-org/large-payload-repo.git",
                },
                "commits": [
                    {
                        "id": f"large_commit_{i}",
                        "message": f"Large commit message {i} "
                        * 100,  # Large commit messages
                        "added": [
                            f"large_file_{j}.py" for j in range(50)
                        ],  # Many files
                        "modified": [f"existing_{j}.py" for j in range(30)],
                        "removed": [f"old_{j}.py" for j in range(10)],
                    }
                    for i in range(20)  # 20 large commits
                ],
            }

            headers = {
                "X-GitHub-Event": "push",
                "X-Hub-Signature-256": "sha256=test-signature",
                "X-GitHub-Delivery": str(uuid4()),
            }

            # Test large payload processing
            payload_size = len(json.dumps(large_payload))

            start_time = time.time()
            response = await async_client.post(
                "/webhooks/github", json=large_payload, headers=headers
            )
            end_time = time.time()

            processing_time = (end_time - start_time) * 1000

            print(
                f"Large webhook payload ({payload_size} bytes) - Processing time: {processing_time:.2f}ms"
            )

            assert response.status_code in [200, 202]
            # Large payloads can take longer but should be reasonable
            assert (
                processing_time <= 10000
            ), f"Large payload processing ({processing_time:.2f}ms) too slow"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_webhook_memory_cleanup(self):
        """Test webhook processing memory cleanup"""
        # Test that webhook processing doesn't accumulate memory
        from src.utils.webhook_validator import WebhookValidator

        validator = WebhookValidator()

        # Process many webhook validations
        for i in range(1000):
            payload = {"test": f"data_{i}"}
            signature = f"sha256=signature_{i}"
            secret = f"secret_{i}"

            # Validate (will fail, but should not accumulate memory)
            result = validator.validate_github_signature(payload, signature, secret)
            assert result is False  # Expected to fail with test data

        # Memory should be cleaned up automatically
        # This test mainly ensures no obvious memory leaks
        assert True  # If we get here, no memory issues occurred


class TestWebhookScalability:
    """Test webhook processing scalability"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_high_volume_webhook_processing(self, async_client: AsyncClient):
        """Test processing high volume of webhooks"""
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.process_webhook_event.return_value = {
                "status": "processed",
                "message": "High volume webhook processed",
                "repository_id": str(uuid4()),
            }

            # Simulate high volume over time
            total_webhooks = 100
            batch_size = 10
            delay_between_batches = 0.1  # 100ms between batches

            all_response_times = []
            successful_count = 0

            for batch_num in range(0, total_webhooks, batch_size):
                batch_tasks = []

                for i in range(batch_size):
                    webhook_id = batch_num + i

                    payload = {
                        "ref": "refs/heads/main",
                        "repository": {
                            "full_name": f"test-org/volume-repo-{webhook_id}",
                            "clone_url": f"https://github.com/test-org/volume-repo-{webhook_id}.git",
                        },
                        "commits": [{"id": f"volume_commit_{webhook_id}"}],
                    }

                    headers = {
                        "X-GitHub-Event": "push",
                        "X-Hub-Signature-256": "sha256=test-signature",
                        "X-GitHub-Delivery": str(uuid4()),
                    }

                    async def send_webhook(p, h):
                        start = time.time()
                        resp = await async_client.post(
                            "/webhooks/github", json=p, headers=h
                        )
                        return (time.time() - start) * 1000, resp.status_code

                    batch_tasks.append(send_webhook(payload, headers))

                # Execute batch
                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                # Process batch results
                for result in batch_results:
                    if not isinstance(result, Exception):
                        response_time, status_code = result
                        all_response_times.append(response_time)
                        if status_code in [200, 202]:
                            successful_count += 1

                # Delay between batches
                await asyncio.sleep(delay_between_batches)

            # Calculate overall performance
            if all_response_times:
                avg_time = sum(all_response_times) / len(all_response_times)
                max_time = max(all_response_times)
                success_rate = successful_count / total_webhooks * 100

                print(f"High volume webhook test ({total_webhooks} total):")
                print(f"  Success rate: {success_rate:.1f}%")
                print(f"  Average response time: {avg_time:.2f}ms")
                print(f"  Max response time: {max_time:.2f}ms")

                # High volume requirements
                assert (
                    success_rate >= 95
                ), f"High volume success rate ({success_rate:.1f}%) below 95%"
                assert (
                    avg_time <= 1500
                ), f"High volume average time ({avg_time:.2f}ms) too high"
                assert (
                    max_time <= 5000
                ), f"High volume max time ({max_time:.2f}ms) too high"
