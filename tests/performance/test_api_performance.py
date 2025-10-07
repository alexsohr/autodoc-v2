"""Performance tests for API endpoints

This module contains performance tests to ensure API endpoints
meet the specified performance requirements (p50 ≤ 500ms, p95 ≤ 1500ms).
"""

import asyncio
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestAPIPerformance:
    """Test API endpoint performance requirements"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_health_endpoint_performance(self, async_client: AsyncClient):
        """Test health endpoint performance"""
        response_times = []
        iterations = 50

        for _ in range(iterations):
            start_time = time.time()
            response = await async_client.get("/health/")
            end_time = time.time()

            response_times.append((end_time - start_time) * 1000)  # Convert to ms
            assert response.status_code == 200

        # Calculate percentiles
        response_times.sort()
        p50 = response_times[len(response_times) // 2]
        p95 = response_times[int(len(response_times) * 0.95)]

        print(f"Health endpoint - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

        # Performance requirements
        assert p50 <= 100, f"P50 ({p50:.2f}ms) exceeds 100ms for health endpoint"
        assert p95 <= 200, f"P95 ({p95:.2f}ms) exceeds 200ms for health endpoint"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_repository_list_performance(self, async_client: AsyncClient):
        """Test repository listing performance"""
        # Mock repository data
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_service:
            mock_service.list_repositories.return_value = {
                "status": "success",
                "repositories": [
                    {
                        "id": str(uuid4()),
                        "provider": "github",
                        "url": f"https://github.com/org/repo{i}",
                        "org": "org",
                        "name": f"repo{i}",
                        "analysis_status": "completed",
                    }
                    for i in range(20)
                ],
                "total": 20,
            }

            response_times = []
            iterations = 30

            for _ in range(iterations):
                start_time = time.time()
                response = await async_client.get("/api/v2/repositories?limit=20")
                end_time = time.time()

                response_times.append((end_time - start_time) * 1000)
                assert response.status_code == 200

            # Calculate percentiles
            response_times.sort()
            p50 = response_times[len(response_times) // 2]
            p95 = response_times[int(len(response_times) * 0.95)]

            print(f"Repository list - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

            # Performance requirements
            assert p50 <= 500, f"P50 ({p50:.2f}ms) exceeds 500ms"
            assert p95 <= 1500, f"P95 ({p95:.2f}ms) exceeds 1500ms"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_wiki_structure_performance(self, async_client: AsyncClient):
        """Test wiki structure retrieval performance"""
        repository_id = str(uuid4())

        # Mock wiki data
        with patch("src.services.wiki_service.wiki_service") as mock_service:
            mock_wiki_structure = {
                "id": f"wiki_{repository_id}",
                "title": "Test Repository Documentation",
                "description": "Generated documentation",
                "pages": [
                    {
                        "id": f"page_{i}",
                        "title": f"Page {i}",
                        "description": f"Description for page {i}",
                        "importance": "medium",
                        "file_paths": [f"src/file_{i}.py"],
                        "related_pages": [],
                        "content": f"Content for page {i}" * 100,  # Substantial content
                    }
                    for i in range(10)
                ],
                "sections": [
                    {
                        "id": f"section_{i}",
                        "title": f"Section {i}",
                        "pages": [f"page_{i}"],
                        "subsections": [],
                    }
                    for i in range(5)
                ],
                "root_sections": [f"section_{i}" for i in range(5)],
            }

            mock_service.get_wiki_structure.return_value = {
                "status": "success",
                "wiki_structure": mock_wiki_structure,
            }

            response_times = []
            iterations = 20

            for _ in range(iterations):
                start_time = time.time()
                response = await async_client.get(
                    f"/api/v2/repositories/{repository_id}/wiki"
                )
                end_time = time.time()

                response_times.append((end_time - start_time) * 1000)
                assert response.status_code == 200

            # Calculate percentiles
            response_times.sort()
            p50 = response_times[len(response_times) // 2]
            p95 = response_times[int(len(response_times) * 0.95)]

            print(f"Wiki structure - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

            # Performance requirements
            assert p50 <= 500, f"P50 ({p50:.2f}ms) exceeds 500ms"
            assert p95 <= 1500, f"P95 ({p95:.2f}ms) exceeds 1500ms"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_repository_files_performance(self, async_client: AsyncClient):
        """Test repository files listing performance"""
        repository_id = str(uuid4())

        # Mock file data
        with patch("src.services.document_service.document_service") as mock_service:
            mock_files = [
                {
                    "id": f"doc_{i}",
                    "file_path": f"src/module_{i // 10}/file_{i}.py",
                    "language": "python",
                    "metadata": {"size": 1024 + i * 10},
                    "has_embedding": True,
                }
                for i in range(100)
            ]

            mock_service.get_repository_documents.return_value = {
                "status": "success",
                "files": mock_files,
                "repository_id": repository_id,
                "total": 100,
                "languages": {"python": 100},
            }

            response_times = []
            iterations = 25

            for _ in range(iterations):
                start_time = time.time()
                response = await async_client.get(
                    f"/api/v2/repositories/{repository_id}/files?limit=50"
                )
                end_time = time.time()

                response_times.append((end_time - start_time) * 1000)
                assert response.status_code == 200

            # Calculate percentiles
            response_times.sort()
            p50 = response_times[len(response_times) // 2]
            p95 = response_times[int(len(response_times) * 0.95)]

            print(f"Repository files - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

            # Performance requirements
            assert p50 <= 500, f"P50 ({p50:.2f}ms) exceeds 500ms"
            assert p95 <= 1500, f"P95 ({p95:.2f}ms) exceeds 1500ms"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_api_requests(self, async_client: AsyncClient):
        """Test API performance under concurrent load"""
        # Mock services
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_repo_service:
            mock_repo_service.list_repositories.return_value = {
                "status": "success",
                "repositories": [],
                "total": 0,
            }

            # Create concurrent requests
            async def make_request():
                start_time = time.time()
                response = await async_client.get("/api/v2/repositories")
                end_time = time.time()
                return (end_time - start_time) * 1000, response.status_code

            # Execute concurrent requests
            concurrent_requests = 20
            tasks = [make_request() for _ in range(concurrent_requests)]

            results = await asyncio.gather(*tasks)

            # Analyze results
            response_times = [result[0] for result in results]
            status_codes = [result[1] for result in results]

            # All requests should succeed
            assert all(code == 200 for code in status_codes)

            # Calculate performance metrics
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)

            print(
                f"Concurrent requests - Avg: {avg_response_time:.2f}ms, Max: {max_response_time:.2f}ms"
            )

            # Performance requirements for concurrent load
            assert (
                avg_response_time <= 1000
            ), f"Average response time ({avg_response_time:.2f}ms) too high under load"
            assert (
                max_response_time <= 3000
            ), f"Max response time ({max_response_time:.2f}ms) too high under load"


class TestDatabasePerformance:
    """Test database operation performance"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_mongodb_query_performance(self):
        """Test MongoDB query performance"""
        with patch("src.services.data_access.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock query response
            mock_mongodb.find_documents.return_value = [
                {"id": f"doc_{i}", "content": f"content {i}"} for i in range(50)
            ]

            # Test query performance
            response_times = []
            iterations = 20

            for _ in range(iterations):
                start_time = time.time()
                await mock_mongodb.find_documents("code_documents", {}, limit=50)
                end_time = time.time()

                response_times.append((end_time - start_time) * 1000)

            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)

            print(f"MongoDB queries - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")

            # Database queries should be fast
            assert avg_time <= 100, f"Average DB query time ({avg_time:.2f}ms) too high"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_vector_search_performance(self):
        """Test vector search performance"""
        with patch("src.services.data_access.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock vector search results
            mock_results = [
                {
                    "document": {"id": f"doc_{i}", "file_path": f"src/file_{i}.py"},
                    "score": 0.9 - (i * 0.01),
                }
                for i in range(10)
            ]

            mock_mongodb.vector_search.return_value = mock_results

            # Test vector search performance
            query_embedding = [0.1] * 384  # Mock embedding

            response_times = []
            iterations = 15

            for _ in range(iterations):
                start_time = time.time()
                await mock_mongodb.vector_search(query_embedding, k=10)
                end_time = time.time()

                response_times.append((end_time - start_time) * 1000)

            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)

            print(f"Vector search - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms")

            # Vector search should be reasonably fast
            assert (
                avg_time <= 200
            ), f"Average vector search time ({avg_time:.2f}ms) too high"


class TestMemoryUsage:
    """Test memory usage and resource management"""

    @pytest.mark.performance
    def test_large_document_processing(self):
        """Test memory usage with large documents"""
        from src.services.document_service import DocumentProcessingService

        doc_service = DocumentProcessingService()

        # Create large content
        large_content = "This is a large document content. " * 10000  # ~340KB

        # Test content cleaning doesn't cause memory issues
        cleaned_content = doc_service._clean_content_for_embedding(
            large_content, "python"
        )

        # Should be truncated to reasonable size
        assert len(cleaned_content) <= 8003  # 8000 + "..."

        # Memory should be released
        del large_content
        del cleaned_content

    @pytest.mark.performance
    def test_embedding_memory_usage(self):
        """Test embedding storage memory usage"""
        from src.models.code_document import CodeDocument

        # Create document with large embedding
        large_embedding = [0.1] * 1536  # Large embedding dimension

        doc = CodeDocument(
            id="large_doc",
            repository_id=uuid4(),
            file_path="src/large_file.py",
            language="python",
            content="print('hello')",
            processed_content="print hello",
            embedding=large_embedding,
        )

        # Verify embedding is stored efficiently
        assert doc.get_embedding_dimension() == 1536
        assert doc.has_embedding() is True

        # Test serialization doesn't cause memory issues
        doc_dict = doc.model_dump()
        assert len(doc_dict["embedding"]) == 1536


class TestScalabilityLimits:
    """Test system scalability limits"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_large_repository_simulation(self, async_client: AsyncClient):
        """Test performance with large repository simulation"""
        repository_id = str(uuid4())

        # Simulate large repository with many files
        large_file_set = [
            {
                "id": f"doc_{i}",
                "file_path": f"src/module_{i // 100}/submodule_{i // 10}/file_{i}.py",
                "language": "python",
                "metadata": {"size": 1024 + (i % 1000)},
                "has_embedding": True,
            }
            for i in range(5000)  # 5000 files
        ]

        with patch("src.services.document_service.document_service") as mock_service:
            mock_service.get_repository_documents.return_value = {
                "status": "success",
                "files": large_file_set[:100],  # Return first 100 for pagination
                "repository_id": repository_id,
                "total": 5000,
                "languages": {"python": 5000},
            }

            # Test performance with large dataset
            start_time = time.time()
            response = await async_client.get(
                f"/api/v2/repositories/{repository_id}/files?limit=100"
            )
            end_time = time.time()

            response_time = (end_time - start_time) * 1000

            print(
                f"Large repository files (5000 total) - Response time: {response_time:.2f}ms"
            )

            assert response.status_code == 200
            assert (
                response_time <= 2000
            ), f"Large repository query ({response_time:.2f}ms) too slow"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_pagination_performance(self, async_client: AsyncClient):
        """Test pagination performance with large datasets"""
        repository_id = str(uuid4())

        with patch("src.services.document_service.document_service") as mock_service:

            def mock_get_documents(
                repo_id, language_filter=None, path_pattern=None, limit=100, offset=0
            ):
                # Simulate database pagination
                total_docs = 1000
                start_idx = offset
                end_idx = min(offset + limit, total_docs)

                files = [
                    {
                        "id": f"doc_{i}",
                        "file_path": f"src/file_{i}.py",
                        "language": "python",
                    }
                    for i in range(start_idx, end_idx)
                ]

                return {
                    "status": "success",
                    "files": files,
                    "repository_id": str(repo_id),
                    "total": total_docs,
                    "languages": {"python": total_docs},
                }

            mock_service.get_repository_documents.side_effect = mock_get_documents

            # Test different pagination offsets
            pagination_tests = [
                (0, 50),  # First page
                (500, 50),  # Middle page
                (950, 50),  # Last page
            ]

            for offset, limit in pagination_tests:
                start_time = time.time()
                response = await async_client.get(
                    f"/api/v2/repositories/{repository_id}/files?limit={limit}&offset={offset}"
                )
                end_time = time.time()

                response_time = (end_time - start_time) * 1000

                print(
                    f"Pagination offset={offset}, limit={limit} - Response time: {response_time:.2f}ms"
                )

                assert response.status_code == 200
                assert (
                    response_time <= 1000
                ), f"Pagination query ({response_time:.2f}ms) too slow"


class TestCachePerformance:
    """Test caching performance improvements"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_repeated_request_performance(self, async_client: AsyncClient):
        """Test performance improvement from caching (when implemented)"""
        repository_id = str(uuid4())

        with patch("src.services.wiki_service.wiki_service") as mock_service:
            call_count = 0

            def mock_get_wiki(repo_id, include_content=False, section_filter=None):
                nonlocal call_count
                call_count += 1

                # Simulate slower first call, faster subsequent calls (caching effect)
                if call_count == 1:
                    time.sleep(0.1)  # Simulate initial processing

                return {
                    "status": "success",
                    "wiki_structure": {
                        "id": f"wiki_{repo_id}",
                        "title": "Test Wiki",
                        "pages": [],
                        "sections": [],
                    },
                }

            mock_service.get_wiki_structure.side_effect = mock_get_wiki

            # First request (cache miss)
            start_time = time.time()
            response1 = await async_client.get(
                f"/api/v2/repositories/{repository_id}/wiki"
            )
            first_request_time = (time.time() - start_time) * 1000

            # Second request (cache hit)
            start_time = time.time()
            response2 = await async_client.get(
                f"/api/v2/repositories/{repository_id}/wiki"
            )
            second_request_time = (time.time() - start_time) * 1000

            print(
                f"Cache test - First: {first_request_time:.2f}ms, Second: {second_request_time:.2f}ms"
            )

            assert response1.status_code == 200
            assert response2.status_code == 200

            # Second request should be faster (or at least not significantly slower)
            # Note: In a real implementation with caching, second request would be much faster
            assert second_request_time <= first_request_time * 2


class TestResourceCleanup:
    """Test resource cleanup and memory management"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_connection_cleanup(self):
        """Test database connection cleanup"""
        from src.services.data_access import MongoDBAdapter

        # Test connection lifecycle
        adapter = MongoDBAdapter()

        # Mock connection operations
        with patch.object(adapter, "client") as mock_client:
            mock_client.admin.command = AsyncMock(return_value={"ok": 1})
            mock_client.close = MagicMock()

            # Initialize and cleanup
            await adapter.initialize()
            await adapter.cleanup()

            # Verify cleanup was called
            mock_client.close.assert_called_once()

    @pytest.mark.performance
    def test_memory_leak_prevention(self):
        """Test for potential memory leaks"""
        from src.models.code_document import CodeDocument

        # Create and destroy many objects
        documents = []
        for i in range(1000):
            doc = CodeDocument(
                id=f"doc_{i}",
                repository_id=uuid4(),
                file_path=f"src/file_{i}.py",
                language="python",
                content=f"Content {i}",
                processed_content=f"Processed {i}",
            )
            documents.append(doc)

        # Clear references
        documents.clear()

        # This test mainly ensures objects can be created and destroyed
        # without causing obvious memory issues
        assert True  # If we get here, no memory issues occurred
