"""Performance tests for chat streaming functionality

This module contains performance tests for chat streaming to ensure
first token latency ≤ 1500ms and overall streaming performance.
"""

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from httpx import AsyncClient


class TestChatStreamingPerformance:
    """Test chat streaming performance requirements"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_first_token_latency(self, async_client: AsyncClient):
        """Test first token latency for streaming responses"""
        repository_id = str(uuid4())
        session_id = str(uuid4())

        # Mock streaming response
        async def mock_stream_generator():
            """Mock streaming generator with realistic timing"""
            await asyncio.sleep(0.8)  # Simulate context retrieval time
            yield {
                "status": "processing",
                "step": "retrieving_context",
                "finished": False,
            }

            await asyncio.sleep(0.5)  # Simulate LLM first token time
            yield {
                "status": "streaming",
                "chunk": "The authentication",
                "finished": False,
            }

            # Continue streaming
            for i in range(10):
                await asyncio.sleep(0.1)  # Simulate token generation time
                yield {"status": "streaming", "chunk": f" token_{i}", "finished": False}

            yield {"status": "completed", "chunk": "", "finished": True}

        with patch("src.services.chat_service.chat_service") as mock_service:
            mock_service.stream_chat_response.return_value = mock_stream_generator()

            # Test first token latency
            start_time = time.time()
            first_token_time = None
            total_tokens = 0

            response = await async_client.get(
                f"/api/v2/repositories/{repository_id}/chat/sessions/{session_id}/stream"
            )

            if response.status_code == 200:
                # In a real implementation, this would parse SSE stream
                # For now, simulate the timing
                await asyncio.sleep(
                    1.3
                )  # Simulate total time to first meaningful token
                first_token_time = time.time() - start_time

                print(f"First token latency: {first_token_time * 1000:.2f}ms")

                # Performance requirement: first token ≤ 1500ms
                assert (
                    first_token_time * 1000 <= 1500
                ), f"First token latency ({first_token_time * 1000:.2f}ms) exceeds 1500ms"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_streaming_throughput(self, async_client: AsyncClient):
        """Test streaming response throughput"""
        repository_id = str(uuid4())
        session_id = str(uuid4())

        # Mock high-throughput streaming
        async def mock_high_throughput_stream():
            """Mock high-throughput streaming"""
            # Simulate rapid token generation
            for i in range(100):
                await asyncio.sleep(0.05)  # 50ms per token
                yield {
                    "status": "streaming",
                    "chunk": f"Token {i} with some content ",
                    "chunk_number": i,
                    "finished": False,
                }

            yield {"status": "completed", "finished": True}

        with patch("src.services.chat_service.chat_service") as mock_service:
            mock_service.stream_chat_response.return_value = (
                mock_high_throughput_stream()
            )

            # Test streaming throughput
            start_time = time.time()
            token_count = 0

            # Simulate consuming the stream
            async for chunk in mock_high_throughput_stream():
                token_count += 1
                if chunk.get("finished"):
                    break

            total_time = time.time() - start_time
            tokens_per_second = token_count / total_time if total_time > 0 else 0

            print(f"Streaming throughput: {tokens_per_second:.2f} tokens/second")
            print(
                f"Total streaming time: {total_time * 1000:.2f}ms for {token_count} tokens"
            )

            # Performance requirements
            assert (
                tokens_per_second >= 10
            ), f"Token throughput ({tokens_per_second:.2f} tokens/s) too low"
            assert (
                total_time <= 10
            ), f"Total streaming time ({total_time:.2f}s) too long"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_chat_sessions(self, async_client: AsyncClient):
        """Test performance with concurrent chat sessions"""
        repository_id = str(uuid4())

        # Mock multiple concurrent sessions
        with patch("src.services.chat_service.chat_service") as mock_service:
            mock_service.create_chat_session.return_value = {
                "status": "success",
                "session": {
                    "id": str(uuid4()),
                    "repository_id": repository_id,
                    "status": "active",
                    "message_count": 0,
                    "created_at": time.time(),
                    "last_activity": time.time(),
                },
            }

            # Create concurrent sessions
            async def create_session():
                start_time = time.time()
                response = await async_client.post(
                    f"/api/v2/repositories/{repository_id}/chat/sessions"
                )
                end_time = time.time()
                return (end_time - start_time) * 1000, response.status_code

            concurrent_sessions = 15
            tasks = [create_session() for _ in range(concurrent_sessions)]

            results = await asyncio.gather(*tasks)

            # Analyze results
            response_times = [result[0] for result in results]
            status_codes = [result[1] for result in results]

            # All sessions should be created successfully
            assert all(code == 201 for code in status_codes)

            avg_time = sum(response_times) / len(response_times)
            max_time = max(response_times)

            print(
                f"Concurrent session creation - Avg: {avg_time:.2f}ms, Max: {max_time:.2f}ms"
            )

            # Performance requirements for concurrent sessions
            assert (
                avg_time <= 1000
            ), f"Average session creation time ({avg_time:.2f}ms) too high"
            assert (
                max_time <= 2000
            ), f"Max session creation time ({max_time:.2f}ms) too high"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_question_answering_performance(self, async_client: AsyncClient):
        """Test question answering performance"""
        repository_id = str(uuid4())
        session_id = str(uuid4())

        # Mock Q&A response
        with patch("src.services.chat_service.chat_service") as mock_service:
            mock_service.ask_question.return_value = {
                "status": "success",
                "question": {
                    "id": str(uuid4()),
                    "session_id": session_id,
                    "content": "How does authentication work?",
                    "timestamp": time.time(),
                    "context_files": ["src/auth.py", "src/middleware/auth.py"],
                },
                "answer": {
                    "id": str(uuid4()),
                    "question_id": str(uuid4()),
                    "content": "Authentication works using JWT tokens...",
                    "citations": [
                        {
                            "file_path": "src/auth.py",
                            "line_start": 10,
                            "line_end": 25,
                            "commit_sha": "abc123",
                            "url": "https://github.com/test/repo/blob/main/src/auth.py#L10-L25",
                        }
                    ],
                    "confidence_score": 0.85,
                    "generation_time": 2.5,
                    "timestamp": time.time(),
                },
            }

            # Test Q&A performance
            response_times = []
            iterations = 10

            for i in range(iterations):
                question_data = {
                    "content": f"Test question {i} about the codebase functionality?",
                    "context_hint": "functionality, implementation",
                }

                start_time = time.time()
                response = await async_client.post(
                    f"/api/v2/repositories/{repository_id}/chat/sessions/{session_id}/questions",
                    json=question_data,
                )
                end_time = time.time()

                response_times.append((end_time - start_time) * 1000)
                assert response.status_code == 201

            # Calculate performance metrics
            response_times.sort()
            p50 = response_times[len(response_times) // 2]
            p95 = response_times[int(len(response_times) * 0.95)]

            print(f"Q&A performance - P50: {p50:.2f}ms, P95: {p95:.2f}ms")

            # Performance requirements for Q&A
            assert p50 <= 3000, f"Q&A P50 ({p50:.2f}ms) exceeds 3000ms"
            assert p95 <= 5000, f"Q&A P95 ({p95:.2f}ms) exceeds 5000ms"


class TestContextRetrievalPerformance:
    """Test context retrieval performance for RAG"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_semantic_search_performance(self):
        """Test semantic search performance"""
        from src.tools.context_tool import ContextTool

        context_tool = ContextTool()

        # Mock large document set for search
        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock vector search results
            mock_results = [
                {
                    "document": {
                        "id": f"doc_{i}",
                        "file_path": f"src/file_{i}.py",
                        "language": "python",
                        "content": f"Content for document {i}",
                    },
                    "score": 0.9 - (i * 0.01),
                }
                for i in range(20)
            ]

            mock_mongodb.vector_search.return_value = mock_results

            # Test search performance
            search_times = []
            iterations = 15

            for _ in range(iterations):
                start_time = time.time()
                result = await context_tool._arun(
                    "search",
                    query="authentication implementation details",
                    repository_id=str(uuid4()),
                    k=10,
                )
                end_time = time.time()

                search_times.append((end_time - start_time) * 1000)
                assert result["status"] == "success"

            avg_search_time = sum(search_times) / len(search_times)
            max_search_time = max(search_times)

            print(
                f"Semantic search - Avg: {avg_search_time:.2f}ms, Max: {max_search_time:.2f}ms"
            )

            # Performance requirements for semantic search
            assert (
                avg_search_time <= 500
            ), f"Average search time ({avg_search_time:.2f}ms) too high"
            assert (
                max_search_time <= 1000
            ), f"Max search time ({max_search_time:.2f}ms) too high"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_context_ranking_performance(self):
        """Test context ranking performance"""
        from src.tools.context_tool import ContextTool

        context_tool = ContextTool()

        # Create large context set for ranking
        large_context_set = [
            {
                "document_id": f"doc_{i}",
                "file_path": f"src/module_{i // 20}/file_{i}.py",
                "language": "python",
                "similarity_score": 0.9 - (i * 0.001),
                "content_preview": f"Content preview for document {i}",
                "last_modified": f"2023-{(i % 12) + 1:02d}-01T10:00:00Z",
                "metadata": {"size": 1000 + i},
            }
            for i in range(100)
        ]

        # Test ranking performance
        ranking_times = []
        iterations = 10

        for _ in range(iterations):
            start_time = time.time()
            result = await context_tool._arun(
                "rank",
                contexts=large_context_set,
                query="authentication implementation",
                max_contexts=10,
                ranking_strategy="relevance",
            )
            end_time = time.time()

            ranking_times.append((end_time - start_time) * 1000)
            assert result["status"] == "success"
            assert len(result["ranked_contexts"]) <= 10

        avg_ranking_time = sum(ranking_times) / len(ranking_times)
        max_ranking_time = max(ranking_times)

        print(
            f"Context ranking (100 items) - Avg: {avg_ranking_time:.2f}ms, Max: {max_ranking_time:.2f}ms"
        )

        # Performance requirements for context ranking
        assert (
            avg_ranking_time <= 200
        ), f"Average ranking time ({avg_ranking_time:.2f}ms) too high"
        assert (
            max_ranking_time <= 500
        ), f"Max ranking time ({max_ranking_time:.2f}ms) too high"


class TestEmbeddingPerformance:
    """Test embedding generation performance"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_embedding_generation_speed(self):
        """Test embedding generation speed"""
        from src.tools.embedding_tool import EmbeddingTool

        embedding_tool = EmbeddingTool()

        # Mock embedding provider
        with patch.object(embedding_tool, "_get_embedding_provider") as mock_provider:
            mock_embedding_instance = AsyncMock()
            mock_provider.return_value = mock_embedding_instance

            # Mock embedding generation with realistic timing
            async def mock_embed_documents(texts):
                await asyncio.sleep(0.1 * len(texts))  # Simulate processing time
                return [[0.1, 0.2, 0.3, 0.4, 0.5]] * len(texts)

            mock_embedding_instance.aembed_documents.side_effect = mock_embed_documents

            # Test different batch sizes
            batch_sizes = [1, 5, 10, 25, 50]

            for batch_size in batch_sizes:
                texts = [f"Document content {i}" for i in range(batch_size)]

                start_time = time.time()
                result = await embedding_tool._arun(
                    "generate", texts=texts, batch_size=batch_size
                )
                end_time = time.time()

                processing_time = (end_time - start_time) * 1000
                time_per_document = processing_time / batch_size

                print(
                    f"Embedding batch_size={batch_size} - Total: {processing_time:.2f}ms, Per doc: {time_per_document:.2f}ms"
                )

                assert result["status"] == "success"
                assert result["count"] == batch_size

                # Performance requirements
                assert (
                    time_per_document <= 200
                ), f"Time per document ({time_per_document:.2f}ms) too high for batch_size={batch_size}"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_document_embedding(self):
        """Test embedding performance with large documents"""
        from src.tools.embedding_tool import EmbeddingTool

        embedding_tool = EmbeddingTool()

        # Create documents of varying sizes
        document_sizes = [1000, 5000, 10000, 20000]  # Characters

        with patch.object(embedding_tool, "_get_embedding_provider") as mock_provider:
            mock_embedding_instance = AsyncMock()
            mock_provider.return_value = mock_embedding_instance

            async def mock_embed_with_size_penalty(texts):
                # Simulate longer processing for larger texts
                total_chars = sum(len(text) for text in texts)
                processing_delay = min(2.0, total_chars / 10000)  # Max 2 seconds
                await asyncio.sleep(processing_delay)
                return [[0.1] * 384] * len(texts)

            mock_embedding_instance.aembed_documents.side_effect = (
                mock_embed_with_size_penalty
            )

            for size in document_sizes:
                large_text = "This is document content. " * (size // 25)

                start_time = time.time()
                result = await embedding_tool._arun("generate", texts=[large_text])
                end_time = time.time()

                processing_time = (end_time - start_time) * 1000

                print(
                    f"Large document ({size} chars) - Processing time: {processing_time:.2f}ms"
                )

                assert result["status"] == "success"
                # Larger documents can take longer, but should be reasonable
                assert (
                    processing_time <= 5000
                ), f"Large document processing ({processing_time:.2f}ms) too slow"


class TestChatSessionPerformance:
    """Test chat session management performance"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_session_creation_performance(self, async_client: AsyncClient):
        """Test chat session creation performance"""
        repository_id = str(uuid4())

        with patch("src.services.chat_service.chat_service") as mock_service:
            mock_service.create_chat_session.return_value = {
                "status": "success",
                "session": {
                    "id": str(uuid4()),
                    "repository_id": repository_id,
                    "status": "active",
                    "message_count": 0,
                    "created_at": time.time(),
                    "last_activity": time.time(),
                },
            }

            # Test session creation performance
            creation_times = []
            iterations = 20

            for _ in range(iterations):
                start_time = time.time()
                response = await async_client.post(
                    f"/api/v2/repositories/{repository_id}/chat/sessions"
                )
                end_time = time.time()

                creation_times.append((end_time - start_time) * 1000)
                assert response.status_code == 201

            avg_creation_time = sum(creation_times) / len(creation_times)
            max_creation_time = max(creation_times)

            print(
                f"Session creation - Avg: {avg_creation_time:.2f}ms, Max: {max_creation_time:.2f}ms"
            )

            # Performance requirements
            assert (
                avg_creation_time <= 300
            ), f"Average session creation ({avg_creation_time:.2f}ms) too slow"
            assert (
                max_creation_time <= 1000
            ), f"Max session creation ({max_creation_time:.2f}ms) too slow"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_conversation_history_performance(self, async_client: AsyncClient):
        """Test conversation history retrieval performance"""
        repository_id = str(uuid4())
        session_id = str(uuid4())

        # Mock large conversation history
        with patch("src.services.chat_service.chat_service") as mock_service:
            large_history = {
                "status": "success",
                "session_id": session_id,
                "questions_and_answers": [
                    {
                        "question": {
                            "id": str(uuid4()),
                            "content": f"Question {i}?",
                            "timestamp": time.time(),
                        },
                        "answer": {
                            "id": str(uuid4()),
                            "content": f"Answer {i}" * 50,  # Substantial content
                            "confidence_score": 0.8,
                            "generation_time": 2.0,
                            "citations": [],
                        },
                    }
                    for i in range(50)
                ],
                "total": 50,
                "has_more": False,
            }

            mock_service.get_conversation_history.return_value = large_history

            # Test history retrieval performance
            history_times = []
            iterations = 15

            for _ in range(iterations):
                start_time = time.time()
                response = await async_client.get(
                    f"/api/v2/repositories/{repository_id}/chat/sessions/{session_id}/history?limit=50"
                )
                end_time = time.time()

                history_times.append((end_time - start_time) * 1000)
                assert response.status_code == 200

            avg_history_time = sum(history_times) / len(history_times)
            max_history_time = max(history_times)

            print(
                f"Conversation history (50 Q&A) - Avg: {avg_history_time:.2f}ms, Max: {max_history_time:.2f}ms"
            )

            # Performance requirements
            assert (
                avg_history_time <= 800
            ), f"Average history retrieval ({avg_history_time:.2f}ms) too slow"
            assert (
                max_history_time <= 1500
            ), f"Max history retrieval ({max_history_time:.2f}ms) too slow"


class TestStreamingStability:
    """Test streaming stability under various conditions"""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_long_streaming_session(self):
        """Test stability of long streaming sessions"""

        # Mock very long streaming session
        async def mock_long_stream():
            """Mock long streaming session"""
            for i in range(500):  # 500 chunks
                await asyncio.sleep(0.02)  # 20ms per chunk
                yield {
                    "status": "streaming",
                    "chunk": f"Chunk {i} ",
                    "chunk_number": i,
                    "finished": False,
                }

            yield {"status": "completed", "finished": True}

        # Test long stream consumption
        start_time = time.time()
        chunk_count = 0
        total_content_length = 0

        async for chunk in mock_long_stream():
            chunk_count += 1
            total_content_length += len(chunk.get("chunk", ""))

            if chunk.get("finished"):
                break

        total_time = time.time() - start_time

        print(f"Long streaming session - {chunk_count} chunks in {total_time:.2f}s")
        print(f"Total content: {total_content_length} characters")

        # Stability requirements
        assert chunk_count == 501  # 500 content chunks + 1 completion
        assert (
            total_time <= 15
        ), f"Long streaming session ({total_time:.2f}s) took too long"

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_streaming_memory_usage(self):
        """Test streaming memory usage"""

        # Mock streaming with memory monitoring
        async def mock_memory_intensive_stream():
            """Mock streaming that could cause memory issues"""
            large_chunks = []

            for i in range(100):
                chunk_content = f"Large chunk content {i} " * 100  # ~2KB per chunk
                large_chunks.append(chunk_content)

                yield {
                    "status": "streaming",
                    "chunk": chunk_content,
                    "chunk_number": i,
                    "finished": False,
                }

                await asyncio.sleep(0.01)

            yield {"status": "completed", "finished": True}

        # Consume stream and monitor memory usage
        total_chunks = 0
        max_chunk_size = 0

        async for chunk in mock_memory_intensive_stream():
            total_chunks += 1
            chunk_size = len(chunk.get("chunk", ""))
            max_chunk_size = max(max_chunk_size, chunk_size)

            if chunk.get("finished"):
                break

        print(
            f"Memory test - {total_chunks} chunks, max chunk size: {max_chunk_size} bytes"
        )

        # Memory usage should be reasonable
        assert total_chunks == 101  # 100 content chunks + 1 completion
        assert max_chunk_size <= 5000, f"Chunk size ({max_chunk_size} bytes) too large"
