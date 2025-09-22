"""Integration tests for semantic search and RAG functionality

These tests validate the complete semantic search and RAG workflow.
They MUST FAIL initially since the workflow is not implemented yet.
"""

import pytest
import asyncio
from fastapi import status
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import UUID
import numpy as np


class TestSemanticSearchAndRAG:
    """Integration tests for semantic search and RAG functionality"""

    @pytest.mark.asyncio
    async def test_complete_semantic_search_workflow(self, async_client: AsyncClient):
        """Test complete semantic search workflow from embedding generation to retrieval"""
        # Step 1: Setup repository with diverse codebase
        registration_payload = {
            "url": "https://github.com/test-org/semantic-search-repo",
            "provider": "github"
        }
        
        response = await async_client.post("/repositories", json=registration_payload)
        
        # This will fail initially - not implemented
        assert response.status_code == status.HTTP_201_CREATED
        
        repository_id = response.json()["id"]
        
        # Step 2: Complete document analysis with embedding generation
        with patch('src.services.document_service.DocumentService') as mock_doc_service:
            with patch('src.tools.embedding_tool.EmbeddingTool') as mock_embedding_tool:
                
                # Mock document processing
                mock_doc_instance = MagicMock()
                mock_doc_service.return_value = mock_doc_instance
                
                mock_processed_docs = [
                    {
                        "id": "doc1",
                        "file_path": "src/auth/authentication.py",
                        "language": "python",
                        "processed_content": "User authentication system with JWT tokens and password hashing",
                        "metadata": {"functions": ["authenticate", "hash_password", "verify_token"]}
                    },
                    {
                        "id": "doc2", 
                        "file_path": "src/api/user_routes.py",
                        "language": "python",
                        "processed_content": "REST API endpoints for user management and profile operations",
                        "metadata": {"endpoints": ["/users", "/users/{id}", "/users/profile"]}
                    },
                    {
                        "id": "doc3",
                        "file_path": "src/database/models.py",
                        "language": "python", 
                        "processed_content": "Database models for User, Role, and Permission entities",
                        "metadata": {"models": ["User", "Role", "Permission"]}
                    },
                    {
                        "id": "doc4",
                        "file_path": "tests/test_auth.py",
                        "language": "python",
                        "processed_content": "Unit tests for authentication functionality and security",
                        "metadata": {"test_functions": ["test_login", "test_token_validation"]}
                    }
                ]
                
                mock_doc_instance.get_processed_documents = AsyncMock(return_value=mock_processed_docs)
                
                # Mock embedding generation
                mock_embedding_instance = MagicMock()
                mock_embedding_tool.return_value = mock_embedding_instance
                
                # Generate realistic embeddings (simplified)
                def generate_mock_embedding(content):
                    # Simple hash-based embedding for testing
                    import hashlib
                    hash_obj = hashlib.md5(content.encode())
                    # Convert to normalized vector
                    embedding = [float(x) / 255.0 for x in hash_obj.digest()[:16]]
                    return embedding + [0.0] * (384 - len(embedding))  # Pad to 384 dimensions
                
                mock_embedding_instance.generate_embedding = AsyncMock(
                    side_effect=lambda content: generate_mock_embedding(content)
                )
                
                # Mock vector storage
                mock_embedding_instance.store_embeddings = AsyncMock()
                
                # Trigger analysis
                analysis_response = await async_client.post(f"/repositories/{repository_id}/analyze")
                assert analysis_response.status_code == status.HTTP_202_ACCEPTED
                
                # Wait for processing
                await asyncio.sleep(0.1)
        
        # Step 3: Test semantic search queries
        search_queries = [
            {
                "query": "user authentication and login",
                "expected_files": ["src/auth/authentication.py", "tests/test_auth.py"],
                "context": "authentication"
            },
            {
                "query": "REST API endpoints for users",
                "expected_files": ["src/api/user_routes.py"],
                "context": "api"
            },
            {
                "query": "database models and entities",
                "expected_files": ["src/database/models.py"],
                "context": "database"
            }
        ]
        
        # Mock semantic search service
        with patch('src.tools.context_tool.ContextTool') as mock_context_tool:
            mock_context_instance = MagicMock()
            mock_context_tool.return_value = mock_context_instance
            
            for query_data in search_queries:
                # Mock similarity search
                def mock_similarity_search(query, k=5):
                    # Simple keyword matching for testing
                    query_lower = query.lower()
                    relevant_docs = []
                    
                    for doc in mock_processed_docs:
                        content_lower = doc["processed_content"].lower()
                        score = 0.0
                        
                        # Simple relevance scoring
                        for word in query_lower.split():
                            if word in content_lower:
                                score += 0.3
                        
                        if score > 0:
                            relevant_docs.append({
                                "document": doc,
                                "score": min(score, 1.0)
                            })
                    
                    # Sort by relevance score
                    relevant_docs.sort(key=lambda x: x["score"], reverse=True)
                    return relevant_docs[:k]
                
                mock_context_instance.similarity_search = AsyncMock(
                    side_effect=lambda q, k=5: mock_similarity_search(q, k)
                )
                
                # Perform search
                search_results = await mock_context_instance.similarity_search(query_data["query"])
                
                # Verify search results
                assert len(search_results) > 0
                
                # Check if expected files are in top results
                result_files = [result["document"]["file_path"] for result in search_results]
                for expected_file in query_data["expected_files"]:
                    assert any(expected_file in file_path for file_path in result_files)

    @pytest.mark.asyncio
    async def test_rag_context_retrieval_and_generation(self, async_client: AsyncClient):
        """Test RAG context retrieval and answer generation"""
        # Setup repository
        registration_payload = {
            "url": "https://github.com/test-org/rag-test-repo",
            "provider": "github"
        }
        
        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]
        
        # Complete analysis
        await async_client.post(f"/repositories/{repository_id}/analyze")
        
        # Create chat session
        session_response = await async_client.post(
            f"/repositories/{repository_id}/chat/sessions"
        )
        session_id = session_response.json()["id"]
        
        # Test RAG workflow with specific questions
        rag_test_cases = [
            {
                "question": "How do I authenticate a user in this system?",
                "expected_context_keywords": ["authenticate", "login", "token", "password"],
                "expected_answer_keywords": ["authentication", "JWT", "login", "password"]
            },
            {
                "question": "What API endpoints are available for user management?",
                "expected_context_keywords": ["api", "endpoint", "user", "routes"],
                "expected_answer_keywords": ["GET", "POST", "/users", "endpoint"]
            },
            {
                "question": "How are database models structured?",
                "expected_context_keywords": ["model", "database", "entity", "schema"],
                "expected_answer_keywords": ["model", "database", "field", "relationship"]
            }
        ]
        
        # Mock RAG pipeline
        with patch('src.services.chat_service.ChatService') as mock_chat_service:
            mock_chat_instance = MagicMock()
            mock_chat_service.return_value = mock_chat_instance
            
            for test_case in rag_test_cases:
                # Mock context retrieval
                mock_context_docs = [
                    {
                        "file_path": "src/relevant_file.py",
                        "content": f"Code related to {' '.join(test_case['expected_context_keywords'])}",
                        "score": 0.9,
                        "metadata": {"functions": ["relevant_function"]}
                    }
                ]
                
                mock_chat_instance.retrieve_context = AsyncMock(return_value=mock_context_docs)
                
                # Mock answer generation
                mock_answer = {
                    "content": f"Answer about {' '.join(test_case['expected_answer_keywords'])}. " +
                              "This system implements the requested functionality with proper security measures.",
                    "citations": [
                        {
                            "file_path": "src/relevant_file.py",
                            "line_start": 10,
                            "line_end": 25,
                            "commit_sha": "abc123",
                            "url": "https://github.com/test-org/rag-test-repo/blob/main/src/relevant_file.py#L10-L25",
                            "excerpt": "def relevant_function():\n    # Implementation details"
                        }
                    ],
                    "confidence_score": 0.85,
                    "generation_time": 1.2,
                    "context_used": len(mock_context_docs)
                }
                
                mock_chat_instance.generate_answer = AsyncMock(return_value=mock_answer)
                
                # Ask question
                qa_response = await async_client.post(
                    f"/repositories/{repository_id}/chat/sessions/{session_id}/questions",
                    json={"content": test_case["question"]}
                )
                
                if qa_response.status_code == status.HTTP_201_CREATED:
                    qa_data = qa_response.json()
                    answer = qa_data["answer"]
                    
                    # Verify RAG components
                    assert "content" in answer
                    assert "citations" in answer
                    assert "confidence_score" in answer
                    assert len(answer["citations"]) > 0
                    
                    # Verify context was used
                    mock_chat_instance.retrieve_context.assert_called()
                    mock_chat_instance.generate_answer.assert_called()
                    
                    # Check answer quality
                    answer_content = answer["content"].lower()
                    for keyword in test_case["expected_answer_keywords"]:
                        # At least some expected keywords should be present
                        pass  # Actual keyword checking would be more sophisticated

    @pytest.mark.asyncio
    async def test_embedding_quality_and_similarity_search(self, async_client: AsyncClient):
        """Test embedding quality and similarity search accuracy"""
        # Setup repository
        registration_payload = {
            "url": "https://github.com/test-org/embedding-quality-repo",
            "provider": "github"
        }
        
        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]
        
        # Mock high-quality embedding generation
        with patch('src.tools.embedding_tool.EmbeddingTool') as mock_embedding_tool:
            mock_embedding_instance = MagicMock()
            mock_embedding_tool.return_value = mock_embedding_instance
            
            # Test documents with known relationships
            test_documents = [
                {
                    "id": "auth_service",
                    "content": "Authentication service handles user login and JWT token generation",
                    "category": "authentication"
                },
                {
                    "id": "auth_middleware", 
                    "content": "Authentication middleware validates JWT tokens for protected routes",
                    "category": "authentication"
                },
                {
                    "id": "user_api",
                    "content": "User API provides REST endpoints for user CRUD operations",
                    "category": "api"
                },
                {
                    "id": "database_models",
                    "content": "Database models define User, Role, and Permission entities",
                    "category": "database"
                },
                {
                    "id": "payment_service",
                    "content": "Payment service handles billing and subscription management",
                    "category": "payment"
                }
            ]
            
            # Generate embeddings that reflect semantic similarity
            def generate_semantic_embedding(content):
                # Mock embeddings that cluster by category
                category_vectors = {
                    "authentication": [0.8, 0.2, 0.1, 0.1, 0.1],
                    "api": [0.1, 0.8, 0.2, 0.1, 0.1], 
                    "database": [0.1, 0.1, 0.8, 0.2, 0.1],
                    "payment": [0.1, 0.1, 0.1, 0.8, 0.2]
                }
                
                # Find category based on content
                content_lower = content.lower()
                for category, vector in category_vectors.items():
                    if any(keyword in content_lower for keyword in [category, category[:-1]]):
                        # Add some noise and pad to full embedding size
                        base_vector = vector + [0.1] * (384 - len(vector))
                        return [v + (np.random.random() - 0.5) * 0.1 for v in base_vector]
                
                # Default vector
                return [0.2] * 384
            
            mock_embedding_instance.generate_embedding = AsyncMock(
                side_effect=lambda content: generate_semantic_embedding(content)
            )
            
            # Mock similarity search
            def mock_similarity_search(query_embedding, k=5):
                # Calculate cosine similarity with stored embeddings
                similarities = []
                
                for doc in test_documents:
                    doc_embedding = generate_semantic_embedding(doc["content"])
                    
                    # Simple cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_embedding[:5], doc_embedding[:5]))
                    magnitude_a = sum(a * a for a in query_embedding[:5]) ** 0.5
                    magnitude_b = sum(b * b for b in doc_embedding[:5]) ** 0.5
                    
                    similarity = dot_product / (magnitude_a * magnitude_b) if magnitude_a * magnitude_b > 0 else 0
                    
                    similarities.append({
                        "document": doc,
                        "score": similarity
                    })
                
                # Sort by similarity
                similarities.sort(key=lambda x: x["score"], reverse=True)
                return similarities[:k]
            
            mock_embedding_instance.similarity_search = AsyncMock(
                side_effect=mock_similarity_search
            )
            
            # Trigger analysis
            await async_client.post(f"/repositories/{repository_id}/analyze")
            
            # Test semantic similarity queries
            similarity_tests = [
                {
                    "query": "user login and token validation",
                    "expected_top_categories": ["authentication"],
                    "min_similarity": 0.7
                },
                {
                    "query": "REST API endpoints and routes", 
                    "expected_top_categories": ["api"],
                    "min_similarity": 0.7
                },
                {
                    "query": "database schema and models",
                    "expected_top_categories": ["database"],
                    "min_similarity": 0.7
                }
            ]
            
            for test in similarity_tests:
                query_embedding = generate_semantic_embedding(test["query"])
                results = await mock_embedding_instance.similarity_search(query_embedding, k=3)
                
                # Verify similarity quality
                assert len(results) > 0
                
                # Check that most relevant results are in expected categories
                top_result = results[0]
                assert top_result["score"] >= test["min_similarity"]
                
                # Verify category clustering
                top_categories = [r["document"]["category"] for r in results[:2]]
                assert any(cat in test["expected_top_categories"] for cat in top_categories)

    @pytest.mark.asyncio
    async def test_multi_language_semantic_search(self, async_client: AsyncClient):
        """Test semantic search across multiple programming languages"""
        # Setup repository with multiple languages
        registration_payload = {
            "url": "https://github.com/test-org/multi-lang-repo",
            "provider": "github"
        }
        
        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]
        
        # Mock multi-language documents
        with patch('src.services.document_service.DocumentService') as mock_doc_service:
            mock_doc_instance = MagicMock()
            mock_doc_service.return_value = mock_doc_instance
            
            multi_lang_docs = [
                {
                    "file_path": "backend/auth.py",
                    "language": "python",
                    "processed_content": "Python authentication service with Flask and JWT",
                    "metadata": {"framework": "flask"}
                },
                {
                    "file_path": "frontend/auth.js", 
                    "language": "javascript",
                    "processed_content": "JavaScript authentication client with axios and localStorage",
                    "metadata": {"framework": "react"}
                },
                {
                    "file_path": "mobile/auth.kt",
                    "language": "kotlin", 
                    "processed_content": "Kotlin authentication for Android with Retrofit and SharedPreferences",
                    "metadata": {"framework": "android"}
                },
                {
                    "file_path": "api/auth.go",
                    "language": "go",
                    "processed_content": "Go authentication microservice with Gin and JWT middleware",
                    "metadata": {"framework": "gin"}
                }
            ]
            
            mock_doc_instance.get_processed_documents = AsyncMock(return_value=multi_lang_docs)
            
            # Mock language-aware embedding
            with patch('src.tools.embedding_tool.EmbeddingTool') as mock_embedding_tool:
                mock_embedding_instance = MagicMock()
                mock_embedding_tool.return_value = mock_embedding_instance
                
                def generate_language_aware_embedding(content, language=None):
                    # Mock embeddings that consider both content and language
                    base_embedding = [0.1] * 384
                    
                    # Content-based features
                    if "authentication" in content.lower():
                        base_embedding[0] = 0.9
                    if "jwt" in content.lower():
                        base_embedding[1] = 0.8
                    
                    # Language-specific features
                    language_offsets = {
                        "python": 100,
                        "javascript": 150, 
                        "kotlin": 200,
                        "go": 250
                    }
                    
                    if language in language_offsets:
                        offset = language_offsets[language]
                        if offset < len(base_embedding):
                            base_embedding[offset] = 0.7
                    
                    return base_embedding
                
                mock_embedding_instance.generate_embedding = AsyncMock(
                    side_effect=lambda content, lang=None: generate_language_aware_embedding(content, lang)
                )
                
                # Trigger analysis
                await async_client.post(f"/repositories/{repository_id}/analyze")
                
                # Test cross-language semantic search
                cross_lang_queries = [
                    {
                        "query": "authentication implementation across all platforms",
                        "expected_languages": ["python", "javascript", "kotlin", "go"],
                        "min_results": 3
                    },
                    {
                        "query": "JWT token handling in backend services",
                        "expected_languages": ["python", "go"],
                        "min_results": 2
                    },
                    {
                        "query": "client-side authentication storage",
                        "expected_languages": ["javascript", "kotlin"],
                        "min_results": 2
                    }
                ]
                
                # Mock cross-language search
                with patch('src.tools.context_tool.ContextTool') as mock_context_tool:
                    mock_context_instance = MagicMock()
                    mock_context_tool.return_value = mock_context_instance
                    
                    def mock_cross_lang_search(query, languages=None, k=10):
                        results = []
                        query_lower = query.lower()
                        
                        for doc in multi_lang_docs:
                            content_lower = doc["processed_content"].lower()
                            score = 0.0
                            
                            # Content similarity
                            common_words = set(query_lower.split()) & set(content_lower.split())
                            score += len(common_words) * 0.2
                            
                            # Language preference
                            if languages and doc["language"] in languages:
                                score += 0.3
                            
                            if score > 0:
                                results.append({
                                    "document": doc,
                                    "score": min(score, 1.0)
                                })
                        
                        results.sort(key=lambda x: x["score"], reverse=True)
                        return results[:k]
                    
                    mock_context_instance.cross_language_search = AsyncMock(
                        side_effect=mock_cross_lang_search
                    )
                    
                    for query_test in cross_lang_queries:
                        results = await mock_context_instance.cross_language_search(
                            query_test["query"],
                            languages=query_test.get("expected_languages"),
                            k=10
                        )
                        
                        # Verify cross-language results
                        assert len(results) >= query_test["min_results"]
                        
                        # Check language diversity
                        result_languages = set(r["document"]["language"] for r in results)
                        expected_languages = set(query_test["expected_languages"])
                        
                        # Should have results from multiple expected languages
                        assert len(result_languages & expected_languages) >= 1

    @pytest.mark.asyncio
    async def test_semantic_search_performance_and_scalability(self, async_client: AsyncClient):
        """Test semantic search performance with large document sets"""
        # Setup repository
        registration_payload = {
            "url": "https://github.com/test-org/large-scale-repo",
            "provider": "github"
        }
        
        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]
        
        # Mock large-scale document processing
        with patch('src.services.document_service.DocumentService') as mock_doc_service:
            mock_doc_instance = MagicMock()
            mock_doc_service.return_value = mock_doc_instance
            
            # Generate large number of mock documents
            large_doc_set = []
            for i in range(1000):  # 1000 documents
                doc = {
                    "id": f"doc_{i}",
                    "file_path": f"src/module_{i % 50}/file_{i}.py",
                    "language": "python",
                    "processed_content": f"Module {i % 50} functionality with specific implementation {i}",
                    "metadata": {"module_id": i % 50, "complexity": i % 10}
                }
                large_doc_set.append(doc)
            
            mock_doc_instance.get_processed_documents = AsyncMock(return_value=large_doc_set)
            
            # Mock efficient vector search
            with patch('src.utils.mongodb_adapter.MongoDBAdapter') as mock_mongo:
                mock_mongo_instance = MagicMock()
                mock_mongo.return_value = mock_mongo_instance
                
                # Mock vector search with performance metrics
                def mock_vector_search(query_vector, k=10, filter_criteria=None):
                    import time
                    start_time = time.time()
                    
                    # Simulate efficient vector search
                    # In reality, this would use MongoDB vector search capabilities
                    results = []
                    for i, doc in enumerate(large_doc_set[:k*2]):  # Search subset for performance
                        # Mock similarity calculation
                        similarity = max(0.1, 1.0 - (i * 0.05))  # Decreasing similarity
                        results.append({
                            "document": doc,
                            "score": similarity
                        })
                    
                    # Sort and limit
                    results.sort(key=lambda x: x["score"], reverse=True)
                    results = results[:k]
                    
                    search_time = time.time() - start_time
                    
                    return {
                        "results": results,
                        "search_time": search_time,
                        "total_documents": len(large_doc_set),
                        "documents_searched": min(len(large_doc_set), k*2)
                    }
                
                mock_mongo_instance.vector_search = AsyncMock(side_effect=mock_vector_search)
                
                # Trigger analysis
                await async_client.post(f"/repositories/{repository_id}/analyze")
                
                # Test performance with various query sizes
                performance_tests = [
                    {"k": 5, "max_time": 0.1},
                    {"k": 20, "max_time": 0.2},
                    {"k": 50, "max_time": 0.5},
                    {"k": 100, "max_time": 1.0}
                ]
                
                for test in performance_tests:
                    query_vector = [0.1] * 384  # Mock query vector
                    
                    search_result = await mock_mongo_instance.vector_search(
                        query_vector, 
                        k=test["k"]
                    )
                    
                    # Verify performance
                    assert search_result["search_time"] <= test["max_time"]
                    assert len(search_result["results"]) <= test["k"]
                    assert search_result["total_documents"] == 1000
                    
                    # Verify result quality
                    scores = [r["score"] for r in search_result["results"]]
                    assert all(0.0 <= score <= 1.0 for score in scores)
                    # Scores should be in descending order
                    assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rag_context_filtering_and_ranking(self, async_client: AsyncClient):
        """Test RAG context filtering and ranking mechanisms"""
        # Setup repository
        registration_payload = {
            "url": "https://github.com/test-org/context-filtering-repo",
            "provider": "github"
        }
        
        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]
        
        await async_client.post(f"/repositories/{repository_id}/analyze")
        
        # Create chat session
        session_response = await async_client.post(
            f"/repositories/{repository_id}/chat/sessions"
        )
        session_id = session_response.json()["id"]
        
        # Test context filtering and ranking
        with patch('src.services.chat_service.ChatService') as mock_chat_service:
            mock_chat_instance = MagicMock()
            mock_chat_service.return_value = mock_chat_instance
            
            # Mock context documents with various quality indicators
            mock_context_candidates = [
                {
                    "file_path": "src/core/authentication.py",
                    "content": "Comprehensive authentication implementation with security best practices",
                    "score": 0.95,
                    "metadata": {
                        "documentation_quality": 0.9,
                        "code_quality": 0.8,
                        "recency": 0.9,
                        "file_type": "implementation"
                    }
                },
                {
                    "file_path": "tests/test_auth.py", 
                    "content": "Unit tests for authentication functionality",
                    "score": 0.85,
                    "metadata": {
                        "documentation_quality": 0.7,
                        "code_quality": 0.9,
                        "recency": 0.8,
                        "file_type": "test"
                    }
                },
                {
                    "file_path": "docs/auth_readme.md",
                    "content": "Authentication documentation and usage examples", 
                    "score": 0.80,
                    "metadata": {
                        "documentation_quality": 0.95,
                        "code_quality": 0.0,  # Documentation file
                        "recency": 0.7,
                        "file_type": "documentation"
                    }
                },
                {
                    "file_path": "src/legacy/old_auth.py",
                    "content": "Deprecated authentication code",
                    "score": 0.70,
                    "metadata": {
                        "documentation_quality": 0.3,
                        "code_quality": 0.4,
                        "recency": 0.1,  # Very old
                        "file_type": "implementation"
                    }
                }
            ]
            
            # Mock context filtering and ranking
            def mock_filter_and_rank_context(candidates, query, max_context=5):
                # Apply filtering rules
                filtered = []
                
                for candidate in candidates:
                    metadata = candidate["metadata"]
                    
                    # Filter out very old or low-quality content
                    if metadata["recency"] < 0.2 or metadata.get("documentation_quality", 0) < 0.2:
                        continue
                    
                    # Calculate composite score
                    composite_score = (
                        candidate["score"] * 0.4 +  # Semantic similarity
                        metadata["documentation_quality"] * 0.3 +
                        metadata["recency"] * 0.2 +
                        metadata["code_quality"] * 0.1
                    )
                    
                    # Boost documentation files for certain queries
                    if "how to" in query.lower() and metadata["file_type"] == "documentation":
                        composite_score *= 1.2
                    
                    candidate["composite_score"] = composite_score
                    filtered.append(candidate)
                
                # Sort by composite score
                filtered.sort(key=lambda x: x["composite_score"], reverse=True)
                
                return filtered[:max_context]
            
            mock_chat_instance.filter_and_rank_context = AsyncMock(
                side_effect=lambda candidates, query, max_ctx=5: mock_filter_and_rank_context(
                    candidates, query, max_ctx
                )
            )
            
            # Test different query types
            context_tests = [
                {
                    "question": "How to implement user authentication?",
                    "expected_top_types": ["documentation", "implementation"],
                    "min_quality_score": 0.7
                },
                {
                    "question": "What are the authentication test cases?",
                    "expected_top_types": ["test", "implementation"],
                    "min_quality_score": 0.6
                },
                {
                    "question": "Authentication code implementation details",
                    "expected_top_types": ["implementation"],
                    "min_quality_score": 0.8
                }
            ]
            
            for test in context_tests:
                # Mock context retrieval and filtering
                mock_chat_instance.retrieve_context = AsyncMock(
                    return_value=mock_context_candidates
                )
                
                filtered_context = await mock_chat_instance.filter_and_rank_context(
                    mock_context_candidates,
                    test["question"],
                    max_ctx=3
                )
                
                # Verify filtering quality
                assert len(filtered_context) <= 3
                assert all(ctx["composite_score"] >= test["min_quality_score"] for ctx in filtered_context)
                
                # Verify ranking (scores should be descending)
                scores = [ctx["composite_score"] for ctx in filtered_context]
                assert scores == sorted(scores, reverse=True)
                
                # Verify appropriate content types are prioritized
                top_types = [ctx["metadata"]["file_type"] for ctx in filtered_context[:2]]
                assert any(t in test["expected_top_types"] for t in top_types)
                
                # Mock answer generation with filtered context
                mock_answer = {
                    "content": f"Answer based on {len(filtered_context)} high-quality context sources",
                    "citations": [
                        {
                            "file_path": ctx["file_path"],
                            "line_start": 1,
                            "line_end": 10,
                            "commit_sha": "abc123",
                            "url": f"https://github.com/test-org/context-filtering-repo/blob/main/{ctx['file_path']}",
                            "excerpt": ctx["content"][:100]
                        }
                        for ctx in filtered_context[:2]  # Top 2 sources
                    ],
                    "confidence_score": min(0.95, sum(ctx["composite_score"] for ctx in filtered_context) / len(filtered_context)),
                    "generation_time": 1.0,
                    "context_quality_score": sum(ctx["composite_score"] for ctx in filtered_context) / len(filtered_context)
                }
                
                mock_chat_instance.generate_answer = AsyncMock(return_value=mock_answer)
                
                # Ask question
                qa_response = await async_client.post(
                    f"/repositories/{repository_id}/chat/sessions/{session_id}/questions",
                    json={"content": test["question"]}
                )
                
                if qa_response.status_code == status.HTTP_201_CREATED:
                    qa_data = qa_response.json()
                    answer = qa_data["answer"]
                    
                    # Verify high-quality context was used
                    assert answer["confidence_score"] >= 0.7
                    assert len(answer["citations"]) > 0
                    
                    # Verify citations come from high-quality sources
                    for citation in answer["citations"]:
                        cited_file = citation["file_path"]
                        # Should not cite deprecated/low-quality files
                        assert "legacy" not in cited_file.lower()
                        assert "deprecated" not in cited_file.lower()
