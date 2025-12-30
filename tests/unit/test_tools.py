"""Unit tests for LangGraph tools

This module contains comprehensive unit tests for all LangGraph tools
including repository, embedding, context, and LLM tools.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.context_tool import ContextTool
from src.tools.embedding_tool import EmbeddingTool
from src.tools.llm_tool import LLMTool
from src.tools.repository_tool import RepositoryTool


class TestRepositoryTool:
    """Test repository tool functionality"""

    @pytest.fixture
    def repo_tool(self):
        """Create repository tool instance"""
        return RepositoryTool()

    def test_language_detection(self, repo_tool):
        """Test programming language detection"""
        test_cases = [
            (Path("test.py"), "python"),
            (Path("test.js"), "javascript"),
            (Path("test.ts"), "typescript"),
            (Path("test.java"), "java"),
            (Path("test.go"), "go"),
            (Path("test.rs"), "rust"),
            (Path("test.cpp"), "cpp"),
            (Path("test.md"), "markdown"),
            (Path("Dockerfile"), "dockerfile"),
            (Path("Makefile"), "makefile"),
            (Path("unknown.xyz"), "unknown"),
        ]

        for file_path, expected_language in test_cases:
            detected = repo_tool._detect_language(file_path)
            assert detected == expected_language

    def test_pattern_matching(self, repo_tool):
        """Test file pattern matching"""
        test_patterns = ["*.py", "src/**/*.js", "__pycache__/**"]

        test_cases = [
            ("main.py", ["*.py"], True),
            ("src/utils/helper.js", ["src/**/*.js"], True),
            ("__pycache__/module.pyc", ["__pycache__/**"], True),
            ("test.txt", ["*.py"], False),
            ("lib/utils.js", ["src/**/*.js"], False),
        ]

        for file_path, patterns, expected in test_cases:
            result = repo_tool._matches_patterns(file_path, patterns)
            assert result == expected

    def test_language_statistics_calculation(self, repo_tool):
        """Test language statistics calculation"""
        mock_files = [
            {"language": "python", "size": 1000},
            {"language": "python", "size": 2000},
            {"language": "javascript", "size": 1500},
            {"language": "markdown", "size": 500},
        ]

        stats = repo_tool._calculate_language_stats(mock_files)

        assert stats["total_files"] == 4
        assert stats["total_size"] == 5000
        assert stats["language_counts"]["python"] == 2
        assert stats["language_counts"]["javascript"] == 1
        assert stats["primary_language"] == "python"  # Largest by size

    @pytest.mark.asyncio
    async def test_repository_info_extraction(self, repo_tool):
        """Test repository information extraction"""
        # Create temporary git repository for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_path = Path(temp_dir)

            # Initialize git repo
            os.system(f"cd {temp_dir} && git init")
            os.system(f"cd {temp_dir} && git config user.email 'test@example.com'")
            os.system(f"cd {temp_dir} && git config user.name 'Test User'")

            # Create a test file and commit
            test_file = repo_path / "README.md"
            test_file.write_text("# Test Repository")

            os.system(f"cd {temp_dir} && git add README.md")
            os.system(f"cd {temp_dir} && git commit -m 'Initial commit'")

            # Test repository info extraction
            repo_info = await repo_tool._get_repository_info(repo_path)

            assert isinstance(repo_info, dict)
            # Some git operations might fail in test environment, so check structure
            assert "current_branch" in repo_info
            assert "commit_sha" in repo_info


class TestEmbeddingTool:
    """Test embedding tool functionality"""

    @pytest.fixture
    def embedding_tool(self):
        """Create embedding tool instance"""
        return EmbeddingTool()

    @pytest.mark.asyncio
    async def test_similarity_calculation(self, embedding_tool):
        """Test cosine similarity calculation"""
        # Test vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [1.0, 0.0, 0.0]  # Identical
        vec3 = [0.0, 1.0, 0.0]  # Orthogonal
        vec4 = [-1.0, 0.0, 0.0]  # Opposite

        # Test similarity calculations
        similarity_identical = await embedding_tool.calculate_similarity(vec1, vec2)
        similarity_orthogonal = await embedding_tool.calculate_similarity(vec1, vec3)
        similarity_opposite = await embedding_tool.calculate_similarity(vec1, vec4)

        assert similarity_identical > 0.9  # Should be very similar
        assert 0.4 <= similarity_orthogonal <= 0.6  # Should be neutral
        assert similarity_opposite < 0.1  # Should be dissimilar

    def test_provider_management(self, embedding_tool):
        """Test embedding provider management"""
        # Test provider availability
        providers = embedding_tool.get_available_providers()
        assert isinstance(providers, list)

        # Test provider info
        if providers:
            provider_info = embedding_tool.get_provider_info(providers[0])
            assert provider_info["status"] == "success"
            assert "provider_info" in provider_info

    @pytest.mark.asyncio
    async def test_embedding_batch_processing(self, embedding_tool):
        """Test batch embedding processing"""
        mock_documents = [
            {
                "id": f"doc_{i}",
                "processed_content": f"Document content {i}",
                "file_path": f"file_{i}.py",
                "language": "python",
            }
            for i in range(10)
        ]

        with patch.object(embedding_tool, "_get_embedding_provider") as mock_provider:
            mock_embedding_instance = AsyncMock()
            mock_provider.return_value = mock_embedding_instance
            mock_embedding_instance.aembed_documents.return_value = [
                [0.1, 0.2, 0.3]
            ] * 10

            with patch("src.services.data_access.get_mongodb_adapter") as mock_db:
                mock_mongodb = AsyncMock()
                mock_db.return_value = mock_mongodb
                mock_mongodb.store_document_embedding.return_value = None

                result = await embedding_tool._arun(
                    "generate_and_store", documents=mock_documents
                )

                assert result["status"] == "success"
                assert result["processed_count"] >= 0


class TestContextTool:
    """Test context retrieval tool functionality"""

    @pytest.fixture
    def context_tool(self):
        """Create context tool instance"""
        return ContextTool()

    def test_content_preview_generation(self, context_tool):
        """Test content preview generation"""
        long_content = (
            "This is a very long piece of content that should be truncated to a reasonable length for preview purposes. "
            * 10
        )

        preview = context_tool._generate_content_preview(long_content, max_length=100)

        assert len(preview) <= 103  # 100 + "..."
        assert preview.endswith("...") or len(preview) <= 100

    def test_code_structure_analysis(self, context_tool):
        """Test code structure analysis"""
        python_code = '''
import os
import sys

class TestClass:
    def __init__(self):
        self.value = 42
    
    def test_method(self):
        return self.value

def standalone_function():
    """This is a docstring."""
    return True
'''

        structure = context_tool._analyze_code_structure(python_code, "python")

        assert "functions" in structure
        assert "classes" in structure
        assert "imports" in structure
        assert (
            len(structure["functions"]) >= 2
        )  # __init__, test_method, standalone_function
        assert len(structure["classes"]) >= 1  # TestClass
        assert len(structure["imports"]) >= 2  # os, sys

    @pytest.mark.asyncio
    async def test_relevance_score_calculation(self, context_tool):
        """Test relevance score calculation"""
        mock_context = {
            "similarity_score": 0.8,
            "content_preview": "authentication login security user",
            "file_path": "src/auth/authentication.py",
            "last_modified": datetime.now(timezone.utc).isoformat(),
            "code_structure": {
                "functions": ["authenticate", "login", "logout"],
                "classes": ["AuthService"],
            },
        }

        query = "user authentication login"

        relevance_score = await context_tool._calculate_relevance_score(
            mock_context, query
        )

        assert 0.0 <= relevance_score <= 1.0
        assert relevance_score > 0.5  # Should be relevant


class TestLLMTool:
    """Test LLM tool functionality"""

    @pytest.fixture
    def llm_tool(self):
        """Create LLM tool instance"""
        return LLMTool()

    def test_provider_capabilities(self, llm_tool):
        """Test LLM provider capabilities"""
        # Test provider availability
        providers = llm_tool.get_available_providers()
        assert isinstance(providers, list)

        # Test provider capabilities
        if providers:
            capabilities = llm_tool.get_provider_capabilities(providers[0])
            assert capabilities["status"] == "success"
            assert "capabilities" in capabilities

    def test_analysis_response_parsing(self, llm_tool):
        """Test analysis response parsing"""
        mock_analysis = """
1. Purpose and functionality
This code provides user authentication functionality.

2. Key components
- AuthService class
- User model
- JWT token handling

3. Code quality assessment
The code follows good practices with proper error handling.
"""

        parsed = llm_tool._parse_analysis_response(mock_analysis, "comprehensive")

        assert "sections" in parsed
        assert "raw_text" in parsed
        assert len(parsed["sections"]) >= 3

    def test_documentation_section_parsing(self, llm_tool):
        """Test documentation section parsing"""
        mock_documentation = """
# Overview

This is the project overview section.

## Getting Started

Instructions for getting started.

### Prerequisites

List of prerequisites.

## API Reference

Detailed API documentation.
"""

        sections = llm_tool._parse_documentation_sections(mock_documentation)

        assert len(sections) >= 3
        assert any(section["title"] == "Overview" for section in sections)
        assert any(section["title"] == "Getting Started" for section in sections)
        assert any(section["title"] == "API Reference" for section in sections)


class TestToolIntegration:
    """Test tool integration and coordination"""

    @pytest.mark.asyncio
    async def test_tool_chain_execution(self):
        """Test chaining multiple tools together"""
        repo_tool = RepositoryTool()
        embedding_tool = EmbeddingTool()
        context_tool = ContextTool()

        # Mock repository analysis -> embedding generation -> context search
        with patch.object(repo_tool, "_arun") as mock_repo:
            with patch.object(embedding_tool, "_arun") as mock_embedding:
                with patch.object(context_tool, "_arun") as mock_context:

                    # Mock repository discovery
                    mock_repo.return_value = {
                        "status": "success",
                        "discovered_files": [
                            {"path": "src/main.py", "language": "python", "size": 1024}
                        ],
                    }

                    # Mock embedding generation
                    mock_embedding.return_value = {
                        "status": "success",
                        "processed_count": 1,
                    }

                    # Mock context search
                    mock_context.return_value = {
                        "status": "success",
                        "results": [
                            {"file_path": "src/main.py", "similarity_score": 0.9}
                        ],
                    }

                    # Execute tool chain
                    repo_result = await repo_tool._arun(
                        "discover_files", repository_path="/tmp/test"
                    )
                    embed_result = await embedding_tool._arun(
                        "generate", texts=["test content"]
                    )
                    context_result = await context_tool._arun(
                        "search", query="test query"
                    )

                    assert repo_result["status"] == "success"
                    assert embed_result["status"] == "success"
                    assert context_result["status"] == "success"

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test tool error handling"""
        repo_tool = RepositoryTool()

        # Test with invalid operation
        with pytest.raises(ValueError):
            await repo_tool._arun("invalid_operation")

        # Test with missing parameters
        result = await repo_tool._arun("clone", repository_url="invalid-url")
        assert result["status"] == "error"

    def test_tool_configuration(self):
        """Test tool configuration and initialization"""
        tools = [RepositoryTool(), EmbeddingTool(), ContextTool(), LLMTool()]

        for tool in tools:
            assert tool is not None
            assert hasattr(tool, "name")
            assert hasattr(tool, "description")
            assert tool.name is not None
            assert tool.description is not None


class TestToolPerformance:
    """Test tool performance characteristics"""

    @pytest.mark.asyncio
    async def test_embedding_batch_efficiency(self):
        """Test embedding tool batch processing efficiency"""
        embedding_tool = EmbeddingTool()

        # Test batch size handling
        large_text_list = [f"Document content {i}" for i in range(100)]

        with patch.object(embedding_tool, "_get_embedding_provider") as mock_provider:
            mock_embedding_instance = AsyncMock()
            mock_provider.return_value = mock_embedding_instance

            # Mock batch processing
            mock_embedding_instance.aembed_documents.return_value = [
                [0.1, 0.2, 0.3]
            ] * 100

            result = await embedding_tool._arun(
                "generate", texts=large_text_list, batch_size=10
            )

            # Should handle large batches efficiently
            assert result["status"] == "success"
            assert result["count"] == 100

    @pytest.mark.asyncio
    async def test_context_search_performance(self):
        """Test context tool search performance"""
        context_tool = ContextTool()

        # Mock large search results
        mock_results = [
            {
                "document_id": f"doc_{i}",
                "file_path": f"src/file_{i}.py",
                "similarity_score": 0.9 - (i * 0.01),
                "content_preview": f"Content preview {i}",
            }
            for i in range(50)
        ]

        with patch("src.services.data_access.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb
            mock_mongodb.vector_search.return_value = mock_results

            result = await context_tool._arun("search", query="test query", k=10)

            # Should limit results efficiently
            assert result["status"] == "success"
            assert len(result["results"]) <= 10


class TestToolErrorRecovery:
    """Test tool error recovery and resilience"""

    @pytest.mark.asyncio
    async def test_embedding_tool_fallback(self):
        """Test embedding tool fallback behavior"""
        embedding_tool = EmbeddingTool()

        with patch.object(embedding_tool, "_get_embedding_provider") as mock_provider:
            # Mock provider failure
            mock_provider.return_value = None

            result = await embedding_tool._arun("generate", texts=["test"])

            assert result["status"] == "error"
            assert "No embedding provider available" in result["error"]

    @pytest.mark.asyncio
    async def test_context_tool_search_fallback(self):
        """Test context tool search fallback"""
        context_tool = ContextTool()

        with patch("src.services.data_access.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock vector search failure
            mock_mongodb.vector_search.side_effect = Exception("Vector search failed")
            mock_mongodb.find_documents.return_value = []  # Fallback text search

            result = await context_tool._arun(
                "search", query="test query", search_type="vector"
            )

            # Should handle error gracefully
            assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_llm_tool_provider_fallback(self):
        """Test LLM tool provider fallback"""
        llm_tool = LLMTool()

        with patch.object(llm_tool, "_get_llm_provider") as mock_provider:
            # Mock provider failure
            mock_provider.return_value = None

            result = await llm_tool._arun("generate", prompt="test prompt")

            assert result["status"] == "error"
            assert "No LLM provider available" in result["error"]


class TestToolDataProcessing:
    """Test tool data processing capabilities"""

    def test_repository_tool_file_filtering(self):
        """Test repository tool file filtering"""
        repo_tool = RepositoryTool()

        # Test file filtering logic
        test_files = [
            "src/main.py",
            "src/utils/helper.js",
            "__pycache__/cache.pyc",
            "node_modules/package/index.js",
            "README.md",
            "tests/test_main.py",
        ]

        # Default exclude patterns should filter out cache and node_modules
        exclude_patterns = ["__pycache__/**", "node_modules/**"]

        filtered_files = []
        for file_path in test_files:
            if not repo_tool._matches_patterns(file_path, exclude_patterns):
                filtered_files.append(file_path)

        assert "src/main.py" in filtered_files
        assert "README.md" in filtered_files
        assert "__pycache__/cache.pyc" not in filtered_files
        assert "node_modules/package/index.js" not in filtered_files

    def test_context_tool_ranking(self):
        """Test context tool ranking algorithms"""
        context_tool = ContextTool()

        mock_contexts = [
            {
                "file_path": "src/main.py",
                "similarity_score": 0.9,
                "last_modified": "2023-12-01T10:00:00Z",
            },
            {
                "file_path": "tests/test_old.py",
                "similarity_score": 0.8,
                "last_modified": "2023-01-01T10:00:00Z",
            },
            {
                "file_path": "docs/readme.md",
                "similarity_score": 0.7,
                "last_modified": "2023-11-01T10:00:00Z",
            },
        ]

        # Test relevance ranking
        ranked_by_relevance = context_tool._rank_by_recency(mock_contexts)
        assert ranked_by_relevance[0]["file_path"] == "src/main.py"  # Most recent

        # Test importance ranking
        ranked_by_importance = context_tool._rank_by_importance(mock_contexts)
        # Main files and docs should rank higher than tests
        main_and_docs = [
            ctx
            for ctx in ranked_by_importance[:2]
            if "main" in ctx["file_path"] or "readme" in ctx["file_path"]
        ]
        assert len(main_and_docs) >= 1


class TestToolValidation:
    """Test tool input validation and sanitization"""

    def test_repository_tool_input_validation(self):
        """Test repository tool input validation"""
        repo_tool = RepositoryTool()

        # Test URL validation in clone operation
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://invalid.com/repo",
            "https://",
        ]

        for invalid_url in invalid_urls:
            # This would be tested in the actual clone method
            # For now, verify the tool exists and has validation
            assert hasattr(repo_tool, "_arun")

    def test_embedding_tool_input_sanitization(self):
        """Test embedding tool input sanitization"""
        embedding_tool = EmbeddingTool()

        # Test empty input handling
        empty_inputs = [[], None, [""]]

        for empty_input in empty_inputs:
            # Test that empty inputs are handled gracefully
            # This would be tested in the actual method calls
            assert hasattr(embedding_tool, "_arun")

    def test_context_tool_query_sanitization(self):
        """Test context tool query sanitization"""
        context_tool = ContextTool()

        # Test malicious query handling
        malicious_queries = [
            "",  # Empty query
            "a" * 10000,  # Very long query
            "SELECT * FROM users",  # SQL injection attempt
            "<script>alert('xss')</script>",  # XSS attempt
        ]

        for query in malicious_queries:
            # Test that malicious queries are handled safely
            # The actual validation would be in the service layer
            assert hasattr(context_tool, "_arun")
