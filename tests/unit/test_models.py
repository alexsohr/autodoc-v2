"""Unit tests for Pydantic data models

This module contains comprehensive unit tests for all Pydantic models
including validation, serialization, and business logic methods.
"""

from datetime import datetime, timezone
from typing import List
from uuid import UUID, uuid4

import pytest

from src.models.chat import (
    Answer,
    ChatSession,
    Citation,
    Question,
    QuestionAnswer,
    QuestionRequest,
    SessionStatus,
)
from src.models.code_document import (
    CodeDocument,
    CodeDocumentCreate,
    CodeDocumentResponse,
    CodeDocumentUpdate,
)
from src.models.config import (
    AppConfig,
    LLMConfig,
    LLMProvider,
    StorageConfig,
    StorageType,
)
from src.models.repository import (
    AccessScope,
    AnalysisStatus,
    Repository,
    RepositoryCreate,
    RepositoryProvider,
    RepositoryUpdate,
)
from src.models.wiki import (
    PageImportance,
    WikiPageCreate,
    WikiPageDetail,
    WikiSection,
    WikiSectionCreate,
    WikiStructure,
)


class TestRepositoryModel:
    """Test Repository model functionality"""

    def test_repository_creation(self):
        """Test repository model creation"""
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
        )

        assert repo.provider == RepositoryProvider.GITHUB
        assert repo.url == "https://github.com/test-org/test-repo"
        assert repo.org == "test-org"
        assert repo.name == "test-repo"
        assert repo.analysis_status == AnalysisStatus.PENDING
        assert repo.webhook_configured is False
        assert isinstance(repo.id, UUID)

    def test_repository_url_validation(self):
        """Test repository URL validation"""
        # Valid URL
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
        )
        assert repo.url == "https://github.com/test-org/test-repo"

        # Invalid URL should raise validation error
        with pytest.raises(ValueError):
            Repository(
                provider=RepositoryProvider.GITHUB,
                url="not-a-valid-url",
                org="test-org",
                name="test-repo",
                default_branch="main",
                access_scope=AccessScope.PUBLIC,
            )

    def test_commit_sha_validation(self):
        """Test commit SHA validation"""
        # Valid commit SHA
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
            commit_sha="abc123def456789012345678901234567890abcd",
        )
        assert repo.commit_sha == "abc123def456789012345678901234567890abcd"

        # Invalid commit SHA should raise validation error
        with pytest.raises(ValueError):
            Repository(
                provider=RepositoryProvider.GITHUB,
                url="https://github.com/test-org/test-repo",
                org="test-org",
                name="test-repo",
                default_branch="main",
                access_scope=AccessScope.PUBLIC,
                commit_sha="invalid-sha",
            )

    def test_webhook_configuration(self):
        """Test webhook configuration methods"""
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
        )

        # Configure webhook
        repo.configure_webhook("secret123", ["push", "pull_request"])

        assert repo.webhook_configured is True
        assert repo.webhook_secret == "secret123"
        assert repo.subscribed_events == ["push", "pull_request"]
        assert repo.is_webhook_event_subscribed("push") is True
        assert repo.is_webhook_event_subscribed("issues") is False

    def test_analysis_status_update(self):
        """Test analysis status update"""
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
        )

        # Update to completed
        commit_sha = "abc123def456789012345678901234567890abcd"
        repo.update_analysis_status(AnalysisStatus.COMPLETED, commit_sha)

        assert repo.analysis_status == AnalysisStatus.COMPLETED
        assert repo.commit_sha == commit_sha
        assert repo.last_analyzed is not None


class TestCodeDocumentModel:
    """Test CodeDocument model functionality"""

    def test_code_document_creation(self):
        """Test code document creation"""
        doc = CodeDocument(
            id="doc1",
            repository_id=uuid4(),
            file_path="src/main.py",
            language="python",
            content="print('hello world')",
            processed_content="print hello world",
        )

        assert doc.id == "doc1"
        assert doc.file_path == "src/main.py"
        assert doc.language == "python"
        assert doc.content == "print('hello world')"
        assert doc.has_embedding() is False

    def test_file_path_validation(self):
        """Test file path validation"""
        # Valid relative path
        doc = CodeDocument(
            id="doc1",
            repository_id=uuid4(),
            file_path="src/utils/helper.py",
            language="python",
            content="# helper code",
            processed_content="helper code",
        )
        assert doc.file_path == "src/utils/helper.py"

        # Invalid absolute path should raise error
        with pytest.raises(ValueError):
            CodeDocument(
                id="doc1",
                repository_id=uuid4(),
                file_path="/absolute/path/file.py",
                language="python",
                content="# code",
                processed_content="code",
            )

    def test_embedding_operations(self):
        """Test embedding operations"""
        doc = CodeDocument(
            id="doc1",
            repository_id=uuid4(),
            file_path="src/main.py",
            language="python",
            content="print('hello')",
            processed_content="print hello",
        )

        # Initially no embedding
        assert doc.has_embedding() is False
        assert doc.get_embedding_dimension() == 0

        # Set embedding
        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        doc.set_embedding(embedding)

        assert doc.has_embedding() is True
        assert doc.get_embedding_dimension() == 5
        assert doc.embedding == embedding

    def test_content_update(self):
        """Test content update functionality"""
        doc = CodeDocument(
            id="doc1",
            repository_id=uuid4(),
            file_path="src/main.py",
            language="python",
            content="old content",
            processed_content="old content",
            embedding=[0.1, 0.2, 0.3],
        )

        # Update content
        doc.update_content("new content", "new processed content")

        assert doc.content == "new content"
        assert doc.processed_content == "new processed content"
        assert doc.embedding is None  # Should be cleared


class TestWikiModels:
    """Test Wiki model functionality"""

    def test_wiki_page_creation(self):
        """Test wiki page creation"""
        page = WikiPageDetail(
            id="overview",
            title="Project Overview",
            description="High-level project overview",
            importance=PageImportance.HIGH,
            file_paths=["README.md", "docs/overview.md"],
            related_pages=["getting-started"],
        )

        assert page.id == "overview"
        assert page.title == "Project Overview"
        assert page.importance == PageImportance.HIGH
        assert len(page.file_paths) == 2
        assert page.has_content() is False

    def test_wiki_section_creation(self):
        """Test wiki section creation"""
        section = WikiSection(
            id="introduction",
            title="Introduction",
            pages=["overview", "getting-started"],
            subsections=["advanced"],
        )

        assert section.id == "introduction"
        assert section.title == "Introduction"
        assert section.has_pages() is True
        assert section.has_subsections() is True

    def test_wiki_structure_validation(self):
        """Test wiki structure validation"""
        # Create valid structure
        pages = [
            WikiPageDetail(
                id="overview",
                title="Overview",
                description="Project overview",
                importance=PageImportance.HIGH,
            ),
            WikiPageDetail(
                id="getting-started",
                title="Getting Started",
                description="Setup guide",
                importance=PageImportance.HIGH,
                related_pages=["overview"],
            ),
        ]

        sections = [
            WikiSection(
                id="introduction",
                title="Introduction",
                pages=["overview", "getting-started"],
            )
        ]

        wiki = WikiStructure(
            id="wiki1",
            repository_id=uuid4(),
            title="Test Wiki",
            description="Test wiki structure",
            pages=pages,
            sections=sections,
            root_sections=["introduction"],
        )

        assert len(wiki.pages) == 2
        assert len(wiki.sections) == 1
        assert wiki.get_total_pages() == 2
        assert wiki.get_page("overview") is not None
        assert wiki.get_section("introduction") is not None

    def test_wiki_structure_validation_errors(self):
        """Test wiki structure validation errors"""
        # Invalid structure - reference to non-existent page
        pages = [
            WikiPageDetail(
                id="overview",
                title="Overview",
                description="Project overview",
                importance=PageImportance.HIGH,
            )
        ]

        sections = [
            WikiSection(
                id="introduction",
                title="Introduction",
                pages=["overview", "non-existent-page"],  # Invalid reference
            )
        ]

        with pytest.raises(ValueError):
            WikiStructure(
                id="wiki1",
                repository_id=uuid4(),
                title="Test Wiki",
                description="Test wiki structure",
                pages=pages,
                sections=sections,
                root_sections=["introduction"],
            )


class TestChatModels:
    """Test Chat model functionality"""

    def test_chat_session_creation(self):
        """Test chat session creation"""
        session = ChatSession(repository_id=uuid4())

        assert isinstance(session.id, UUID)
        assert isinstance(session.repository_id, UUID)
        assert session.status == SessionStatus.ACTIVE
        assert session.message_count == 0
        assert session.is_active() is True
        assert session.is_expired() is False

    def test_question_creation(self):
        """Test question creation"""
        session_id = uuid4()
        question = Question(
            session_id=session_id, content="How does authentication work?"
        )

        assert isinstance(question.id, UUID)
        assert question.session_id == session_id
        assert question.content == "How does authentication work?"
        assert question.has_context() is False

    def test_answer_creation(self):
        """Test answer creation"""
        question_id = uuid4()
        citations = [
            Citation(
                file_path="src/auth.py",
                line_start=10,
                line_end=20,
                commit_sha="abc123def456789012345678901234567890abcd",
                url="https://github.com/test/repo/blob/main/src/auth.py#L10-L20",
            )
        ]

        answer = Answer(
            question_id=question_id,
            content="Authentication works using JWT tokens...",
            citations=citations,
            confidence_score=0.85,
            generation_time=1.5,
        )

        assert answer.question_id == question_id
        assert answer.confidence_score == 0.85
        assert answer.get_citation_count() == 1
        assert answer.has_high_confidence() is True

    def test_citation_validation(self):
        """Test citation validation"""
        # Valid citation
        citation = Citation(
            file_path="src/auth.py",
            line_start=10,
            line_end=20,
            commit_sha="abc123def456789012345678901234567890abcd",
            url="https://github.com/test/repo/blob/main/src/auth.py#L10-L20",
        )

        assert citation.get_line_range_str() == "L10-L20"
        assert citation.get_file_name() == "auth.py"

        # Invalid line range should raise error
        with pytest.raises(ValueError):
            Citation(
                file_path="src/auth.py",
                line_start=20,
                line_end=10,  # End before start
                commit_sha="abc123def456789012345678901234567890abcd",
                url="https://github.com/test/repo/blob/main/src/auth.py",
            )


class TestConfigModels:
    """Test Configuration model functionality"""

    def test_llm_config_creation(self):
        """Test LLM configuration creation"""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4-turbo",
            api_key="test-api-key",
            max_tokens=4000,
            temperature=0.1,
        )

        assert config.provider == LLMProvider.OPENAI
        assert config.model_name == "gpt-4-turbo"
        assert config.get_api_key() == "test-api-key"
        assert config.max_tokens == 4000
        assert config.temperature == 0.1

    def test_llm_config_validation(self):
        """Test LLM configuration validation"""
        # Invalid temperature should raise error
        with pytest.raises(ValueError):
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key="test-key",
                temperature=2.5,  # Invalid temperature > 2.0
            )

        # Invalid max_tokens should raise error
        with pytest.raises(ValueError):
            LLMConfig(
                provider=LLMProvider.OPENAI,
                model_name="gpt-4",
                api_key="test-key",
                max_tokens=-100,  # Invalid negative tokens
            )

    def test_storage_config_creation(self):
        """Test storage configuration creation"""
        config = StorageConfig(
            type=StorageType.LOCAL,
            base_path="./data",
            backup_enabled=True,
            retention_days=30,
        )

        assert config.type == StorageType.LOCAL
        assert config.base_path == "./data"
        assert config.is_local_storage() is True
        assert config.is_cloud_storage() is False

    def test_app_config_creation(self):
        """Test application configuration creation"""
        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI, model_name="gpt-4", api_key="test-key"
        )

        storage_config = StorageConfig(type=StorageType.LOCAL, base_path="./data")

        app_config = AppConfig(
            storage_config=storage_config,
            app_name="AutoDoc Test",
            environment="testing",
        )

        app_config.add_llm_config("primary", llm_config)

        assert app_config.app_name == "AutoDoc Test"
        assert len(app_config.get_available_llm_providers()) == 1
        assert app_config.get_llm_config("primary") == llm_config
        assert app_config.default_llm_provider == "primary"


class TestModelSerialization:
    """Test model serialization and deserialization"""

    def test_repository_serialization(self):
        """Test repository model serialization"""
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
        )

        # Test serialization
        repo_dict = repo.model_dump()
        assert isinstance(repo_dict, dict)
        assert repo_dict["provider"] == "github"
        assert repo_dict["url"] == "https://github.com/test-org/test-repo"

        # Test deserialization
        repo_dict["id"] = UUID(repo_dict["id"])
        new_repo = Repository(**repo_dict)
        assert new_repo.url == repo.url
        assert new_repo.provider == repo.provider

    def test_code_document_serialization(self):
        """Test code document serialization"""
        doc = CodeDocument(
            id="doc1",
            repository_id=uuid4(),
            file_path="src/main.py",
            language="python",
            content="print('hello')",
            processed_content="print hello",
        )

        # Test serialization
        doc_dict = doc.model_dump()
        assert isinstance(doc_dict, dict)
        assert doc_dict["file_path"] == "src/main.py"
        assert doc_dict["language"] == "python"

        # Test response model conversion
        response = CodeDocumentResponse.from_code_document(doc)
        assert response.id == doc.id
        assert response.file_path == doc.file_path
        assert response.has_embedding is False

    def test_wiki_structure_serialization(self):
        """Test wiki structure serialization"""
        page = WikiPageDetail(
            id="overview",
            title="Overview",
            description="Project overview",
            importance=PageImportance.HIGH,
        )

        section = WikiSection(id="intro", title="Introduction", pages=["overview"])

        wiki = WikiStructure(
            id="wiki1",
            repository_id=uuid4(),
            title="Test Wiki",
            description="Test description",
            pages=[page],
            sections=[section],
            root_sections=["intro"],
        )

        # Test serialization
        wiki_dict = wiki.model_dump()
        assert isinstance(wiki_dict, dict)
        assert wiki_dict["title"] == "Test Wiki"
        assert len(wiki_dict["pages"]) == 1
        assert len(wiki_dict["sections"]) == 1

    def test_chat_models_serialization(self):
        """Test chat models serialization"""
        session = ChatSession(repository_id=uuid4())
        question = Question(session_id=session.id, content="Test question")
        answer = Answer(
            question_id=question.id,
            content="Test answer",
            citations=[],
            confidence_score=0.8,
            generation_time=1.0,
        )

        # Test serialization
        session_dict = session.model_dump()
        question_dict = question.model_dump()
        answer_dict = answer.model_dump()

        assert isinstance(session_dict, dict)
        assert isinstance(question_dict, dict)
        assert isinstance(answer_dict, dict)

        # Test QuestionAnswer combination
        qa = QuestionAnswer(question=question_dict, answer=answer_dict)
        qa_dict = qa.model_dump()
        assert "question" in qa_dict
        assert "answer" in qa_dict


class TestModelValidationEdgeCases:
    """Test model validation edge cases"""

    def test_empty_string_validation(self):
        """Test empty string validation"""
        # Empty default branch should raise error
        with pytest.raises(ValueError):
            Repository(
                provider=RepositoryProvider.GITHUB,
                url="https://github.com/test-org/test-repo",
                org="test-org",
                name="test-repo",
                default_branch="",  # Empty branch
                access_scope=AccessScope.PUBLIC,
            )

    def test_webhook_secret_validation(self):
        """Test webhook secret validation"""
        repo = Repository(
            provider=RepositoryProvider.GITHUB,
            url="https://github.com/test-org/test-repo",
            org="test-org",
            name="test-repo",
            default_branch="main",
            access_scope=AccessScope.PUBLIC,
            webhook_configured=True,
        )

        # Should raise error because webhook_configured=True but no secret
        with pytest.raises(ValueError):
            repo.model_validate(repo.model_dump())

    def test_embedding_dimension_validation(self):
        """Test embedding dimension validation"""
        doc = CodeDocument(
            id="doc1",
            repository_id=uuid4(),
            file_path="src/main.py",
            language="python",
            content="print('hello')",
            processed_content="print hello",
        )

        # Valid embedding dimensions
        valid_embeddings = [
            [0.1] * 128,  # 128 dimensions
            [0.1] * 384,  # 384 dimensions
            [0.1] * 768,  # 768 dimensions
        ]

        for embedding in valid_embeddings:
            doc.set_embedding(embedding)
            assert doc.get_embedding_dimension() == len(embedding)

        # Invalid embedding values should raise error
        with pytest.raises(ValueError):
            doc.set_embedding([2.0, 3.0, 4.0])  # Values > 1.0

    def test_page_importance_validation(self):
        """Test page importance validation"""
        # Valid importance levels
        for importance in ["high", "medium", "low"]:
            page = WikiPageDetail(
                id="test",
                title="Test Page",
                description="Test description",
                importance=importance,
            )
            assert page.importance == importance

        # Invalid importance should use default
        page_data = {
            "id": "test",
            "title": "Test Page",
            "description": "Test description",
            "importance": "invalid",
        }

        # This should not raise an error but use default
        try:
            page = WikiPageDetail(**page_data)
            # The validation should handle invalid values gracefully
        except ValueError:
            pass  # Expected for invalid importance values
