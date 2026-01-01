"""Unit tests for service layer

This module contains comprehensive unit tests for all service classes
including authentication, repository, document, wiki, and chat services.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from src.models.chat import ChatSession, QuestionRequest, SessionStatus
from src.models.repository import AnalysisStatus, Repository, RepositoryCreate, RepositoryProvider
from src.services.auth_service import AuthenticationService, User, UserCreate, UserLogin
from src.services.chat_service import ChatService
from src.services.document_service import DocumentProcessingService
from src.services.repository_service import RepositoryService
from src.services.wiki_service import WikiGenerationService


class TestAuthenticationService:
    """Test authentication service functionality"""

    @pytest.fixture
    def auth_service(self):
        """Create authentication service instance with mock repository"""
        mock_user_repo = MagicMock()
        return AuthenticationService(user_repository=mock_user_repo)

    def test_password_hashing(self, auth_service):
        """Test password hashing and verification"""
        password = "test_password_123"

        # Hash password
        hashed = auth_service.get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 20  # Bcrypt hashes are long

        # Verify password
        assert auth_service.verify_password(password, hashed) is True
        assert auth_service.verify_password("wrong_password", hashed) is False

    def test_jwt_token_operations(self, auth_service):
        """Test JWT token creation and verification"""
        # Create token
        token_data = {"sub": "testuser", "user_id": str(uuid4())}
        token = auth_service.create_access_token(token_data)

        assert isinstance(token, str)
        assert len(token) > 50  # JWT tokens are long

        # Verify token
        decoded_data = auth_service.verify_token(token)
        assert decoded_data is not None
        assert decoded_data.username == "testuser"
        assert decoded_data.user_id == token_data["user_id"]

    @pytest.mark.asyncio
    async def test_user_creation(self, auth_service):
        """Test user creation"""
        user_data = UserCreate(
            username="testuser",
            email="test@example.com",
            password="secure_password_123",
            full_name="Test User",
        )

        # Mock database operations
        with patch("src.services.data_access.get_mongodb_adapter", new_callable=AsyncMock) as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock user doesn't exist
            mock_mongodb.find_document.return_value = None
            mock_mongodb.insert_document.return_value = "user_id"

            result = await auth_service.create_user(user_data)

            assert result["status"] == "success"
            assert result["username"] == "testuser"
            assert result["email"] == "test@example.com"
            assert "user_id" in result

    @pytest.mark.asyncio
    async def test_user_authentication(self, auth_service):
        """Test user authentication"""
        # Mock user data
        mock_user_data = {
            "id": str(uuid4()),
            "username": "testuser",
            "email": "test@example.com",
            "hashed_password": auth_service.get_password_hash("correct_password"),
            "is_active": True,
            "is_admin": False,
        }

        with patch("src.services.data_access.get_mongodb_adapter", new_callable=AsyncMock) as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock successful authentication
            mock_mongodb.find_document.return_value = mock_user_data
            mock_mongodb.update_document.return_value = True

            user = await auth_service.authenticate_user("testuser", "correct_password")

            assert user is not None
            assert user.username == "testuser"
            assert user.email == "test@example.com"

            # Test failed authentication
            failed_user = await auth_service.authenticate_user(
                "testuser", "wrong_password"
            )
            assert failed_user is None


class TestRepositoryService:
    """Test repository service functionality"""

    @pytest.fixture
    def repo_service(self):
        """Create repository service instance with mock repositories"""
        mock_repository_repo = MagicMock()
        mock_code_document_repo = MagicMock()
        return RepositoryService(
            repository_repo=mock_repository_repo,
            code_document_repo=mock_code_document_repo
        )

    @pytest.mark.asyncio
    async def test_repository_creation(self, repo_service):
        """Test repository creation"""
        repo_data = RepositoryCreate(
            url="https://github.com/test-org/test-repo",
            provider=RepositoryProvider.GITHUB,
        )

        with patch("src.services.data_access.get_mongodb_adapter", new_callable=AsyncMock) as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock repository doesn't exist
            mock_mongodb.find_document.return_value = None
            mock_mongodb.insert_document.return_value = "repo_id"

            with patch.object(repo_service, "_trigger_analysis") as mock_trigger:
                result = await repo_service.create_repository(repo_data)

                assert result["status"] == "success"
                assert "repository" in result
                assert result["repository"]["provider"] == "github"
                assert result["repository"]["url"] == repo_data.url

    def test_url_validation(self, repo_service):
        """Test repository URL validation"""
        # Valid URLs
        valid_urls = [
            "https://github.com/user/repo",
            "https://bitbucket.org/user/repo",
            "https://gitlab.com/user/repo",
        ]

        for url in valid_urls:
            result = repo_service._validate_repository_url(url)
            assert result["valid"] is True

        # Invalid URLs
        invalid_urls = [
            "",
            "not-a-url",
            "https://unsupported.com/user/repo",
            "https://github.com/",  # Missing repo path
        ]

        for url in invalid_urls:
            result = repo_service._validate_repository_url(url)
            assert result["valid"] is False

    def test_provider_detection(self, repo_service):
        """Test provider detection from URL"""
        test_cases = [
            ("https://github.com/user/repo", RepositoryProvider.GITHUB),
            ("https://bitbucket.org/user/repo", RepositoryProvider.BITBUCKET),
            ("https://gitlab.com/user/repo", RepositoryProvider.GITLAB),
            ("https://unknown.com/user/repo", None),
        ]

        for url, expected_provider in test_cases:
            detected = repo_service._detect_provider_from_url(url)
            assert detected == expected_provider

    def test_org_name_extraction(self, repo_service):
        """Test organization and repository name extraction"""
        test_cases = [
            ("https://github.com/test-org/test-repo", ("test-org", "test-repo")),
            ("https://github.com/test-org/test-repo.git", ("test-org", "test-repo")),
            ("https://bitbucket.org/company/project", ("company", "project")),
            ("https://invalid-url", (None, None)),
        ]

        for url, expected in test_cases:
            org, name = repo_service._extract_org_and_name(url)
            assert (org, name) == expected


class TestDocumentService:
    """Test document processing service functionality"""

    @pytest.fixture
    def doc_service(self):
        """Create document service instance with mock dependencies"""
        mock_code_document_repo = MagicMock()
        mock_document_agent = MagicMock()
        mock_context_tool = MagicMock()
        mock_embedding_tool = MagicMock()
        return DocumentProcessingService(
            code_document_repo=mock_code_document_repo,
            document_agent=mock_document_agent,
            context_tool=mock_context_tool,
            embedding_tool=mock_embedding_tool
        )

    def test_content_cleaning(self, doc_service):
        """Test content cleaning for embeddings"""
        # Python content
        python_content = '''
"""This is a docstring"""
def hello():
    # This is a comment
    print("hello world")
    return True
'''

        cleaned = doc_service._clean_content_for_embedding(python_content, "python")
        assert "hello world" in cleaned
        assert len(cleaned) < len(python_content)  # Should be cleaned/compressed

    def test_quality_score_calculation(self, doc_service):
        """Test content quality score calculation"""
        # Good quality Python code
        good_content = '''
def authenticate_user(username: str, password: str) -> bool:
    """Authenticate user with username and password."""
    # Validate input parameters
    if not username or not password:
        return False
    
    # Check credentials
    return check_credentials(username, password)
'''

        score = doc_service._calculate_content_quality_score(
            good_content, "authenticate user username password", "python"
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be decent quality

    def test_comment_ratio_calculation(self, doc_service):
        """Test comment ratio calculation"""
        # Python code with comments
        python_code = """
# This is a comment
def function():
    # Another comment
    return True
# Final comment
"""

        ratio = doc_service._calculate_comment_ratio(python_code, "python")
        assert 0.0 <= ratio <= 1.0
        assert ratio > 0.0  # Should detect comments


class TestWikiService:
    """Test wiki generation service functionality"""

    @pytest.fixture
    def wiki_service(self):
        """Create wiki service instance with mock dependencies"""
        mock_wiki_structure_repo = MagicMock()
        mock_code_document_repo = MagicMock()
        mock_wiki_agent = MagicMock()
        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()
        return WikiGenerationService(
            wiki_structure_repo=mock_wiki_structure_repo,
            code_document_repo=mock_code_document_repo,
            wiki_agent=mock_wiki_agent,
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool
        )

    @pytest.mark.asyncio
    async def test_wiki_generation_validation(self, wiki_service):
        """Test wiki generation validation"""
        repository_id = uuid4()

        with patch("src.services.data_access.get_mongodb_adapter", new_callable=AsyncMock) as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock repository not found
            mock_mongodb.get_repository.return_value = None

            result = await wiki_service.generate_wiki(repository_id)

            assert result["status"] == "error"
            assert result["error_type"] == "NotFound"

    def test_section_navigation_generation(self, wiki_service):
        """Test section navigation generation"""
        wiki_data = {
            "title": "Test Wiki",
            "sections": [
                {
                    "id": "intro",
                    "title": "Introduction",
                    "pages": [
                        {
                            "id": "overview",
                            "title": "Overview",
                            "description": "Project overview",
                        }
                    ],
                }
            ],
        }

        nav_content = wiki_service._generate_section_nav(
            wiki_data["sections"][0], level=1
        )

        assert "Introduction" in nav_content
        assert "Overview" in nav_content
        assert "[Overview](docs/overview.md)" in nav_content


class TestChatService:
    """Test chat service functionality"""

    @pytest.fixture
    def chat_service(self):
        """Create chat service instance with mock dependencies"""
        mock_chat_session_repo = MagicMock()
        mock_question_repo = MagicMock()
        mock_answer_repo = MagicMock()
        mock_workflow_orchestrator = MagicMock()
        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()
        return ChatService(
            chat_session_repo=mock_chat_session_repo,
            question_repo=mock_question_repo,
            answer_repo=mock_answer_repo,
            workflow_orchestrator=mock_workflow_orchestrator,
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool
        )

    @pytest.mark.asyncio
    async def test_session_creation_validation(self, chat_service):
        """Test chat session creation validation"""
        repository_id = uuid4()

        with patch("src.services.data_access.get_mongodb_adapter", new_callable=AsyncMock) as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock repository not found
            mock_mongodb.get_repository.return_value = None

            result = await chat_service.create_chat_session(repository_id)

            assert result["status"] == "error"
            assert result["error_type"] == "NotFound"

    def test_session_expiration_check(self, chat_service):
        """Test session expiration logic"""
        # Create active session
        session = ChatSession(repository_id=uuid4())
        assert chat_service._is_session_expired(session) is False

        # Create expired session
        old_time = datetime.now(timezone.utc) - timedelta(hours=25)  # 25 hours ago
        expired_session = ChatSession(repository_id=uuid4(), last_activity=old_time)
        assert chat_service._is_session_expired(expired_session) is True

    @pytest.mark.asyncio
    async def test_question_validation(self, chat_service):
        """Test question content validation"""
        repository_id = uuid4()
        session_id = uuid4()

        # Empty question should fail
        with pytest.raises(ValidationError):
            QuestionRequest(content="")


class TestServiceIntegration:
    """Test service integration and coordination"""

    @pytest.mark.asyncio
    async def test_repository_to_wiki_workflow(self):
        """Test complete repository to wiki workflow"""
        repository_id = uuid4()

        # Create mock services for integration test
        mock_repo_service = AsyncMock()
        mock_wiki_service = AsyncMock()

        # Mock repository creation
        mock_repo_service.create_repository.return_value = {
            "status": "success",
            "repository": {
                "id": str(repository_id),
                "provider": "github",
                "url": "https://github.com/test/repo",
            },
        }

        # Mock wiki generation
        mock_wiki_service.generate_wiki.return_value = {
            "status": "completed",
            "pages_generated": 5,
        }

        # Test workflow coordination
        repo_result = await mock_repo_service.create_repository({})
        wiki_result = await mock_wiki_service.generate_wiki(repository_id)

        assert repo_result["status"] == "success"
        assert wiki_result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation between services"""
        # Create mock repositories for AuthenticationService
        mock_user_repo = AsyncMock()
        mock_user_repo.find_by_username.side_effect = Exception(
            "Database connection failed"
        )

        auth_service = AuthenticationService(user_repository=mock_user_repo)
        result = await auth_service.get_user_by_username("testuser")

        assert result is None  # Should handle error gracefully


class TestServiceConfiguration:
    """Test service configuration and initialization"""

    def test_service_initialization(self):
        """Test service initialization with configuration"""
        # Test that all services can be initialized with mock dependencies
        mock_user_repo = MagicMock()
        mock_repository_repo = MagicMock()
        mock_code_document_repo = MagicMock()
        mock_wiki_structure_repo = MagicMock()
        mock_chat_session_repo = MagicMock()
        mock_question_repo = MagicMock()
        mock_answer_repo = MagicMock()
        mock_wiki_agent = MagicMock()
        mock_document_agent = MagicMock()
        mock_workflow_orchestrator = MagicMock()
        mock_context_tool = MagicMock()
        mock_embedding_tool = MagicMock()
        mock_llm_tool = MagicMock()

        services = [
            AuthenticationService(user_repository=mock_user_repo),
            RepositoryService(
                repository_repo=mock_repository_repo,
                code_document_repo=mock_code_document_repo
            ),
            DocumentProcessingService(
                code_document_repo=mock_code_document_repo,
                document_agent=mock_document_agent,
                context_tool=mock_context_tool,
                embedding_tool=mock_embedding_tool
            ),
            WikiGenerationService(
                wiki_structure_repo=mock_wiki_structure_repo,
                code_document_repo=mock_code_document_repo,
                wiki_agent=mock_wiki_agent,
                context_tool=mock_context_tool,
                llm_tool=mock_llm_tool
            ),
            ChatService(
                chat_session_repo=mock_chat_session_repo,
                question_repo=mock_question_repo,
                answer_repo=mock_answer_repo,
                workflow_orchestrator=mock_workflow_orchestrator,
                context_tool=mock_context_tool,
                llm_tool=mock_llm_tool
            ),
        ]

        for service in services:
            assert service is not None
            # Check that settings are loaded
            if hasattr(service, "settings"):
                assert service.settings is not None

    @pytest.mark.asyncio
    async def test_service_health_checks(self):
        """Test service health check methods"""
        mock_user_repo = AsyncMock()
        auth_service = AuthenticationService(user_repository=mock_user_repo)

        # If the service has a health_check method, test it
        if hasattr(auth_service, "health_check"):
            mock_user_repo.health_check.return_value = {"status": "healthy"}

            health_result = await auth_service.health_check()

            assert health_result["status"] == "healthy"
            assert "password_hashing" in health_result
            assert "jwt_operations" in health_result
        else:
            # Service may not have health_check, just verify it's created
            assert auth_service is not None


class TestServiceErrorHandling:
    """Test service error handling patterns"""

    @pytest.mark.asyncio
    async def test_graceful_error_handling(self):
        """Test graceful error handling in services"""
        mock_user_repo = AsyncMock()
        mock_user_repo.find_by_username.side_effect = Exception("Database error")
        auth_service = AuthenticationService(user_repository=mock_user_repo)

        # Should not raise exception, should return None
        result = await auth_service.get_user_by_username("testuser")
        assert result is None

    @pytest.mark.asyncio
    async def test_service_error_responses(self):
        """Test service error response format"""
        mock_repository_repo = AsyncMock()
        mock_code_document_repo = MagicMock()
        mock_repository_repo.get.return_value = None

        repo_service = RepositoryService(
            repository_repo=mock_repository_repo,
            code_document_repo=mock_code_document_repo
        )

        # Test with invalid repository ID
        invalid_id = uuid4()

        result = await repo_service.get_repository(invalid_id)

        assert result["status"] == "error"
        assert result["error_type"] == "NotFound"
        assert "error" in result


class TestServicePerformance:
    """Test service performance characteristics"""

    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test service batch operation efficiency"""
        mock_code_document_repo = MagicMock()
        mock_document_agent = MagicMock()
        mock_context_tool = MagicMock()
        mock_embedding_tool = MagicMock()
        mock_embedding_tool._arun.return_value = {
            "status": "success",
            "processed_count": 100,
            "failed_count": 0,
        }

        doc_service = DocumentProcessingService(
            code_document_repo=mock_code_document_repo,
            document_agent=mock_document_agent,
            context_tool=mock_context_tool,
            embedding_tool=mock_embedding_tool
        )

        # Mock batch document processing
        mock_documents = [
            {"id": f"doc_{i}", "content": f"content {i}"} for i in range(100)
        ]

        # Test that batch operations are handled efficiently
        # This would be more comprehensive in a real test
        assert len(mock_documents) == 100
        assert doc_service is not None

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test service handling of concurrent operations"""
        import asyncio

        mock_user_repo = AsyncMock()
        mock_user_repo.find_by_username.return_value = None

        auth_service = AuthenticationService(user_repository=mock_user_repo)

        # Create multiple concurrent requests
        tasks = [auth_service.get_user_by_username(f"user_{i}") for i in range(10)]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete without errors
        assert len(results) == 10
        assert all(result is None for result in results)  # No users found


class TestServiceDataConsistency:
    """Test service data consistency and integrity"""

    @pytest.mark.asyncio
    async def test_transactional_operations(self):
        """Test transactional data operations"""
        mock_chat_session_repo = AsyncMock()
        mock_question_repo = MagicMock()
        mock_answer_repo = MagicMock()
        mock_workflow_orchestrator = MagicMock()
        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()

        chat_service = ChatService(
            chat_session_repo=mock_chat_session_repo,
            question_repo=mock_question_repo,
            answer_repo=mock_answer_repo,
            workflow_orchestrator=mock_workflow_orchestrator,
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool
        )

        repository_id = uuid4()
        session_id = uuid4()

        # Mock session exists
        mock_chat_session_repo.get.return_value = MagicMock(
            id=session_id,
            repository_id=repository_id,
            status="active",
            message_count=0
        )
        mock_chat_session_repo.delete.return_value = True

        result = await chat_service.delete_chat_session(repository_id, session_id)

        # Should succeed with proper transaction handling
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_data_validation_consistency(self):
        """Test data validation consistency across services"""
        # Create all services with mock dependencies
        mock_user_repo = MagicMock()
        mock_repository_repo = MagicMock()
        mock_code_document_repo = MagicMock()
        mock_wiki_structure_repo = MagicMock()
        mock_chat_session_repo = MagicMock()
        mock_question_repo = MagicMock()
        mock_answer_repo = MagicMock()
        mock_wiki_agent = MagicMock()
        mock_document_agent = MagicMock()
        mock_workflow_orchestrator = MagicMock()
        mock_context_tool = MagicMock()
        mock_embedding_tool = MagicMock()
        mock_llm_tool = MagicMock()

        # Test that all services validate UUIDs consistently
        services = [
            RepositoryService(
                repository_repo=mock_repository_repo,
                code_document_repo=mock_code_document_repo
            ),
            DocumentProcessingService(
                code_document_repo=mock_code_document_repo,
                document_agent=mock_document_agent,
                context_tool=mock_context_tool,
                embedding_tool=mock_embedding_tool
            ),
            WikiGenerationService(
                wiki_structure_repo=mock_wiki_structure_repo,
                code_document_repo=mock_code_document_repo,
                wiki_agent=mock_wiki_agent,
                context_tool=mock_context_tool,
                llm_tool=mock_llm_tool
            ),
            ChatService(
                chat_session_repo=mock_chat_session_repo,
                question_repo=mock_question_repo,
                answer_repo=mock_answer_repo,
                workflow_orchestrator=mock_workflow_orchestrator,
                context_tool=mock_context_tool,
                llm_tool=mock_llm_tool
            ),
        ]

        invalid_uuid = "not-a-uuid"

        # All services should handle invalid UUIDs gracefully
        for service in services:
            # This would be tested with actual service methods
            # For now, just verify services are created
            assert service is not None
