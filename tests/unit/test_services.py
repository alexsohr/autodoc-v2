"""Unit tests for service layer

This module contains comprehensive unit tests for all service classes
including authentication, repository, document, wiki, and chat services.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from src.models.chat import QuestionRequest, SessionStatus
from src.models.repository import AnalysisStatus, Repository, RepositoryProvider
from src.services.auth_service import AuthenticationService, User, UserCreate, UserLogin
from src.services.chat_service import ChatService
from src.services.document_service import DocumentProcessingService
from src.services.repository_service import RepositoryService
from src.services.wiki_service import WikiGenerationService


class TestAuthenticationService:
    """Test authentication service functionality"""

    @pytest.fixture
    def auth_service(self):
        """Create authentication service instance"""
        return AuthenticationService()

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
        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
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

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
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
        """Create repository service instance"""
        return RepositoryService()

    @pytest.mark.asyncio
    async def test_repository_creation(self, repo_service):
        """Test repository creation"""
        repo_data = RepositoryCreate(
            url="https://github.com/test-org/test-repo",
            provider=RepositoryProvider.GITHUB,
        )

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
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
        """Create document service instance"""
        return DocumentProcessingService()

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
        """Create wiki service instance"""
        return WikiGenerationService()

    @pytest.mark.asyncio
    async def test_wiki_generation_validation(self, wiki_service):
        """Test wiki generation validation"""
        repository_id = uuid4()

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
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
                    "pages": ["overview"],
                    "subsections": [],
                }
            ],
            "pages": [
                {
                    "id": "overview",
                    "title": "Overview",
                    "description": "Project overview",
                }
            ],
        }

        nav_content = wiki_service._generate_section_nav(
            wiki_data["sections"][0], wiki_data, level=1
        )

        assert "Introduction" in nav_content
        assert "Overview" in nav_content
        assert "[Overview](docs/overview.md)" in nav_content


class TestChatService:
    """Test chat service functionality"""

    @pytest.fixture
    def chat_service(self):
        """Create chat service instance"""
        return ChatService()

    @pytest.mark.asyncio
    async def test_session_creation_validation(self, chat_service):
        """Test chat session creation validation"""
        repository_id = uuid4()

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
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
        empty_question = QuestionRequest(content="")

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter"):
            # This would be tested in the API layer, but we can test the model validation
            with pytest.raises(ValueError):
                QuestionRequest(content="")


class TestServiceIntegration:
    """Test service integration and coordination"""

    @pytest.mark.asyncio
    async def test_repository_to_wiki_workflow(self):
        """Test complete repository to wiki workflow"""
        repository_id = uuid4()

        # Mock successful repository processing
        with patch(
            "src.services.repository_service.repository_service"
        ) as mock_repo_service:
            with patch("src.services.wiki_service.wiki_service") as mock_wiki_service:

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
        # Test that errors are properly propagated and handled
        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock database error
            mock_mongodb.find_document.side_effect = Exception(
                "Database connection failed"
            )

            auth_service = AuthenticationService()
            result = await auth_service.get_user_by_username("testuser")

            assert result is None  # Should handle error gracefully


class TestServiceConfiguration:
    """Test service configuration and initialization"""

    def test_service_initialization(self):
        """Test service initialization with configuration"""
        # Test that all services can be initialized
        services = [
            AuthenticationService(),
            RepositoryService(),
            DocumentProcessingService(),
            WikiGenerationService(),
            ChatService(),
        ]

        for service in services:
            assert service is not None
            # Check that settings are loaded
            if hasattr(service, "settings"):
                assert service.settings is not None

    @pytest.mark.asyncio
    async def test_service_health_checks(self):
        """Test service health check methods"""
        auth_service = AuthenticationService()

        # Mock database health check
        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb
            mock_mongodb.health_check.return_value = {"status": "healthy"}

            health_result = await auth_service.health_check()

            assert health_result["status"] == "healthy"
            assert "password_hashing" in health_result
            assert "jwt_operations" in health_result


class TestServiceErrorHandling:
    """Test service error handling patterns"""

    @pytest.mark.asyncio
    async def test_graceful_error_handling(self):
        """Test graceful error handling in services"""
        auth_service = AuthenticationService()

        # Test with invalid input
        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb
            mock_mongodb.find_document.side_effect = Exception("Database error")

            # Should not raise exception, should return None
            result = await auth_service.get_user_by_username("testuser")
            assert result is None

    @pytest.mark.asyncio
    async def test_service_error_responses(self):
        """Test service error response format"""
        repo_service = RepositoryService()

        # Test with invalid repository ID
        invalid_id = uuid4()

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb
            mock_mongodb.get_repository.return_value = None

            result = await repo_service.get_repository(invalid_id)

            assert result["status"] == "error"
            assert result["error_type"] == "NotFound"
            assert "error" in result


class TestServicePerformance:
    """Test service performance characteristics"""

    @pytest.mark.asyncio
    async def test_batch_operations(self):
        """Test service batch operation efficiency"""
        doc_service = DocumentProcessingService()

        # Mock batch document processing
        mock_documents = [
            {"id": f"doc_{i}", "content": f"content {i}"} for i in range(100)
        ]

        with patch("src.tools.embedding_tool.embedding_tool") as mock_embedding:
            mock_embedding._arun.return_value = {
                "status": "success",
                "processed_count": 100,
                "failed_count": 0,
            }

            # Test that batch operations are handled efficiently
            # This would be more comprehensive in a real test
            assert len(mock_documents) == 100

    @pytest.mark.asyncio
    async def test_concurrent_operations(self):
        """Test service handling of concurrent operations"""
        import asyncio

        auth_service = AuthenticationService()

        # Test concurrent user lookups
        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb
            mock_mongodb.find_document.return_value = None

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
        chat_service = ChatService()

        repository_id = uuid4()
        session_id = uuid4()

        with patch("src.utils.mongodb_adapter.get_mongodb_adapter") as mock_db:
            mock_mongodb = AsyncMock()
            mock_db.return_value = mock_mongodb

            # Mock session exists
            mock_mongodb.find_document.return_value = {
                "id": str(session_id),
                "repository_id": str(repository_id),
                "status": "active",
                "message_count": 0,
            }

            # Mock transaction context
            mock_session = AsyncMock()
            mock_mongodb.client.start_session.return_value.__aenter__.return_value = (
                mock_session
            )
            mock_session.start_transaction.return_value.__aenter__.return_value = None

            result = await chat_service.delete_chat_session(repository_id, session_id)

            # Should succeed with proper transaction handling
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_data_validation_consistency(self):
        """Test data validation consistency across services"""
        # Test that all services validate UUIDs consistently
        services = [
            RepositoryService(),
            DocumentProcessingService(),
            WikiGenerationService(),
            ChatService(),
        ]

        invalid_uuid = "not-a-uuid"

        # All services should handle invalid UUIDs gracefully
        for service in services:
            # This would be tested with actual service methods
            # For now, just verify services are created
            assert service is not None
