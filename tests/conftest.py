"""Pytest configuration and shared fixtures for full E2E testing

This module provides fixtures for true end-to-end testing with:
- Real MongoDB database
- Real Git repository cloning
- Real OpenAI LLM/embedding API calls
- No mocking of any services
"""

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Add src to path for testing
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.api.main import create_app


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables before any tests run.
    
    Uses a separate test database to isolate from development data.
    OPENAI_API_KEY is expected to be in .env file and loaded by Settings.
    """
    # Set test database name to isolate from development data
    os.environ["MONGODB_DATABASE"] = "autodoc_e2e_test"
    os.environ["ENVIRONMENT"] = "testing"
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    yield
    
    # Environment cleanup happens automatically when process ends


@pytest.fixture
def test_app():
    """Create a test FastAPI application for full E2E testing.
    
    This fixture uses the real application with NO mocking:
    - Real MongoDB connection (test database)
    - Real repository classes (RepositoryRepository, CodeDocumentRepository, etc.)
    - Real service classes (RepositoryService, ChatService, etc.)
    - Real LLMTool (OpenAI API calls)
    - Real RepositoryTool (Git cloning)
    - Real EmbeddingTool (OpenAI embeddings)
    """
    app = create_app()
    
    # No dependency overrides - use everything real
    yield app


@pytest.fixture
def client(test_app) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application.
    
    Uses real MongoDB and real external services.
    Data persists between requests within a test but is cleaned up
    by the BDD context fixture after each scenario.
    """
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI application."""
    from httpx import ASGITransport

    transport = ASGITransport(app=test_app)
    async with AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as async_test_client:
        yield async_test_client


@pytest.fixture
def sample_repository_data():
    """Sample repository data for testing - uses real public repository"""
    return {
        "url": "https://github.com/yusufocaliskan/python-flask-mvc",
        "provider": "github",
        "branch": "main",
    }


@pytest.fixture
def sample_wiki_data():
    """Sample wiki data for testing"""
    return {
        "id": "test-wiki-1",
        "title": "Test Repository Documentation",
        "description": "Documentation for the test repository",
        "sections": [
            {
                "id": "getting-started",
                "title": "Getting Started",
                "pages": [
                    {
                        "id": "overview",
                        "title": "Overview",
                        "description": "Project overview and architecture",
                        "importance": "high",
                        "file_paths": ["README.md", "docs/overview.md"],
                        "related_pages": [],
                        "content": "# Overview\n\nThis is a test repository...",
                    }
                ],
            }
        ],
    }


@pytest.fixture
def sample_chat_data():
    """Sample chat data for testing"""
    return {
        "question": "How does authentication work in this codebase?",
        "context_hint": "authentication, login, security",
    }
