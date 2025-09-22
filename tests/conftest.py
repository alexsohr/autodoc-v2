"""Pytest configuration and shared fixtures"""

import asyncio
import os
import pytest
from typing import AsyncGenerator, Generator
from httpx import AsyncClient
from fastapi.testclient import TestClient

# Add src to path for testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.main import create_app


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_app():
    """Create a test FastAPI application"""
    app = create_app()
    return app


@pytest.fixture
def client(test_app) -> Generator[TestClient, None, None]:
    """Create a test client for the FastAPI application"""
    with TestClient(test_app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(test_app) -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI application"""
    async with AsyncClient(app=test_app, base_url="http://testserver") as async_test_client:
        yield async_test_client


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        "ENVIRONMENT": "testing",
        "DEBUG": "true",
        "LOG_LEVEL": "DEBUG",
        "MONGODB_URL": "mongodb://localhost:27017",
        "MONGODB_DATABASE": "autodoc_test",
        "STORAGE_TYPE": "local",
        "STORAGE_BASE_PATH": "./test_data",
        "SECRET_KEY": "test-secret-key-for-testing-only",
        "OPENAI_API_KEY": "test-openai-key",
        "OPENAI_MODEL": "gpt-3.5-turbo",
    }
    
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    
    return env_vars


@pytest.fixture
def sample_repository_data():
    """Sample repository data for testing"""
    return {
        "url": "https://github.com/test-org/test-repo",
        "provider": "github",
        "branch": "main"
    }


@pytest.fixture
def sample_wiki_data():
    """Sample wiki data for testing"""
    return {
        "id": "test-wiki-1",
        "title": "Test Repository Documentation",
        "description": "Documentation for the test repository",
        "pages": [
            {
                "id": "overview",
                "title": "Overview",
                "description": "Project overview and architecture",
                "importance": "high",
                "file_paths": ["README.md", "docs/overview.md"],
                "related_pages": ["getting-started"],
                "content": "# Overview\n\nThis is a test repository..."
            }
        ],
        "sections": [
            {
                "id": "getting-started",
                "title": "Getting Started",
                "pages": ["overview"],
                "subsections": []
            }
        ],
        "root_sections": ["getting-started"]
    }


@pytest.fixture
def sample_chat_data():
    """Sample chat data for testing"""
    return {
        "question": "How does authentication work in this codebase?",
        "context_hint": "authentication, login, security"
    }
