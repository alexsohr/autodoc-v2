"""BDD-specific pytest configuration and fixtures

This module provides shared fixtures and context management for BDD tests.
It integrates with the main conftest.py fixtures for test app and clients.
"""

import pytest


@pytest.fixture
def context(client, request):
    """Shared context dictionary for passing data between BDD steps.
    
    This fixture provides a mutable dictionary that persists across
    all steps within a single scenario. It also handles cleanup of
    created resources after each scenario.
    """
    ctx = {
        "created_repositories": [],
        "created_sessions": [],
    }
    
    yield ctx
    
    # Cleanup: Delete all created resources after the scenario
    _cleanup_test_data(client, ctx)


def _cleanup_test_data(client, ctx):
    """Clean up all test data created during the scenario.
    
    This function deletes repositories, chat sessions, and other
    resources that were created during test execution.
    """
    # Delete chat sessions first (they depend on repositories)
    for session_info in ctx.get("created_sessions", []):
        repository_id = session_info.get("repository_id")
        session_id = session_info.get("session_id")
        if repository_id and session_id:
            try:
                client.delete(
                    f"/repositories/{repository_id}/chat/sessions/{session_id}"
                )
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Delete repositories (this should cascade delete related data)
    for repository_id in ctx.get("created_repositories", []):
        try:
            client.delete(f"/repositories/{repository_id}")
        except Exception:
            pass  # Ignore errors during cleanup


@pytest.fixture
def response_context(context):
    """Helper fixture to access the last HTTP response from context."""
    return context.get("response")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_database():
    """Clean up test database before and after the entire test session.
    
    This fixture runs once at the start and end of the test session
    to ensure a clean state.
    """
    # Setup: Could clean existing test data here if needed
    yield
    # Teardown: Final cleanup after all tests complete
    # Note: Individual test cleanup happens in the context fixture
