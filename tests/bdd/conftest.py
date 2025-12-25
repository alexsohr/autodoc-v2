"""BDD-specific pytest configuration and fixtures

This module provides shared fixtures and context management for BDD tests.
It integrates with the main conftest.py fixtures for test app and clients.

For E2E testing, this module handles real database cleanup using Motor/Beanie.

Usage:
    # Run tests with cleanup (default)
    pytest tests/bdd/test_repositories.py -v

    # Run tests WITHOUT cleanup (for troubleshooting)
    pytest tests/bdd/test_repositories.py -v --no-cleanup
"""

import asyncio

import pytest

# API prefix for cleanup operations
API_PREFIX = "/api/v2"


def pytest_addoption(parser):
    """Add custom command line options for BDD tests."""
    parser.addoption(
        "--no-cleanup",
        action="store_true",
        default=False,
        help="Disable cleanup of test data after scenarios (for troubleshooting)"
    )


@pytest.fixture
def no_cleanup(request):
    """Fixture to check if cleanup is disabled."""
    return request.config.getoption("--no-cleanup")


@pytest.fixture
def context(client, request, no_cleanup):
    """Shared context dictionary for passing data between BDD steps.
    
    This fixture provides a mutable dictionary that persists across
    all steps within a single scenario. It also handles cleanup of
    created resources after each scenario using real HTTP DELETE calls.
    
    Use --no-cleanup flag to disable cleanup for troubleshooting.
    """
    ctx = {
        "created_repositories": [],
        "created_sessions": [],
    }
    
    yield ctx
    
    # Cleanup: Delete all created resources after the scenario (unless disabled)
    if no_cleanup:
        print("\n[CLEANUP DISABLED] Skipping scenario cleanup. Test data preserved in database.")
        print(f"  Created repositories: {ctx.get('created_repositories', [])}")
        print(f"  Created sessions: {ctx.get('created_sessions', [])}")
    else:
        _cleanup_test_data(client, ctx)


def _cleanup_test_data(client, ctx):
    """Clean up all test data created during the scenario.
    
    This function deletes repositories, chat sessions, and other
    resources that were created during test execution.
    Uses real HTTP DELETE calls to the API which will delete from MongoDB.
    """
    # Delete chat sessions first (they depend on repositories)
    for session_info in ctx.get("created_sessions", []):
        repository_id = session_info.get("repository_id")
        session_id = session_info.get("session_id")
        if repository_id and session_id:
            try:
                client.delete(
                    f"{API_PREFIX}/repositories/{repository_id}/chat/sessions/{session_id}"
                )
            except Exception:
                pass  # Ignore errors during cleanup
    
    # Delete repositories (this should cascade delete related data)
    for repository_id in ctx.get("created_repositories", []):
        try:
            client.delete(f"{API_PREFIX}/repositories/{repository_id}")
        except Exception:
            pass  # Ignore errors during cleanup


@pytest.fixture
def response_context(context):
    """Helper fixture to access the last HTTP response from context."""
    return context.get("response")


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_database(request):
    """Clean up test database before and after the entire test session.
    
    This fixture runs once at the start and end of the test session
    to ensure a clean state. It drops all collections in the test database
    to provide isolation between test runs.
    
    Use --no-cleanup flag to disable this cleanup for troubleshooting.
    """
    no_cleanup = request.config.getoption("--no-cleanup", default=False)
    
    # Run async cleanup in a new event loop
    loop = asyncio.new_event_loop()
    
    try:
        # Setup: Clean database before tests (unless disabled)
        if no_cleanup:
            print("\n[CLEANUP DISABLED] Skipping pre-test database cleanup.")
        else:
            loop.run_until_complete(_drop_all_collections())
        
        yield
        
        # Teardown: Clean database after all tests complete (unless disabled)
        if no_cleanup:
            print("\n[CLEANUP DISABLED] Skipping post-test database cleanup. Data preserved for inspection.")
        else:
            loop.run_until_complete(_drop_all_collections())
    finally:
        loop.close()


async def _drop_all_collections():
    """Drop all collections in the test database.
    
    This provides a clean slate for E2E tests by removing all data
    from the test database. Uses Motor/Beanie to interact with MongoDB.
    """
    # Import here to avoid circular imports and ensure settings are loaded
    from src.repository.database import close_mongodb, get_database, init_mongodb

    try:
        # Initialize database connection
        await init_mongodb()
        
        # Get database instance
        db = await get_database()
        
        # Get list of all collections
        collection_names = await db.list_collection_names()
        
        # Drop each collection
        for collection_name in collection_names:
            try:
                await db.drop_collection(collection_name)
                print(f"Dropped collection: {collection_name}")
            except Exception as e:
                print(f"Warning: Could not drop collection {collection_name}: {e}")
        
        print(f"Test database cleanup complete. Dropped {len(collection_names)} collections.")
        
    except Exception as e:
        print(f"Warning: Database cleanup failed: {e}")
        # Don't fail the test run if cleanup fails - tests may still work
    finally:
        # Close database connection
        try:
            await close_mongodb()
        except Exception:
            pass  # Ignore close errors
