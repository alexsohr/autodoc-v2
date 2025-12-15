"""Repository step definitions for BDD tests

This module contains step definitions for repository management scenarios
including registration, listing, analysis, and deletion.
"""

import pytest
from pytest_bdd import given, when, then, parsers


# =============================================================================
# Repository Registration Steps
# =============================================================================

@when(parsers.parse('I register a repository with URL "{url}"'))
def register_repository(client, context, url: str):
    """Register a new repository with the given URL."""
    payload = {"url": url}
    response = client.post("/repositories", json=payload)
    context["response"] = response
    
    if response.status_code == 201:
        repo_id = response.json()["id"]
        context["repository_id"] = repo_id
        context["repository"] = response.json()
        # Track for cleanup
        context["created_repositories"].append(repo_id)


@when(parsers.parse('I register a repository with URL "{url}" and branch "{branch}"'))
def register_repository_with_branch(client, context, url: str, branch: str):
    """Register a new repository with URL and custom branch."""
    payload = {"url": url, "branch": branch}
    response = client.post("/repositories", json=payload)
    context["response"] = response
    
    if response.status_code == 201:
        repo_id = response.json()["id"]
        context["repository_id"] = repo_id
        context["repository"] = response.json()
        # Track for cleanup
        context["created_repositories"].append(repo_id)


@given(parsers.parse('I have a registered repository with URL "{url}"'))
def have_registered_repository(client, context, url: str):
    """Ensure a repository exists with the given URL."""
    payload = {"url": url}
    response = client.post("/repositories", json=payload)
    
    if response.status_code == 201:
        repo_id = response.json()["id"]
        context["repository_id"] = repo_id
        context["repository"] = response.json()
        # Track for cleanup
        context["created_repositories"].append(repo_id)
    elif response.status_code == 409:
        # Repository already exists, that's fine for this step
        pass
    else:
        pytest.fail(
            f"Failed to create repository: {response.status_code} - {response.text}"
        )


@given("I have a registered repository")
def have_any_registered_repository(client, context):
    """Create a repository with a default URL if none exists."""
    if "repository_id" not in context:
        payload = {"url": "https://github.com/test-org/default-test-repo"}
        response = client.post("/repositories", json=payload)
        
        if response.status_code == 201:
            repo_id = response.json()["id"]
            context["repository_id"] = repo_id
            context["repository"] = response.json()
            # Track for cleanup
            context["created_repositories"].append(repo_id)


# =============================================================================
# Repository Listing Steps
# =============================================================================

@when("I list all repositories")
def list_all_repositories(client, context):
    """List all repositories."""
    response = client.get("/repositories")
    context["response"] = response


@when(parsers.parse('I list repositories with status filter "{status}"'))
def list_repositories_with_status(client, context, status: str):
    """List repositories filtered by status."""
    response = client.get("/repositories", params={"status": status})
    context["response"] = response


@when(parsers.parse("I list repositories with limit {limit:d} and offset {offset:d}"))
def list_repositories_with_pagination(client, context, limit: int, offset: int):
    """List repositories with pagination parameters."""
    response = client.get("/repositories", params={"limit": limit, "offset": offset})
    context["response"] = response


@then(parsers.parse('all repositories in the response should have status "{status}"'))
def all_repositories_should_have_status(context, status: str):
    """Assert all repositories in the response have the expected status."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    repositories = data.get("repositories", [])
    
    for repo in repositories:
        assert repo["analysis_status"] == status, (
            f"Repository {repo['id']} has status '{repo['analysis_status']}', "
            f"expected '{status}'"
        )


# =============================================================================
# Repository Details Steps
# =============================================================================

@when("I get the repository details")
def get_repository_details(client, context):
    """Get details of the current repository in context."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}")
    context["response"] = response


@when(parsers.parse('I get repository with ID "{repository_id}"'))
def get_repository_by_id(client, context, repository_id: str):
    """Get details of a repository by its ID."""
    response = client.get(f"/repositories/{repository_id}")
    context["response"] = response


# =============================================================================
# Repository Analysis Steps
# =============================================================================

@when("I trigger analysis for the repository")
def trigger_analysis(client, context):
    """Trigger analysis for the current repository in context."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.post(f"/repositories/{repository_id}/analyze")
    context["response"] = response


@when("I trigger forced analysis for the repository")
def trigger_forced_analysis(client, context):
    """Trigger forced re-analysis for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.post(
        f"/repositories/{repository_id}/analyze",
        json={"force": True}
    )
    context["response"] = response


@when("I get the analysis status for the repository")
def get_analysis_status(client, context):
    """Get the analysis status for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}/status")
    context["response"] = response


@then(parsers.parse('the analysis response status should be "{status}"'))
def analysis_status_should_be(context, status: str):
    """Assert the analysis response status field."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("status") == status, (
        f"Expected analysis status '{status}', got '{data.get('status')}'"
    )


# =============================================================================
# Repository Deletion Steps
# =============================================================================

@when("I delete the repository")
def delete_repository(client, context):
    """Delete the current repository in context."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.delete(f"/repositories/{repository_id}")
    context["response"] = response
    context["deleted_repository_id"] = repository_id


@when(parsers.parse('I delete repository with ID "{repository_id}"'))
def delete_repository_by_id(client, context, repository_id: str):
    """Delete a repository by its ID."""
    response = client.delete(f"/repositories/{repository_id}")
    context["response"] = response


@then("the repository should no longer exist")
def repository_should_not_exist(client, context):
    """Verify the deleted repository no longer exists."""
    repository_id = context.get("deleted_repository_id")
    assert repository_id is not None, "No deleted_repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}")
    assert response.status_code == 404, (
        f"Expected 404 for deleted repository, got {response.status_code}"
    )


# =============================================================================
# Webhook Configuration Steps
# =============================================================================

@when(parsers.parse('I configure the webhook with secret "{secret}"'))
def configure_webhook(client, context, secret: str):
    """Configure webhook for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    payload = {
        "webhook_secret": secret,
        "subscribed_events": ["push", "pull_request"]
    }
    response = client.put(f"/repositories/{repository_id}/webhook", json=payload)
    context["response"] = response


@when("I get the webhook configuration")
def get_webhook_config(client, context):
    """Get webhook configuration for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}/webhook")
    context["response"] = response


# =============================================================================
# Repository Property Assertions
# =============================================================================

@then(parsers.parse('the repository should have status "{status}"'))
def repository_should_have_status(context, status: str):
    """Assert the repository has the expected analysis status."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("analysis_status") == status, (
        f"Expected status '{status}', got '{data.get('analysis_status')}'"
    )


@then(parsers.parse('the repository provider should be "{provider}"'))
def repository_provider_should_be(context, provider: str):
    """Assert the repository provider matches."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("provider") == provider, (
        f"Expected provider '{provider}', got '{data.get('provider')}'"
    )


@then(parsers.parse('the repository default branch should be "{branch}"'))
def repository_default_branch_should_be(context, branch: str):
    """Assert the repository default branch matches."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("default_branch") == branch, (
        f"Expected branch '{branch}', got '{data.get('default_branch')}'"
    )


@then(parsers.parse('the repository org should be "{org}"'))
def repository_org_should_be(context, org: str):
    """Assert the repository organization matches."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("org") == org, (
        f"Expected org '{org}', got '{data.get('org')}'"
    )


@then(parsers.parse('the repository name should be "{name}"'))
def repository_name_should_be(context, name: str):
    """Assert the repository name matches."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert data.get("name") == name, (
        f"Expected name '{name}', got '{data.get('name')}'"
    )

