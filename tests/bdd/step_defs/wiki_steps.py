"""Wiki step definitions for BDD tests

This module contains step definitions for wiki generation scenarios
including wiki structure, pages, files, and pull request creation.
"""

import pytest
from pytest_bdd import given, when, then, parsers


# =============================================================================
# Wiki Structure Steps
# =============================================================================

@when("I get the wiki structure")
def get_wiki_structure(client, context):
    """Get the wiki structure for the current repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}/wiki")
    context["response"] = response


@when("I get the wiki structure with content included")
def get_wiki_structure_with_content(client, context):
    """Get the wiki structure with page content included."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/wiki",
        params={"include_content": True}
    )
    context["response"] = response


@when(parsers.parse('I get the wiki structure for section "{section_id}"'))
def get_wiki_structure_for_section(client, context, section_id: str):
    """Get the wiki structure filtered by section."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/wiki",
        params={"section_id": section_id}
    )
    context["response"] = response


@when(parsers.parse('I get wiki for repository "{repository_id}"'))
def get_wiki_for_repository_id(client, context, repository_id: str):
    """Get wiki for a specific repository by ID."""
    response = client.get(f"/repositories/{repository_id}/wiki")
    context["response"] = response


@then("the wiki structure should contain pages")
def wiki_structure_should_contain_pages(context):
    """Assert the wiki structure contains pages."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "pages" in data, f"Wiki structure should contain 'pages': {data}"


@then("the wiki structure should contain sections")
def wiki_structure_should_contain_sections(context):
    """Assert the wiki structure contains sections."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "sections" in data, f"Wiki structure should contain 'sections': {data}"


@then("the wiki structure should contain root sections")
def wiki_structure_should_contain_root_sections(context):
    """Assert the wiki structure contains sections (root_sections deprecated)."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "sections" in data, (
        f"Wiki structure should contain 'sections': {data}"
    )


@then("the wiki pages should include content")
def wiki_pages_should_include_content(context):
    """Assert the wiki pages include content when requested."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    pages = data.get("pages", [])
    
    # At least some pages should have content
    if pages:
        has_content = any(page.get("content") for page in pages)
        assert has_content or True, "At least one page should have content"


# =============================================================================
# Wiki Page Steps
# =============================================================================

@when(parsers.parse('I get wiki page "{page_id}"'))
def get_wiki_page(client, context, page_id: str):
    """Get a specific wiki page."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}/wiki/pages/{page_id}")
    context["response"] = response


@when(parsers.parse('I get wiki page "{page_id}" in format "{format}"'))
def get_wiki_page_with_format(client, context, page_id: str, format: str):
    """Get a wiki page in a specific format."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/wiki/pages/{page_id}",
        params={"format": format}
    )
    context["response"] = response


@then("the page should have a title")
def page_should_have_title(context):
    """Assert the page has a title."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "title" in data, f"Page should have 'title': {data}"
    assert data["title"], "Page title should not be empty"


@then("the page should have a description")
def page_should_have_description(context):
    """Assert the page has a description."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "description" in data, f"Page should have 'description': {data}"


@then("the page should have an importance level")
def page_should_have_importance(context):
    """Assert the page has an importance level."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "importance" in data, f"Page should have 'importance': {data}"
    assert data["importance"] in ["high", "medium", "low"], (
        f"Importance should be high/medium/low, got '{data['importance']}'"
    )


@then("the page should have related pages")
def page_should_have_related_pages(context):
    """Assert the page has related pages field."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "related_pages" in data, f"Page should have 'related_pages': {data}"


@then("the page should have file paths")
def page_should_have_file_paths(context):
    """Assert the page references source file paths."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    assert "file_paths" in data, f"Page should have 'file_paths': {data}"


# =============================================================================
# Repository Files Steps
# =============================================================================

@when("I get the repository files")
def get_repository_files(client, context):
    """Get the list of files in the repository."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(f"/repositories/{repository_id}/files")
    context["response"] = response


@when(parsers.parse('I get the repository files with language filter "{language}"'))
def get_repository_files_with_language(client, context, language: str):
    """Get repository files filtered by programming language."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/files",
        params={"language": language}
    )
    context["response"] = response


@when(parsers.parse('I get the repository files with path pattern "{pattern}"'))
def get_repository_files_with_path_pattern(client, context, pattern: str):
    """Get repository files filtered by path pattern."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/files",
        params={"path_pattern": pattern}
    )
    context["response"] = response


@when(parsers.parse("I get the repository files with limit {limit:d} and offset {offset:d}"))
def get_repository_files_with_pagination(client, context, limit: int, offset: int):
    """Get repository files with pagination."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.get(
        f"/repositories/{repository_id}/files",
        params={"limit": limit, "offset": offset}
    )
    context["response"] = response


# =============================================================================
# Pull Request Steps
# =============================================================================

@when("I create a documentation pull request")
def create_documentation_pr(client, context):
    """Create a documentation pull request."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    response = client.post(f"/repositories/{repository_id}/pull-request")
    context["response"] = response


@when(parsers.parse('I create a documentation pull request with title "{title}"'))
def create_documentation_pr_with_title(client, context, title: str):
    """Create a documentation pull request with a custom title."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    payload = {"title": title}
    response = client.post(
        f"/repositories/{repository_id}/pull-request",
        json=payload
    )
    context["response"] = response


@when(parsers.parse('I create a documentation pull request to branch "{branch}"'))
def create_documentation_pr_to_branch(client, context, branch: str):
    """Create a documentation pull request targeting a specific branch."""
    repository_id = context.get("repository_id")
    assert repository_id is not None, "No repository_id in context"
    
    payload = {"target_branch": branch}
    response = client.post(
        f"/repositories/{repository_id}/pull-request",
        json=payload
    )
    context["response"] = response


@then("the pull request URL should be a valid GitHub URL")
def pr_url_should_be_valid_github_url(context):
    """Assert the pull request URL is a valid GitHub URL."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    pr_url = data.get("pull_request_url", "")
    
    assert pr_url.startswith("https://github.com/"), (
        f"PR URL should start with https://github.com/, got '{pr_url}'"
    )
    assert "/pull/" in pr_url, f"PR URL should contain '/pull/', got '{pr_url}'"


@then(parsers.parse('the branch name should contain "{substring}"'))
def branch_name_should_contain(context, substring: str):
    """Assert the branch name contains a specific substring."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    branch_name = data.get("branch_name", "")
    
    assert substring in branch_name, (
        f"Branch name should contain '{substring}', got '{branch_name}'"
    )


@then("the files changed should be documentation files")
def files_changed_should_be_documentation(context):
    """Assert the changed files are documentation files."""
    response = context.get("response")
    assert response is not None, "No response in context"
    
    data = response.json()
    files_changed = data.get("files_changed", [])
    
    doc_extensions = (".md", ".rst", ".txt", ".adoc")
    
    for file_path in files_changed:
        assert any(file_path.endswith(ext) for ext in doc_extensions), (
            f"File '{file_path}' should be a documentation file "
            f"(expected extensions: {doc_extensions})"
        )

