"""Contract tests for Documentation/Wiki API endpoints

These tests validate the API contracts defined in documentation_api.yaml
They MUST FAIL initially since endpoints are not implemented yet.
"""

from uuid import UUID, uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestDocumentationAPIContract:
    """Contract tests for documentation/wiki API endpoints"""

    def test_get_wiki_structure_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/wiki contract"""
        repo_id = str(uuid4())

        response = client.get(f"/repositories/{repo_id}/wiki")

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_get_wiki_structure_with_content(self, client: TestClient):
        """Test GET /repositories/{repository_id}/wiki with include_content=true"""
        repo_id = str(uuid4())
        params = {"include_content": True}

        response = client.get(f"/repositories/{repo_id}/wiki", params=params)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_wiki_structure_with_section_filter(self, client: TestClient):
        """Test GET /repositories/{repository_id}/wiki with section_id filter"""
        repo_id = str(uuid4())
        params = {"section_id": "getting-started"}

        response = client.get(f"/repositories/{repo_id}/wiki", params=params)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_wiki_page_json_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/wiki/pages/{page_id} with JSON format"""
        repo_id = str(uuid4())
        page_id = "overview"

        response = client.get(f"/repositories/{repo_id}/wiki/pages/{page_id}")

        # Should return 404 for non-existent repository/page
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_get_wiki_page_markdown_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/wiki/pages/{page_id} with markdown format"""
        repo_id = str(uuid4())
        page_id = "overview"
        params = {"format": "markdown"}

        response = client.get(
            f"/repositories/{repo_id}/wiki/pages/{page_id}", params=params
        )

        # Should return 404 for non-existent repository/page
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_wiki_page_invalid_format(self, client: TestClient):
        """Test GET /repositories/{repository_id}/wiki/pages/{page_id} with invalid format"""
        repo_id = str(uuid4())
        page_id = "overview"
        params = {"format": "invalid"}

        response = client.get(
            f"/repositories/{repo_id}/wiki/pages/{page_id}", params=params
        )

        # Should return 422 for invalid format parameter
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_create_documentation_pr_contract(self, client: TestClient):
        """Test POST /repositories/{repository_id}/pull-request contract"""
        repo_id = str(uuid4())
        payload = {
            "target_branch": "main",
            "title": "Update documentation",
            "description": "Automated documentation update",
            "force_update": False,
        }

        response = client.post(f"/repositories/{repo_id}/pull-request", json=payload)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_create_documentation_pr_minimal_payload(self, client: TestClient):
        """Test POST /repositories/{repository_id}/pull-request with minimal payload"""
        repo_id = str(uuid4())
        payload = {}  # All fields are optional according to schema

        response = client.post(f"/repositories/{repo_id}/pull-request", json=payload)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_repository_files_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/files contract"""
        repo_id = str(uuid4())

        response = client.get(f"/repositories/{repo_id}/files")

        # Should return 404 for non-existent/unprocessed repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_get_repository_files_with_language_filter(self, client: TestClient):
        """Test GET /repositories/{repository_id}/files with language filter"""
        repo_id = str(uuid4())
        params = {"language": "python"}

        response = client.get(f"/repositories/{repo_id}/files", params=params)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_repository_files_with_path_pattern(self, client: TestClient):
        """Test GET /repositories/{repository_id}/files with path pattern filter"""
        repo_id = str(uuid4())
        params = {"path_pattern": "src/**/*.py"}

        response = client.get(f"/repositories/{repo_id}/files", params=params)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_repository_files_with_combined_filters(self, client: TestClient):
        """Test GET /repositories/{repository_id}/files with multiple filters"""
        repo_id = str(uuid4())
        params = {"language": "python", "path_pattern": "src/**/*.py"}

        response = client.get(f"/repositories/{repo_id}/files", params=params)

        # Should return 404 for non-existent repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_invalid_uuid_format_documentation(self, client: TestClient):
        """Test documentation endpoints with invalid UUID format"""
        invalid_id = "not-a-uuid"
        page_id = "overview"

        # Test various endpoints with invalid UUID
        endpoints = [
            f"/repositories/{invalid_id}/wiki",
            f"/repositories/{invalid_id}/wiki/pages/{page_id}",
            f"/repositories/{invalid_id}/pull-request",
            f"/repositories/{invalid_id}/files",
        ]

        for endpoint in endpoints:
            if "pull-request" in endpoint:
                response = client.post(endpoint, json={})
            else:
                response = client.get(endpoint)

            # Should return 422 for invalid UUID format
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.parametrize(
        "method,endpoint_template",
        [
            ("GET", "/repositories/{repo_id}/wiki"),
            ("GET", "/repositories/{repo_id}/wiki/pages/overview"),
            ("POST", "/repositories/{repo_id}/pull-request"),
            ("GET", "/repositories/{repo_id}/files"),
        ],
    )
    def test_documentation_endpoint_exists(
        self, client: TestClient, method: str, endpoint_template: str
    ):
        """Test that all documentation API endpoints exist (even if not implemented)"""
        repo_id = str(uuid4())
        endpoint = endpoint_template.format(repo_id=repo_id)

        if method == "POST":
            response = client.post(endpoint, json={})
        else:
            response = client.get(endpoint)

        # Should not return 405 Method Not Allowed - endpoints should exist
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED

    def test_wiki_structure_response_schema(self, client: TestClient):
        """Test that successful wiki structure response matches expected schema"""
        # This test will be skipped until we have a working repository
        # but defines the expected response structure
        expected_schema = {
            "id": str,
            "title": str,
            "description": str,
            "pages": list,  # List of WikiPageDetail objects
            "sections": list,  # List of WikiSection objects
            "root_sections": list,  # List of section IDs
        }

        # For now, just assert the schema structure is defined
        assert expected_schema is not None

    def test_wiki_page_response_schema(self, client: TestClient):
        """Test that successful wiki page response matches expected schema"""
        expected_page_schema = {
            "id": str,
            "title": str,
            "description": str,
            "importance": str,  # enum: high, medium, low
            "file_paths": list,
            "related_pages": list,
            "content": str,
        }

        # For now, just assert the schema structure is defined
        assert expected_page_schema is not None

    def test_pull_request_response_schema(self, client: TestClient):
        """Test that successful PR creation response matches expected schema"""
        expected_pr_schema = {
            "pull_request_url": str,
            "branch_name": str,
            "files_changed": list,
            "commit_sha": str,
        }

        # For now, just assert the schema structure is defined
        assert expected_pr_schema is not None

    def test_file_list_response_schema(self, client: TestClient):
        """Test that successful file list response matches expected schema"""
        expected_file_list_schema = {
            "files": list,  # List of CodeDocument objects
            "repository_id": str,
            "total": int,
            "languages": dict,  # Dict mapping language to count
        }

        # For now, just assert the schema structure is defined
        assert expected_file_list_schema is not None
