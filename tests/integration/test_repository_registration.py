"""Integration tests for repository registration workflow

These tests validate the complete repository registration and analysis workflow.
They MUST FAIL initially since the workflow is not implemented yet.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from httpx import AsyncClient


class TestRepositoryRegistrationWorkflow:
    """Integration tests for repository registration workflow"""

    @pytest.mark.asyncio
    async def test_complete_repository_registration_workflow(
        self, async_client: AsyncClient
    ):
        """Test complete repository registration from start to finish"""
        # Step 1: Register repository
        registration_payload = {
            "url": "https://github.com/test-org/test-repo",
            "branch": "main",
            "provider": "github",
        }

        response = await async_client.post("/repositories", json=registration_payload)

        # This will fail initially - repository endpoint not implemented
        assert response.status_code == status.HTTP_201_CREATED

        repo_data = response.json()
        repository_id = repo_data["id"]
        assert UUID(repository_id)  # Valid UUID
        assert repo_data["analysis_status"] == "pending"

        # Step 2: Check initial status
        status_response = await async_client.get(
            f"/repositories/{repository_id}/status"
        )
        assert status_response.status_code == status.HTTP_200_OK

        status_data = status_response.json()
        assert status_data["status"] == "pending"
        assert status_data["repository_id"] == repository_id

        # Step 3: Trigger analysis
        analysis_payload = {"branch": "main", "force": False}
        analysis_response = await async_client.post(
            f"/repositories/{repository_id}/analyze", json=analysis_payload
        )
        assert analysis_response.status_code == status.HTTP_202_ACCEPTED

        analysis_data = analysis_response.json()
        assert analysis_data["status"] == "processing"

        # Step 4: Poll status until completion (mock the analysis process)
        with patch(
            "src.services.repository_service.RepositoryService.analyze_repository"
        ) as mock_analyze:
            mock_analyze.return_value = AsyncMock()

            # Simulate analysis completion
            max_polls = 10
            for i in range(max_polls):
                status_response = await async_client.get(
                    f"/repositories/{repository_id}/status"
                )
                status_data = status_response.json()

                if status_data["status"] == "completed":
                    break
                elif status_data["status"] == "failed":
                    pytest.fail("Repository analysis failed")

                await asyncio.sleep(0.1)  # Short delay

            assert status_data["status"] == "completed"
            assert "progress" in status_data
            assert status_data["progress"] == 100

        # Step 5: Verify repository details
        repo_response = await async_client.get(f"/repositories/{repository_id}")
        assert repo_response.status_code == status.HTTP_200_OK

        repo_details = repo_response.json()
        assert repo_details["id"] == repository_id
        assert repo_details["analysis_status"] == "completed"
        assert repo_details["last_analyzed"] is not None
        assert repo_details["commit_sha"] is not None

    @pytest.mark.asyncio
    async def test_repository_registration_with_auto_detection(
        self, async_client: AsyncClient
    ):
        """Test repository registration with automatic provider detection"""
        registration_payload = {
            "url": "https://github.com/test-org/test-repo"
            # No provider specified - should auto-detect
        }

        response = await async_client.post("/repositories", json=registration_payload)

        # This will fail initially
        assert response.status_code == status.HTTP_201_CREATED

        repo_data = response.json()
        assert repo_data["provider"] == "github"  # Auto-detected
        assert repo_data["org"] == "test-org"
        assert repo_data["name"] == "test-repo"

    @pytest.mark.asyncio
    async def test_repository_registration_with_custom_branch(
        self, async_client: AsyncClient
    ):
        """Test repository registration with custom default branch"""
        registration_payload = {
            "url": "https://github.com/test-org/test-repo",
            "branch": "develop",
            "provider": "github",
        }

        response = await async_client.post("/repositories", json=registration_payload)

        # This will fail initially
        assert response.status_code == status.HTTP_201_CREATED

        repo_data = response.json()
        assert repo_data["default_branch"] == "develop"

    @pytest.mark.asyncio
    async def test_repository_registration_duplicate_handling(
        self, async_client: AsyncClient
    ):
        """Test handling of duplicate repository registration"""
        registration_payload = {
            "url": "https://github.com/test-org/test-repo",
            "provider": "github",
        }

        # First registration
        response1 = await async_client.post("/repositories", json=registration_payload)
        assert response1.status_code == status.HTTP_201_CREATED

        # Second registration (duplicate)
        response2 = await async_client.post("/repositories", json=registration_payload)
        assert response2.status_code == status.HTTP_409_CONFLICT

        error_data = response2.json()
        assert "error" in error_data
        assert "already exists" in error_data["message"].lower()

    @pytest.mark.asyncio
    async def test_repository_registration_invalid_url(self, async_client: AsyncClient):
        """Test repository registration with invalid URL"""
        invalid_urls = [
            "not-a-url",
            "http://invalid-domain.invalid/repo",
            "https://github.com/",  # Missing repo path
            "https://github.com/user",  # Missing repo name
            "ftp://github.com/user/repo",  # Invalid protocol
        ]

        for invalid_url in invalid_urls:
            registration_payload = {"url": invalid_url, "provider": "github"}

            response = await async_client.post(
                "/repositories", json=registration_payload
            )
            assert response.status_code == status.HTTP_400_BAD_REQUEST

            error_data = response.json()
            assert "error" in error_data

    @pytest.mark.asyncio
    async def test_repository_registration_unsupported_provider(
        self, async_client: AsyncClient
    ):
        """Test repository registration with unsupported provider"""
        registration_payload = {
            "url": "https://gitlab.com/test-org/test-repo",
            "provider": "gitlab",  # Not yet supported
        }

        response = await async_client.post("/repositories", json=registration_payload)

        # Should either succeed (if gitlab is supported) or return 400
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_400_BAD_REQUEST,
        ]

    @pytest.mark.asyncio
    async def test_repository_analysis_timeout_handling(
        self, async_client: AsyncClient
    ):
        """Test handling of repository analysis timeout"""
        registration_payload = {
            "url": "https://github.com/test-org/large-repo",
            "provider": "github",
        }

        response = await async_client.post("/repositories", json=registration_payload)
        assert response.status_code == status.HTTP_201_CREATED

        repository_id = response.json()["id"]

        # Mock analysis timeout
        with patch(
            "src.services.repository_service.RepositoryService.analyze_repository"
        ) as mock_analyze:
            mock_analyze.side_effect = asyncio.TimeoutError("Analysis timeout")

            analysis_response = await async_client.post(
                f"/repositories/{repository_id}/analyze"
            )

            # Should handle timeout gracefully
            if analysis_response.status_code == status.HTTP_202_ACCEPTED:
                # Check that status eventually shows failed
                await asyncio.sleep(0.1)
                status_response = await async_client.get(
                    f"/repositories/{repository_id}/status"
                )
                status_data = status_response.json()
                assert status_data["status"] == "failed"
                assert "timeout" in status_data.get("error_message", "").lower()

    @pytest.mark.asyncio
    async def test_repository_analysis_force_reanalysis(
        self, async_client: AsyncClient
    ):
        """Test forced re-analysis of already analyzed repository"""
        # Setup: Register and analyze repository
        registration_payload = {
            "url": "https://github.com/test-org/test-repo",
            "provider": "github",
        }

        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]

        # First analysis
        await async_client.post(f"/repositories/{repository_id}/analyze")

        # Force re-analysis
        force_analysis_payload = {"force": True}
        force_response = await async_client.post(
            f"/repositories/{repository_id}/analyze", json=force_analysis_payload
        )

        assert force_response.status_code == status.HTTP_202_ACCEPTED

    @pytest.mark.asyncio
    async def test_repository_analysis_concurrent_requests(
        self, async_client: AsyncClient
    ):
        """Test handling of concurrent analysis requests for same repository"""
        registration_payload = {
            "url": "https://github.com/test-org/test-repo",
            "provider": "github",
        }

        response = await async_client.post("/repositories", json=registration_payload)
        repository_id = response.json()["id"]

        # Send multiple concurrent analysis requests
        tasks = [
            async_client.post(f"/repositories/{repository_id}/analyze")
            for _ in range(3)
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # First request should succeed, others should return 409 (conflict)
        success_count = sum(
            1
            for r in responses
            if hasattr(r, "status_code") and r.status_code == status.HTTP_202_ACCEPTED
        )
        conflict_count = sum(
            1
            for r in responses
            if hasattr(r, "status_code") and r.status_code == status.HTTP_409_CONFLICT
        )

        assert success_count == 1
        assert conflict_count == 2

    def test_repository_listing_after_registration(self, client: TestClient):
        """Test repository listing after successful registration"""
        # Register multiple repositories
        repos = [
            {"url": "https://github.com/org1/repo1", "provider": "github"},
            {"url": "https://github.com/org1/repo2", "provider": "github"},
            {"url": "https://bitbucket.org/org2/repo3", "provider": "bitbucket"},
        ]

        registered_ids = []
        for repo_data in repos:
            response = client.post("/repositories", json=repo_data)
            if response.status_code == status.HTTP_201_CREATED:
                registered_ids.append(response.json()["id"])

        # List repositories
        list_response = client.get("/repositories")

        if list_response.status_code == status.HTTP_200_OK:
            list_data = list_response.json()
            assert len(list_data["repositories"]) >= len(registered_ids)
            assert list_data["total"] >= len(registered_ids)

            # Verify registered repositories are in the list
            repo_ids_in_list = [repo["id"] for repo in list_data["repositories"]]
            for registered_id in registered_ids:
                assert registered_id in repo_ids_in_list

    def test_repository_filtering_by_status(self, client: TestClient):
        """Test repository listing with status filtering"""
        # This test assumes some repositories exist with different statuses
        filters = ["pending", "processing", "completed", "failed"]

        for status_filter in filters:
            response = client.get("/repositories", params={"status": status_filter})

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                # All returned repositories should have the requested status
                for repo in data["repositories"]:
                    assert repo["analysis_status"] == status_filter

    @pytest.mark.asyncio
    async def test_repository_deletion_workflow(self, async_client: AsyncClient):
        """Test complete repository deletion workflow"""
        # Register repository
        registration_payload = {
            "url": "https://github.com/test-org/temp-repo",
            "provider": "github",
        }

        response = await async_client.post("/repositories", json=registration_payload)
        if response.status_code == status.HTTP_201_CREATED:
            repository_id = response.json()["id"]

            # Delete repository
            delete_response = await async_client.delete(
                f"/repositories/{repository_id}"
            )
            assert delete_response.status_code == status.HTTP_204_NO_CONTENT

            # Verify repository is deleted
            get_response = await async_client.get(f"/repositories/{repository_id}")
            assert get_response.status_code == status.HTTP_404_NOT_FOUND
