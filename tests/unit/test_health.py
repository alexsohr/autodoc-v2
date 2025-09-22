"""Unit tests for health check endpoints"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_check(self, client: TestClient):
        """Test basic health check endpoint"""
        response = client.get("/health/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "python_version" in data
        assert "environment" in data
        assert data["version"] == "2.0.0"

    def test_readiness_check(self, client: TestClient):
        """Test readiness check endpoint"""
        response = client.get("/health/ready")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "ready"
        assert "timestamp" in data

    def test_liveness_check(self, client: TestClient):
        """Test liveness check endpoint"""
        response = client.get("/health/live")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "alive"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_check_async(self, async_client):
        """Test health check with async client"""
        response = await async_client.get("/health/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["version"] == "2.0.0"
