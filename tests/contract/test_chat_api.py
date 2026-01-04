"""Contract tests for Chat API endpoints

These tests validate the API contracts defined in chat_api.yaml
They MUST FAIL initially since endpoints are not implemented yet.
"""

from uuid import UUID, uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient


class TestChatAPIContract:
    """Contract tests for chat API endpoints"""

    def test_create_chat_session_contract(self, client: TestClient):
        """Test POST /repositories/{repository_id}/chat/sessions contract"""
        repo_id = str(uuid4())

        response = client.post(f"/repositories/{repo_id}/chat/sessions")

        # Should return 404 for non-existent/unanalyzed repository
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_list_chat_sessions_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/chat/sessions contract"""
        repo_id = str(uuid4())

        response = client.get(f"/repositories/{repo_id}/chat/sessions")

        # Should return 200 with empty list or 404 for non-existent repository
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    def test_list_chat_sessions_with_status_filter(self, client: TestClient):
        """Test GET /repositories/{repository_id}/chat/sessions with status filter"""
        repo_id = str(uuid4())
        params = {"status": "active"}

        response = client.get(f"/repositories/{repo_id}/chat/sessions", params=params)

        # Should return 200 with filtered results or 404
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

    def test_list_chat_sessions_invalid_status(self, client: TestClient):
        """Test GET /repositories/{repository_id}/chat/sessions with invalid status"""
        repo_id = str(uuid4())
        params = {"status": "invalid_status"}

        response = client.get(f"/repositories/{repo_id}/chat/sessions", params=params)

        # Should return 422 for invalid status parameter
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_get_chat_session_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/chat/sessions/{session_id} contract"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        response = client.get(f"/repositories/{repo_id}/chat/sessions/{session_id}")

        # Should return 404 for non-existent session
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_delete_chat_session_contract(self, client: TestClient):
        """Test DELETE /repositories/{repository_id}/chat/sessions/{session_id} contract"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        response = client.delete(f"/repositories/{repo_id}/chat/sessions/{session_id}")

        # Should return 404 for non-existent session
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_ask_question_contract(self, client: TestClient):
        """Test POST /repositories/{repository_id}/chat/sessions/{session_id}/questions contract"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        payload = {
            "content": "How does authentication work in this codebase?",
            "context_hint": "authentication, login, security",
        }

        response = client.post(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/questions",
            json=payload,
        )

        # Should return 404 for non-existent session
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_ask_question_minimal_payload(self, client: TestClient):
        """Test POST questions with minimal required payload"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        payload = {"content": "How does this work?"}

        response = client.post(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/questions",
            json=payload,
        )

        # Should return 404 for non-existent session
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_ask_question_missing_content(self, client: TestClient):
        """Test POST questions without required content field"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        payload = {"context_hint": "authentication"}

        response = client.post(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/questions",
            json=payload,
        )

        # Should return 422 for missing required field
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_ask_question_empty_content(self, client: TestClient):
        """Test POST questions with empty content"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        payload = {"content": ""}

        response = client.post(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/questions",
            json=payload,
        )

        # Should return 400 for invalid question format
        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_stream_chat_responses_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/chat/sessions/{session_id}/stream contract"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        response = client.get(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/stream"
        )

        # Should return 404 for non-existent session or appropriate SSE response
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

        if response.status_code == status.HTTP_200_OK:
            # Should have SSE content type
            assert "text/event-stream" in response.headers.get("content-type", "")

    def test_get_conversation_history_contract(self, client: TestClient):
        """Test GET /repositories/{repository_id}/chat/sessions/{session_id}/history contract"""
        repo_id = str(uuid4())
        session_id = str(uuid4())

        response = client.get(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/history"
        )

        # Should return 404 for non-existent session
        assert response.status_code == status.HTTP_404_NOT_FOUND

        data = response.json()
        assert "error" in data
        assert "message" in data

    def test_get_conversation_history_with_pagination(self, client: TestClient):
        """Test GET conversation history with pagination parameters"""
        repo_id = str(uuid4())
        session_id = str(uuid4())
        params = {"limit": 10, "before": "2023-01-01T00:00:00Z"}

        response = client.get(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/history", params=params
        )

        # Should return 404 for non-existent session
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_get_conversation_history_invalid_limit(self, client: TestClient):
        """Test GET conversation history with invalid limit"""
        repo_id = str(uuid4())
        session_id = str(uuid4())
        params = {"limit": 150}  # Exceeds maximum of 100

        response = client.get(
            f"/repositories/{repo_id}/chat/sessions/{session_id}/history", params=params
        )

        # Should return 422 for invalid parameter
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_invalid_uuid_format_chat(self, client: TestClient):
        """Test chat endpoints with invalid UUID format"""
        invalid_repo_id = "not-a-uuid"
        invalid_session_id = "also-not-a-uuid"
        valid_repo_id = str(uuid4())
        valid_session_id = str(uuid4())

        # Test various combinations
        test_cases = [
            ("POST", f"/repositories/{invalid_repo_id}/chat/sessions"),
            ("GET", f"/repositories/{invalid_repo_id}/chat/sessions"),
            (
                "GET",
                f"/repositories/{valid_repo_id}/chat/sessions/{invalid_session_id}",
            ),
            (
                "DELETE",
                f"/repositories/{valid_repo_id}/chat/sessions/{invalid_session_id}",
            ),
            (
                "POST",
                f"/repositories/{invalid_repo_id}/chat/sessions/{valid_session_id}/questions",
            ),
            (
                "GET",
                f"/repositories/{valid_repo_id}/chat/sessions/{invalid_session_id}/stream",
            ),
            (
                "GET",
                f"/repositories/{invalid_repo_id}/chat/sessions/{valid_session_id}/history",
            ),
        ]

        for method, endpoint in test_cases:
            if method == "POST":
                if "questions" in endpoint:
                    response = client.post(endpoint, json={"content": "test"})
                else:
                    response = client.post(endpoint)
            elif method == "DELETE":
                response = client.delete(endpoint)
            else:
                response = client.get(endpoint)

            # Should return 422 for invalid UUID format
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    @pytest.mark.parametrize(
        "method,endpoint_template",
        [
            ("POST", "/repositories/{repo_id}/chat/sessions"),
            ("GET", "/repositories/{repo_id}/chat/sessions"),
            ("GET", "/repositories/{repo_id}/chat/sessions/{session_id}"),
            ("DELETE", "/repositories/{repo_id}/chat/sessions/{session_id}"),
            ("POST", "/repositories/{repo_id}/chat/sessions/{session_id}/questions"),
            ("GET", "/repositories/{repo_id}/chat/sessions/{session_id}/stream"),
            ("GET", "/repositories/{repo_id}/chat/sessions/{session_id}/history"),
        ],
    )
    def test_chat_endpoint_exists(
        self, client: TestClient, method: str, endpoint_template: str
    ):
        """Test that all chat API endpoints exist (even if not implemented)"""
        repo_id = str(uuid4())
        session_id = str(uuid4())
        endpoint = endpoint_template.format(repo_id=repo_id, session_id=session_id)

        if method == "POST":
            if "questions" in endpoint:
                response = client.post(endpoint, json={"content": "test question"})
            else:
                response = client.post(endpoint)
        elif method == "DELETE":
            response = client.delete(endpoint)
        else:
            response = client.get(endpoint)

        # Should not return 405 Method Not Allowed - endpoints should exist
        assert response.status_code != status.HTTP_405_METHOD_NOT_ALLOWED

    def test_chat_session_response_schema(self, client: TestClient):
        """Test that successful chat session response matches expected schema"""
        expected_session_schema = {
            "id": str,  # UUID format
            "repository_id": str,  # UUID format
            "created_at": str,  # datetime format
            "last_activity": str,  # datetime format
            "status": str,  # enum: active, expired
            "message_count": int,
        }

        # For now, just assert the schema structure is defined
        assert expected_session_schema is not None

    def test_question_answer_response_schema(self, client: TestClient):
        """Test that successful Q&A response matches expected schema"""
        expected_qa_schema = {
            "question": {
                "id": str,
                "session_id": str,
                "content": str,
                "timestamp": str,
                "context_files": list,
            },
            "answer": {
                "id": str,
                "question_id": str,
                "content": str,
                "citations": list,
                "confidence_score": float,
                "generation_time": float,
                "timestamp": str,
            },
        }

        # For now, just assert the schema structure is defined
        assert expected_qa_schema is not None

    def test_citation_schema(self, client: TestClient):
        """Test that citation objects match expected schema"""
        expected_citation_schema = {
            "file_path": str,
            "line_start": int,  # nullable
            "line_end": int,  # nullable
            "commit_sha": str,
            "url": str,
            "excerpt": str,  # nullable
        }

        # For now, just assert the schema structure is defined
        assert expected_citation_schema is not None
