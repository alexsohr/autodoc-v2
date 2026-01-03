"""Tests for WikiGenerationAgent with Deep Agent integration"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4


class TestWikiAgentDeepAgentIntegration:
    """Tests for Deep Agent integration in WikiGenerationAgent"""

    @pytest.mark.asyncio
    async def test_generate_structure_node_uses_deep_agent(self):
        """Test that _generate_structure_node uses the deep agent"""
        from src.agents.wiki_agent import WikiGenerationAgent

        # Mock dependencies
        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()
        mock_wiki_repo = MagicMock()
        mock_repo_repo = MagicMock()
        mock_code_doc_repo = MagicMock()

        # Mock repository with clone_path
        mock_repository = MagicMock()
        mock_repository.clone_path = "/tmp/test-repo"
        mock_repository.org = "test-org"
        mock_repository.name = "test-repo"
        mock_repo_repo.find_one = AsyncMock(return_value=mock_repository)

        agent = WikiGenerationAgent(
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool,
            wiki_structure_repo=mock_wiki_repo,
            repository_repo=mock_repo_repo,
            code_document_repo=mock_code_doc_repo,
        )

        state = {
            "repository_id": str(uuid4()),
            "file_tree": "├── src/",
            "readme_content": "# Test",
            "wiki_structure": None,
            "generated_pages": [],
            "current_page": None,
            "current_step": "starting",
            "error_message": None,
            "progress": 0.0,
            "start_time": "2024-01-01T00:00:00Z",
            "messages": [],
        }

        expected_structure = {
            "title": "Test Wiki",
            "description": "Test description",
            "pages": [{"title": "Overview", "slug": "overview", "section": "Overview", "file_paths": [], "description": "Test"}]
        }

        with patch("src.agents.deep_structure_agent.run_structure_agent", new_callable=AsyncMock) as mock_run:
            mock_run.return_value = expected_structure

            # Need to also patch Path.exists to return True
            with patch("pathlib.Path.exists", return_value=True):
                result = await agent._generate_structure_node(state)

        assert result["wiki_structure"] == expected_structure
        assert result["error_message"] is None
        assert result["progress"] == 50.0


    @pytest.mark.asyncio
    async def test_generate_structure_node_handles_no_clone_path(self):
        """Test error handling when clone_path is missing"""
        from src.agents.wiki_agent import WikiGenerationAgent

        mock_context_tool = MagicMock()
        mock_llm_tool = MagicMock()
        mock_wiki_repo = MagicMock()
        mock_repo_repo = MagicMock()
        mock_code_doc_repo = MagicMock()

        # Repository without clone_path
        mock_repository = MagicMock()
        mock_repository.clone_path = None
        mock_repo_repo.find_one = AsyncMock(return_value=mock_repository)

        agent = WikiGenerationAgent(
            context_tool=mock_context_tool,
            llm_tool=mock_llm_tool,
            wiki_structure_repo=mock_wiki_repo,
            repository_repo=mock_repo_repo,
            code_document_repo=mock_code_doc_repo,
        )

        state = {
            "repository_id": str(uuid4()),
            "file_tree": "",
            "readme_content": "",
            "wiki_structure": None,
            "generated_pages": [],
            "current_page": None,
            "current_step": "starting",
            "error_message": None,
            "progress": 0.0,
            "start_time": "2024-01-01T00:00:00Z",
            "messages": [],
        }

        result = await agent._generate_structure_node(state)

        assert result["error_message"] is not None
        assert "clone" in result["error_message"].lower() or "path" in result["error_message"].lower()
