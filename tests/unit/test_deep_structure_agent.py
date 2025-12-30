"""Unit tests for Deep Agent structure generator"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


class TestDeepStructureAgent:
    """Tests for DeepStructureAgent"""

    @pytest.mark.asyncio
    async def test_create_structure_agent_returns_agent(self):
        """Test that create_structure_agent returns a configured agent"""
        from src.agents.deep_structure_agent import create_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="├── src/\n│   └── main.py",
                readme_content="# Test Project",
            )

            assert agent is not None
            mock_create.assert_called_once()


    @pytest.mark.asyncio
    async def test_finalize_tool_captures_structure(self):
        """Test that finalize_wiki_structure tool captures the structure"""
        from src.agents.deep_structure_agent import create_finalize_tool

        captured = {}
        tool = create_finalize_tool(captured)

        # Simulate agent calling the tool
        result = tool.invoke({
            "title": "Test Wiki",
            "description": "A test wiki",
            "pages": [{"title": "Overview", "slug": "overview", "section": "intro", "file_paths": [], "description": "Overview page"}]
        })

        assert captured["title"] == "Test Wiki"
        assert captured["description"] == "A test wiki"
        assert len(captured["pages"]) == 1
        assert "success" in result.lower()
