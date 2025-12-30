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

    @pytest.mark.asyncio
    async def test_run_structure_agent_timeout(self):
        """Test that run_structure_agent handles timeout gracefully"""
        import asyncio
        from src.agents.deep_structure_agent import run_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            # Mock agent that never completes
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_create.return_value = mock_agent

            result = await run_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="",
                readme_content="",
                timeout=0.1,
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_run_structure_agent_empty_output(self):
        """Test handling when agent doesn't call finalize tool"""
        from src.agents.deep_structure_agent import run_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
            mock_agent._structure_capture = {}  # Empty - agent didn't call tool
            mock_create.return_value = mock_agent

            result = await run_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="",
                readme_content="",
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_run_structure_agent_success(self):
        """Test successful structure generation"""
        from src.agents.deep_structure_agent import run_structure_agent

        expected_structure = {
            "title": "Test Project Wiki",
            "description": "Documentation for test project",
            "pages": [
                {"title": "Overview", "slug": "overview", "section": "Overview", "file_paths": ["README.md"], "description": "Project overview"}
            ]
        }

        with patch("src.agents.deep_structure_agent.create_structure_agent") as mock_create_agent:
            mock_agent = MagicMock()
            mock_agent.ainvoke = AsyncMock(return_value={"messages": []})
            mock_agent._structure_capture = expected_structure
            mock_create_agent.return_value = mock_agent

            result = await run_structure_agent(
                clone_path="/tmp/test-repo",
                owner="test-org",
                repo="test-repo",
                file_tree="├── README.md",
                readme_content="# Test",
            )

            assert result == expected_structure
            assert result["title"] == "Test Project Wiki"
            assert len(result["pages"]) == 1

    def test_get_structure_prompt_includes_context(self):
        """Test that prompt includes all context"""
        from src.agents.deep_structure_agent import get_structure_prompt

        prompt = get_structure_prompt(
            owner="my-org",
            repo="my-repo",
            file_tree="├── src/",
            readme_content="# My Project"
        )

        assert "my-org" in prompt
        assert "my-repo" in prompt
        assert "├── src/" in prompt
        assert "# My Project" in prompt
        assert "finalize_wiki_structure" in prompt
