"""Integration tests for Deep Agent wiki structure generation.

These tests use a real sample repository fixture to verify the agent works.
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "sample_repo"


class TestDeepAgentStructureIntegration:
    """Integration tests for Deep Agent structure generation"""

    @pytest.fixture
    def sample_repo_path(self) -> str:
        """Get path to sample repository fixture"""
        assert FIXTURES_DIR.exists(), f"Fixture directory not found: {FIXTURES_DIR}"
        return str(FIXTURES_DIR)

    @pytest.fixture
    def sample_file_tree(self) -> str:
        """Generate file tree for sample repo"""
        return """├── README.md
├── pyproject.toml
├── docs/
└── src/
    ├── main.py
    └── utils.py"""

    @pytest.fixture
    def sample_readme(self, sample_repo_path: str) -> str:
        """Read sample README"""
        readme_path = Path(sample_repo_path) / "README.md"
        if readme_path.exists():
            return readme_path.read_text()
        return "# Sample Project"

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_deep_agent_generates_structure_for_sample_repo(
        self,
        sample_repo_path: str,
        sample_file_tree: str,
        sample_readme: str,
    ):
        """Test that Deep Agent generates valid structure for sample repo"""
        from src.agents.deep_structure_agent import run_structure_agent

        # Skip if no API key configured (CI environment)
        import os
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

        result = await run_structure_agent(
            clone_path=sample_repo_path,
            owner="test-org",
            repo="sample-project",
            file_tree=sample_file_tree,
            readme_content=sample_readme,
            timeout=120.0,  # 2 minute timeout for test
        )

        # Verify structure was generated
        assert result is not None, "Agent should produce a structure"
        assert "title" in result
        assert "description" in result
        assert "pages" in result
        assert len(result["pages"]) >= 1, "Should have at least one page"

        # Verify page structure
        for page in result["pages"]:
            assert "title" in page
            assert "slug" in page
            assert "section" in page
            assert "description" in page

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_deep_agent_with_mocked_backend(
        self,
        sample_repo_path: str,
        sample_file_tree: str,
        sample_readme: str,
    ):
        """Test agent creation with mocked deep agent (no API call)"""
        from src.agents.deep_structure_agent import create_structure_agent

        with patch("src.agents.deep_structure_agent.create_deep_agent") as mock_create:
            mock_agent = MagicMock()
            mock_create.return_value = mock_agent

            agent = create_structure_agent(
                clone_path=sample_repo_path,
                owner="test-org",
                repo="sample-project",
                file_tree=sample_file_tree,
                readme_content=sample_readme,
            )

            # Verify agent was created with correct backend
            call_kwargs = mock_create.call_args.kwargs
            assert "backend" in call_kwargs
            assert "tools" in call_kwargs
            assert "system_prompt" in call_kwargs
            assert len(call_kwargs["tools"]) == 1  # finalize tool
