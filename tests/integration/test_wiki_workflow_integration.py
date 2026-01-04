"""Integration tests for wiki workflow with React agents.

Tests the full wiki generation workflow with mocked MCP tools and LLM responses
to verify state transitions and data flow through the workflow nodes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.wiki_workflow import (
    WikiWorkflowState,
    create_wiki_workflow,
    extract_structure_node,
    generate_pages_node,
    aggregate_node,
    finalize_node,
    _convert_llm_structure_to_wiki_structure,
)
from src.models.wiki import WikiStructure, WikiSection, WikiPageDetail, PageImportance


@pytest.fixture
def anyio_backend():
    """Use asyncio backend only."""
    return "asyncio"


@pytest.fixture
def mock_llm_structure_response():
    """Mock LLM response for structure extraction."""
    return {
        "title": "Test Project Wiki",
        "description": "Documentation for test project",
        "sections": [
            {
                "id": "overview",
                "title": "Overview",
                "pages": [
                    {
                        "id": "getting-started",
                        "title": "Getting Started",
                        "description": "Introduction to the project",
                        "importance": "high",
                        "file_paths": ["README.md"],
                    }
                ],
            },
            {
                "id": "reference",
                "title": "Reference",
                "pages": [
                    {
                        "id": "api-reference",
                        "title": "API Reference",
                        "description": "API documentation for the project",
                        "importance": "medium",
                        "file_paths": ["src/api/main.py"],
                    }
                ],
            },
        ],
    }


@pytest.fixture
def sample_initial_state():
    """Create sample initial workflow state."""
    return WikiWorkflowState(
        repository_id=str(uuid4()),
        clone_path="/tmp/test-repo",
        file_tree="src/\n  main.py\ntests/\nREADME.md",
        readme_content="# Test Project\n\nA sample project for testing.",
        structure=None,
        pages=[],
        error=None,
        current_step="start",
    )


@pytest.fixture
def sample_wiki_structure():
    """Create sample WikiStructure for testing."""
    pages = [
        WikiPageDetail(
            id="getting-started",
            title="Getting Started",
            description="Introduction to the project",
            importance=PageImportance.HIGH,
            file_paths=["README.md"],
        ),
        WikiPageDetail(
            id="api-reference",
            title="API Reference",
            description="API documentation",
            importance=PageImportance.MEDIUM,
            file_paths=["src/api/main.py"],
        ),
    ]

    sections = [
        WikiSection(id="overview", title="Overview", pages=[pages[0]]),
        WikiSection(id="reference", title="Reference", pages=[pages[1]]),
    ]

    return WikiStructure(
        id="wiki-test123",
        repository_id=uuid4(),
        title="Test Project Wiki",
        description="Documentation for test project",
        sections=sections,
    )


class TestLLMStructureConversion:
    """Test LLM structure to WikiStructure conversion."""

    def test_convert_llm_structure_to_wiki_structure(
        self, mock_llm_structure_response
    ):
        """Test conversion of LLM output to WikiStructure model."""
        repository_id = str(uuid4())

        result = _convert_llm_structure_to_wiki_structure(
            mock_llm_structure_response, repository_id
        )

        assert isinstance(result, WikiStructure)
        assert result.title == "Test Project Wiki"
        assert result.description == "Documentation for test project"
        assert len(result.sections) == 2
        assert result.sections[0].id == "overview"
        assert len(result.sections[0].pages) == 1
        assert result.sections[0].pages[0].id == "getting-started"
        assert result.sections[0].pages[0].importance == PageImportance.HIGH

    def test_convert_llm_structure_handles_invalid_importance(self):
        """Test conversion handles invalid importance values gracefully."""
        repository_id = str(uuid4())
        llm_response = {
            "title": "Test Wiki",
            "description": "Test description",
            "sections": [
                {
                    "id": "test-section",
                    "title": "Test Section",
                    "pages": [
                        {
                            "id": "test-page",
                            "title": "Test Page",
                            "description": "Test page description",
                            "importance": "invalid_importance",
                            "file_paths": [],
                        }
                    ],
                }
            ],
        }

        result = _convert_llm_structure_to_wiki_structure(llm_response, repository_id)

        # Should default to MEDIUM for invalid importance
        assert result.sections[0].pages[0].importance == PageImportance.MEDIUM


class TestExtractStructureNode:
    """Test the extract_structure workflow node."""

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_extract_structure_success(
        self, sample_initial_state, mock_llm_structure_response
    ):
        """Test successful structure extraction."""
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [],
                "structured_response": mock_llm_structure_response,
            }
        )

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ) as mock_create_agent:
            result = await extract_structure_node(sample_initial_state)

            # Verify agent was created and called
            mock_create_agent.assert_called_once()
            mock_agent.ainvoke.assert_called_once()

            assert result["current_step"] == "structure_extracted"
            assert result.get("error") is None
            assert result["structure"] is not None
            assert isinstance(result["structure"], WikiStructure)
            assert result["structure"].title == "Test Project Wiki"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_extract_structure_no_structured_response(self, sample_initial_state):
        """Test structure extraction when agent returns no structured response."""
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [MagicMock(content="Some text without structure")],
                "structured_response": None,
            }
        )

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ) as mock_create_agent:
            result = await extract_structure_node(sample_initial_state)

            # Verify agent was created and called
            mock_create_agent.assert_called_once()
            mock_agent.ainvoke.assert_called_once()

            assert result["current_step"] == "error"
            assert result["error"] is not None
            assert "structured output" in result["error"].lower()

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_extract_structure_agent_exception(self, sample_initial_state):
        """Test structure extraction handles agent exceptions."""
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("LLM API error"))

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ) as mock_create_agent:
            result = await extract_structure_node(sample_initial_state)

            # Verify agent was created and called
            mock_create_agent.assert_called_once()
            mock_agent.ainvoke.assert_called_once()

            assert result["current_step"] == "error"
            assert result["error"] is not None
            assert "Structure extraction failed" in result["error"]


class TestGeneratePagesNode:
    """Test the generate_pages workflow node."""

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_generate_pages_success(
        self, sample_initial_state, sample_wiki_structure
    ):
        """Test successful page content generation."""
        state_with_structure = {
            **sample_initial_state,
            "structure": sample_wiki_structure,
            "current_step": "structure_extracted",
        }

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    MagicMock(content="# Getting Started\n\nThis is the generated content.")
                ]
            }
        )

        with patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ) as mock_create_agent:
            result = await generate_pages_node(state_with_structure)

            # Verify agent was created and called for each page (2 pages)
            mock_create_agent.assert_called_once()
            assert mock_agent.ainvoke.call_count == 2

            assert result["current_step"] == "pages_generated"
            assert result.get("error") is None
            assert len(result["pages"]) == 2
            assert result["pages"][0].content == "# Getting Started\n\nThis is the generated content."

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_generate_pages_with_existing_error(self, sample_initial_state):
        """Test generate_pages preserves existing error state."""
        state_with_error = {
            **sample_initial_state,
            "error": "Previous error",
            "current_step": "error",
        }

        result = await generate_pages_node(state_with_error)

        assert result["current_step"] == "error"
        assert result["error"] == "Previous error"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_generate_pages_no_structure(self, sample_initial_state):
        """Test generate_pages handles missing structure."""
        result = await generate_pages_node(sample_initial_state)

        assert result["current_step"] == "error"
        assert result["error"] == "No structure available"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_generate_pages_handles_page_error(
        self, sample_initial_state, sample_wiki_structure
    ):
        """Test generate_pages continues on individual page errors."""
        state_with_structure = {
            **sample_initial_state,
            "structure": sample_wiki_structure,
            "current_step": "structure_extracted",
        }

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            side_effect=Exception("Page generation error")
        )

        with patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_agent,
        ) as mock_create_agent:
            result = await generate_pages_node(state_with_structure)

            # Verify agent was created and called for each page
            mock_create_agent.assert_called_once()
            assert mock_agent.ainvoke.call_count == 2

            assert result["current_step"] == "pages_generated"
            # Pages should be generated but with error content
            assert len(result["pages"]) == 2
            for page in result["pages"]:
                assert "Error generating content" in page.content


class TestAggregateNode:
    """Test the aggregate workflow node."""

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_aggregate_merges_content(
        self, sample_initial_state, sample_wiki_structure
    ):
        """Test aggregate node merges page content into structure."""
        generated_pages = [
            WikiPageDetail(
                id="getting-started",
                title="Getting Started",
                description="Introduction",
                importance=PageImportance.HIGH,
                file_paths=["README.md"],
                content="# Getting Started\n\nWelcome to the project!",
            ),
            WikiPageDetail(
                id="api-reference",
                title="API Reference",
                description="API docs",
                importance=PageImportance.MEDIUM,
                file_paths=["src/api/main.py"],
                content="# API Reference\n\nAPI documentation here.",
            ),
        ]

        state = {
            **sample_initial_state,
            "structure": sample_wiki_structure,
            "pages": generated_pages,
            "current_step": "pages_generated",
        }

        result = await aggregate_node(state)

        assert result["current_step"] == "aggregated"
        assert result.get("error") is None
        assert result["structure"] is not None

        # Verify content was merged
        all_pages = result["structure"].get_all_pages()
        for page in all_pages:
            assert page.content is not None
            assert len(page.content) > 0

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_aggregate_with_existing_error(self, sample_initial_state):
        """Test aggregate preserves existing error state and message."""
        error_message = "Previous error from structure extraction"
        state_with_error = {
            **sample_initial_state,
            "error": error_message,
            "current_step": "error",
        }

        result = await aggregate_node(state_with_error)

        assert result["current_step"] == "error"
        assert result["error"] == error_message

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_aggregate_no_structure(self, sample_initial_state):
        """Test aggregate handles missing structure."""
        state = {
            **sample_initial_state,
            "pages": [],
            "current_step": "pages_generated",
        }

        result = await aggregate_node(state)

        assert result["current_step"] == "error"
        assert "No structure" in result["error"]


class TestFinalizeNode:
    """Test the finalize workflow node."""

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_finalize_saves_to_database(
        self, sample_initial_state, sample_wiki_structure
    ):
        """Test finalize node saves wiki to database."""
        state = {
            **sample_initial_state,
            "structure": sample_wiki_structure,
            "current_step": "aggregated",
        }

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        with patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            result = await finalize_node(state)

            assert result["current_step"] == "completed"
            mock_repo.upsert.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_finalize_with_existing_error(self, sample_initial_state):
        """Test finalize preserves existing error state."""
        state_with_error = {
            **sample_initial_state,
            "error": "Previous error",
            "current_step": "error",
        }

        result = await finalize_node(state_with_error)

        assert result["current_step"] == "error"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_finalize_database_error(
        self, sample_initial_state, sample_wiki_structure
    ):
        """Test finalize handles database errors."""
        state = {
            **sample_initial_state,
            "structure": sample_wiki_structure,
            "current_step": "aggregated",
        }

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock(side_effect=Exception("Database connection error"))

        with patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            result = await finalize_node(state)

            assert result["current_step"] == "error"
            assert "Failed to save wiki" in result["error"]


class TestFullWorkflowIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_full_workflow_with_mocked_agents(
        self,
        mock_llm_structure_response,
        sample_initial_state,
    ):
        """Test full workflow with mocked agents and repository."""
        # Mock the structure agent
        mock_structure_agent = MagicMock()
        mock_structure_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [],
                "structured_response": mock_llm_structure_response,
            }
        )

        # Mock the page agent
        mock_page_agent = MagicMock()
        mock_page_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    MagicMock(content="# Getting Started\n\nThis is the generated content.")
                ]
            }
        )

        # Mock repository
        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        # Note: No need to patch get_mcp_tools since we're mocking the agent factories
        # themselves. The agents are fully mocked, so get_mcp_tools is never called.
        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_structure_agent,
        ) as mock_create_structure, patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_page_agent,
        ) as mock_create_page, patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

            # Verify agents were created and called
            mock_create_structure.assert_called_once()
            mock_structure_agent.ainvoke.assert_called_once()
            mock_create_page.assert_called_once()
            # Page agent called for each page (2 pages in the mock structure)
            assert mock_page_agent.ainvoke.call_count == 2

            # Verify workflow completed
            assert result["current_step"] == "completed"
            assert result["error"] is None

            # Verify structure was extracted
            assert result["structure"] is not None
            assert result["structure"].title == "Test Project Wiki"

            # Verify pages were generated
            assert len(result["pages"]) >= 1

            # Verify repository was called to save
            mock_repo.upsert.assert_called_once()

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_workflow_handles_structure_extraction_error(
        self, sample_initial_state
    ):
        """Test workflow handles errors during structure extraction."""
        mock_structure_agent = MagicMock()
        mock_structure_agent.ainvoke = AsyncMock(
            side_effect=Exception("LLM service unavailable")
        )

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_structure_agent,
        ) as mock_create_structure:
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

            # Verify agent was created and called (even though it failed)
            mock_create_structure.assert_called_once()
            mock_structure_agent.ainvoke.assert_called_once()

            # Verify error is captured and workflow reaches end state
            # Note: Due to LangGraph's sequential flow, workflow may still
            # proceed through nodes but with error state
            assert result["error"] is not None
            assert "Structure extraction failed" in result["error"]

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_workflow_handles_page_generation_error(
        self,
        mock_llm_structure_response,
        sample_initial_state,
    ):
        """Test workflow handles errors during page generation gracefully."""
        mock_structure_agent = MagicMock()
        mock_structure_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [],
                "structured_response": mock_llm_structure_response,
            }
        )

        mock_page_agent = MagicMock()
        mock_page_agent.ainvoke = AsyncMock(
            side_effect=Exception("Page generation error")
        )

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_structure_agent,
        ) as mock_create_structure, patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_page_agent,
        ) as mock_create_page, patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

            # Verify agents were called
            mock_create_structure.assert_called_once()
            mock_structure_agent.ainvoke.assert_called_once()
            mock_create_page.assert_called_once()
            # Page agent called for each page (2 pages), but all fail
            assert mock_page_agent.ainvoke.call_count == 2

            # Pages should have error content but workflow should complete
            assert result["pages"] is not None
            # The pages should contain error message
            for page in result["pages"]:
                assert "Error generating content" in page.content

            # Workflow should still complete (with error pages saved)
            assert result["current_step"] == "completed"

    @pytest.mark.integration
    @pytest.mark.anyio
    async def test_workflow_state_transitions(
        self,
        mock_llm_structure_response,
        sample_initial_state,
    ):
        """Test workflow progresses through expected state and reaches completion.

        This test verifies the actual workflow implementation by running the
        real workflow graph and verifying the final state reflects all
        intermediate transitions were successful.
        """
        mock_structure_agent = MagicMock()
        mock_structure_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [],
                "structured_response": mock_llm_structure_response,
            }
        )

        mock_page_agent = MagicMock()
        mock_page_agent.ainvoke = AsyncMock(
            return_value={
                "messages": [MagicMock(content="# Page Content")]
            }
        )

        mock_repo = MagicMock()
        mock_repo.upsert = AsyncMock()

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_structure_agent,
        ) as mock_create_structure, patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_page_agent,
        ) as mock_create_page, patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            # Use the actual workflow from the module
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

            # Verify all agents and repository were called (proves all nodes executed)
            mock_create_structure.assert_called_once()
            mock_structure_agent.ainvoke.assert_called_once()
            mock_create_page.assert_called_once()
            # Page agent called for each page (2 pages in the mock structure)
            assert mock_page_agent.ainvoke.call_count == 2
            mock_repo.upsert.assert_called_once()

            # Verify final state reflects successful completion
            assert result["current_step"] == "completed"
            assert result["error"] is None

            # Verify structure was properly extracted and aggregated
            assert result["structure"] is not None
            assert result["structure"].title == "Test Project Wiki"

            # Verify pages were generated
            assert len(result["pages"]) == 2
