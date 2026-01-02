"""Integration tests for wiki workflow with React agents.

Tests the full wiki generation workflow with mocked MCP tools and LLM responses
to verify state transitions and data flow through the workflow nodes.
"""

import operator
import pytest
from typing import Annotated, List, Optional
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
def mock_mcp_tools():
    """Create mock MCP tools that simulate filesystem operations."""
    mock_read_file = MagicMock()
    mock_read_file.name = "read_text_file"
    mock_read_file.ainvoke = AsyncMock(
        return_value="# Sample file content\nclass Example:\n    pass"
    )

    mock_list_dir = MagicMock()
    mock_list_dir.name = "list_directory"
    mock_list_dir.ainvoke = AsyncMock(return_value=["src/", "tests/", "README.md"])

    return [mock_read_file, mock_list_dir]


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
        ):
            result = await extract_structure_node(sample_initial_state)

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
        ):
            result = await extract_structure_node(sample_initial_state)

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
        ):
            result = await extract_structure_node(sample_initial_state)

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
        ):
            result = await generate_pages_node(state_with_structure)

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
        ):
            result = await generate_pages_node(state_with_structure)

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
        """Test aggregate preserves existing error state."""
        state_with_error = {
            **sample_initial_state,
            "error": "Previous error",
            "current_step": "error",
        }

        result = await aggregate_node(state_with_error)

        assert result["current_step"] == "error"

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
        mock_mcp_tools,
        mock_llm_structure_response,
        sample_initial_state,
    ):
        """Test full workflow with mocked MCP client and LLM."""
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

        with patch(
            "src.agents.wiki_react_agents.get_mcp_tools",
            new_callable=AsyncMock,
            return_value=mock_mcp_tools,
        ), patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_structure_agent,
        ), patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_page_agent,
        ), patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

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
        ):
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

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
        ), patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_page_agent,
        ), patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ):
            workflow = create_wiki_workflow()
            result = await workflow.ainvoke(sample_initial_state)

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
        """Test workflow progresses through expected state transitions."""
        state_history = []

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

        # Create interceptor to track state changes
        original_extract = extract_structure_node
        original_generate = generate_pages_node
        original_aggregate = aggregate_node
        original_finalize = finalize_node

        async def track_extract(state):
            result = await original_extract(state)
            state_history.append(("extract_structure", result.get("current_step")))
            return result

        async def track_generate(state):
            result = await original_generate(state)
            state_history.append(("generate_pages", result.get("current_step")))
            return result

        async def track_aggregate(state):
            result = await original_aggregate(state)
            state_history.append(("aggregate", result.get("current_step")))
            return result

        async def track_finalize(state):
            result = await original_finalize(state)
            state_history.append(("finalize", result.get("current_step")))
            return result

        with patch(
            "src.agents.wiki_workflow.create_structure_agent",
            new_callable=AsyncMock,
            return_value=mock_structure_agent,
        ), patch(
            "src.agents.wiki_workflow.create_page_agent",
            new_callable=AsyncMock,
            return_value=mock_page_agent,
        ), patch(
            "src.agents.wiki_workflow.WikiStructureRepository",
            return_value=mock_repo,
        ), patch(
            "src.agents.wiki_workflow.extract_structure_node",
            side_effect=track_extract,
        ), patch(
            "src.agents.wiki_workflow.generate_pages_node",
            side_effect=track_generate,
        ), patch(
            "src.agents.wiki_workflow.aggregate_node",
            side_effect=track_aggregate,
        ), patch(
            "src.agents.wiki_workflow.finalize_node",
            side_effect=track_finalize,
        ):
            # Need to recreate workflow after patching nodes
            from langgraph.graph import StateGraph, START, END

            builder = StateGraph(WikiWorkflowState)
            builder.add_node("extract_structure", track_extract)
            builder.add_node("generate_pages", track_generate)
            builder.add_node("aggregate", track_aggregate)
            builder.add_node("finalize", track_finalize)
            builder.add_edge(START, "extract_structure")
            builder.add_edge("extract_structure", "generate_pages")
            builder.add_edge("generate_pages", "aggregate")
            builder.add_edge("aggregate", "finalize")
            builder.add_edge("finalize", END)
            workflow = builder.compile()

            result = await workflow.ainvoke(sample_initial_state)

            # Verify state transitions
            expected_steps = [
                ("extract_structure", "structure_extracted"),
                ("generate_pages", "pages_generated"),
                ("aggregate", "aggregated"),
                ("finalize", "completed"),
            ]

            assert state_history == expected_steps
            assert result["current_step"] == "completed"
