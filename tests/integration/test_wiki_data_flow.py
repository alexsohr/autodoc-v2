"""Integration test for wiki agent data flow from document processing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.workflow import WorkflowOrchestrator


@pytest.fixture
def anyio_backend():
    """Use asyncio backend only."""
    return "asyncio"


@pytest.mark.integration
@pytest.mark.anyio
async def test_document_processing_data_flows_to_wiki_agent():
    """Test that document processing output flows correctly to wiki agent."""
    repository_id = str(uuid4())

    # Mock document processing result
    doc_processing_result = {
        "status": "success",
        "clone_path": "/tmp/test_repo",
        "file_tree": "├── README.md\n└── src/\n    └── main.py",
        "documentation_files": [
            {"path": "README.md", "content": "# Test Project"},
            {"path": "docs/API.md", "content": "# API Docs"},
        ],
    }

    # Create orchestrator with mocked dependencies
    with patch.object(WorkflowOrchestrator, "__init__", lambda self: None):
        orchestrator = WorkflowOrchestrator()
        orchestrator._document_agent = AsyncMock()
        orchestrator._document_agent.process_repository = AsyncMock(
            return_value=doc_processing_result
        )
        orchestrator._wiki_agent = AsyncMock()
        orchestrator._wiki_agent.generate_wiki = AsyncMock(
            return_value={"status": "completed", "pages_generated": 3}
        )
        orchestrator._repository_repo = AsyncMock()
        orchestrator._code_document_repo = AsyncMock()
        orchestrator._code_document_repo.count = AsyncMock(return_value=0)

        # Call _process_documents_node
        state = {
            "repository_id": repository_id,
            "repository_url": "https://github.com/test/repo",
            "branch": "main",
            "force_update": True,
            "stages_completed": [],
            "results": {},
            "messages": [],
            "progress": 0.0,
        }

        state = await orchestrator._process_documents_node(state)

        # Verify document processing stored correctly
        assert "document_processing" in state["results"]
        doc_result = state["results"]["document_processing"]
        assert doc_result["file_tree"] == doc_processing_result["file_tree"]
        assert "--- README.md ---" in doc_result["readme_content"]
        assert "--- docs/API.md ---" in doc_result["readme_content"]

        # Call _generate_wiki_node
        state = await orchestrator._generate_wiki_node(state)

        # Verify wiki agent was called with correct data
        orchestrator._wiki_agent.generate_wiki.assert_called_once()
        call_kwargs = orchestrator._wiki_agent.generate_wiki.call_args.kwargs
        assert call_kwargs["file_tree"] == doc_processing_result["file_tree"]
        assert "--- README.md ---" in call_kwargs["readme_content"]
