# Wiki Agent Data Flow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Pass document processing output directly to wiki agent, eliminating redundant database queries.

**Architecture:** The orchestrator extracts file_tree and formatted readme_content from process_documents result, stores clone_path in Repository, then passes data to wiki agent's generate_wiki method.

**Tech Stack:** Python, Beanie ODM, LangGraph, Pydantic

---

### Task 1: Add clone_path Field to Repository Model

**Files:**
- Modify: `src/models/repository.py:65-68`
- Test: `tests/unit/models/test_repository.py`

**Step 1: Write the failing test**

Add to `tests/unit/models/test_repository.py`:

```python
def test_repository_clone_path_field():
    """Test that Repository has optional clone_path field."""
    repo = Repository(
        provider=RepositoryProvider.GITHUB,
        url="https://github.com/test/repo",
        org="test",
        name="repo",
        default_branch="main",
        access_scope=AccessScope.PUBLIC,
        clone_path="/tmp/repo_clone",
    )
    assert repo.clone_path == "/tmp/repo_clone"


def test_repository_clone_path_defaults_to_none():
    """Test that clone_path defaults to None."""
    repo = Repository(
        provider=RepositoryProvider.GITHUB,
        url="https://github.com/test/repo",
        org="test",
        name="repo",
        default_branch="main",
        access_scope=AccessScope.PUBLIC,
    )
    assert repo.clone_path is None
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/models/test_repository.py::test_repository_clone_path_field -v`
Expected: FAIL with validation error (field doesn't exist)

**Step 3: Write minimal implementation**

In `src/models/repository.py`, add after line 68 (after `commit_sha` field):

```python
    clone_path: Optional[str] = Field(
        default=None, description="Local path where repository was cloned"
    )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/models/test_repository.py::test_repository_clone_path_field tests/unit/models/test_repository.py::test_repository_clone_path_defaults_to_none -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/models/repository.py tests/unit/models/test_repository.py
git commit -m "feat(repository): add clone_path field"
```

---

### Task 2: Add Documentation Files Formatter to Orchestrator

**Files:**
- Modify: `src/agents/workflow.py`
- Test: `tests/unit/agents/test_workflow.py`

**Step 1: Write the failing test**

Add to `tests/unit/agents/test_workflow.py`:

```python
def test_format_documentation_files_single_file():
    """Test formatting a single documentation file."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    doc_files = [{"path": "README.md", "content": "# Hello World"}]

    result = orchestrator._format_documentation_files(doc_files)

    assert "--- README.md ---" in result
    assert "# Hello World" in result


def test_format_documentation_files_multiple_files():
    """Test formatting multiple documentation files."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)
    doc_files = [
        {"path": "README.md", "content": "# Project"},
        {"path": "docs/API.md", "content": "# API Reference"},
    ]

    result = orchestrator._format_documentation_files(doc_files)

    assert "--- README.md ---" in result
    assert "# Project" in result
    assert "--- docs/API.md ---" in result
    assert "# API Reference" in result


def test_format_documentation_files_empty_list():
    """Test formatting empty documentation files list."""
    orchestrator = WorkflowOrchestrator.__new__(WorkflowOrchestrator)

    result = orchestrator._format_documentation_files([])

    assert result == ""
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/unit/agents/test_workflow.py::test_format_documentation_files_single_file -v`
Expected: FAIL with AttributeError (method doesn't exist)

**Step 3: Write minimal implementation**

Add to `WorkflowOrchestrator` class in `src/agents/workflow.py`:

```python
    def _format_documentation_files(self, doc_files: List[Dict[str, str]]) -> str:
        """Format documentation files into a single string with citations.

        Args:
            doc_files: List of dicts with 'path' and 'content' keys.

        Returns:
            Formatted string with file citations.
        """
        if not doc_files:
            return ""

        parts = []
        for doc in doc_files:
            path = doc.get("path", "unknown")
            content = doc.get("content", "")
            parts.append(f"--- {path} ---\n{content}")

        return "\n\n".join(parts)
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/agents/test_workflow.py::test_format_documentation_files_single_file tests/unit/agents/test_workflow.py::test_format_documentation_files_multiple_files tests/unit/agents/test_workflow.py::test_format_documentation_files_empty_list -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/agents/workflow.py tests/unit/agents/test_workflow.py
git commit -m "feat(workflow): add documentation files formatter"
```

---

### Task 3: Update WikiGenerationState to Remove repository_info

**Files:**
- Modify: `src/agents/wiki_agent.py:69-83`

**Step 1: Modify WikiGenerationState**

Update `WikiGenerationState` in `src/agents/wiki_agent.py`:

```python
class WikiGenerationState(TypedDict):
    """State for wiki generation workflow"""

    repository_id: str
    file_tree: str
    readme_content: str
    wiki_structure: Optional[Dict[str, Any]]
    generated_pages: List[Dict[str, Any]]
    current_page: Optional[str]
    current_step: str
    error_message: Optional[str]
    progress: float
    start_time: str
    messages: List[BaseMessage]
```

**Step 2: Run existing tests to check for breakage**

Run: `pytest tests/unit/agents/test_wiki_agent.py -v`
Expected: May fail if tests use repository_info - fix in next steps

**Step 3: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "refactor(wiki_agent): remove repository_info from state"
```

---

### Task 4: Update generate_wiki Method Signature

**Files:**
- Modify: `src/agents/wiki_agent.py:279-342`
- Test: `tests/unit/agents/test_wiki_agent.py`

**Step 1: Update generate_wiki method**

Modify `generate_wiki` in `src/agents/wiki_agent.py`:

```python
    async def generate_wiki(
        self,
        repository_id: str,
        file_tree: str = "",
        readme_content: str = "",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Generate complete wiki for repository

        Args:
            repository_id: Repository identifier
            file_tree: ASCII file tree structure from document processing
            readme_content: Formatted documentation files content
            force_regenerate: Force regeneration even if wiki exists

        Returns:
            Dictionary with wiki generation results
        """
        try:
            # Check if wiki already exists
            if not force_regenerate:
                existing_wiki = await self._wiki_structure_repo.find_one(
                    {"repository_id": repository_id}
                )
                if existing_wiki:
                    return {
                        "status": "exists",
                        "message": "Wiki already exists for this repository",
                        "wiki_id": str(existing_wiki.id),
                    }

            # Initialize state with provided values
            initial_state: WikiGenerationState = {
                "repository_id": repository_id,
                "file_tree": file_tree,
                "readme_content": readme_content,
                "wiki_structure": None,
                "generated_pages": [],
                "current_page": None,
                "current_step": "starting",
                "error_message": None,
                "progress": 0.0,
                "start_time": datetime.now(timezone.utc).isoformat(),
                "messages": [
                    HumanMessage(
                        content=f"Generate wiki for repository: {repository_id}"
                    )
                ],
            }

            # Execute workflow
            result = await self.workflow.ainvoke(initial_state)

            return {
                "status": "completed" if not result.get("error_message") else "failed",
                "repository_id": repository_id,
                "wiki_structure": result.get("wiki_structure"),
                "pages_generated": len(result.get("generated_pages", [])),
                "error_message": result.get("error_message"),
            }

        except Exception as e:
            logger.error(f"Wiki generation failed for repository {repository_id}: {e}")
            return {
                "status": "failed",
                "repository_id": repository_id,
                "error": str(e),
                "error_type": type(e).__name__,
            }
```

**Step 2: Run tests**

Run: `pytest tests/unit/agents/test_wiki_agent.py -v`
Expected: PASS (signature is backward compatible with defaults)

**Step 3: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "feat(wiki_agent): accept file_tree and readme_content parameters"
```

---

### Task 5: Simplify _analyze_repository_node

**Files:**
- Modify: `src/agents/wiki_agent.py:344-392`

**Step 1: Update _analyze_repository_node**

Replace the method in `src/agents/wiki_agent.py`:

```python
    async def _analyze_repository_node(
        self, state: WikiGenerationState
    ) -> WikiGenerationState:
        """Analyze repository for wiki generation - uses pre-populated values"""
        try:
            state["current_step"] = "analyzing_repository"
            state["progress"] = 10.0

            # Validate we have the required data
            if not state["file_tree"]:
                state["error_message"] = "No file tree provided"
                return state

            if not state["readme_content"]:
                logger.warning("No readme content provided, wiki may be less detailed")

            # Verify repository exists
            repository = await self._repository_repo.find_one(
                {"id": UUID(state["repository_id"])}
            )
            if not repository:
                state["error_message"] = "Repository not found"
                return state

            state["progress"] = 20.0

            # Count files in tree for logging
            file_count = state["file_tree"].count("├") + state["file_tree"].count("└")
            state["messages"].append(
                AIMessage(content=f"Analyzed repository with ~{file_count} files")
            )

            return state

        except Exception as e:
            state["error_message"] = f"Repository analysis failed: {str(e)}"
            return state
```

**Step 2: Run tests**

Run: `pytest tests/unit/agents/test_wiki_agent.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "refactor(wiki_agent): simplify _analyze_repository_node to use pre-populated values"
```

---

### Task 6: Update _process_documents_node to Store clone_path

**Files:**
- Modify: `src/agents/workflow.py:396-450`

**Step 1: Update _process_documents_node**

Modify the success path in `_process_documents_node` in `src/agents/workflow.py`. After the line `if processing_result["status"] != "success":` block, update:

```python
            # Store clone_path in Repository
            if processing_result.get("clone_path"):
                await self._repository_repo.update_one(
                    {"id": UUID(state["repository_id"])},
                    {"$set": {"clone_path": processing_result["clone_path"]}}
                )

            # Format documentation files for wiki agent
            doc_files = processing_result.get("documentation_files", [])
            readme_content = self._format_documentation_files(doc_files)

            state["stages_completed"].append("process_documents")
            state["progress"] = 60.0
            state["results"]["document_processing"] = {
                "file_tree": processing_result.get("file_tree", ""),
                "readme_content": readme_content,
                "clone_path": processing_result.get("clone_path"),
                "documentation_files_count": len(doc_files),
            }

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Processed repository with {len(doc_files)} documentation files"
                )
            )

            return state
```

**Step 2: Add UUID import if not present**

Ensure `from uuid import UUID` is imported at the top of `src/agents/workflow.py`.

**Step 3: Run tests**

Run: `pytest tests/unit/agents/test_workflow.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/agents/workflow.py
git commit -m "feat(workflow): store clone_path and format docs in _process_documents_node"
```

---

### Task 7: Update _generate_wiki_node to Pass Data

**Files:**
- Modify: `src/agents/workflow.py:452-505`

**Step 1: Update _generate_wiki_node**

Modify the wiki agent call in `_generate_wiki_node`:

```python
    async def _generate_wiki_node(self, state: WorkflowState) -> WorkflowState:
        """Generate wiki node"""
        try:
            state["current_stage"] = "generate_wiki"
            state["progress"] = 70.0

            # Check if wiki already exists (unless force update)
            if not state["force_update"]:
                existing_wiki = await self._wiki_structure_repo.find_one(
                    {"repository_id": state["repository_id"]}
                )

                if existing_wiki:
                    state["stages_completed"].append("generate_wiki")
                    state["progress"] = 90.0
                    state["results"]["wiki_generation"] = {
                        "status": "exists",
                        "wiki_id": str(existing_wiki.id),
                    }

                    state["messages"].append(
                        AIMessage(content="Using existing wiki structure")
                    )

                    return state

            # Extract data from document processing results
            doc_result = state["results"].get("document_processing", {})
            file_tree = doc_result.get("file_tree", "")
            readme_content = doc_result.get("readme_content", "")

            # Generate wiki using wiki agent with pre-processed data
            wiki_result = await self._wiki_agent.generate_wiki(
                repository_id=state["repository_id"],
                file_tree=file_tree,
                readme_content=readme_content,
                force_regenerate=state["force_update"],
            )

            if wiki_result["status"] not in ["completed", "exists"]:
                state["error_message"] = (
                    f"Wiki generation failed: {wiki_result.get('error_message', 'Unknown error')}"
                )
                return state

            state["stages_completed"].append("generate_wiki")
            state["progress"] = 90.0
            state["results"]["wiki_generation"] = wiki_result

            # Add success message
            state["messages"].append(
                AIMessage(
                    content=f"Generated wiki with {wiki_result.get('pages_generated', 0)} pages"
                )
            )

            return state

        except Exception as e:
            state["error_message"] = f"Wiki generation node failed: {str(e)}"
            return state
```

**Step 2: Run tests**

Run: `pytest tests/unit/agents/test_workflow.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add src/agents/workflow.py
git commit -m "feat(workflow): pass file_tree and readme_content to wiki agent"
```

---

### Task 8: Clean Up Unused Methods in Wiki Agent

**Files:**
- Modify: `src/agents/wiki_agent.py`

**Step 1: Remove or simplify unused methods**

The following methods in `WikiGenerationAgent` may now be unused:
- `_build_file_tree` - no longer needed (file tree comes from document agent)
- `_format_tree_structure` - no longer needed
- `_find_readme_content` - no longer needed (readme comes from orchestrator)

Check if these are used elsewhere before removing:

Run: `grep -r "_build_file_tree\|_format_tree_structure\|_find_readme_content" src/`

If only used in `_analyze_repository_node` (which we simplified), remove them.

**Step 2: Remove unused methods**

Delete the methods if confirmed unused.

**Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/agents/wiki_agent.py
git commit -m "refactor(wiki_agent): remove unused tree and readme methods"
```

---

### Task 9: Integration Test

**Files:**
- Test: `tests/integration/test_wiki_data_flow.py`

**Step 1: Write integration test**

Create `tests/integration/test_wiki_data_flow.py`:

```python
"""Integration test for wiki agent data flow from document processing."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.agents.workflow import WorkflowOrchestrator


@pytest.mark.integration
@pytest.mark.asyncio
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
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_wiki_data_flow.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_wiki_data_flow.py
git commit -m "test(integration): add wiki data flow integration test"
```

---

### Task 10: Update Design Document Status

**Files:**
- Modify: `docs/plans/2025-12-29-wiki-agent-data-flow-design.md`

**Step 1: Update status**

Change status from "Approved" to "Implemented" in the design document.

**Step 2: Commit**

```bash
git add docs/plans/2025-12-29-wiki-agent-data-flow-design.md
git commit -m "docs: mark wiki agent data flow design as implemented"
```

---

## Summary

| Task | Description | Estimated Complexity |
|------|-------------|---------------------|
| 1 | Add clone_path field to Repository | Simple |
| 2 | Add documentation files formatter | Simple |
| 3 | Update WikiGenerationState | Simple |
| 4 | Update generate_wiki signature | Medium |
| 5 | Simplify _analyze_repository_node | Medium |
| 6 | Update _process_documents_node | Medium |
| 7 | Update _generate_wiki_node | Simple |
| 8 | Clean up unused methods | Simple |
| 9 | Integration test | Medium |
| 10 | Update design document | Simple |
