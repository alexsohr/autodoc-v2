"""Unit tests for the deep page agent."""

import pytest

from src.agents.deep_page_agent import (
    PageContent,
    PageSection,
    create_page_finalize_tool,
    get_page_prompt,
)


class TestPageModels:
    """Test Pydantic models."""

    def test_page_section_model(self):
        section = PageSection(heading="Overview", content="This is the overview.")
        assert section.heading == "Overview"
        assert section.content == "This is the overview."

    def test_page_content_model(self):
        content = PageContent(
            title="Test Page",
            content="# Test\n\nContent here",
            source_files=["a.py", "b.py", "c.py", "d.py", "e.py"],
        )
        assert content.title == "Test Page"
        assert len(content.source_files) == 5

    def test_page_content_model_empty_source_files(self):
        content = PageContent(
            title="Test Page",
            content="# Test\n\nContent here",
        )
        assert content.source_files == []


class TestFinalizePageTool:
    """Test the finalize_page tool."""

    def test_finalize_captures_content(self):
        captured = {}
        tool = create_page_finalize_tool(captured)

        result = tool.func(
            title="Test",
            content="# Test\n\nContent",
            source_files=["a.py", "b.py", "c.py", "d.py", "e.py"],
        )

        assert "finalized successfully" in result
        assert captured["title"] == "Test"
        assert captured["content"] == "# Test\n\nContent"
        assert len(captured["source_files"]) == 5

    def test_finalize_rejects_insufficient_sources(self):
        captured = {}
        tool = create_page_finalize_tool(captured)

        result = tool.func(
            title="Test",
            content="# Test",
            source_files=["a.py", "b.py"],  # Only 2, need 5
        )

        assert "Error" in result
        assert "at least 5" in result
        assert "title" not in captured  # Should not be captured

    def test_finalize_rejects_empty_sources(self):
        captured = {}
        tool = create_page_finalize_tool(captured)

        result = tool.func(
            title="Test",
            content="# Test",
            source_files=[],
        )

        assert "Error" in result
        assert "title" not in captured

    def test_finalize_accepts_exactly_5_sources(self):
        captured = {}
        tool = create_page_finalize_tool(captured)

        result = tool.func(
            title="Test",
            content="# Test",
            source_files=["1.py", "2.py", "3.py", "4.py", "5.py"],
        )

        assert "finalized successfully" in result
        assert len(captured["source_files"]) == 5


class TestGetPagePrompt:
    """Test prompt generation."""

    def test_prompt_includes_page_info(self):
        prompt = get_page_prompt(
            page_title="Authentication System",
            page_description="How auth works",
            file_hints=["src/auth.py", "src/login.py"],
            clone_path="/tmp/repo",
            repo_name="myrepo",
            repo_description="A test repo",
            use_mcp_tools=True,
        )

        assert "Authentication System" in prompt
        assert "How auth works" in prompt
        assert "src/auth.py" in prompt
        assert "/tmp/repo" in prompt
        assert "myrepo" in prompt
        assert "read_text_file" in prompt  # MCP tools

    def test_prompt_without_mcp_tools(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp/repo",
            repo_name="repo",
            repo_description="desc",
            use_mcp_tools=False,
        )

        assert "read_text_file" not in prompt  # MCP tool not present
        assert "Test" in prompt

    def test_prompt_with_mcp_tools_includes_examples(self):
        """When use_mcp_tools=True, MCP tool examples are included."""
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp/repo",
            repo_name="repo",
            repo_description="desc",
            use_mcp_tools=True,
        )

        # MCP tools and examples should be present
        assert "read_text_file" in prompt
        assert "search_files" in prompt
        assert "head=50" in prompt  # Context-efficient reading
        assert "/tmp/repo" in prompt  # Clone path in examples

    def test_prompt_requires_5_sources(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        # Check for the 5 source file requirement
        assert "5" in prompt
        assert "source" in prompt.lower()

    def test_prompt_includes_mermaid_instructions(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        assert "Mermaid" in prompt
        assert "graph TD" in prompt
        assert "LR" in prompt  # Warning about not using LR

    def test_prompt_includes_citation_instructions(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        assert "citation" in prompt.lower() or "cite" in prompt.lower()
        assert "line" in prompt.lower()

    def test_prompt_includes_details_block_instruction(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        assert "<details>" in prompt
        assert "Relevant source files" in prompt

    def test_prompt_with_file_hints(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=["src/main.py", "src/utils.py", "tests/test_main.py"],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        assert "src/main.py" in prompt
        assert "src/utils.py" in prompt
        assert "tests/test_main.py" in prompt

    def test_prompt_without_file_hints(self):
        prompt = get_page_prompt(
            page_title="Test",
            page_description="Desc",
            file_hints=[],
            clone_path="/tmp",
            repo_name="repo",
            repo_description="desc",
        )

        assert "No specific files provided" in prompt
