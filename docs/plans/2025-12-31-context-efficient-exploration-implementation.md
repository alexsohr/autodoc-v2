# Context-Efficient Exploration Strategy - Implementation Plan

**Date:** 2025-12-31
**Status:** Ready for Implementation
**Prerequisites:** None - minimal code changes required
**Estimated Effort:** 1-2 hours

---

## Overview

This plan implements the Context-Efficient Exploration Strategy for the Deep Agent. The change is **minimal** - only the system prompt in `deep_structure_agent.py` needs to be modified to instruct the agent to use `head=50` parameter when reading files.

### Key Finding: No Code Changes Needed (Only Prompt)

The MCP filesystem server already supports the `head` parameter:

```typescript
// Already supported by @modelcontextprotocol/server-filesystem
read_text_file: {
  path: string,      // File path
  head?: number      // Read only first N lines
}
```

The Deep Agent uses MCP tools directly (not through wrapper methods):

```python
# deep_structure_agent.py:279
mcp_tools = list(mcp_client._tools.values())  # Raw MCP tools
agent_kwargs["tools"] = list(mcp_tools) + [finalize_tool]  # Passed directly
```

---

## Task 1: Update System Prompt for MCP Mode

**File:** `src/agents/deep_structure_agent.py`
**Function:** `get_structure_prompt()` (lines 70-148)
**Target:** The `exploration_instructions` variable for MCP mode (lines 91-107)

### Current Code (lines 91-107)

```python
if use_mcp_tools and clone_path:
    exploration_instructions = f"""## Exploration Strategy
The repository is located at: {clone_path}

The file tree above already shows the complete directory structure - use it to identify files to read.

You have access to MCP filesystem tools for reading file contents:
1. Use `read_text_file` to read individual files with absolute paths
2. Use `read_multiple_files` to read several files at once (more efficient)
3. Use `search_files` to find files matching a pattern if needed

Focus on reading key files to understand the codebase:
- Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
- Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
- Core modules and their purposes

IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files."""
```

### New Code (Replace lines 91-107)

```python
if use_mcp_tools and clone_path:
    exploration_instructions = f"""## Exploration Strategy
The repository is located at: {clone_path}

The file tree above already shows the complete directory structure - use it to identify files to read.

### Context-Efficient Reading (IMPORTANT)
To minimize context usage and improve efficiency, follow this reading strategy:

1. **Read File Headers First (50 lines)**
   Use `read_text_file` with `head=50` to read only the first 50 lines:
   ```
   read_text_file(path="{clone_path}/src/main.py", head=50)
   ```
   The first 50 lines typically contain imports, docstrings, and class/function signatures -
   enough to understand the file's purpose without loading full content.

2. **Read More Only When Needed**
   If 50 lines aren't enough to understand a file:
   - Use `head=100` or `head=150` for larger files
   - Only read full files for small config files (< 50 lines anyway)

3. **Use search_files for Discovery**
   Find related files by pattern:
   ```
   search_files(path="{clone_path}", pattern="**/*controller*.py")
   search_files(path="{clone_path}", pattern="**/test_*.py")
   ```

4. **Use read_multiple_files Efficiently**
   When reading multiple small files (like configs), batch them:
   ```
   read_multiple_files(paths=["{clone_path}/package.json", "{clone_path}/tsconfig.json"])
   ```

### What to Read
Focus on understanding the codebase architecture:
- Config files: package.json, pyproject.toml, Cargo.toml, setup.py (read in full - usually small)
- Entry points: main.py, index.ts, App.tsx (use head=50 first)
- Core modules: understand structure before reading details

### Exploration Workflow
1. Start with config files (usually small, read in full)
2. Read entry points with head=50 to understand structure
3. Use search_files to discover related components
4. Read additional files only as needed for page decisions

IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files.
Do NOT read full files unless absolutely necessary - prefer head=50 for initial exploration."""
```

### Validation Checklist

- [ ] Verify the `head` parameter is documented in MCP filesystem server
- [ ] Test that `read_text_file(path="...", head=50)` works correctly
- [ ] Confirm the prompt fits within token limits
- [ ] Run a test repository analysis to verify reduced context usage

---

## Task 2: Update User Message for MCP Mode

**File:** `src/agents/deep_structure_agent.py`
**Function:** `run_structure_agent()` (lines 296-304)

### Current Code (lines 297-304)

```python
if mcp_tools:
    user_message = (
        f"Analyze this repository and create a comprehensive wiki structure. "
        f"The repository is located at: {clone_path}\n\n"
        f"The file tree is already provided above. Use `read_text_file` or `read_multiple_files` "
        f"to read key files. Use absolute paths like '{clone_path}/filename'.\n\n"
        f"Explore thoroughly, then call finalize_wiki_structure with your findings."
    )
```

### New Code (Replace lines 297-304)

```python
if mcp_tools:
    user_message = (
        f"Analyze this repository and create a comprehensive wiki structure. "
        f"The repository is located at: {clone_path}\n\n"
        f"The file tree is already provided above.\n\n"
        f"CONTEXT-EFFICIENT EXPLORATION:\n"
        f"- Use `read_text_file` with `head=50` to read only file headers first\n"
        f"- Example: read_text_file(path='{clone_path}/src/main.py', head=50)\n"
        f"- Only read more if 50 lines aren't enough to understand the file's purpose\n"
        f"- Use `search_files` to discover related files by pattern\n\n"
        f"Explore efficiently, then call finalize_wiki_structure with your findings."
    )
```

---

## Task 3: Add Logging for Context Tracking (Optional)

**File:** `src/agents/deep_structure_agent.py`
**Purpose:** Track context efficiency improvements

### Add After Line 326

```python
# Log context-efficient exploration mode
if mcp_tools:
    logger.info(
        "Deep Agent using context-efficient exploration",
        strategy="head=50 first, then targeted reads",
        clone_path=clone_path,
    )
```

---

## Validation Against Codebase

### Verified Components

| Component | Location | Status |
|-----------|----------|--------|
| `get_structure_prompt()` | `deep_structure_agent.py:70-148` | Target for Task 1 |
| `run_structure_agent()` | `deep_structure_agent.py:246-352` | Target for Task 2 |
| MCP tools extraction | `deep_structure_agent.py:274-280` | No changes needed |
| Tool passing to agent | `deep_structure_agent.py:195-198` | No changes needed |
| MCP client initialization | `mcp_filesystem_client.py:67-132` | No changes needed |

### MCP Tool Availability

The `@modelcontextprotocol/server-filesystem` provides:

| Tool | Available | head/tail Support |
|------|-----------|-------------------|
| `read_text_file` | Yes | Yes (`head` parameter) |
| `read_multiple_files` | Yes | No (reads full files) |
| `search_files` | Yes | N/A (returns paths) |
| `list_directory` | Yes | N/A |
| `directory_tree` | Yes | N/A |
| `get_file_info` | Yes | N/A |
| `grep` | **No** | Not in standard server |

### Important Note: No grep in MCP Mode

The standard `@modelcontextprotocol/server-filesystem` does NOT include a `grep` tool. Content search is only available when using deepagents' `FilesystemBackend` (non-MCP mode).

For MCP mode, the workaround is:
1. Use `search_files` to find files by name pattern
2. Use `read_text_file` with `head=50` to scan file headers
3. If needed, add a separate grep MCP server in the future

---

## Dependencies

### Python Packages (No Changes)

```toml
# pyproject.toml - all dependencies already present
"langgraph>=1.0.0,<2.0.0"
"langchain>=1.0.0,<2.0.0"
"langchain-mcp-adapters>=0.1.0"
"deepagents>=0.2.0"
```

### MCP Server (No Changes)

```env
# .env - already configured
MCP_FILESYSTEM_ENABLED=true
MCP_FILESYSTEM_COMMAND=npx
MCP_FILESYSTEM_ARGS=-y,@modelcontextprotocol/server-filesystem
```

---

## Testing Plan

### Unit Test (Add to Existing File)

**File:** `tests/unit/test_deep_structure_agent.py`

Add the following test method to the existing `TestDeepStructureAgent` class (after line 142):

```python
def test_get_structure_prompt_mcp_includes_head_parameter(self):
    """Test that MCP mode prompt instructs agent to use head=50."""
    from src.agents.deep_structure_agent import get_structure_prompt

    prompt = get_structure_prompt(
        owner="test-org",
        repo="test-repo",
        file_tree="├── src/",
        readme_content="# Test",
        clone_path="/test/path",
        use_mcp_tools=True,
    )

    # Verify context-efficient exploration instructions
    assert "head=50" in prompt
    assert "read_text_file" in prompt
    assert "Context-Efficient" in prompt or "context-efficient" in prompt.lower()
    # Verify it still includes absolute path warning
    assert "/test/path/" in prompt


def test_get_structure_prompt_non_mcp_unchanged(self):
    """Test that non-MCP mode prompt is unchanged."""
    from src.agents.deep_structure_agent import get_structure_prompt

    prompt = get_structure_prompt(
        owner="test-org",
        repo="test-repo",
        file_tree="├── src/",
        readme_content="# Test",
        clone_path=None,
        use_mcp_tools=False,
    )

    # Non-MCP mode should not have head parameter
    assert "head=50" not in prompt
    # But should have grep (FilesystemBackend)
    assert "grep" in prompt
```

### Integration Test (Existing File)

**File:** `tests/integration/test_deep_agent_structure.py`

Add verification that tool calls include the `head` parameter when analyzing LangSmith traces.

### Manual Verification

1. Run the wiki generation on a test repository
2. Check LangSmith trace for tool calls
3. Verify `read_text_file` calls include `head=50` parameter
4. Compare token usage before/after (expect 60-80% reduction)

---

## Rollback Plan

If issues occur, revert to the original prompt by restoring lines 91-107 and 297-304 in `deep_structure_agent.py`. No database migrations or infrastructure changes are involved.

---

## Sub-Agent Execution Instructions

### For Claude Code Sub-Agent

Execute the following tasks in order:

**Task 1: Update exploration_instructions**
```
File: src/agents/deep_structure_agent.py
Action: Replace lines 91-107 with the new exploration_instructions
Verify: The new text includes "head=50" and "Context-Efficient"
```

**Task 2: Update user_message**
```
File: src/agents/deep_structure_agent.py
Action: Replace lines 297-304 with the new user_message
Verify: The new text includes "head=50" example
```

**Task 3: Add logging (optional)**
```
File: src/agents/deep_structure_agent.py
Action: Add logging statement after line 326
Verify: Logger call includes "context-efficient exploration"
```

**Task 4: Create unit test**
```
File: tests/unit/test_deep_structure_agent.py
Action: Add test function test_get_structure_prompt_includes_head_parameter
Verify: Test passes when running pytest
```

---

## Success Criteria

1. **Prompt Updated:** System prompt includes `head=50` instructions
2. **User Message Updated:** User message reinforces context-efficient reading
3. **Tests Pass:** Unit test verifies prompt content
4. **LangSmith Verification:** Tool calls show `head` parameter usage
5. **Token Reduction:** Context usage reduced by 60-80% for exploration phase

---

## Appendix: Full File Diff Preview

```diff
--- a/src/agents/deep_structure_agent.py
+++ b/src/agents/deep_structure_agent.py
@@ -88,20 +88,46 @@ def get_structure_prompt(
         Formatted system prompt
     """
     if use_mcp_tools and clone_path:
-        exploration_instructions = f"""## Exploration Strategy
+        exploration_instructions = f"""## Exploration Strategy
 The repository is located at: {clone_path}

 The file tree above already shows the complete directory structure - use it to identify files to read.

-You have access to MCP filesystem tools for reading file contents:
-1. Use `read_text_file` to read individual files with absolute paths
-2. Use `read_multiple_files` to read several files at once (more efficient)
-3. Use `search_files` to find files matching a pattern if needed
+### Context-Efficient Reading (IMPORTANT)
+To minimize context usage and improve efficiency, follow this reading strategy:

-Focus on reading key files to understand the codebase:
-- Config files: package.json, pyproject.toml, Cargo.toml, setup.py, etc.
-- Entry points: main.py, index.ts, App.tsx, __init__.py, etc.
-- Core modules and their purposes
+1. **Read File Headers First (50 lines)**
+   Use `read_text_file` with `head=50` to read only the first 50 lines:
+   ```
+   read_text_file(path="{clone_path}/src/main.py", head=50)
+   ```
+   The first 50 lines typically contain imports, docstrings, and class/function signatures.

-IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files."""
+2. **Read More Only When Needed**
+   If 50 lines aren't enough:
+   - Use `head=100` or `head=150` for larger files
+   - Only read full files for small config files

+3. **Use search_files for Discovery**
+   Find related files by pattern:
+   ```
+   search_files(path="{clone_path}", pattern="**/*controller*.py")
+   ```

+4. **Use read_multiple_files Efficiently**
+   Batch small files like configs.

+### What to Read
+- Config files: read in full (usually small)
+- Entry points: use head=50 first
+- Core modules: understand structure before details

+IMPORTANT: Always use absolute paths starting with "{clone_path}/" when accessing files.
+Do NOT read full files unless absolutely necessary - prefer head=50 for initial exploration."""
```

---

*Document generated: 2025-12-31*
