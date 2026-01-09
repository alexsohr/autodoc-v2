# Wiki Workflow Improvement Proposal

**Date**: 2025-01-07
**Status**: Proposed
**Author**: Claude (AI Assistant)

## Executive Summary

The current wiki generation workflow has a critical bug where page content is being replaced by memory tool outputs. Investigation reveals deeper architectural issues that should be addressed holistically.

## Problem Analysis

### Immediate Bug (Root Cause)

**Location**: `src/agents/wiki_workflow.py:323-328`

```python
# Current problematic code
messages = result.get("messages", [])
content = ""
if messages:
    last_message = messages[-1]  # BUG: Takes last message
    content = last_message.content
```

**What happens**:
1. Agent generates wiki content (AI message with markdown)
2. Agent calls `store_memory` (tool call)
3. Tool returns success message (tool message)
4. Agent acknowledges completion (AI message: "The documentation has been successfully generated...")
5. Extraction takes `messages[-1]` → Gets acknowledgment, NOT wiki content

### Architectural Issues

| Issue | Impact | Severity |
|-------|--------|----------|
| No separation between content and conversation | Content mixed with tool results and acks | High |
| No structured output for page content | Fragile message parsing | High |
| Memory workflow mixed with generation | Side effects interfere with extraction | Medium |
| Single agent doing multiple responsibilities | Harder to debug/maintain | Medium |
| No dedicated state field for outputs | Must parse messages | Medium |

---

## Proposed Solutions (3 Options)

### Option A: Quick Fix (Minimal Changes)

**Approach**: Fix content extraction logic only

```python
def extract_wiki_content_from_messages(messages: list) -> str:
    """Extract the longest AI message content (likely the wiki page)."""
    ai_messages = [
        msg for msg in messages
        if hasattr(msg, 'type') and msg.type == 'ai'
        and hasattr(msg, 'content') and msg.content
        and not msg.tool_calls  # Exclude messages with tool calls
    ]

    if not ai_messages:
        return ""

    # Return the longest content (wiki pages are much longer than acks)
    return max(ai_messages, key=lambda m: len(m.content or "")).content
```

**Pros**: Minimal code changes, quick to implement
**Cons**: Fragile, doesn't fix root architectural issues

---

### Option B: State-Based Output (Recommended)

**Approach**: Add `generated_content` field to agent state, use structured output pattern

#### 1. Define Extended Agent State

```python
from langchain.agents.middleware.types import AgentState
from typing_extensions import NotRequired

class WikiPageAgentState(AgentState):
    """Extended state with dedicated output field."""
    repository_id: NotRequired[str]
    generated_content: NotRequired[str]  # Dedicated field for wiki content
```

#### 2. Create Response Extraction Node

Based on [LangGraph best practices](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/):

```python
def should_continue(state: WikiPageAgentState) -> str:
    """Determine if agent should continue tool calling or respond."""
    messages = state["messages"]
    last_message = messages[-1]

    # If last message has no tool calls, extract content
    if not getattr(last_message, 'tool_calls', None):
        return "extract_content"
    return "continue"

def extract_content_node(state: WikiPageAgentState) -> dict:
    """Extract wiki content from the last AI response without tool calls."""
    messages = state["messages"]

    # Find the last AI message with substantial content (not tool calls)
    for msg in reversed(messages):
        if (hasattr(msg, 'type') and msg.type == 'ai'
            and msg.content and len(msg.content) > 500  # Wiki pages are long
            and not getattr(msg, 'tool_calls', None)):
            return {"generated_content": msg.content}

    return {"generated_content": ""}
```

#### 3. Modify Workflow Graph

```python
workflow = StateGraph(WikiPageAgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("extract_content", extract_content_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools", "extract_content": "extract_content"}
)
workflow.add_edge("tools", "agent")
workflow.add_edge("extract_content", END)
```

#### 4. Update Page Extraction in Workflow

```python
# In generate_pages_node
result = await agent.ainvoke({...})
content = result.get("generated_content", "")  # Clean extraction
```

**Pros**: Clean separation, follows LangGraph patterns, maintainable
**Cons**: Requires moderate refactoring

---

### Option C: Multi-Agent Architecture (Most Robust)

**Approach**: Separate content generation from memory operations using supervisor pattern

Based on [ByteByteGo patterns](https://blog.bytebytego.com/p/top-ai-agentic-workflow-patterns) and [LangGraph multi-agent](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/):

```
┌─────────────────────────────────────────────────────┐
│                   Page Supervisor                   │
│   (Coordinates content + memory agents)             │
└──────────────────────┬──────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
         ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│  Content Agent  │         │  Memory Agent   │
│  - File reading │         │  - recall_memories │
│  - Wiki writing │         │  - store_memory │
│  - Returns MD   │         │  - get_file_memories │
└─────────────────┘         └─────────────────┘
```

#### Workflow:
1. **Memory Agent** recalls relevant memories first
2. **Content Agent** generates wiki page (pure content, no memory tools)
3. **Memory Agent** stores new learnings
4. **Supervisor** extracts final content from Content Agent

#### Benefits:
- **Single responsibility**: Each agent has one job
- **Clean extraction**: Content agent only outputs wiki content
- **No interference**: Memory operations don't pollute content messages
- **Better debugging**: Can trace issues to specific agents
- **Reusable**: Memory agent can be used across different workflows

**Pros**: Most robust, follows best practices, scalable
**Cons**: Significant refactoring, more complex

---

## Prompt Engineering Improvements

Based on [OpenAI's prompt engineering guide](https://platform.openai.com/docs/guides/prompt-engineering) and [PromptHub best practices](https://www.prompthub.us/blog/prompt-engineering-for-ai-agents):

### 1. Separate Content from Actions

**Current** (problematic):
```
**FAILURE TO USE MEMORY TOOLS = INCOMPLETE TASK**
Store at least one memory with your decisions
```

**Improved** (for Option B/C):
```
## Output Format
Your FINAL response must be the complete wiki page in markdown.
Do NOT include conversational text like "Here is the documentation" or "I've completed the task".
The response should START with the wiki page title (## Page Title) and contain ONLY the wiki content.
```

### 2. Add Explicit Output Markers

```
When you have finished generating the wiki page, output it with these markers:
<wiki_content>
## Page Title
[Your complete wiki page in markdown]
</wiki_content>

Do NOT add any text after the closing </wiki_content> tag.
```

### 3. Two-Phase Approach

```
PHASE 1 - RESEARCH:
1. Recall relevant memories
2. Read source files
3. Analyze the codebase

PHASE 2 - OUTPUT:
Generate the wiki page. Your response should contain ONLY the wiki content.
Memory storage will happen automatically after you complete the page.
```

---

## Memory Middleware Improvements

### Remove Memory Storage from Agent Prompt

The current prompt forces the agent to call `store_memory` before completing, which adds messages after the wiki content. Instead:

1. **Post-processing approach**: After extracting content, have a separate node store memories
2. **Or remove mandatory storage**: Make memory usage optional, not required

```python
# Option: Move memory storage to post-processing
def post_content_memory_node(state: WikiPageAgentState) -> dict:
    """Store memories after content is extracted."""
    content = state.get("generated_content", "")
    if content:
        # Store memory about what was generated
        await memory_service.store_memory(
            content=f"Generated wiki page with key topics: {extract_topics(content)}",
            memory_type="structural_decision",
            ...
        )
    return {}
```

---

## Implementation Recommendation

### Phase 1 (Immediate - Fix Bug)
1. Implement Option A quick fix to unblock users
2. Update extraction logic to find actual wiki content

### Phase 2 (Short-term - Architecture)
1. Implement Option B state-based output
2. Add `generated_content` field to agent state
3. Create conditional routing with extraction node
4. Update prompts to separate content from conversation

### Phase 3 (Medium-term - Optimization)
1. Consider Option C multi-agent architecture
2. Separate memory agent from content agent
3. Implement supervisor coordination
4. Add quality validation node

---

## Additional Resources

- [LangGraph Structured Output](https://langchain-ai.github.io/langgraph/how-tos/react-agent-structured-output/)
- [Multi-Agent Supervisor Pattern](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- [Production-Grade Agentic AI Workflows (arXiv)](https://arxiv.org/abs/2512.08769)
- [Building Effective Agents - Anthropic](https://www.anthropic.com/research/building-effective-agents)
- [ByteByteGo AI Workflow Patterns](https://blog.bytebytego.com/p/top-ai-agentic-workflow-patterns)

---

## Changelog

- **2025-01-07**: Initial proposal created after investigating trace `019b9c07-ab6e-7852-8751-a4d14976fe1d`
