# Wiki Agent Orchestration Redesign Proposal

**Date:** 2025-01-02
**Author:** Claude (Research Agent)
**Status:** Draft for Review

---

## Executive Summary

This proposal presents four approaches to replace the current autonomous deep agent implementation with a predictable, deterministic workflow for wiki documentation generation. Based on research of LangGraph patterns and industry best practices, **Approach 2 (Map-Reduce with Send API)** is recommended as the primary solution, with **Approach 1 (Sequential Workflow)** as an MVP starting point.

---

## Problem Statement

### Current Implementation Issues

The current implementation uses `create_deep_agent` from the deepagents library with autonomous sub-agents:

1. **Unpredictable Execution** - Sub-agents decide their own steps dynamically
2. **Path Compatibility** - MCP filesystem tools fail with Windows absolute paths
3. **No Guaranteed Order** - Structure extraction may not happen before page generation
4. **Debugging Difficulty** - Autonomous agents are hard to trace and debug

**Trace Evidence (019b7d00-1063-7730-a6c5-1c8fbc7921c1):**
- Sub-agent `wiki_agent_yusufocaliskan_python-flask-mvc` crashed
- Error: `ValueError('Windows absolute paths are not supported')`
- `ls` tool failed when agent tried to explore filesystem

### Desired Flow

```
1. Clone Repository (existing)
2. Extract Wiki Structure → Sections, Pages, File Paths (STRUCTURE FIRST)
3. For Each Page → Generate Content (PARALLEL OK)
4. Finalize → Combine into Complete Wiki
```

---

## Research Summary

### Key Findings

#### Industry Best Practice
Both OpenAI and Anthropic explicitly state: **"You do not always need agents."** Workflows are simpler, more reliable, cheaper, and faster for well-defined tasks.

#### LangGraph Capabilities
LangGraph provides multiple orchestration patterns ranging from simple sequential workflows to complex multi-agent hierarchies. Key patterns relevant to our use case:

| Pattern | Predictability | Parallelism | Complexity |
|---------|----------------|-------------|------------|
| Sequential Workflow | Highest | None | Lowest |
| Map-Reduce (Send API) | High | Yes | Medium |
| Subgraph Composition | High | Yes | Higher |
| Supervisor Pattern | Medium | Yes | High |

### Sources
- [LangGraph Multi-Agent Workflows](https://blog.langchain.com/langgraph-multi-agent-workflows/)
- [How and When to Build Multi-Agent Systems](https://blog.langchain.com/how-and-when-to-build-multi-agent-systems/)
- [LangGraph: Controlled Workflows, Not Autonomous Agents](https://medium.com/@egrois/langgraph-controlled-workflows-not-autonomous-agents-37332efa753f)
- [AI Agent Workflows: LangGraph vs LangChain](https://towardsdatascience.com/ai-agent-workflows-a-complete-guide-on-whether-to-build-with-langgraph-or-langchain-117025509fa0/)

---

## Proposed Approaches

### Approach 1: Sequential Workflow (Simplest - MVP)

**Architecture:**
```
START → extract_structure → generate_pages_loop → finalize → END
```

**Implementation:**
```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class WikiState(TypedDict):
    clone_path: str
    file_tree: str
    readme: str
    structure: Optional[WikiStructure]
    pages: Annotated[list[WikiPageContent], operator.add]
    final_wiki: Optional[dict]

def extract_structure(state: WikiState) -> dict:
    """Use LLM with structured output to extract wiki structure."""
    llm = get_llm().with_structured_output(WikiStructure)
    structure = llm.invoke(create_structure_prompt(state))
    return {"structure": structure}

def generate_pages(state: WikiState) -> dict:
    """Generate content for all pages sequentially."""
    pages = []
    for page in state["structure"].get_all_pages():
        content = generate_single_page(state, page)
        pages.append(content)
    return {"pages": pages}

def finalize(state: WikiState) -> dict:
    """Combine all pages into final wiki."""
    return {"final_wiki": build_wiki_dict(state)}

# Build graph
builder = StateGraph(WikiState)
builder.add_node("extract_structure", extract_structure)
builder.add_node("generate_pages", generate_pages)
builder.add_node("finalize", finalize)

builder.add_edge(START, "extract_structure")
builder.add_edge("extract_structure", "generate_pages")
builder.add_edge("generate_pages", "finalize")
builder.add_edge("finalize", END)

wiki_workflow = builder.compile()
```

**Pros:**
- Most predictable execution order
- Easiest to debug and trace
- Minimal code changes from current structure
- Clear state transitions in LangSmith

**Cons:**
- Sequential page generation (slower for many pages)
- No parallelization benefit

**Best For:** MVP, debugging, small repositories

---

### Approach 2: Map-Reduce with Send API (Recommended)

**Architecture:**
```
START → extract_structure → fan_out_pages → [PARALLEL: generate_page] → aggregate → finalize → END
```

**Implementation:**
```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

class WikiState(TypedDict):
    clone_path: str
    file_tree: str
    readme: str
    structure: Optional[WikiStructure]
    pages: Annotated[list[WikiPageContent], operator.add]
    final_wiki: Optional[dict]

def extract_structure(state: WikiState) -> dict:
    """Extract wiki structure with structured output."""
    llm = get_llm().with_structured_output(WikiStructure)
    structure = llm.invoke(create_structure_prompt(state))
    return {"structure": structure}

def fan_out_pages(state: WikiState) -> list[Send]:
    """Create parallel tasks for each page."""
    return [
        Send("generate_page", {
            "page": page,
            "clone_path": state["clone_path"],
            "file_tree": state["file_tree"],
        })
        for page in state["structure"].get_all_pages()
    ]

def generate_page(state: dict) -> dict:
    """Generate content for a single page (runs in parallel)."""
    page = state["page"]
    content = llm.invoke(create_page_prompt(page, state["clone_path"]))
    return {"pages": [WikiPageContent(
        page_id=page.id,
        content=content,
    )]}

def aggregate(state: WikiState) -> dict:
    """All pages collected via reducer - just pass through."""
    return {}

def finalize(state: WikiState) -> dict:
    """Build final wiki from collected pages."""
    return {"final_wiki": build_wiki_dict(state)}

# Build graph
builder = StateGraph(WikiState)
builder.add_node("extract_structure", extract_structure)
builder.add_node("generate_page", generate_page)
builder.add_node("aggregate", aggregate)
builder.add_node("finalize", finalize)

builder.add_edge(START, "extract_structure")
builder.add_conditional_edges(
    "extract_structure",
    fan_out_pages,
    ["generate_page"]
)
builder.add_edge("generate_page", "aggregate")
builder.add_edge("aggregate", "finalize")
builder.add_edge("finalize", END)

wiki_workflow = builder.compile()
```

**Pros:**
- Parallel page generation (significant speedup)
- Deterministic structure extraction
- Clear fan-out/fan-in pattern visible in traces
- State reducer handles result collection automatically

**Cons:**
- Slightly more complex state management
- Need to handle partial failures

**Best For:** Production use, large repositories

---

### Approach 3: Subgraph Composition

**Architecture:**
```
Main: START → structure_subgraph → page_subgraph → finalize → END
Structure Subgraph: analyze → design → validate → END
Page Subgraph: START → fan_out → generate → collect → END
```

**Implementation Sketch:**
```python
# Structure extraction as subgraph
structure_graph = StateGraph(StructureState)
structure_graph.add_node("analyze", analyze_repository)
structure_graph.add_node("design", design_structure)
structure_graph.add_node("validate", validate_structure)
# ... edges ...
structure_subgraph = structure_graph.compile()

# Page generation as subgraph
page_graph = StateGraph(PageState)
# ... similar pattern with Send API ...
page_subgraph = page_graph.compile()

# Main graph invokes subgraphs
def call_structure_subgraph(state: WikiState) -> dict:
    result = structure_subgraph.invoke({...})
    return {"structure": result["structure"]}

main_graph = StateGraph(WikiState)
main_graph.add_node("structure", call_structure_subgraph)
main_graph.add_node("pages", call_page_subgraph)
main_graph.add_node("finalize", finalize)
```

**Pros:**
- Clean separation of concerns
- Reusable subgraphs
- Independent testing of components
- Separate state schemas per subgraph

**Cons:**
- More boilerplate code
- State mapping between graphs
- More complex debugging

**Best For:** Large teams, highly modular systems

---

### Approach 4: Functional API (@task pattern)

**Architecture:**
```python
@entrypoint()
async def generate_wiki(repo_info: RepoInfo) -> WikiResult:
    structure = await extract_structure(repo_info)
    page_futures = [generate_page(p) for p in structure.pages]
    pages = await asyncio.gather(*page_futures)
    return finalize(structure, pages)
```

**Implementation:**
```python
from langgraph.func import entrypoint, task

@task
async def extract_structure(repo: RepoInfo) -> WikiStructure:
    """Extract wiki structure."""
    llm = get_llm().with_structured_output(WikiStructure)
    return await llm.ainvoke(create_structure_prompt(repo))

@task
async def generate_page(page: WikiPageDetail, repo: RepoInfo) -> PageContent:
    """Generate content for single page."""
    return await llm.ainvoke(create_page_prompt(page, repo))

@task
async def finalize_wiki(structure: WikiStructure, pages: list[PageContent]) -> dict:
    """Combine into final wiki."""
    return build_wiki_dict(structure, pages)

@entrypoint()
async def wiki_generation_workflow(repo: RepoInfo) -> dict:
    # Extract structure first
    structure = await extract_structure(repo)

    # Generate all pages in parallel
    page_tasks = [generate_page(page, repo) for page in structure.get_all_pages()]
    pages = await asyncio.gather(*page_tasks)

    # Finalize
    return await finalize_wiki(structure, pages)
```

**Pros:**
- Most Pythonic syntax
- Natural async/await patterns
- Less boilerplate than StateGraph
- Familiar to Python developers

**Cons:**
- Less visual graph representation
- Newer API (less documentation)
- Different mental model from StateGraph

**Best For:** Python-native teams, simpler workflows

---

## Comparison Matrix

| Criterion | Approach 1 | Approach 2 | Approach 3 | Approach 4 |
|-----------|------------|------------|------------|------------|
| **Predictability** | Excellent | Excellent | Excellent | Excellent |
| **Simplicity** | Excellent | Good | Fair | Good |
| **Performance** | Fair | Excellent | Good | Good |
| **Debuggability** | Excellent | Good | Fair | Good |
| **Extensibility** | Fair | Good | Excellent | Good |
| **Learning Curve** | Low | Medium | High | Low |

---

## Recommendation

### Primary Recommendation: Approach 2 (Map-Reduce with Send API)

**Rationale:**
1. **Predictable** - Structure extraction is deterministic (structured output)
2. **Parallel** - Pages generate concurrently for performance
3. **Observable** - Clear fan-out/fan-in pattern in LangSmith traces
4. **Maintainable** - Single graph, clear state management
5. **Proven** - Well-documented LangGraph pattern

### Implementation Strategy

**Phase 1: MVP (Approach 1)**
- Implement sequential workflow
- Validate structure extraction with structured output
- Ensure page generation produces quality content
- Establish baseline metrics

**Phase 2: Production (Approach 2)**
- Refactor to Send API for parallelization
- Add error handling for partial failures
- Implement retry logic per page
- Performance benchmarking

---

## Migration Path

### Files to Modify

1. `src/agents/deep_structure_agent.py` → **Delete or archive**
2. `src/agents/wiki_agent.py` → **Refactor completely**
3. `src/models/wiki.py` → **Add PageContent model**
4. `src/agents/workflow.py` → **Update wiki generation step**

### Dependencies to Remove

```python
# Remove from requirements
- deepagents (or equivalent deep agent library)

# Keep
- langgraph >= 1.0.0
- langchain-core
- pydantic
```

### New Components

```
src/agents/
├── wiki_workflow.py      # New: LangGraph workflow
├── wiki_prompts.py       # New: Prompt templates
└── wiki_models.py        # New: State/output models (or extend wiki.py)
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Structure extraction fails | Medium | High | Fallback to simpler structure |
| Page generation quality varies | Medium | Medium | Quality validation step |
| Parallel failures cascade | Low | Medium | Individual page retry |
| Performance regression | Low | Low | Benchmark against current |

---

## Success Metrics

1. **Reliability**: 95%+ successful wiki generations
2. **Performance**: <5 min for repositories with <50 pages
3. **Observability**: Clear traces in LangSmith for every run
4. **Quality**: Page content accurately reflects codebase

---

## Next Steps

1. [ ] Review and approve this proposal
2. [ ] Create detailed implementation plan
3. [ ] Set up feature branch for development
4. [ ] Implement Approach 1 (MVP)
5. [ ] Test with sample repositories
6. [ ] Refactor to Approach 2 (Production)
7. [ ] Performance benchmarking
8. [ ] Deploy to production

---

## Appendix: Research Sources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Multi-Agent Orchestration Guide 2025](https://latenode.com/blog/ai-frameworks-technical-infrastructure/langgraph-multi-agent-orchestration/)
- [How to Think About Agent Frameworks](https://blog.langchain.com/how-to-think-about-agent-frameworks/)
- [AI Workflow Design: Agents vs Structured Execution](https://medium.com/@karanbhutani477/ai-workflow-design-when-to-use-agents-vs-structured-execution-and-when-to-combine-them-bdb99473e385)
- [Advanced Multi-Agent Development with LangGraph 2025](https://medium.com/@kacperwlodarczyk/advanced-multi-agent-development-with-langgraph-expert-guide-best-practices-2025-4067b9cec634)
