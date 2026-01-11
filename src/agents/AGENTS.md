# AGENTS.md - LangGraph Agents Subsystem

## Overview

This subsystem implements AI-powered workflows using LangGraph for repository documentation generation. Key patterns:

- **LangGraph Workflows**: StateGraph-based orchestration with typed state classes
- **React Agents**: `create_agent()` factories with MCP filesystem tools and structured output
- **Fan-out/Fan-in**: Parallel page generation using `Send()` API with `Annotated[List, operator.add]` reducers

## Setup

- **Python**: 3.12+
- **Dependencies**: `langgraph`, `langchain-core`, `langchain-google-genai`, `pydantic`
- **MCP Tools**: Filesystem tools via `MCPFilesystemClient` for codebase exploration

## Build/Tests

```bash
# Run agent tests
pytest tests/unit/test_wiki_workflow.py
pytest tests/unit/test_wiki_react_agents.py

# Run with coverage
pytest tests/unit/test_wiki*.py --cov=src/agents
```

## Code Style

### State Classes

Use `TypedDict` for all workflow state. For fan-in fields, use `Annotated` with `operator.add`:

```python
class WikiWorkflowState(TypedDict):
    repository_id: str
    structure: Optional[WikiStructure]
    pages: Annotated[List[WikiPageDetail], operator.add]  # Fan-in reducer
    error: Optional[str]
    current_step: str
```

### Fan-out with Send()

Create parallel workers using the `Send()` API:

```python
def fan_out_to_page_workers(state: WikiWorkflowState) -> List[Send]:
    structure = state["structure"]
    return [
        Send("generate_single_page", {
            "page": page,
            "clone_path": state["clone_path"],
        })
        for page in structure.get_all_pages()
    ]
```

### Structured Output

Always use Pydantic schemas with `response_format` for LLM responses:

```python
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=prompt,
    response_format=LLMWikiStructureSchema,  # Structured output
)
```

## Security

- **File Paths**: Validate all paths are within the cloned repository root
- **LLM Output**: Sanitize generated content before storage
- **MCP Tools**: Use wrapped tools with parameter validation (e.g., `_wrap_read_text_file_tool`)

## PR Checklist

- [ ] State schema changes: Update TypedDict and verify reducers
- [ ] Workflow node changes: Update graph edges and test fan-out/fan-in
- [ ] Agent tool changes: Update MCP tool list and wrap new tools
- [ ] Recursion limits: Verify `STRUCTURE_RECURSION_LIMIT = 2 * MAX_ITERATIONS + 1`
- [ ] Tests pass: `pytest tests/unit/test_wiki*.py`

## Examples

### WikiWorkflowState Pattern

```python
class WikiWorkflowState(TypedDict):
    repository_id: str
    clone_path: str
    file_tree: str
    structure: Optional[WikiStructure]
    pages: Annotated[List[WikiPageDetail], operator.add]  # Reducer for fan-in
    error: Optional[str]
    current_step: str
```

### Creating a React Agent

```python
async def create_page_agent() -> Any:
    tools = await get_mcp_tools(names=["read_text_file", "read_multiple_files"])
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=PROMPTS["page_generation"]["system_prompt"],
        response_format=LLMPageSchema,
    )
```

### Workflow Assembly

```python
def create_wiki_workflow():
    builder = StateGraph(WikiWorkflowState)
    builder.add_node("extract_structure", extract_structure_node)
    builder.add_node("generate_single_page", generate_single_page_node)
    builder.add_node("aggregate", aggregate_node)

    builder.add_edge(START, "extract_structure")
    builder.add_conditional_edges("extract_structure", should_fan_out,
                                   ["generate_single_page", "aggregate"])
    builder.add_edge("generate_single_page", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()
```

## When Stuck

- **Recursion limit errors**: Increase `recursion_limit` in workflow config (formula: `2 * max_iterations + 1`)
- **State not merging**: Verify `Annotated[List[T], operator.add]` reducer on fan-in fields
- **MCP tool failures**: Check `MCPFilesystemClient.is_initialized` and wrap tools with validation
- **Structured output missing**: Ensure `response_format` is set and access via `result["structured_response"]`
- **Agent hangs**: Check for infinite loops in conditional edges; add timeout to `ainvoke()`

## House Rules

1. **All workflow nodes are async** - Use `async def` for every node function
2. **Use `Annotated[List[T], operator.add]` for fan-in** - Required for parallel worker convergence
3. **Set reasonable recursion limits** - Default: `2 * max_iterations + 1`
4. **Always use structured output** - All LLM responses must use Pydantic schemas via `response_format`
5. **Wrap MCP tools** - Add parameter validation wrappers for error-prone tools
6. **Load prompts from YAML** - Store prompts in `src/prompts/wiki_prompts.yaml`
