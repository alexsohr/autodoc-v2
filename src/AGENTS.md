<!-- Managed by agent. Last updated: 2025-01-11 -->

# AGENTS.md (src/)

**Precedence**: The **closest AGENTS.md** to changed files wins. This root holds global defaults only.

## Global Rules

- **Architecture**: API → Service → Repository → Database (no layer skipping)
- All code is **async** (`async/await`)
- Use **constructor dependency injection** (FastAPI `Depends()`)
- Never access MongoDB directly in services — use repositories
- Never import `motor` or `pymongo` in service files
- Python 3.12+ required

## Pre-commit Checks

```bash
python -m black src/ tests/
python -m flake8 src/ tests/
python -m mypy src/
pytest tests/unit/ -q
```

## Scoped AGENTS.md Index

| Directory | Focus |
|-----------|-------|
| [agents/](agents/AGENTS.md) | LangGraph workflows, React agents |
| [api/](api/AGENTS.md) | FastAPI routes, middleware |
| [models/](models/AGENTS.md) | Beanie ODM, Pydantic models |
| [prompts/](prompts/AGENTS.md) | LLM prompt templates |
| [repository/](repository/AGENTS.md) | Data access layer |
| [services/](services/AGENTS.md) | Business logic |
| [tools/](tools/AGENTS.md) | LangChain tools |
| [utils/](utils/AGENTS.md) | Configuration, utilities |

## Conflict Resolution

Nearest AGENTS.md wins. User prompts override all files.
