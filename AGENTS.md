# Repository Guidelines

Use this guide to keep contributions aligned with AutoDoc v2's architecture, agents, and automation tooling.

## Project Structure & Module Organization
- `src/` holds production code: `api/` exposes FastAPI endpoints, `agents/` wires LangGraph workflows, `services/` wraps external integrations, and `models/` stores persistence layers. Group new modules with their domain.
- `prompts/` and `tools/` house agent prompt assets and shared executors; keep prompts versioned alongside workflow changes.
- `tests/` mirrors runtime layout with `unit/`, `integration`, `contract`, `performance`, and `security`. Shared fixtures live in `tests/fixtures/` and `tests/conftest.py`.
- Design references live under `specs/001-build-the-autodoc/`, API docs under `docs/`, and sample corpora under `data/`.

## Build, Test, and Development Commands
- Use the PowerShell terminal for all local commands; the Make targets wrap cross-platform scripts.
- `make dev-install` installs dependencies plus formatting, linting, and typing toolchains.
- `make run` starts the FastAPI app (`python -m src.api.main`) at http://localhost:8000; use `make run-prod` for gunicorn workers.
- `make lint`, `make format`, and `make type-check` run flake8, black/isort, and mypy. `make check` combines them for pre-commit gates, and `make test`, `make test-fast`, or `make test-cov` execute the relevant pytest suites.

## Coding Style & Naming Conventions
- Target Python 3.12, 4-space indentation, exhaustive type hints, and brief module docstrings for non-obvious behavior.
- `black` enforces 88-character lines and `isort` uses the black profile; never hand-format around these tools.
- Prefer PascalCase Pydantic models, snake_case fields and functions, and prefix async LangGraph helpers with `async_`.

## Testing Guidelines
- Pytest runs with strict markers; label expensive suites with `@pytest.mark.slow`, `integration`, `contract`, `performance`, or `security` so selectors stay accurate.
- Place regression specs next to their modules (`tests/unit/test_<module>.py`) and keep fixtures reusable in `tests/fixtures/`.
- Aim for >90% coverage via `make test-cov`; investigate uncovered critical paths before merging.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `refactor:`) with <=72 character subjects and bodies describing intent, trade-offs, and validation.
- Reference relevant specs or issue IDs (e.g., `Refs specs/001-build-the-autodoc`) and list verification commands run.
- Pull requests must include a summary, configuration or schema changes, screenshots for UI or docs diffs, and links to updated API docs when applicable.

## MCP Tooling & Automation
- **Context7 MCP**: Resolve the library ID before fetching docs (`context7.resolve-library-id` then `context7.get-library-docs`); request focused topics to minimize output and cite commands in PR descriptions.
- **Serena MCP**: Use `serena.find_symbol`, pattern search, and code rewrite helpers for navigation, refactors, and cross-reference mapping; respect existing modifications and validate before writing.
- **Knowledge Memory**: Store durable findings with `serena.write_memory`, revisit context with `serena.list_memories`, and prune stale notes via `serena.delete_memory` when they no longer apply.
- **Terminal Expectations**: PowerShell is the default shell; prefer `Make` targets or `python -m ...` invocations rather than Unix-specific scripts.

## Environment Setup
- Always activate the Python virtual environment at the start of each session: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix/macOS).
- Verify the correct Python interpreter is active with `python --version` and `which python` before running any `python -m ...` commands.
- If dependencies seem missing or outdated, run `make dev-install` to refresh the development environment.


