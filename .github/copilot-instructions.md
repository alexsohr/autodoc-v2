# Copilot Instructions for AutoDoc v2

## Project Overview
AutoDoc v2 is an AI-powered documentation generator that automatically analyzes Git repositories and creates comprehensive, searchable documentation using LangGraph workflows, vector embeddings, and conversational AI. It supports multi-provider LLMs (OpenAI, Google Gemini, Ollama), real-time webhook updates, and integrates with GitHub, Bitbucket, and GitLab.

## Architecture & Key Components
- **src/api/**: FastAPI app with lifespan management (`main.py`), OpenAPI docs, middleware stack (auth, logging, CORS), and versioned routes (`/api/v2/*`)
- **src/agents/**: LangGraph workflow orchestration with typed state management, document processing agents, and wiki generation agents
- **src/models/**: Pydantic v2 models with field validation, enums for status/providers, and UUID-based identifiers
- **src/services/**: Business logic layer with async patterns, MongoDB integration, and workflow coordination (never put logic in routes)
- **src/tools/**: AI tools for LLM calls, embeddings, context retrieval, and repository cloning with structured outputs
- **src/utils/**: Configuration via Pydantic settings, pluggable storage adapters, MongoDB vector search, and structured logging
- **tests/**: Comprehensive test suite with fixtures, performance benchmarks (P50 ≤ 500ms), and security validation

## Developer Workflows & Commands
- **Install**: `pip install -e .` (editable install from repo root)
- **Run API**: `python -m src.api.main` (serves on http://localhost:8000 with auto-reload)
- **Interactive Docs**: http://localhost:8000/docs (Swagger UI), http://localhost:8000/redoc (ReDoc)
- **Test Categories**: `pytest tests/unit/`, `pytest tests/integration/`, `pytest tests/contract/`, `pytest tests/performance/`
- **Code Quality**: `python -m black src/ tests/`, `python -m isort src/ tests/ --profile black`, `python -m mypy src/`
- **MongoDB Setup**: `docker run -d --name autodoc-mongo -p 27017:27017 mongo:7.0` (requires vector search support)
- **Environment**: Copy `.env.example` to `.env` with provider API keys, MongoDB URL, and storage config

## Patterns & Conventions
- **Async Everything**: All I/O operations use `async def` and `await` - database calls, file operations, HTTP requests, and LLM calls
- **Service Layer Pattern**: All business logic lives in `src/services/` classes with async methods; routes only handle HTTP concerns and delegate to services
- **Pydantic Models**: Use Pydantic v2 with field validators, enums for constants, and UUID fields for IDs (example: `Repository`, `RepositoryCreate`)
- **LangGraph Workflows**: AI workflows use typed state (`WorkflowState(TypedDict)`) with `StateGraph`, checkpoint memory, and structured agent responses
- **Vector Search Pipeline**: Embed documents → store in MongoDB with vector index → semantic search via embeddings for RAG context retrieval
- **Webhook Security**: HMAC-SHA256 signature validation in middleware, provider-specific payload parsing, async processing with workflow orchestration
- **Multi-Provider Architecture**: LLM provider switching via env config (`OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OLLAMA_BASE_URL`) with fallback handling
- **Storage Adapters**: Pluggable storage via abstract base class - `LocalStorageAdapter`, `S3StorageAdapter`, `MongoDBStorageAdapter` based on `STORAGE_TYPE`
- **Structured Logging**: Use `structlog` with correlation IDs, performance metrics, and error context throughout the application
- **Performance Targets**: API responses P50 ≤ 500ms, chat streaming first token ≤ 1500ms, webhook processing ≤ 3000ms

## Integration Points & External APIs
- **LLM Providers**: `src/tools/llm_tool.py` handles OpenAI GPT, Google Gemini, and Ollama with structured output validation
- **Vector Database**: MongoDB Atlas with `$vectorSearch` aggregation pipelines for semantic similarity (see `src/utils/mongodb_adapter.py`)
- **Git Provider Webhooks**: GitHub/GitLab/Bitbucket POST to `/webhooks/{provider}` with HMAC validation and async workflow triggering
- **Repository Cloning**: `src/tools/repository_tool.py` for Git operations with authentication, branch handling, and incremental updates
- **Interactive API Docs**: FastAPI auto-generates Swagger UI with comprehensive examples, auth testing, and schema validation

## Critical Workflow Sequences
- **Repository Registration**: URL validation → provider detection → clone → document analysis agent → embedding generation → MongoDB storage → wiki generation
- **Webhook Processing**: Signature validation → repository lookup → incremental analysis → selective re-embedding → wiki update → response
- **Chat/RAG Pipeline**: Question → embedding → vector search → context ranking → LLM prompt with schema → structured response → Q&A storage
- **Analysis Workflow**: `WorkflowType.FULL_ANALYSIS` → document agent → embedding tool → wiki agent → status updates in MongoDB

## Testing & Quality Assurance
- **Test Structure**: Unit tests in `tests/unit/` for individual components, integration in `tests/integration/` for workflows
- **Performance Tests**: `tests/performance/` with specific targets (P50 ≤ 500ms API, ≤ 1500ms chat first token)
- **Contract Tests**: `tests/contract/` validate API schemas match OpenAPI specs for all endpoints
- **Security Tests**: `tests/security/` for webhook validation, JWT handling, and injection prevention
- **Fixtures**: Use `tests/fixtures/` for shared test data, MongoDB test collections, and mock LLM responses

## References
- See `README.md` for full setup, usage, and API details
- See `specs/001-build-the-autodoc/` for architecture, data model, and planning docs

---
_If any section is unclear or missing, please provide feedback for further refinement._
