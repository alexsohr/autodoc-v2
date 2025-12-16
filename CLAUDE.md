# CLAUDE.md - Claude Code Project Guide

This file provides guidance for Claude Code when working on this project.

## Project Overview

AutoDoc v2 is an AI-powered repository documentation generator built with FastAPI, LangGraph, MongoDB (with vector search), and Beanie ODM. It automatically analyzes Git repositories and creates comprehensive documentation using AI.

## Environment Setup

### Python Version
- **Required**: Python 3.12+
- **Virtual Environment**: `venv` directory exists in project root

### Activating the Environment (Windows PowerShell)

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If execution policy blocks this, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Activating the Environment (Windows Command Prompt)

```cmd
venv\Scripts\activate.bat
```

### Installing Dependencies

```powershell
# Install main dependencies
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## Running the Application

### Development Server (Recommended)

```powershell
# Windows PowerShell - includes cache cleaning
.\scripts\dev-run.ps1

# Windows PowerShell - clean cache only
.\scripts\dev-run.ps1 -CleanOnly

# Windows PowerShell - skip cache cleaning
.\scripts\dev-run.ps1 -SkipClean

# Windows Command Prompt
dev-run.bat
```

### Direct Startup

```powershell
python -m src.api.main
```

### API Endpoints

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

## Project Architecture

### Layered Architecture Pattern

```
API Layer → Service Layer → Repository Layer → Database
```

1. **API Layer** (`src/api/`): HTTP handling, request validation, authentication
2. **Service Layer** (`src/services/`): Business logic, workflow orchestration
3. **Repository Layer** (`src/repository/`): Data access abstraction, domain-specific queries
4. **Data Layer** (`src/models/`): Pydantic models, Beanie ODM documents

### Directory Structure

```
src/
├── api/                    # FastAPI endpoints & middleware
│   ├── main.py            # Application entry point
│   ├── middleware/        # Auth, logging, error handling
│   └── routes/            # API route handlers
├── services/               # Business logic (NO direct data access)
│   ├── auth_service.py
│   ├── chat_service.py
│   ├── document_service.py
│   ├── repository_service.py
│   └── wiki_service.py
├── repository/             # Data access layer
│   ├── base.py            # BaseRepository generic class
│   ├── database.py        # MongoDB connection management
│   └── *_repository.py    # Domain-specific repositories
├── agents/                 # LangGraph AI workflows
│   ├── document_agent.py  # Document processing
│   ├── wiki_agent.py      # Wiki generation
│   └── workflow.py        # Workflow orchestration
├── tools/                  # Shared AI tools
│   ├── context_tool.py    # RAG context retrieval
│   ├── embedding_tool.py  # Vector embeddings
│   ├── llm_tool.py        # LLM interactions
│   └── repository_tool.py # Git repository operations
├── models/                 # Data models (Pydantic/Beanie)
│   ├── base.py            # Base model classes
│   ├── chat.py            # ChatSession, Question, Answer
│   ├── code_document.py   # CodeDocument
│   ├── repository.py      # Repository model
│   ├── user.py            # UserDocument
│   └── wiki.py            # WikiStructure
├── prompts/                # LLM prompt templates
└── utils/                  # Configuration & utilities
    ├── config_loader.py   # Settings management
    ├── logging_config.py  # Structured logging
    ├── retry_utils.py     # Tenacity retry logic
    ├── storage_adapters.py # Local/S3/MongoDB storage
    └── webhook_validator.py # Webhook signature validation
```

## Key Patterns to Follow

### 1. Dependency Injection Pattern

Services receive repositories via constructor injection:

```python
class DocumentProcessingService:
    def __init__(self, code_document_repo=None):
        self._code_document_repo = code_document_repo

    async def _get_code_document_repo(self):
        if self._code_document_repo is None:
            from ..repository.code_document_repository import CodeDocumentRepository
            self._code_document_repo = CodeDocumentRepository(CodeDocument)
        return self._code_document_repo
```

### 2. Repository Pattern

All database access goes through repositories extending `BaseRepository`:

```python
from ..repository.base import BaseRepository
from ..models.code_document import CodeDocument

class CodeDocumentRepository(BaseRepository[CodeDocument]):
    def __init__(self, document: Type[CodeDocument]):
        super().__init__(document)

    # Domain-specific methods here
```

### 3. Service Layer Rules

- Services contain business logic only
- Services do NOT import `motor` or `pymongo` directly
- Services access data through repository methods
- Services use lazy loading for repository dependencies

### 4. API Route Pattern

Routes are thin wrappers that delegate to services:

```python
@router.post("/repositories")
async def create_repository(data: RepositoryCreate):
    service = RepositoryService()
    return await service.create_repository(data)
```

## Beanie Document Models

All database entities use Beanie ODM. Key documents registered in `database.py`:

- `Repository` - Git repository metadata
- `CodeDocument` - Processed source code files with embeddings
- `WikiStructure` - Generated wiki pages
- `ChatSession`, `Question`, `Answer` - Chat entities
- `UserDocument` - User accounts

### UUID Handling

UUIDs are configured to use standard representation:

```python
# In database.py
codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
```

## Testing

### Running Tests

```powershell
# Clean cache first (recommended)
python scripts/clean_cache.py

# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/bdd/           # BDD/Cucumber tests
pytest tests/contract/      # Contract tests
pytest tests/performance/   # Performance tests
pytest tests/security/      # Security tests

# Run with coverage
pytest --cov=src --cov-report=html
```

### Test Structure

```
tests/
├── unit/          # Test individual components in isolation
├── integration/   # Test component interactions
├── contract/      # API contract validation
├── bdd/           # Gherkin-style scenarios
│   ├── features/  # .feature files
│   └── step_defs/ # Step implementations
├── performance/   # Load and latency tests
└── security/      # Security validation
```

### Test Markers

```python
@pytest.mark.slow           # Long-running tests
@pytest.mark.integration    # Integration tests
@pytest.mark.contract       # Contract tests
@pytest.mark.performance    # Performance tests
@pytest.mark.security       # Security tests
@pytest.mark.bdd           # BDD tests
```

## Code Quality

```powershell
# Format code
python -m black src/ tests/

# Sort imports
python -m isort src/ tests/ --profile black

# Type checking
python -m mypy src/

# Linting
python -m flake8 src/ tests/
```

### Code Style Configuration

- **Line length**: 88 characters (Black default)
- **Python version**: 3.12
- **Import style**: isort with black profile

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `langgraph` | AI workflow orchestration |
| `langchain` | LLM integration |
| `beanie` | Async MongoDB ODM |
| `motor` | Async MongoDB driver |
| `pydantic` | Data validation |
| `structlog` | Structured logging |
| `tenacity` | Retry logic |

## Important Notes for Claude

### Avoid Code Duplication

1. **Check existing repositories** before creating database queries
2. **Check existing services** before implementing business logic
3. **Use BaseRepository** for any new data access patterns
4. **Reuse existing tools** in `src/tools/` for AI operations

### Database Access Rules

- NEVER access MongoDB directly in services - use repositories
- NEVER import `motor` or `pymongo` in service files
- ALWAYS use `await get_database()` for database access in infrastructure code
- ALWAYS go through repository layer for domain operations

### Environment Variables

Key settings configured via `.env`:

```env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=autodoc_v2
OPENAI_API_KEY=your-key
GOOGLE_API_KEY=your-key
SECRET_KEY=your-secret
```

Settings are loaded via `src/utils/config_loader.py` using `get_settings()`.

### Git Branch

Current branch: `003-updating-mongodb-to-beanie` (migration to Beanie ODM)
Main branch: `main`

### Common Commands Reference

```powershell
# Start development server
.\scripts\dev-run.ps1

# Run tests
pytest

# Format code
python -m black src/ tests/

# Clean Python cache
python scripts/clean_cache.py

# Check types
python -m mypy src/
```
