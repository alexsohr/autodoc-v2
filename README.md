# AutoDoc v2

**AI-Powered Repository Documentation Generator**

AutoDoc v2 is an intelligent documentation generation system that automatically analyzes Git repositories and creates comprehensive, searchable documentation using advanced AI technologies including LangGraph workflows, vector embeddings, and conversational AI.

## 🚀 Features

### Core Capabilities
- **🤖 AI-Powered Analysis**: Automatic code analysis using LangGraph agents and LLM providers
- **📚 Wiki Generation**: Structured documentation with intelligent page organization
- **💬 Conversational AI**: Chat interface for codebase questions with RAG (Retrieval-Augmented Generation)
- **🔍 Semantic Search**: Vector-based search across your entire codebase
- **🔗 Multi-Provider Support**: GitHub, Bitbucket, GitLab integration
- **⚡ Real-time Updates**: Webhook-driven automatic documentation updates

### Technical Excellence
- **🏗️ Clean Architecture**: Repository pattern with dependency injection and proper separation of concerns
- **⚡ Modern Stack**: FastAPI + LangGraph + MongoDB with vector search + Beanie ODM
- **🔒 Enterprise Security**: JWT authentication, webhook signature validation, injection prevention
- **📊 Performance Optimized**: P50 ≤ 500ms API responses, streaming chat with ≤ 1500ms first token
- **📚 Interactive API Documentation**: Comprehensive Swagger UI with examples and real-time testing
- **🧪 Test-Driven Development**: 100% TDD implementation with comprehensive test coverage
- **🔄 Multi-LLM Support**: OpenAI GPT, Google Gemini, AWS Bedrock, Ollama local models
- **📦 Flexible Storage**: Local filesystem, AWS S3, MongoDB with vector capabilities
- **🧠 Wiki Memory System**: Persistent agent memory across wiki regenerations for consistent documentation
- **🔧 MCP Filesystem Tools**: Model Context Protocol integration for intelligent codebase exploration

## 📋 Quick Start

### Prerequisites

- **Python 3.12+**
- **MongoDB 7.0+** (with vector search support)
- **Git** (for repository cloning)
- **API Keys** for your chosen LLM provider (OpenAI, Google, or local Ollama)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/your-org/autodoc-v2.git
cd autodoc-v2
```

2. **Install dependencies:**
```bash
pip install -e .
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your settings (see Configuration section)
```

4. **Start MongoDB:**
```bash
# Using Docker
docker run -d --name autodoc-mongo -p 27017:27017 mongo:7.0

# Or use your existing MongoDB instance
```

5. **Run the application:**
```bash
# Recommended for development (includes cache cleaning)
make dev-run                    # Linux/macOS/WSL (via Makefile)
./scripts/dev-run.sh           # Unix/macOS (direct script)
.\scripts\dev-run.ps1          # Windows PowerShell
dev-run.bat                    # Windows Command Prompt

# Alternative: Direct startup (without cache cleaning)
python -m src.api.main
```

6. **Access the API:**
- **Interactive API Documentation (Swagger UI)**: http://localhost:8000/docs
- **Alternative API Documentation (ReDoc)**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health
- **OpenAPI Specification**: http://localhost:8000/openapi.json

## 🔧 Configuration

Create a `.env` file with your configuration:

```env
# Environment
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=INFO

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
API_PREFIX=/api/v2
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# Database
MONGODB_URL=mongodb://localhost:27017
MONGODB_DATABASE=autodoc_v2

# Storage
STORAGE_TYPE=local
STORAGE_BASE_PATH=./data

# Security
SECRET_KEY=your-super-secure-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# LLM Providers (choose one or more)
OPENAI_API_KEY=your-openai-api-key
GOOGLE_API_KEY=your-google-api-key
OLLAMA_BASE_URL=http://localhost:11434

# AWS Bedrock (optional enterprise LLM)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1

# LangSmith (Development Tracing & Monitoring)
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=autodoc-v2
LANGSMITH_TRACING=true

# MCP Filesystem (for wiki generation agent tools)
MCP_FILESYSTEM_ENABLED=true
MCP_FILESYSTEM_COMMAND=npx
MCP_FILESYSTEM_ARGS=-y,@anthropic/mcp-filesystem

# Optional: AWS S3 (if using S3 storage)
S3_BUCKET_NAME=autodoc-storage
```

### 🔍 LangSmith Integration

AutoDoc v2 includes built-in LangSmith integration for comprehensive tracing and monitoring of AI operations during development:

**Key Features:**
- **🔍 LLM Call Tracing**: Monitor all OpenAI, Google, and other LLM provider interactions
- **🤖 Agent Workflow Tracking**: Trace LangGraph agent decision-making and tool usage
- **📊 Performance Analytics**: Track response times, token usage, and costs
- **🐛 Debugging Support**: Inspect conversation flows and identify bottlenecks
- **📈 Usage Insights**: Understand patterns and optimize AI operations

**Configuration:**
```env
# Enable LangSmith tracing (development only)
LANGSMITH_API_KEY=your-langsmith-api-key    # Required for tracing
LANGSMITH_PROJECT=autodoc-v2                # Project name in dashboard
LANGSMITH_TRACING=true                      # Enable/disable tracing
```

**Getting Started:**
1. Sign up for a free LangSmith account at [smith.langchain.com](https://smith.langchain.com)
2. Generate an API key in your LangSmith dashboard
3. Add the API key to your `.env` file
4. Start the server - all AI operations will be automatically traced

When enabled, you'll see detailed traces for:
- Repository analysis workflows
- Wiki generation processes
- Chat conversations and RAG operations
- Document embedding and retrieval

## 📖 Usage

### 1. Create Your First Repository

```bash
# Register a repository for analysis
curl -X POST http://localhost:8000/api/v2/repositories \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/your-org/your-repo",
    "branch": "main"
  }'
```

### 2. Monitor Analysis Progress

```bash
# Check analysis status
curl -X GET http://localhost:8000/api/v2/repositories/$REPO_ID/status \
  -H "Authorization: Bearer $TOKEN"
```

### 3. Access Generated Documentation

```bash
# Get wiki structure
curl -X GET http://localhost:8000/api/v2/repositories/$REPO_ID/wiki \
  -H "Authorization: Bearer $TOKEN"

# Get specific page
curl -X GET http://localhost:8000/api/v2/repositories/$REPO_ID/wiki/pages/overview \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Start Conversational AI

```bash
# Create chat session
curl -X POST http://localhost:8000/api/v2/repositories/$REPO_ID/chat/sessions \
  -H "Authorization: Bearer $TOKEN"

# Ask questions about your codebase
curl -X POST http://localhost:8000/api/v2/repositories/$REPO_ID/chat/sessions/$SESSION_ID/questions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How does user authentication work in this codebase?",
    "context_hint": "authentication, security, login"
  }'
```

## 📚 API Documentation

AutoDoc v2 provides comprehensive, interactive API documentation powered by **Swagger UI** and **ReDoc**.

### 🎯 Interactive Documentation Features

- **🔍 Swagger UI**: Full interactive API explorer at `/docs`
  - Try out API endpoints directly in your browser
  - Comprehensive request/response examples
  - Built-in authentication support
  - Real-time API testing with custom parameters

- **📖 ReDoc**: Clean, professional documentation at `/redoc`
  - Three-panel layout for easy navigation
  - Detailed schema documentation
  - Code samples in multiple formats
  - Responsive design for mobile and desktop

### 🚀 Key Documentation Features

#### **Rich Examples & Use Cases**
Every endpoint includes multiple real-world examples:

- **Repository Management**
  - GitHub public/private repository registration
  - GitLab and Bitbucket integration examples
  - Analysis triggering with different options
  - Webhook configuration scenarios

- **Conversational AI**
  - Architecture questions ("How does authentication work?")
  - Function-specific queries ("What does create_user do?")
  - Debugging help ("How are errors handled?")
  - Best practices questions ("What testing patterns are used?")

- **Webhook Integration**
  - GitHub push event payloads
  - Pull request webhook examples
  - Bitbucket event handling
  - Response format examples

#### **Authentication Documentation**
- **JWT Bearer Token**: Primary authentication method
- **API Key Authentication**: Service-to-service communication
- **Interactive Auth Testing**: Test authenticated endpoints directly in Swagger UI

#### **Enhanced Developer Experience**
- **Detailed Descriptions**: Every endpoint has comprehensive descriptions
- **Parameter Documentation**: Clear explanations of all query parameters and request bodies
- **Response Examples**: Multiple response scenarios (success, error, edge cases)
- **Schema Validation**: Real-time request validation with helpful error messages

### 🛠️ Accessing the Documentation

```bash
# Start the server
python -m src.api.main

# Access documentation
open http://localhost:8000/docs      # Swagger UI (interactive)
open http://localhost:8000/redoc     # ReDoc (clean reading)
open http://localhost:8000/openapi.json  # Raw OpenAPI specification
```

### 📋 API Endpoints Overview

| Category | Endpoints | Description |
|----------|-----------|-------------|
| **Health** | `/health/*` | System health and readiness checks |
| **Repositories** | `/api/v2/repositories/*` | Repository management and analysis |
| **Chat** | `/api/v2/repositories/{id}/chat/*` | Conversational AI interface |
| **Wiki** | `/api/v2/repositories/{id}/wiki/*` | Documentation generation |
| **Webhooks** | `/webhooks/*` | Git provider webhook handling |

### 🔧 Customization

The API documentation is automatically generated from your code with:
- **Contact Information**: support@autodoc.dev
- **License**: MIT License
- **Terms of Service**: https://autodoc.dev/terms
- **Multiple Server Environments**: Local, development, and production

## 🏗️ Architecture

### System Overview

```mermaid
graph TB
    GP[Git Provider<br/>GitHub/Bitbucket/GitLab] --> API[AutoDoc v2<br/>FastAPI Application]
    API --> DB[(MongoDB<br/>Vector Search)]
    API --> LG[LangGraph<br/>AI Workflows]
    LG --> LLM[LLM Provider<br/>OpenAI/Gemini/Bedrock/Ollama]
    API --> SA[Storage Adapters<br/>Local/S3/MongoDB]

    subgraph "FastAPI Application"
        API --> AUTH[Auth Middleware]
        API --> LOG[Logging Middleware]
        API --> ERR[Error Handler]
        API --> ROUTES[API Routes]
    end

    subgraph "LangGraph Workflows"
        LG --> DA[Document Agent]
        LG --> WA[Wiki React Agent]
        LG --> WO[Workflow Orchestrator]
    end

    subgraph "AI Tools"
        RT[Repository Tool]
        ET[Embedding Tool]
        CT[Context Tool]
        LT[LLM Tool]
        MCP[MCP Filesystem Tools]

        DA --> RT
        DA --> ET
        WA --> CT
        WA --> LT
        WA --> MCP
    end

    subgraph "Wiki Memory"
        WM[(Wiki Memory<br/>MongoDB)]
        WA --> WM
    end
```

### Detailed Service Flow

```mermaid
flowchart TD
    Start([Repository Registration]) --> Validate{Validate Repository}
    Validate -->|Valid| Clone[Clone Repository]
    Validate -->|Invalid| Error1[Return Error]
    
    Clone --> Analyze[Document Analysis Agent]
    Analyze --> Extract[Extract Code Files]
    Extract --> Process[Process Documents]
    Process --> Embed[Generate Embeddings]
    Embed --> Store[Store in MongoDB]
    
    Store --> Wiki[Wiki Generation Agent]
    Wiki --> Structure[Generate Wiki Structure]
    Structure --> Pages[Create Wiki Pages]
    Pages --> Complete[Analysis Complete]
    
    Complete --> Webhook{Webhook Configured?}
    Webhook -->|Yes| Listen[Listen for Updates]
    Webhook -->|No| Ready[Ready for Queries]
    
    Listen --> Update[Process Webhook]
    Update --> Reanalyze[Incremental Analysis]
    Reanalyze --> Ready
    
    Ready --> Chat[Chat Interface Available]
    Chat --> Query[User Queries]
    Query --> RAG[RAG Pipeline]
    RAG --> Response[AI Response]
    
    subgraph "Storage Layer"
        MongoDB[(MongoDB)]
        S3[(AWS S3)]
        Local[(Local FS)]
    end
    
    subgraph "AI Layer"
        OpenAI[OpenAI GPT]
        Gemini[Google Gemini]
        Ollama[Ollama Local]
    end
    
    Store --> MongoDB
    Store --> S3
    Store --> Local
    
    RAG --> OpenAI
    RAG --> Gemini
    RAG --> Ollama
```

### Key Components

- **🌐 FastAPI Application**: High-performance async API server
- **🤖 LangGraph Workflows**: AI agent orchestration for document processing
- **🗃️ MongoDB**: Document storage with vector search capabilities
- **🔧 Storage Adapters**: Pluggable storage backends (Local, S3, MongoDB)
- **🛡️ Security Layer**: JWT authentication, webhook validation, rate limiting
- **📊 Observability**: Structured logging, performance monitoring, health checks

### Wiki Generation Architecture

AutoDoc v2 uses a sophisticated wiki generation system built on LangGraph with React agents:

#### React Agent Pattern
Wiki generation uses the `create_react_agent` pattern with MCP filesystem tools for intelligent codebase exploration:

- **Structure Agent**: Analyzes repository structure, designs wiki organization using Google Gemini
- **Page Agent**: Generates detailed documentation with code citations using LLM tools

#### Fan-out/Fan-in Page Generation
Pages are generated in parallel using LangGraph's Send API for optimal performance:

```mermaid
flowchart TD
    START([Start]) --> extract_structure[Extract Structure<br/>React Agent]
    extract_structure --> should_fan_out{Fan-out?}
    should_fan_out -->|Send per page| generate_single_page[Generate Single Page<br/>×N Workers]
    should_fan_out -->|error| aggregate[Aggregate]
    generate_single_page -->|pages reducer| aggregate
    aggregate --> finalize[Finalize<br/>Save to MongoDB]
    finalize --> END([End])
```

**Workflow Benefits:**
- Fresh agent context per page (no context pollution)
- Parallel execution with configurable concurrency (default: 5)
- Isolated failures (one page failing doesn't affect others)
- Better observability in LangSmith traces

#### MCP Filesystem Tools
Agents access the codebase through Model Context Protocol tools:
- `read_text_file`: Read individual source files with head/tail options
- `read_multiple_files`: Batch file reading for efficiency
- `directory_tree`: Explore repository structure
- `list_directory_with_sizes`: Understand file organization

#### Wiki Memory System
Persistent memory across wiki regenerations for consistent documentation:

| Memory Type | Purpose |
|-------------|---------|
| `structural_decision` | Wiki organization choices |
| `pattern_found` | Coding patterns discovered |
| `cross_reference` | Relationships between code areas |

Memory is stored in MongoDB with vector embeddings for semantic search and recall.

#### Agent Middleware Stack
Wiki agents use a composable middleware stack for enhanced capabilities:

| Middleware | Purpose |
|------------|---------|
| `WikiMemoryMiddleware` | Persistent memory for structural decisions |
| `TodoListMiddleware` | Task tracking during exploration |
| `SummarizationMiddleware` | Context window management |
| `PatchToolCallsMiddleware` | Tool parameter validation |
| `ModelRetryMiddleware` | Automatic retry on LLM failures |
| `ToolRetryMiddleware` | Automatic retry on tool failures |

### Architectural Improvements

AutoDoc v2 follows a **clean, layered architecture** with proper separation of concerns:

#### **🏗️ Repository Pattern Implementation**
- **API Layer** → **Service Layer** → **Repository Layer** → **Data Access Layer**
- Clean dependency injection with lazy loading
- Testable components with mocked dependencies
- Type-safe operations throughout the stack

#### **📁 Organized Code Structure**
```
src/
├── api/                    # FastAPI endpoints & middleware
│   ├── main.py            # Application entry point
│   ├── routes/            # API route handlers
│   └── middleware/        # Auth, logging, error handling
├── services/               # Business logic (clean, no data access)
│   └── mcp_filesystem_client.py  # MCP tool provider
├── repository/             # Data access layer & repository implementations
│   ├── database.py        # MongoDB/Beanie connection management
│   ├── base.py            # BaseRepository[TDocument] generic class
│   └── *_repository.py    # Domain-specific repositories
├── agents/                 # LangGraph AI workflows
│   ├── document_agent.py  # Document processing workflow
│   ├── wiki_agent.py      # Wiki generation orchestration
│   ├── wiki_workflow.py   # Fan-out/fan-in wiki workflow
│   ├── wiki_react_agents.py  # React agent factories
│   ├── workflow.py        # Workflow orchestrator
│   └── middleware/        # Agent middleware (WikiMemory, etc.)
├── tools/                  # Shared AI tools & utilities
├── models/                 # Pydantic/Beanie data models
│   └── wiki_memory.py     # Wiki memory model
├── prompts/                # LLM prompt templates
│   └── wiki_prompts.yaml  # Wiki generation prompts
└── utils/                  # Configuration & utilities
```

#### **🔄 Dependency Flow**
- **APIs** call **Services** for business logic
- **Services** use **Repositories** for data operations
- **Repositories** use **Data Access Layer** for database operations
- **Agents & Tools** access data through the same clean interfaces

### Webhook Processing Sequence

```mermaid
sequenceDiagram
    participant GP as Git Provider
    participant WH as Webhook Endpoint
    participant WV as Webhook Validator
    participant RS as Repository Service
    participant DA as Document Agent
    participant ET as Embedding Tool
    participant DB as MongoDB
    participant WA as Wiki Agent

    GP->>WH: POST /webhooks/github
    Note over GP,WH: Push event with commits
    
    WH->>WV: Validate signature
    WV->>WV: HMAC-SHA256 verification
    WV-->>WH: Signature valid
    
    WH->>RS: Process webhook event
    RS->>RS: Extract repository info
    RS->>DB: Find repository record
    DB-->>RS: Repository found
    
    RS->>DA: Trigger document analysis
    Note over RS,DA: Async workflow execution
    
    DA->>DA: Clone updated repository
    DA->>DA: Detect changed files
    DA->>DA: Process new/modified files
    
    DA->>ET: Generate embeddings
    ET->>ET: Create vector embeddings
    ET->>DB: Store embeddings
    
    DA->>DB: Update documents
    DA-->>RS: Analysis complete
    
    RS->>WA: Trigger wiki update
    WA->>DB: Retrieve updated documents
    WA->>WA: Generate wiki structure
    WA->>WA: Create/update pages
    WA->>DB: Store wiki content
    WA-->>RS: Wiki updated
    
    RS-->>WH: Processing complete
    WH-->>GP: 200 OK
    
    Note over GP,WA: Documentation automatically updated
```

### Chat/RAG Sequence

```mermaid
sequenceDiagram
    participant U as User
    participant API as Chat API
    participant CS as Chat Service
    participant CT as Context Tool
    participant DB as MongoDB
    participant LT as LLM Tool
    participant LLM as LLM Provider

    U->>API: POST /chat/sessions/{id}/questions
    Note over U,API: "How does auth work?"
    
    API->>CS: Process question
    CS->>DB: Create question record
    CS->>CT: Retrieve relevant context
    
    CT->>CT: Generate query embedding
    CT->>DB: Vector search
    Note over CT,DB: Semantic similarity search
    
    DB-->>CT: Relevant documents
    CT->>CT: Rank by relevance
    CT->>CT: Extract code snippets
    CT-->>CS: Context with citations
    
    CS->>LT: Generate answer
    Note over CS,LT: Question + Context + Schema
    
    LT->>LLM: Structured prompt
    Note over LT,LLM: Pydantic schema enforcement
    
    LLM-->>LT: Structured response
    LT->>LT: Validate response format
    LT-->>CS: Formatted answer
    
    CS->>DB: Store Q&A pair
    CS->>CS: Calculate confidence score
    CS-->>API: Complete response
    
    API-->>U: JSON response
    Note over API,U: Answer + citations + confidence
    
    opt Streaming Mode
        CS->>U: Server-Sent Events
        Note over CS,U: Real-time token streaming
    end
```

### Component Architecture

```mermaid
graph TB
    subgraph "API Layer"
        FastAPI[FastAPI Application]
        Routes[Route Handlers]
        Middleware[Middleware Stack]
        
        FastAPI --> Routes
        FastAPI --> Middleware
    end
    
    subgraph "Service Layer"
        AuthSvc[Auth Service]
        RepoSvc[Repository Service]
        DocSvc[Document Service]
        WikiSvc[Wiki Service]
        ChatSvc[Chat Service]
    end
    
    subgraph "Repository Layer"
        RepoRepo[Repository Repositories]
        DataAccess[Data Access Layer]
        
        RepoRepo --> DataAccess
    end
    
    subgraph "Agent Layer (LangGraph)"
        DocAgent[Document Processing Agent]
        WikiAgent[Wiki Generation Agent]
        Orchestrator[Workflow Orchestrator]
        
        DocAgent --> Orchestrator
        WikiAgent --> Orchestrator
    end
    
    subgraph "Tool Layer"
        RepoTool[Repository Tool]
        EmbedTool[Embedding Tool]
        ContextTool[Context Tool]
        LLMTool[LLM Tool]
    end
    
    subgraph "Data Layer"
        Models[Pydantic Models]
        MongoDB[(MongoDB)]
        Storage[Storage Adapters]
        
        Models --> MongoDB
        Models --> Storage
    end
    
    subgraph "External Services"
        GitHub[GitHub API]
        OpenAI[OpenAI API]
        Gemini[Google Gemini]
        Ollama[Ollama Local]
        S3[AWS S3]
    end
    
    Routes --> AuthSvc
    Routes --> RepoSvc
    Routes --> DocSvc
    Routes --> WikiSvc
    Routes --> ChatSvc
    
    AuthSvc --> RepoRepo
    RepoSvc --> RepoRepo
    DocSvc --> RepoRepo
    WikiSvc --> RepoRepo
    ChatSvc --> RepoRepo
    
    RepoSvc --> DocAgent
    WikiSvc --> WikiAgent
    ChatSvc --> ContextTool
    ChatSvc --> LLMTool
    
    DocAgent --> RepoTool
    DocAgent --> EmbedTool
    DocAgent --> DataAccess
    WikiAgent --> ContextTool
    WikiAgent --> LLMTool
    WikiAgent --> DataAccess
    
    RepoTool --> GitHub
    EmbedTool --> OpenAI
    EmbedTool --> Gemini
    EmbedTool --> Ollama
    LLMTool --> OpenAI
    LLMTool --> Gemini
    LLMTool --> Ollama
    
    Storage --> S3
    Storage --> MongoDB
```

## 🏷️ Tech Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern async web framework
- **[LangGraph](https://langchain-ai.github.io/langgraph/)**: AI workflow orchestration
- **[LangChain](https://langchain.com/)**: LLM integration and tooling
- **[Pydantic](https://pydantic.dev/)**: Data validation and structured output
- **[MongoDB](https://www.mongodb.com/)**: Document storage with vector search
- **[Beanie](https://beanie-odm.dev/)**: Async MongoDB ODM with repository pattern

### AI & ML
- **[OpenAI GPT](https://openai.com/)**: Primary LLM provider
- **[Google Gemini](https://ai.google.dev/)**: Alternative LLM provider
- **[AWS Bedrock](https://aws.amazon.com/bedrock/)**: Enterprise LLM provider
- **[Ollama](https://ollama.ai/)**: Local LLM support
- **Vector Embeddings**: Semantic search and RAG capabilities
- **[LangSmith](https://smith.langchain.com/)**: LLM observability and tracing

### Agent Infrastructure
- **[langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)**: MCP filesystem tools for codebase exploration
- **[deepagents](https://pypi.org/project/deepagents/)**: Agent middleware stack (retry, summarization, memory)
- **[ChromaDB](https://www.trychroma.com/)**: Vector store for embeddings

### DevOps & Quality
- **[pytest](https://pytest.org/)**: Comprehensive testing framework
- **[pytest-bdd](https://pytest-bdd.readthedocs.io/)**: BDD/Cucumber-style testing with Gherkin syntax
- **[Black](https://black.readthedocs.io/)**: Code formatting
- **[isort](https://pycqa.github.io/isort/)**: Import organization
- **[structlog](https://structlog.org/)**: Structured logging
- **[tenacity](https://tenacity.readthedocs.io/)**: Retry logic with backoff

## 📊 Performance

AutoDoc v2 is designed for enterprise-scale performance:

- **⚡ API Response Times**: P50 ≤ 500ms, P95 ≤ 1500ms
- **🗨️ Chat Streaming**: First token ≤ 1500ms
- **🔗 Webhook Processing**: ≤ 3000ms for large payloads
- **📈 Concurrent Users**: Supports 100+ concurrent sessions
- **🔄 Repository Analysis**: 5-15 minutes for typical repositories

## 🧪 Development

### Running Tests

```bash
# Install test dependencies
pip install -e ".[test]"

# Clean cache before testing (recommended)
make clean-cache               # Linux/macOS/WSL
python scripts/clean_cache.py  # Any platform

# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests
pytest tests/security/      # Security tests
pytest tests/contract/      # Contract tests

# Run wiki generation tests
pytest tests/unit/test_wiki_workflow.py -v       # Wiki workflow unit tests
pytest tests/unit/test_wiki_react_agents.py -v   # React agent tests
pytest tests/integration/test_wiki_generation.py -v  # Wiki integration tests
pytest tests/integration/test_wiki_workflow_integration.py -v  # Workflow integration

# Run BDD/Cucumber tests
pytest tests/bdd/ -v                    # All BDD tests
pytest tests/bdd/test_repositories.py   # Repository API scenarios
pytest tests/bdd/test_chat.py           # Chat API scenarios
pytest tests/bdd/test_wiki.py           # Wiki API scenarios

# Run with coverage
pytest --cov=src --cov-report=html
```

### BDD Testing (Cucumber-Style)

AutoDoc v2 includes BDD tests using **pytest-bdd** with Gherkin syntax for readable, behavior-driven API testing.

#### Test Structure
```
tests/bdd/
├── features/               # Gherkin .feature files
│   ├── repositories.feature   # Repository management scenarios
│   ├── chat.feature          # Chat workflow scenarios
│   └── wiki.feature          # Wiki generation scenarios
├── step_defs/              # Step implementations
│   ├── common_steps.py       # Shared auth & assertion steps
│   ├── repository_steps.py   # Repository API steps
│   ├── chat_steps.py         # Chat API steps
│   └── wiki_steps.py         # Wiki API steps
├── test_repositories.py    # Test runner for repositories
├── test_chat.py            # Test runner for chat
└── test_wiki.py            # Test runner for wiki
```

#### Running BDD Tests
```bash
# Run all BDD tests (55 scenarios)
pytest tests/bdd/ -v

# Run specific feature
pytest tests/bdd/test_repositories.py -v
pytest tests/bdd/test_chat.py -v
pytest tests/bdd/test_wiki.py -v

# Run by keyword
pytest tests/bdd/ -k "register" -v    # All registration scenarios
pytest tests/bdd/ -k "webhook" -v     # Webhook-related scenarios

# Run specific scenario
pytest tests/bdd/test_repositories.py -k "trigger_repository_analysis" -v
```

#### Troubleshooting BDD Tests
```bash
# Run tests WITHOUT cleanup (preserves test data in MongoDB for inspection)
pytest tests/bdd/test_repositories.py -v --no-cleanup

# Run specific scenario without cleanup
pytest tests/bdd/test_repositories.py -k "trigger_repository_analysis" -v --no-cleanup
```

The `--no-cleanup` flag:
- Skips pre-test database cleanup (existing data preserved)
- Skips post-scenario cleanup (created repositories/sessions remain)
- Skips post-test database cleanup (all test data remains for inspection)
- Useful for debugging failed tests by examining the database state

#### Example Feature File
```gherkin
Feature: Repository Management
  As an API user
  I want to manage code repositories
  So that I can generate documentation

  Scenario: Register a new GitHub repository
    Given I am authenticated as an admin user
    When I register a repository with URL "https://github.com/test-org/test-repo"
    Then the response status should be 201
    And the repository should have status "pending"
    And the repository provider should be "github"
```

### Code Quality

```bash
# Format code
python -m black src/ tests/

# Sort imports
python -m isort src/ tests/ --profile black

# Type checking
python -m mypy src/

# Linting
python -m flake8 src/ tests/
```

### Development Setup

```bash
# Install in development mode
pip install -e ".[dev]"

# Start development server (recommended - includes cache cleaning)
make dev-run                    # Linux/macOS/WSL
.\scripts\dev-run.ps1          # Windows PowerShell
dev-run.bat                    # Windows Command Prompt

# Alternative: Start without cache cleaning
python -m src.api.main

# Clean cache manually (if needed)
make clean-cache               # Linux/macOS/WSL
python scripts/clean_cache.py  # Any platform

# Run with debug logging
LOG_LEVEL=DEBUG python -m src.api.main

# Access interactive API documentation during development
open http://localhost:8000/docs
```

#### 🧹 Cache Management

AutoDoc v2 includes automatic cache cleaning for development to prevent issues with Python cache files:

**What gets cleaned:**
- `__pycache__` directories (recursively)
- `*.egg-info` directories  
- `.mypy_cache`, `.pytest_cache` directories
- `build/`, `dist/`, `htmlcov/` directories
- `.coverage` files and `*.pyc`/`*.pyo` files

**When to use cache cleaning:**
- Starting a new development session
- After switching Git branches
- When experiencing import errors
- Before running tests
- After installing/updating dependencies

**Available commands:**
```bash
# Cross-platform cache cleaning
python scripts/clean_cache.py

# Development server with cache cleaning (recommended)
make dev-run                    # Linux/macOS/WSL
.\scripts\dev-run.ps1          # Windows PowerShell
dev-run.bat                    # Windows Command Prompt

# Cache cleaning only
make clean-cache               # Linux/macOS/WSL
.\scripts\dev-run.ps1 -CleanOnly  # Windows PowerShell

# Start server without cleaning (original behavior)
make run                       # Linux/macOS/WSL
python -m src.api.main         # Any platform
```

### Development Scripts

AutoDoc v2 includes several utility scripts in the `scripts/` directory to streamline development:

#### Cache Management Scripts
- **`scripts/clean_cache.py`**: Cross-platform Python cache cleaner
- **`scripts/dev-run.ps1`**: Windows PowerShell development server with cache cleaning
- **`scripts/dev-run.sh`**: Unix/macOS development server with cache cleaning
- **`scripts/README.md`**: Comprehensive documentation for all development scripts
- **`dev-run.bat`**: Windows batch file for easy server startup

#### Database Management Scripts
- **`scripts/drop_indexes.py`**: Drop all MongoDB indexes (for schema migrations)
- **`scripts/drop_text_indexes.py`**: Drop text indexes only
- **`scripts/fix_index_conflict.py`**: Resolve index conflicts during Beanie migration
- **`scripts/fix_index_conflict_v2.py`**: Enhanced index conflict resolution
- **`scripts/test_startup.py`**: Test application startup and database connectivity

#### Usage Examples
```bash
# Cross-platform cache cleaning
python scripts/clean_cache.py

# Unix/macOS development
./scripts/dev-run.sh           # Clean cache + start server
./scripts/dev-run.sh -c        # Clean cache only
./scripts/dev-run.sh -s        # Start server without cleaning
./scripts/dev-run.sh -h        # Show help

# Windows PowerShell (multiple options)
.\scripts\dev-run.ps1           # Clean cache + start server
.\scripts\dev-run.ps1 -CleanOnly # Clean cache only
.\scripts\dev-run.ps1 -SkipClean # Start server without cleaning
.\scripts\dev-run.ps1 -Help     # Show help

# Windows Command Prompt
dev-run.bat

# Make targets (Linux/macOS/WSL)
make clean-cache    # Clean cache only
make dev-run        # Clean cache + start server
make run           # Start server without cleaning

# Database management
python scripts/drop_indexes.py           # Drop all indexes
python scripts/drop_text_indexes.py      # Drop text indexes only
python scripts/fix_index_conflict.py     # Fix index conflicts
```

### API Development Workflow

1. **Setup**: Use `make dev-run` or `.\scripts\dev-run.ps1` to start with a clean environment
2. **Design First**: Use the interactive Swagger UI to design and test endpoints
3. **Implement**: Write FastAPI route handlers with Pydantic models
4. **Document**: Add comprehensive examples and descriptions using OpenAPI extras
5. **Test**: Use the Swagger UI to test endpoints during development
6. **Validate**: Ensure all examples work correctly in the documentation

### Architecture Guidelines

When developing new features, follow these architectural principles:

#### **🎯 Separation of Concerns**
- **API Layer**: Only handle HTTP concerns (request/response, validation, auth)
- **Service Layer**: Contain all business logic, orchestrate workflows
- **Repository Layer**: Abstract data access, provide domain-specific operations
- **Data Access Layer**: Handle generic database operations, connections

#### **💉 Dependency Injection**
```python
# Services receive repositories via constructor injection
class DocumentService:
    def __init__(self, code_document_repo=None):
        self._code_document_repo = code_document_repo
    
    async def _get_code_document_repo(self):
        if self._code_document_repo is None:
            from ..repository import data_access
            dal = await data_access.get_data_access()
            self._code_document_repo = dal.code_documents
        return self._code_document_repo
```

#### **🧪 Testing Strategy**
- **Unit Tests**: Mock repository dependencies in services
- **Integration Tests**: Test full request/response cycles
- **Repository Tests**: Test data access patterns
- **Contract Tests**: Validate API contracts

#### **📁 File Organization**
```
src/
├── api/                    # FastAPI endpoints & middleware
├── services/               # Business logic (clean, no data access)
├── repository/             # Data access layer & repository implementations
│   ├── data_access.py     # Core database infrastructure
│   └── *_repository.py    # Domain-specific repositories
├── agents/                 # LangGraph AI workflows
├── tools/                  # Shared AI tools & utilities
├── models/                 # Pydantic/Beanie data models
└── utils/                  # Configuration & utilities
```

## 🚀 Deployment

### Docker Deployment

```bash
# Build container
docker build -t autodoc-v2 .

# Run with Docker Compose
docker-compose up -d
```

### Production Configuration

```env
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security (CHANGE THESE!)
SECRET_KEY=your-production-secret-key-minimum-32-characters
JWT_ALGORITHM=HS256

# Database (production MongoDB)
MONGODB_URL=mongodb://username:password@your-mongo-cluster:27017
MONGODB_DATABASE=autodoc_production

# Storage (S3 recommended for production)
STORAGE_TYPE=s3
S3_BUCKET_NAME=your-production-bucket
AWS_ACCESS_KEY_ID=your-production-access-key
AWS_SECRET_ACCESS_KEY=your-production-secret-key

# Performance
WORKERS=4
MAX_CONNECTIONS=1000
```

## 🎯 Use Cases

### Software Teams
- **📋 Onboarding**: Generate comprehensive documentation for new team members
- **🔍 Code Discovery**: Find relevant code sections using natural language queries
- **📖 Documentation Maintenance**: Keep docs synchronized with code changes
- **🤝 Knowledge Sharing**: Capture and share architectural decisions

### Open Source Projects
- **👥 Contributor Onboarding**: Help new contributors understand the codebase
- **📚 User Documentation**: Generate user-friendly guides from code
- **🔍 Issue Triage**: Quickly find relevant code for bug reports
- **📈 Project Growth**: Scale documentation with project complexity

### Enterprise Organizations
- **🏢 Compliance**: Maintain up-to-date documentation for audits
- **🔄 Legacy Systems**: Document undocumented legacy codebases
- **🚀 Migration Planning**: Understand system architecture for migrations
- **📊 Technical Debt**: Identify and document technical debt areas

## 📞 Support & Community

- **🐛 Bug Reports**: [GitHub Issues](https://github.com/your-org/autodoc-v2/issues)
- **💡 Feature Requests**: [GitHub Discussions](https://github.com/your-org/autodoc-v2/discussions)
- **📧 Email Support**: support@autodoc.dev
- **💬 Community Chat**: [Discord Server](https://discord.gg/autodoc)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain Team** for the excellent LLM orchestration framework
- **FastAPI** for the high-performance async web framework
- **MongoDB** for vector search capabilities
- **OpenAI** for GPT models that power the AI features

---

**Built with ❤️ using Test-Driven Development and AI-First Architecture**

*AutoDoc v2 - Making code documentation effortless through AI*