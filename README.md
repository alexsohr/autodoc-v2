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
- **🏗️ Modern Architecture**: FastAPI + LangGraph + MongoDB with vector search
- **🔒 Enterprise Security**: JWT authentication, webhook signature validation, injection prevention
- **📊 Performance Optimized**: P50 ≤ 500ms API responses, streaming chat with ≤ 1500ms first token
- **🧪 Test-Driven Development**: 100% TDD implementation with comprehensive test coverage
- **🔄 Multi-LLM Support**: OpenAI GPT, Google Gemini, Ollama local models
- **📦 Flexible Storage**: Local filesystem, AWS S3, MongoDB with vector capabilities

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
python -m src.api.main
```

6. **Access the API:**
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **OpenAPI Spec**: http://localhost:8000/openapi.json

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

# Optional: AWS S3 (if using S3 storage)
AWS_ACCESS_KEY_ID=your-aws-access-key
AWS_SECRET_ACCESS_KEY=your-aws-secret-key
AWS_REGION=us-east-1
S3_BUCKET_NAME=autodoc-storage
```

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

## 🏗️ Architecture

### System Overview

```mermaid
graph TB
    GP[Git Provider<br/>GitHub/Bitbucket/GitLab] --> API[AutoDoc v2<br/>FastAPI Application]
    API --> DB[(MongoDB<br/>Vector Search)]
    API --> LG[LangGraph<br/>AI Workflows]
    LG --> LLM[LLM Provider<br/>OpenAI/Gemini/Ollama]
    API --> SA[Storage Adapters<br/>Local/S3/MongoDB]
    
    subgraph "FastAPI Application"
        API --> AUTH[Auth Middleware]
        API --> LOG[Logging Middleware]
        API --> ERR[Error Handler]
        API --> ROUTES[API Routes]
    end
    
    subgraph "LangGraph Workflows"
        LG --> DA[Document Agent]
        LG --> WA[Wiki Agent]
        LG --> WO[Workflow Orchestrator]
    end
    
    subgraph "AI Tools"
        RT[Repository Tool]
        ET[Embedding Tool]
        CT[Context Tool]
        LT[LLM Tool]
        
        DA --> RT
        DA --> ET
        WA --> CT
        WA --> LT
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
    
    RepoSvc --> DocAgent
    WikiSvc --> WikiAgent
    ChatSvc --> ContextTool
    ChatSvc --> LLMTool
    
    DocAgent --> RepoTool
    DocAgent --> EmbedTool
    WikiAgent --> ContextTool
    WikiAgent --> LLMTool
    
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

### AI & ML
- **[OpenAI GPT](https://openai.com/)**: Primary LLM provider
- **[Google Gemini](https://ai.google.dev/)**: Alternative LLM provider  
- **[Ollama](https://ollama.ai/)**: Local LLM support
- **Vector Embeddings**: Semantic search and RAG capabilities

### DevOps & Quality
- **[pytest](https://pytest.org/)**: Comprehensive testing framework
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

# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests
pytest tests/security/      # Security tests

# Run with coverage
pytest --cov=src --cov-report=html
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

# Start with auto-reload
python -m src.api.main --reload

# Run with debug logging
LOG_LEVEL=DEBUG python -m src.api.main
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