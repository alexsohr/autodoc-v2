# Data Model: AutoDoc v2

## Core Entities

### Repository
**Purpose**: Represents a source code repository being analyzed
**Fields**:
- `id`: UUID - Unique identifier
- `provider`: Enum[github, bitbucket, gitlab] - Repository provider
- `url`: str - Repository URL
- `org`: str - Organization/owner name
- `name`: str - Repository name
- `default_branch`: str - Default branch name (e.g., "main", "master")
- `access_scope`: Enum[public, private] - Repository visibility
- `last_analyzed`: datetime - Last analysis timestamp
- `analysis_status`: Enum[pending, processing, completed, failed] - Current status
- `commit_sha`: str - Last analyzed commit SHA
- `webhook_configured`: bool - Whether webhook is configured (default: False)
- `webhook_secret`: str - Secret for validating webhook signatures (optional)
- `subscribed_events`: List[str] - Events that trigger documentation updates
- `last_webhook_event`: datetime - Timestamp of last received webhook event (optional)
- `created_at`: datetime
- `updated_at`: datetime

**Relationships**:
- One-to-many with CodeDocument
- One-to-many with WikiStructure
- One-to-many with Question/Answer sessions

**Validation Rules**:
- URL must be valid repository URL
- Provider must match URL domain
- Default branch cannot be empty
- Commit SHA must be valid git hash
- Webhook secret must be non-empty if webhook_configured is True
- Subscribed events must be valid for the provider

### CodeDocument
**Purpose**: Represents a processed code file for semantic search
**Fields**:
- `id`: str - Unique identifier
- `repository_id`: UUID - Foreign key to Repository
- `file_path`: str - Path relative to repository root
- `language`: str - Programming language
- `content`: str - File content
- `processed_content`: str - Cleaned content for embedding
- `metadata`: dict - File metadata (size, last_modified, etc.)
- `embedding`: List[float] - Vector embedding (optional, stored separately)
- `created_at`: datetime
- `updated_at`: datetime

**Relationships**:
- Many-to-one with Repository

**Validation Rules**:
- File path must be relative to repository root
- Language must be supported language code
- Content cannot be empty for text files

### WikiStructure
**Purpose**: Complete wiki structure for a repository
**Fields**:
- `id`: str - Unique identifier
- `title`: str - Wiki title
- `description`: str - Wiki description
- `pages`: List[WikiPageDetail] - All wiki pages
- `sections`: List[WikiSection] - All wiki sections
- `root_sections`: List[str] - Top-level section IDs

**Relationships**:
- One-to-one with Repository
- Contains WikiPageDetail and WikiSection entities

**Validation Rules**:
- Root sections must exist in sections list
- All page and section IDs must be unique

### WikiPageDetail
**Purpose**: Individual wiki page with content and metadata
**Fields**:
- `id`: str - Unique page identifier
- `title`: str - Page title
- `description`: str - Page description
- `importance`: str - Priority level ('high', 'medium', 'low')
- `file_paths`: List[str] - Source file paths this page covers
- `related_pages`: List[str] - IDs of related pages
- `content`: str - Generated markdown content (default: empty)

**Relationships**:
- Part of WikiStructure
- References other WikiPageDetail via related_pages

**Validation Rules**:
- Importance must be one of: 'high', 'medium', 'low'
- File paths must be valid relative paths
- Related pages must reference existing page IDs
- Content defaults to empty string

### WikiSection
**Purpose**: Organizational section containing pages and subsections
**Fields**:
- `id`: str - Unique section identifier
- `title`: str - Section title
- `pages`: List[str] - Page IDs in this section
- `subsections`: List[str] - Subsection IDs (default: empty)

**Relationships**:
- Part of WikiStructure
- Contains WikiPageDetail references
- Can contain other WikiSection references

**Validation Rules**:
- Page IDs must reference existing pages
- Subsection IDs must reference existing sections
- Subsections default to empty list

### ChatSession
**Purpose**: Conversational query session for a repository
**Fields**:
- `id`: UUID - Session identifier
- `repository_id`: UUID - Foreign key to Repository
- `created_at`: datetime
- `last_activity`: datetime
- `status`: Enum[active, expired] - Session status

**Relationships**:
- Many-to-one with Repository
- One-to-many with Question/Answer

**State Transitions**:
- Active → Expired (after inactivity timeout)

### Question
**Purpose**: User query about the codebase
**Fields**:
- `id`: UUID - Unique identifier
- `session_id`: UUID - Foreign key to ChatSession
- `content`: str - Question text
- `timestamp`: datetime
- `context_nodes`: List[str] - Relevant analysis node IDs used for context

**Relationships**:
- Many-to-one with ChatSession
- One-to-one with Answer

### Answer
**Purpose**: AI-generated response to user question
**Fields**:
- `id`: UUID - Unique identifier
- `question_id`: UUID - Foreign key to Question
- `content`: str - Answer text
- `citations`: List[Citation] - Source code references
- `confidence_score`: float - Answer confidence (0.0-1.0)
- `generation_time`: float - Response generation time (seconds)
- `timestamp`: datetime

**Relationships**:
- One-to-one with Question

### Citation
**Purpose**: Reference to source code supporting an answer
**Fields**:
- `file_path`: str - Path to source file
- `line_start`: int - Starting line number (optional)
- `line_end`: int - Ending line number (optional)
- `commit_sha`: str - Commit SHA of referenced code
- `url`: str - Direct link to source code
- `excerpt`: str - Relevant code snippet (optional)

**Validation Rules**:
- File path must exist in repository
- Line numbers must be valid if provided
- Commit SHA must be valid git hash
- URL must be accessible

## Configuration Models

### LLMConfig
**Purpose**: LLM provider configuration
**Fields**:
- `provider`: Enum[openai, gemini, bedrock, ollama] - Provider name
- `model_name`: str - Specific model identifier
- `api_key`: str - Provider API key (encrypted)
- `endpoint_url`: str - Custom endpoint (optional)
- `max_tokens`: int - Maximum response tokens
- `temperature`: float - Generation temperature (0.0-1.0)
- `timeout`: int - Request timeout (seconds)

### StorageConfig
**Purpose**: Storage adapter configuration
**Fields**:
- `type`: Enum[local, s3] - Storage type
- `base_path`: str - Base directory/bucket path
- `connection_params`: dict - Provider-specific parameters
- `backup_enabled`: bool - Backup configuration flag
- `retention_days`: int - Data retention period

## Environment-Specific Variations

### Development Environment
- Repository storage: Local filesystem (`./data/repos/`)
- Primary storage: Local MongoDB (`mongodb://localhost:27017/autodoc_dev`)
- Vector storage: MongoDB vector search capabilities
- Configuration: Environment variables from `.env`
- Minimal authentication requirements

### Production Environment
- Repository storage: AWS S3 buckets with organization structure
- Primary storage: MongoDB with connection pooling and replica sets
- Vector storage: MongoDB vector search capabilities
- Configuration: AWS Parameter Store
- Full authentication and authorization

## Data Flow Patterns

### Repository Analysis Flow
1. Repository → CodeDocuments (created and processed)
2. CodeDocuments → Embeddings generated and stored
3. CodeDocuments → WikiStructure (generated using semantic search)
4. WikiStructure → WikiPageDetail content generation via RAG
5. Repository.last_analyzed updated

### Query Processing Flow
1. ChatSession → Question (created)
2. Question → Semantic search across CodeDocuments
3. Relevant CodeDocuments retrieved for context
4. Answer generated with Citations using RAG
5. ChatSession.last_activity updated

### Webhook-Triggered Update Flow
1. Webhook event received (GitHub/Bitbucket push or PR merge)
2. Event validated using webhook signature and repository lookup
3. Repository identified from webhook payload
4. Repository.last_webhook_event updated
5. Document generation workflow triggered immediately
6. Changed files → CodeDocuments updated
7. New embeddings generated for changed files
8. WikiStructure updated based on semantic analysis
9. WikiPageDetail content regenerated using updated context
10. Pull request created with wiki changes

### Manual Update Flow (Fallback)
1. Repository change detected via manual trigger
2. Changed files → CodeDocuments updated
3. New embeddings generated for changed files
4. WikiStructure updated based on semantic analysis
5. WikiPageDetail content regenerated using updated context
6. Pull request created with wiki changes
