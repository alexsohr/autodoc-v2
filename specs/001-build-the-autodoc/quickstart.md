# AutoDoc v2 Quickstart Guide

This guide walks through the core AutoDoc v2 workflows to verify the implementation meets all functional requirements.

## Prerequisites

- AutoDoc v2 API server running locally or deployed
- Access to test repositories (public GitHub/Bitbucket repos)
- API client (curl, Postman, or HTTP client library)

## Workflow 1: Repository Registration and Analysis

### Step 1: Register a Repository
```bash
# Register a public repository for analysis
curl -X POST http://localhost:8000/repositories \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/fastapi/fastapi",
    "provider": "github"
  }'
```

**Expected Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "provider": "github",
  "url": "https://github.com/fastapi/fastapi",
  "org": "fastapi",
  "name": "fastapi",
  "default_branch": "master",
  "access_scope": "public",
  "analysis_status": "pending",
  "created_at": "2025-09-21T10:00:00Z",
  "updated_at": "2025-09-21T10:00:00Z"
}
```

**Validation Criteria:**
- ✅ Repository URL parsed correctly
- ✅ Provider auto-detected from URL
- ✅ Organization and name extracted
- ✅ Analysis status starts as "pending"

### Step 2: Monitor Analysis Progress
```bash
# Check analysis status
curl http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/status
```

**Expected Response (Processing):**
```json
{
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 45.0,
  "current_step": "Analyzing code structure",
  "estimated_completion": "2025-09-21T10:05:00Z"
}
```

**Expected Response (Completed):**
```json
{
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100.0,
  "current_step": "Analysis complete"
}
```

**Validation Criteria:**
- ✅ Status transitions: pending → processing → completed
- ✅ Progress percentage updates (0-100)
- ✅ Current step descriptions provided
- ✅ Analysis completes within reasonable time

## Workflow 2: Documentation Generation and Retrieval

### Step 3: Retrieve Generated Wiki Structure
```bash
# Get complete wiki structure
curl http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/wiki
```

**Expected Response:**
```json
{
  "id": "wiki-1",
  "title": "FastAPI Documentation Wiki",
  "description": "Comprehensive documentation for the FastAPI framework",
  "pages": [
    {
      "id": "overview",
      "title": "FastAPI Overview",
      "description": "High-level architecture and core concepts",
      "importance": "high",
      "file_paths": ["fastapi/main.py", "fastapi/__init__.py"],
      "related_pages": ["routing", "dependencies"],
      "content": ""
    },
    {
      "id": "routing",
      "title": "Request Routing",
      "description": "How FastAPI handles HTTP request routing",
      "importance": "high",
      "file_paths": ["fastapi/routing.py"],
      "related_pages": ["overview", "middleware"],
      "content": ""
    }
  ],
  "sections": [
    {
      "id": "core-concepts",
      "title": "Core Concepts",
      "pages": ["overview", "routing"],
      "subsections": []
    }
  ],
  "root_sections": ["core-concepts"]
}
```

**Validation Criteria:**
- ✅ Hierarchical wiki structure with sections and pages
- ✅ Page importance ranking (high/medium/low)
- ✅ File path associations for each page
- ✅ Related page references for navigation

### Step 3b: Retrieve Individual Wiki Page
```bash
# Get specific page with content
curl http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/wiki/pages/overview?format=markdown
```

**Expected Response (Markdown):**
```markdown
# FastAPI Overview

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints.

## Key Features

- **Fast**: Very high performance, on par with NodeJS and Go (thanks to Starlette and Pydantic)
- **Fast to code**: Increase the speed to develop features by about 200% to 300%
- **Fewer bugs**: Reduce about 40% of human (developer) induced errors

## Architecture

The main application class is defined in `fastapi/main.py`:

```python
class FastAPI:
    def __init__(self, ...):
        # Application initialization
```

This class handles request routing through the routing module...
```

**Validation Criteria:**
- ✅ Page content generated with meaningful documentation
- ✅ Code examples included from source files
- ✅ Markdown formatting properly applied
- ✅ File references match the page's file_paths

### Step 4: Get Repository File List
```bash
# Retrieve processed files for semantic search
curl http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/files
```

**Expected Response:**
```json
{
  "files": [
    {
      "id": "doc-1",
      "repository_id": "550e8400-e29b-41d4-a716-446655440000",
      "file_path": "fastapi/main.py",
      "language": "python",
      "metadata": {
        "size": 15420,
        "last_modified": "2025-09-21T09:30:00Z",
        "lines": 456
      },
      "created_at": "2025-09-21T10:02:00Z",
      "updated_at": "2025-09-21T10:02:00Z"
    },
    {
      "id": "doc-2",
      "repository_id": "550e8400-e29b-41d4-a716-446655440000",
      "file_path": "fastapi/routing.py",
      "language": "python",
      "metadata": {
        "size": 8920,
        "last_modified": "2025-09-21T09:15:00Z",
        "lines": 234
      },
      "created_at": "2025-09-21T10:02:15Z",
      "updated_at": "2025-09-21T10:02:15Z"
    }
  ],
  "total": 155,
  "languages": {
    "python": 150,
    "yaml": 5
  }
}
```

**Validation Criteria:**
- ✅ Files processed and indexed for semantic search
- ✅ Language detection working correctly
- ✅ File metadata captured (size, lines, modification time)
- ✅ Language statistics aggregated

## Workflow 3: Pull Request Creation

### Step 5: Create Documentation Pull Request
```bash
# Create PR with generated documentation
curl -X POST http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/pull-request \
  -H "Content-Type: application/json" \
  -d '{
    "title": "docs: Update AutoDoc analysis",
    "description": "Automated documentation update from AutoDoc v2 analysis"
  }'
```

**Expected Response:**
```json
{
  "pull_request_url": "https://github.com/fastapi/fastapi/pull/1234",
  "branch_name": "autodoc/wiki-update-2025-09-21",
  "files_changed": [
    "docs/wiki/overview.md",
    "docs/wiki/routing.md",
    "docs/wiki/README.md"
  ],
  "commit_sha": "def789ghi012"
}
```

**Validation Criteria:**
- ✅ Pull request created in target repository
- ✅ Wiki pages added to structured directory
- ✅ README.md includes navigation structure
- ✅ Commit message follows standards
- ✅ Branch name includes timestamp/identifier

## Workflow 4: Conversational Querying

### Step 6: Start Chat Session
```bash
# Create new chat session
curl -X POST http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/chat/sessions
```

**Expected Response:**
```json
{
  "id": "session-1",
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2025-09-21T10:10:00Z",
  "status": "active",
  "message_count": 0
}
```

### Step 7: Ask Questions About Codebase
```bash
# Ask a question about the codebase
curl -X POST http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/chat/sessions/session-1/questions \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How does FastAPI handle request validation?"
  }'
```

**Expected Response:**
```json
{
  "question": {
    "id": "question-1",
    "content": "How does FastAPI handle request validation?",
    "timestamp": "2025-09-21T10:11:00Z",
    "context_files": ["fastapi/main.py", "fastapi/dependencies.py", "pydantic_core/validators.py"]
  },
  "answer": {
    "id": "answer-1",
    "content": "FastAPI handles request validation through Pydantic models. When you define a route with a Pydantic model parameter, FastAPI automatically validates the incoming request data against the model schema...",
    "citations": [
      {
        "file_path": "fastapi/main.py",
        "line_start": 120,
        "line_end": 135,
        "commit_sha": "abc123def456",
        "url": "https://github.com/fastapi/fastapi/blob/abc123def456/fastapi/main.py#L120-L135",
        "excerpt": "@app.post(\"/items/\")\nasync def create_item(item: Item):\n    return item"
      }
    ],
    "confidence_score": 0.92,
    "generation_time": 1.2,
    "timestamp": "2025-09-21T10:11:01Z"
  }
}
```

**Validation Criteria:**
- ✅ Response generated within 1500ms (first token)
- ✅ Answer includes relevant citations with file paths
- ✅ Citations include commit SHA and direct links
- ✅ Code excerpts are relevant to question
- ✅ Confidence score provided

### Step 8: Test Streaming Responses
```bash
# Connect to streaming endpoint
curl -N http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/chat/sessions/session-1/stream
```

**Expected Stream Events:**
```
event: token
data: {"content": "FastAPI", "type": "token"}

event: token
data: {"content": " handles", "type": "token"}

event: citation
data: {"file_path": "fastapi/main.py", "line_start": 120, "url": "..."}

event: complete
data: {"answer_id": "answer-2", "total_time": 1.1}
```

**Validation Criteria:**
- ✅ Streaming starts within 1500ms
- ✅ Tokens arrive incrementally
- ✅ Citations provided during stream
- ✅ Completion event signals end

## Workflow 5: Webhook Configuration and Testing

### Step 9: Configure Repository Webhook
```bash
# Configure webhook settings for automatic updates
curl -X PUT http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/webhook \
  -H "Content-Type: application/json" \
  -d '{
    "webhook_secret": "your-secure-secret-key",
    "subscribed_events": ["push", "pull_request", "merge"]
  }'
```

**Expected Response:**
```json
{
  "webhook_configured": true,
  "webhook_secret": "your-secure-secret-key",
  "subscribed_events": ["push", "pull_request", "merge"],
  "setup_instructions": {
    "github": {
      "webhook_url": "http://localhost:8000/webhooks/github",
      "content_type": "application/json",
      "events": ["push", "pull_request"],
      "instructions": "Go to Settings > Webhooks > Add webhook in your GitHub repository"
    },
    "bitbucket": {
      "webhook_url": "http://localhost:8000/webhooks/bitbucket",
      "events": ["repo:push", "pullrequest:fulfilled"],
      "instructions": "Go to Settings > Webhooks > Add webhook in your Bitbucket repository"
    }
  }
}
```

**Validation Criteria:**
- ✅ Webhook configuration saved successfully
- ✅ Setup instructions provided for both GitHub and Bitbucket
- ✅ Webhook URLs point to correct global endpoints
- ✅ Event mappings correct for each provider

### Step 9b: Test GitHub Webhook Event
```bash
# Simulate GitHub webhook event
curl -X POST http://localhost:8000/webhooks/github \
  -H "Content-Type: application/json" \
  -H "X-GitHub-Event: push" \
  -H "X-Hub-Signature-256: sha256=calculated_signature" \
  -H "X-GitHub-Delivery: test-delivery-123" \
  -d '{
    "repository": {
      "full_name": "fastapi/fastapi",
      "clone_url": "https://github.com/fastapi/fastapi.git",
      "default_branch": "master"
    },
    "ref": "refs/heads/master",
    "after": "abc123def456",
    "commits": [
      {
        "id": "abc123def456",
        "message": "Update documentation"
      }
    ]
  }'
```

**Expected Response:**
```json
{
  "status": "processed",
  "message": "Webhook event processed successfully, documentation update triggered",
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "push",
  "processing_time": 0.15
}
```

**Validation Criteria:**
- ✅ Webhook event accepted and validated
- ✅ Repository identified from payload
- ✅ Documentation generation triggered immediately
- ✅ Response includes processing details

### Step 9c: Test Bitbucket Webhook Event
```bash
# Simulate Bitbucket webhook event
curl -X POST http://localhost:8000/webhooks/bitbucket \
  -H "Content-Type: application/json" \
  -H "X-Event-Key: repo:push" \
  -H "X-Hook-UUID: test-hook-uuid" \
  -d '{
    "repository": {
      "full_name": "fastapi/fastapi",
      "links": {
        "clone": [
          {
            "name": "https",
            "href": "https://bitbucket.org/fastapi/fastapi.git"
          }
        ]
      }
    },
    "push": {
      "changes": [
        {
          "new": {
            "name": "master",
            "target": {
              "hash": "abc123def456"
            }
          }
        }
      ]
    }
  }'
```

**Expected Response:**
```json
{
  "status": "processed",
  "message": "Webhook event processed successfully, documentation update triggered",
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "event_type": "repo:push",
  "processing_time": 0.12
}
```

**Validation Criteria:**
- ✅ Bitbucket webhook event processed correctly
- ✅ Different payload format handled properly
- ✅ Repository matching works across providers
- ✅ Documentation generation triggered

## Workflow 6: Repository Updates and Freshness

### Step 10: Trigger Manual Re-analysis (Fallback)
```bash
# Force re-analysis of repository
curl -X POST http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "force": true
  }'
```

**Expected Response:**
```json
{
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 0,
  "current_step": "Fetching latest changes"
}
```

### Step 10: Verify Update Detection
```bash
# Check if new wiki reflects changes
curl http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/wiki?include_content=true
```

**Validation Criteria:**
- ✅ Re-analysis triggered successfully
- ✅ Changes detected and processed
- ✅ Wiki structure updated with new/modified pages
- ✅ Page content reflects latest code changes
- ✅ Update completed within 10 minutes

## Performance Validation

### Response Time Tests
```bash
# Test API response times
time curl http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000

# Test chat response streaming
time curl -N http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/chat/sessions/session-1/stream
```

**Performance Criteria:**
- ✅ API responses: p50 ≤ 500ms, p95 ≤ 1500ms
- ✅ Chat streaming: First token ≤ 1500ms
- ✅ Repository analysis: Reasonable time for repo size
- ✅ No 5xx errors during normal operation

## Error Handling Validation

### Test Error Scenarios
```bash
# Test invalid repository URL
curl -X POST http://localhost:8000/repositories \
  -H "Content-Type: application/json" \
  -d '{"url": "invalid-url"}'

# Test non-existent repository
curl http://localhost:8000/repositories/00000000-0000-0000-0000-000000000000

# Test malformed question
curl -X POST http://localhost:8000/repositories/550e8400-e29b-41d4-a716-446655440000/chat/sessions/session-1/questions \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Error Handling Criteria:**
- ✅ Clear error messages for invalid inputs
- ✅ Appropriate HTTP status codes
- ✅ Graceful degradation on failures
- ✅ No system crashes or undefined behavior

## Success Criteria Summary

This quickstart validates all core AutoDoc v2 functionality:

- ✅ Repository registration and analysis
- ✅ Structured wiki generation with hierarchical organization
- ✅ Page importance ranking and file path associations
- ✅ Individual page content generation with markdown formatting
- ✅ Pull request creation with wiki structure
- ✅ Webhook configuration for GitHub and Bitbucket
- ✅ GitHub webhook event processing and validation
- ✅ Bitbucket webhook event processing with different payload format
- ✅ Automatic documentation updates via webhooks
- ✅ Conversational querying with citations
- ✅ Streaming responses within latency targets
- ✅ Repository update detection and freshness
- ✅ Performance targets met
- ✅ Error handling and edge cases covered

The implementation is ready for production deployment when all validation criteria pass.
