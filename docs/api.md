# AutoDoc v2 API Documentation

## Overview

AutoDoc v2 provides a comprehensive REST API for automated documentation generation from Git repositories. The API enables repository analysis, wiki generation, conversational queries, and webhook-driven updates.

## Base URL

```
https://your-autodoc-instance.com/api/v2
```

## Authentication

All API endpoints (except webhooks and health checks) require JWT authentication:

```http
Authorization: Bearer <your-jwt-token>
```

### Getting a Token

```bash
curl -X POST https://your-autodoc-instance.com/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "your-username", "password": "your-password"}'
```

## API Endpoints

### Repository Management

#### Create Repository
Register a new repository for analysis.

```http
POST /api/v2/repositories
Content-Type: application/json

{
  "url": "https://github.com/owner/repo",
  "branch": "main",
  "provider": "github"
}
```

**Response (201 Created):**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "provider": "github",
  "url": "https://github.com/owner/repo",
  "org": "owner",
  "name": "repo",
  "default_branch": "main",
  "access_scope": "public",
  "analysis_status": "pending",
  "webhook_configured": false,
  "created_at": "2023-12-01T10:00:00Z",
  "updated_at": "2023-12-01T10:00:00Z"
}
```

#### List Repositories
Get paginated list of repositories.

```http
GET /api/v2/repositories?limit=50&offset=0&status=completed
```

**Response (200 OK):**
```json
{
  "repositories": [...],
  "total": 150,
  "limit": 50,
  "offset": 0
}
```

#### Get Repository
Get detailed repository information.

```http
GET /api/v2/repositories/{repository_id}
```

#### Delete Repository
Remove repository and all associated data.

```http
DELETE /api/v2/repositories/{repository_id}
```

#### Trigger Analysis
Manually trigger repository analysis.

```http
POST /api/v2/repositories/{repository_id}/analyze
Content-Type: application/json

{
  "branch": "main",
  "force": false
}
```

#### Get Analysis Status
Check repository analysis progress.

```http
GET /api/v2/repositories/{repository_id}/status
```

**Response (200 OK):**
```json
{
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 75.5,
  "current_step": "Generating embeddings",
  "last_analyzed": "2023-12-01T10:00:00Z",
  "documents_processed": 245,
  "embeddings_generated": 180
}
```

### Webhook Configuration

#### Configure Webhook
Set up webhook for automatic updates.

```http
PUT /api/v2/repositories/{repository_id}/webhook
Content-Type: application/json

{
  "webhook_secret": "your-secure-webhook-secret",
  "subscribed_events": ["push", "pull_request"]
}
```

#### Get Webhook Configuration
Get webhook setup instructions.

```http
GET /api/v2/repositories/{repository_id}/webhook
```

**Response (200 OK):**
```json
{
  "webhook_configured": true,
  "subscribed_events": ["push", "pull_request"],
  "setup_instructions": {
    "github": {
      "webhook_url": "https://your-autodoc-instance.com/webhooks/github",
      "content_type": "application/json",
      "events": ["push", "pull_request"],
      "instructions": "Go to Settings > Webhooks > Add webhook in your GitHub repository"
    }
  }
}
```

### Documentation/Wiki

#### Get Wiki Structure
Retrieve complete wiki structure for repository.

```http
GET /api/v2/repositories/{repository_id}/wiki?include_content=false&section_id=overview
```

**Response (200 OK):**
```json
{
  "id": "wiki_550e8400-e29b-41d4-a716-446655440000",
  "title": "Repository Documentation",
  "description": "Comprehensive documentation for the repository",
  "pages": [
    {
      "id": "overview",
      "title": "Project Overview",
      "description": "High-level project overview and architecture",
      "importance": "high",
      "file_paths": ["README.md", "docs/overview.md"],
      "related_pages": ["getting-started"],
      "content": "# Project Overview\n\n..."
    }
  ],
  "sections": [
    {
      "id": "introduction",
      "title": "Introduction",
      "pages": ["overview", "getting-started"],
      "subsections": []
    }
  ],
  "root_sections": ["introduction"]
}
```

#### Get Wiki Page
Get specific wiki page content.

```http
GET /api/v2/repositories/{repository_id}/wiki/pages/{page_id}?format=json
```

```http
GET /api/v2/repositories/{repository_id}/wiki/pages/{page_id}?format=markdown
```

#### Create Documentation PR
Generate pull request with updated documentation.

```http
POST /api/v2/repositories/{repository_id}/pull-request
Content-Type: application/json

{
  "target_branch": "main",
  "title": "Update documentation",
  "description": "Automated documentation update",
  "force_update": false
}
```

#### List Repository Files
Get processed files available for semantic search.

```http
GET /api/v2/repositories/{repository_id}/files?language=python&path_pattern=src/**/*.py
```

### Chat/Conversational AI

#### Create Chat Session
Start new conversational session.

```http
POST /api/v2/repositories/{repository_id}/chat/sessions
```

**Response (201 Created):**
```json
{
  "id": "660e8400-e29b-41d4-a716-446655440000",
  "repository_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2023-12-01T10:00:00Z",
  "last_activity": "2023-12-01T10:00:00Z",
  "status": "active",
  "message_count": 0
}
```

#### List Chat Sessions
Get active chat sessions.

```http
GET /api/v2/repositories/{repository_id}/chat/sessions?status=active
```

#### Ask Question
Submit question about the codebase.

```http
POST /api/v2/repositories/{repository_id}/chat/sessions/{session_id}/questions
Content-Type: application/json

{
  "content": "How does user authentication work in this codebase?",
  "context_hint": "authentication, login, security"
}
```

**Response (201 Created):**
```json
{
  "question": {
    "id": "770e8400-e29b-41d4-a716-446655440000",
    "session_id": "660e8400-e29b-41d4-a716-446655440000",
    "content": "How does user authentication work in this codebase?",
    "timestamp": "2023-12-01T10:05:00Z",
    "context_files": ["src/auth/authentication.py", "src/middleware/auth.py"]
  },
  "answer": {
    "id": "880e8400-e29b-41d4-a716-446655440000",
    "question_id": "770e8400-e29b-41d4-a716-446655440000",
    "content": "The authentication system uses JWT tokens...",
    "citations": [
      {
        "file_path": "src/auth/authentication.py",
        "line_start": 25,
        "line_end": 45,
        "commit_sha": "abc123def456789012345678901234567890abcd",
        "url": "https://github.com/owner/repo/blob/main/src/auth/authentication.py#L25-L45",
        "excerpt": "def authenticate_user(username: str, password: str):"
      }
    ],
    "confidence_score": 0.92,
    "generation_time": 2.3,
    "timestamp": "2023-12-01T10:05:02Z"
  }
}
```

#### Stream Chat Responses
Real-time streaming responses via Server-Sent Events.

```http
GET /api/v2/repositories/{repository_id}/chat/sessions/{session_id}/stream
Accept: text/event-stream
```

#### Get Conversation History
Retrieve conversation history with pagination.

```http
GET /api/v2/repositories/{repository_id}/chat/sessions/{session_id}/history?limit=50
```

### Webhooks

#### GitHub Webhook
Receive GitHub webhook events.

```http
POST /webhooks/github
X-GitHub-Event: push
X-Hub-Signature-256: sha256=<signature>
X-GitHub-Delivery: <delivery-id>
Content-Type: application/json

{
  "ref": "refs/heads/main",
  "repository": {
    "full_name": "owner/repo",
    "clone_url": "https://github.com/owner/repo.git"
  },
  "commits": [...]
}
```

#### Bitbucket Webhook
Receive Bitbucket webhook events.

```http
POST /webhooks/bitbucket
X-Event-Key: repo:push
X-Hook-UUID: <hook-uuid>
Content-Type: application/json

{
  "repository": {
    "full_name": "owner/repo",
    "links": {
      "clone": [{"name": "https", "href": "https://bitbucket.org/owner/repo.git"}]
    }
  },
  "push": {...}
}
```

## Error Responses

All endpoints return structured error responses:

```json
{
  "error": "ErrorType",
  "message": "Human-readable error message",
  "correlation_id": "req_123456789",
  "timestamp": "2023-12-01T10:00:00Z",
  "path": "/api/v2/repositories",
  "method": "POST",
  "details": {
    "additional_error_info": "..."
  }
}
```

### Common Error Codes

- `400 Bad Request` - Invalid request data
- `401 Unauthorized` - Authentication required
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `409 Conflict` - Resource already exists or conflict
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Rate Limiting

API endpoints are rate limited to ensure fair usage:

- **General API**: 1000 requests per hour per user
- **Webhooks**: 100 requests per minute per repository
- **Chat**: 50 questions per hour per session
- **Analysis**: 10 analysis triggers per hour per repository

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1701432000
```

## Performance Characteristics

AutoDoc v2 is designed for high performance:

- **API Response Times**: P50 ≤ 500ms, P95 ≤ 1500ms
- **Chat First Token**: ≤ 1500ms for streaming responses
- **Webhook Processing**: ≤ 3000ms for complex payloads
- **Repository Analysis**: 5-15 minutes depending on size

## SDKs and Examples

### Python SDK Example

```python
import asyncio
from autodoc_client import AutoDocClient

async def main():
    client = AutoDocClient(
        base_url="https://your-autodoc-instance.com",
        api_key="your-api-key"
    )
    
    # Create repository
    repo = await client.repositories.create(
        url="https://github.com/owner/repo",
        branch="main"
    )
    
    # Wait for analysis
    while repo.analysis_status != "completed":
        await asyncio.sleep(30)
        repo = await client.repositories.get(repo.id)
    
    # Generate wiki
    wiki = await client.wiki.get_structure(repo.id)
    print(f"Generated wiki with {len(wiki.pages)} pages")
    
    # Start chat session
    session = await client.chat.create_session(repo.id)
    
    # Ask question
    qa = await client.chat.ask_question(
        session.id,
        "How does authentication work in this codebase?"
    )
    print(f"Answer: {qa.answer.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

### JavaScript/TypeScript Example

```javascript
import { AutoDocClient } from '@autodoc/client';

const client = new AutoDocClient({
  baseUrl: 'https://your-autodoc-instance.com',
  apiKey: 'your-api-key'
});

// Create and analyze repository
const repo = await client.repositories.create({
  url: 'https://github.com/owner/repo',
  branch: 'main'
});

// Wait for analysis completion
await client.repositories.waitForAnalysis(repo.id);

// Get wiki structure
const wiki = await client.wiki.getStructure(repo.id, {
  includeContent: true
});

console.log(`Wiki generated with ${wiki.pages.length} pages`);

// Start chat session
const session = await client.chat.createSession(repo.id);

// Ask question with streaming
const stream = client.chat.streamQuestion(session.id, {
  content: 'Explain the main architecture of this project'
});

for await (const chunk of stream) {
  if (chunk.status === 'streaming') {
    process.stdout.write(chunk.content);
  }
}
```

### cURL Examples

#### Create Repository
```bash
curl -X POST https://your-autodoc-instance.com/api/v2/repositories \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://github.com/owner/repo",
    "branch": "main"
  }'
```

#### Get Wiki Structure
```bash
curl -X GET https://your-autodoc-instance.com/api/v2/repositories/$REPO_ID/wiki \
  -H "Authorization: Bearer $TOKEN"
```

#### Ask Question
```bash
curl -X POST https://your-autodoc-instance.com/api/v2/repositories/$REPO_ID/chat/sessions/$SESSION_ID/questions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "How do I configure the database connection?",
    "context_hint": "database, configuration"
  }'
```

## Webhook Setup

### GitHub Webhook Setup

1. Go to your repository Settings → Webhooks
2. Click "Add webhook"
3. Set Payload URL: `https://your-autodoc-instance.com/webhooks/github`
4. Set Content type: `application/json`
5. Set Secret: Your webhook secret from AutoDoc
6. Select events: `Push` and `Pull requests`
7. Ensure webhook is Active

### Bitbucket Webhook Setup

1. Go to repository Settings → Webhooks
2. Click "Add webhook"
3. Set URL: `https://your-autodoc-instance.com/webhooks/bitbucket`
4. Select triggers: `Repository push` and `Pull request fulfilled`
5. Save webhook

## Best Practices

### Repository Management
- Use descriptive branch names for analysis
- Configure webhooks for automatic updates
- Monitor analysis status for large repositories
- Use force=true sparingly for re-analysis

### Chat Usage
- Provide context hints for better answers
- Use specific questions for better results
- Check confidence scores in answers
- Review citations for accuracy

### Performance Optimization
- Use pagination for large result sets
- Filter by language/path for focused searches
- Cache wiki structures when possible
- Use streaming for real-time chat responses

### Security
- Rotate webhook secrets regularly
- Use HTTPS for all API calls
- Store API keys securely
- Monitor webhook signature validation failures

## Troubleshooting

### Common Issues

**Repository Analysis Stuck**
```bash
# Check analysis status
curl -X GET https://your-autodoc-instance.com/api/v2/repositories/$REPO_ID/status \
  -H "Authorization: Bearer $TOKEN"

# Force re-analysis if needed
curl -X POST https://your-autodoc-instance.com/api/v2/repositories/$REPO_ID/analyze \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"force": true}'
```

**Webhook Not Triggering**
1. Verify webhook secret matches AutoDoc configuration
2. Check webhook delivery history in Git provider
3. Verify webhook URL is accessible
4. Check AutoDoc logs for webhook processing errors

**Chat Responses Low Quality**
1. Ensure repository is fully analyzed
2. Provide more specific questions
3. Use context hints effectively
4. Check if embeddings are generated

### Error Codes Reference

| Code | Description | Solution |
|------|-------------|----------|
| 400 | Bad Request | Check request format and parameters |
| 401 | Unauthorized | Verify JWT token is valid |
| 403 | Forbidden | Check user permissions |
| 404 | Not Found | Verify resource exists |
| 409 | Conflict | Resource already exists or analysis in progress |
| 422 | Validation Error | Check request data format |
| 429 | Rate Limited | Wait and retry with backoff |
| 500 | Server Error | Check AutoDoc service status |

## API Versioning

AutoDoc v2 uses URL-based versioning:
- Current version: `/api/v2`
- Previous version: `/api/v1` (deprecated)

Version changes are backwards compatible within major versions.

## Support

- **Documentation**: https://docs.autodoc.dev
- **API Reference**: https://your-autodoc-instance.com/docs
- **Issues**: https://github.com/autodoc/autodoc-v2/issues
- **Community**: https://discord.gg/autodoc
