# AGENTS.md - FastAPI API Layer

## Overview

FastAPI API layer with middleware chain and thin route handlers. Routes delegate business logic
to services via dependency injection. The middleware chain processes requests in order:
ErrorHandling -> Logging -> CORS -> Routes.

## Setup

```bash
pip install fastapi uvicorn pydantic structlog
```

Key dependencies: `fastapi`, `pydantic`, `uvicorn`, `structlog`, `python-jose[cryptography]`

## Build/Tests

```bash
# Run API layer tests
pytest tests/unit/test_health.py

# Start development server
python -m src.api.main

# Or via script
./scripts/dev-run.ps1
```

## Code Style

- Middleware registration order in `main.py`:
  1. `error_handling_middleware` (catches all exceptions)
  2. `request_logging_middleware` (logs requests/responses)
  3. `CORSMiddleware` (handles cross-origin requests)

- Routes use `Depends()` for dependency injection:
  ```python
  @router.get("/{id}")
  async def get_item(
      id: UUID,
      current_user: User = Depends(get_current_user),
      service: MyService = Depends(get_my_service),
  ):
      return await service.get_item(id)
  ```

- Use `Annotated[Type, Depends()]` for reusable dependencies
- Add `openapi_extra` for request/response examples in Swagger UI
- Use `response_model_by_alias=False` to return Python field names

## Security

- Use `get_current_user` dependency for authenticated endpoints
- Use `get_current_active_user` to verify active account status
- Use `get_admin_user` for admin-only endpoints
- Use `require_scopes(["scope"])` for scope-based authorization
- All inputs validated via Pydantic models before reaching handlers
- Public paths defined in `AuthMiddleware.public_paths`

## PR Checklist

- [ ] Middleware order preserved (ErrorHandling -> Logging -> CORS)
- [ ] Routes use authentication dependencies
- [ ] OpenAPI docs updated with examples and descriptions
- [ ] Pydantic models for all request/response bodies
- [ ] Error responses use consistent format with `correlation_id`
- [ ] New routes added to appropriate router with prefix

## Examples

### Route with Dependency Injection
```python
@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_item(
    data: ItemCreate,
    current_user: User = Depends(get_current_user),
    service: ItemService = Depends(get_item_service),
):
    result = await service.create(data)
    if result["status"] != "success":
        raise HTTPException(status_code=400, detail=result["error"])
    return ItemResponse(**result["item"])
```

### Middleware Registration (main.py)
```python
app.middleware("http")(request_logging_middleware)
app.middleware("http")(error_handling_middleware)
app.add_exception_handler(HTTPException, http_exception_handler)
```

### Consistent Error Response Format
```json
{
  "error": "ValidationError",
  "message": "Request validation failed",
  "correlation_id": "abc123",
  "timestamp": "2024-01-01T12:00:00Z",
  "path": "/api/v2/repositories",
  "method": "POST"
}
```

## When Stuck

1. Check middleware order in `main.py` - order matters for exception handling
2. Verify DI wiring in `dependencies.py` for service factories
3. Inspect `request.state` for user context and correlation_id
4. Check `AuthMiddleware.public_paths` if auth is bypassed unexpectedly
5. Review exception handlers registration order

## House Rules

- Routes are thin wrappers - no business logic, delegate to services
- All routes require authentication except health checks and webhooks
- Use Pydantic models for request/response validation
- Return consistent error format with `error`, `message`, `correlation_id`
- Include OpenAPI metadata: `summary`, `description`, `openapi_extra`
- Log errors with `structlog` including `correlation_id`
- Health endpoints at `/health`, `/health/ready`, `/health/live`
