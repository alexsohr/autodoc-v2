# Utils Subsystem

## Overview

Configuration, logging, retry utilities, and storage adapters for AutoDoc v2. This subsystem provides centralized application settings via Pydantic, structured logging with structlog, retry logic with tenacity, and pluggable storage backends.

## Setup

Required dependencies for this subsystem:
- `pydantic-settings` - Environment-aware configuration management
- `structlog` - Structured logging with JSON output and context binding
- `tenacity` - Retry decorators with exponential backoff

## Build/Tests

Utils are tested primarily through integration tests. Run with:
```bash
pytest tests/integration/ -k "config or logging or retry"
```

## Code Style

### Settings Pattern
Use `get_settings()` singleton - settings are loaded once and cached:
```python
from src.utils.config_loader import get_settings
settings = get_settings()
db_url = settings.mongodb_url
```

### Logging Pattern
Use structlog with context binding and correlation IDs:
```python
import structlog
logger = structlog.get_logger(__name__)
logger.info("Processing started", repository_id=repo_id, user_id=user_id)
```

### Retry Pattern
Use `@retry_async_on_failure` decorator with exponential backoff:
```python
from src.utils.retry_utils import retry_async_on_failure

@retry_async_on_failure(max_attempts=3, base_delay=1.0)
async def call_external_api():
    ...
```

### Storage Pattern
Factory pattern for interchangeable backends (local, S3, MongoDB):
```python
storage_type = settings.storage_type  # StorageType.LOCAL or StorageType.S3
```

## Security

- Never log secrets (API keys, tokens, passwords)
- Mask sensitive config in debug output using field validators
- Validate environment before startup (required API keys, database URLs)
- Log sensitive fields as `***` in function call logging

## PR Checklist

- [ ] Settings field changes: Update `.env.example` and document in README
- [ ] Logging format changes: Verify JSON output in production mode
- [ ] Retry policy changes: Test with simulated failures
- [ ] New environment variables: Add to `Settings` class with `Field()`

## Examples

### Using Settings
```python
from src.utils.config_loader import get_settings

settings = get_settings()
if settings.is_production:
    settings.configure_langsmith()
llm_config = settings.get_llm_config(LLMProvider.OPENAI)
```

### Using Retry Decorator
```python
from src.utils.retry_utils import retry_async_on_failure, RetryConfig

@retry_async_on_failure(
    retry_config=RetryConfig.LLM_RETRY,
    max_attempts=3
)
async def generate_content():
    return await llm.invoke(prompt)
```

### Using Structured Logging
```python
from src.utils.logging_config import LoggingContext

with LoggingContext(request_id=req_id, user_id=user_id) as log:
    log.info("Processing request")
```

## When Stuck

- Check env vars are set: `echo $MONGODB_URL`
- Verify settings loading: `python -c "from src.utils.config_loader import get_settings; print(get_settings().mongodb_url)"`
- Inspect log output format: Check `logs/autodoc.log` for JSON structure
- Test retry behavior: Use `RetryConfig.DATABASE_RETRY` for quick iteration

## House Rules

1. **All settings via `get_settings()`** - Never access `os.environ` directly in application code
2. **Use structlog for all logging** - Standard `logging` module only in utils internals
3. **Retry only on transient failures** - ConnectionError, TimeoutError, OSError (not ValueError, KeyError)
4. **Storage adapters are interchangeable** - Code should work with local, S3, or MongoDB storage
5. **Settings are immutable after load** - Do not modify settings at runtime
