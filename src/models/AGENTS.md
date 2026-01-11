# AGENTS.md - Data Models Subsystem

## Overview

This subsystem contains Beanie ODM documents for MongoDB persistence and Pydantic models for API request/response handling. All documents extend `BaseDocument` which provides consistent serialization for UUIDs and datetimes.

**Key Documents:** `Repository`, `WikiStructure`, `CodeDocument`, `UserDocument`, `ChatSession`
**Key Patterns:** `BaseDocument` with `_SerializerMixin`, `Settings` inner class for indexes, separate response models

## Setup

```bash
pip install beanie motor pydantic pymongo
```

Required imports for new documents:
```python
from beanie import Document
from pydantic import Field, field_validator, model_validator, ConfigDict
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from .base import BaseDocument, BaseSerializers
```

## Build/Tests

```bash
pytest tests/unit/test_models.py -v
pytest tests/unit/test_models.py -k "test_repository" --tb=short
```

## Code Style

- **All Beanie documents** must extend `BaseDocument` (not `Document` directly)
- **Settings inner class** defines collection name and indexes:
  ```python
  class Settings:
      name = "collection_name"
      indexes = [IndexModel("field_name", unique=True)]
  ```
- **UUID normalization** uses `@field_validator(mode="before")` pattern
- **Response models** (Pydantic) are separate from Beanie documents to control serialization
- **ConfigDict** for model configuration: `validate_assignment=True`, `use_enum_values=True`

## Security

- **Never expose sensitive fields** in API responses: `webhook_secret`, `hashed_password`
- Use `exclude=True` in Field definition for write-only fields:
  ```python
  webhook_secret: Optional[str] = Field(default=None, exclude=True)
  ```
- Create separate `*Response` models that inherit from document but override sensitive fields

## PR Checklist

- [ ] Index changes reviewed (unique constraints, compound indexes)
- [ ] Field validators tested with edge cases
- [ ] Settings class has correct collection name
- [ ] Response model excludes sensitive fields
- [ ] UUID fields use `default_factory=uuid4`
- [ ] Datetime fields default to `datetime.now(timezone.utc)`

## Examples

### BaseDocument Pattern
```python
from .base import BaseDocument

class MyDocument(BaseDocument):
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Document name")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Settings:
        name = "my_documents"
        indexes = [IndexModel("name", unique=True)]

    model_config = ConfigDict(validate_assignment=True)
```

### Field Validator Usage
```python
@field_validator("commit_sha")
@classmethod
def validate_commit_sha(cls, v: Optional[str]) -> Optional[str]:
    if v is None:
        return v
    if not re.match(r"^[a-f0-9]{40}$", v, re.IGNORECASE):
        raise ValueError("Invalid git commit SHA format")
    return v.lower()
```

### Settings Class with Indexes
```python
class Settings:
    name = "repositories"
    indexes = [
        IndexModel("url", unique=True),
        IndexModel([("provider", ASCENDING), ("org", ASCENDING)]),
        IndexModel([("created_at", DESCENDING)]),
        IndexModel([("title", TEXT), ("description", TEXT)]),
    ]
```

## When Stuck

1. **UUID serialization issues**: Check `UuidRepresentation.STANDARD` in database.py
2. **Index not created**: Verify `init_beanie()` is called with document class
3. **Validator not running**: Ensure `validate_assignment=True` in ConfigDict
4. **Field not serializing**: Check `by_alias` setting in `model_dump()`
5. **Datetime format wrong**: Use `_SerializerMixin` from base.py

## House Rules

1. **UUIDs use STANDARD representation** - configured in database.py codec_options
2. **All timestamps in UTC** - use `datetime.now(timezone.utc)`, never naive datetimes
3. **Indexes defined in Settings class** - not created manually elsewhere
4. **Validate foreign keys before save** - use `@model_validator(mode="after")` for cross-field validation
5. **Enums use string values** - `class Status(str, Enum)` pattern with `use_enum_values=True`
6. **No direct motor/pymongo imports in services** - models can import for IndexModel only
