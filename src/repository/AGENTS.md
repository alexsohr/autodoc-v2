# AGENTS.md - Repository Layer

## 1. Overview

Data access layer implementing the Repository Pattern with Beanie ODM. All database operations flow through `BaseRepository[TDocument]`, which provides UUID normalization, CRUD helpers, and serialization. Specialized repositories extend this base for domain-specific operations like vector search.

Key files:
- `base.py` - Generic `BaseRepository[TDocument]` with CRUD operations
- `database.py` - MongoDB/Beanie initialization and connection management
- `code_document_repository.py` - Vector search and hybrid search implementation

## 2. Setup

Dependencies (from `pyproject.toml`):
- `beanie` - Async MongoDB ODM built on Pydantic
- `motor` - Async MongoDB driver
- `pymongo` - MongoDB Python driver (used for codec options, errors)

Database initialization happens via `await get_database()` which initializes Beanie with all document models and configures UUID representation as STANDARD.

## 3. Build/Tests

```bash
# Run repository unit tests
pytest tests/unit/test_repositories.py -v

# Run integration tests (requires MongoDB)
pytest tests/integration/ -v

# Run with coverage
pytest tests/unit/ --cov=src/repository --cov-report=term-missing
```

## 4. Code Style

- **Extend BaseRepository**: All repositories must extend `BaseRepository[TDocument]`
- **UUID Handling**: Use `_prepare_query()` for all query dicts to normalize UUIDs
- **Async Only**: All repository methods must be `async def`
- **Vector Search**: Uses MongoDB Atlas `$vectorSearch` aggregation stage
- **Type Hints**: Use `TDocument` generic and `Dict[str, Any]` for queries

```python
# Query preparation pattern
prepared = self._prepare_query({"repository_id": some_uuid})
return await self.document.find_one(prepared)
```

## 5. Security

- **Sanitize Queries**: Always use `_prepare_query()` and `_prepare_updates()`
- **Parameterized Updates**: Use `{"$set": updates}` pattern, never string interpolation
- **Connection Strings**: Never log or expose `MONGODB_URL` - load via `get_settings()`
- **Embedding Data**: Strip embeddings from results before returning (`raw.pop("embedding", None)`)

## 6. PR Checklist

- [ ] Query normalization via `_prepare_query()` for all queries
- [ ] Index changes documented (vector indexes, text indexes)
- [ ] Connection handling uses `await get_database()` only
- [ ] New document models registered in `database.py` `init_beanie()` call
- [ ] Graceful handling of `None` results
- [ ] OperationFailure caught for vector search fallback

## 7. Examples

### Extending BaseRepository

```python
from .base import BaseRepository
from ..models.my_document import MyDocument

class MyDocumentRepository(BaseRepository[MyDocument]):
    def __init__(self):
        super().__init__(MyDocument)

    async def find_by_status(self, status: str) -> List[MyDocument]:
        return await self.find_many({"status": status})
```

### Vector Search Pipeline

```python
pipeline = [
    {
        "$vectorSearch": {
            "index": "code_embeddings_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": k * 10,
            "limit": k,
            "filter": {"repository_id": str(repo_id)},
        }
    },
    {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}},
    {"$match": {"similarity_score": {"$gte": score_threshold}}},
]
async for doc in self.collection.aggregate(pipeline):
    # Process results
```

### Upsert Pattern

```python
async def upsert(self, query: Dict, data: Dict) -> TDocument:
    existing = await self.find_one(query)
    if existing:
        await self.update_one(query, data)
        return await self.find_one(query)
    return await self.insert({**query, **data})
```

## 8. When Stuck

- **UUID Mismatch**: Ensure `_prepare_query()` is called; check `UuidRepresentation.STANDARD`
- **Vector Search Fails**: Verify `code_embeddings_index` exists in MongoDB Atlas
- **Empty Results**: Inspect aggregation pipeline with `collection.aggregate(pipeline).explain()`
- **Connection Issues**: Check `MONGODB_URL` in `.env`, run `await health_check()`
- **Beanie Not Initialized**: Ensure `await get_database()` called before any repository operation

## 9. House Rules

1. **ALL database access goes through repositories** - Services never import motor/pymongo
2. **Use `await get_database()` only in this layer** - Never in services or API routes
3. **Never expose raw motor/pymongo outside this directory** - Return Beanie documents or dicts
4. **Handle None results gracefully** - Check before accessing document attributes
5. **Strip sensitive data** - Remove embeddings and internal fields before returning
6. **Use lazy initialization** - Repositories are instantiated in services, not imported globally
