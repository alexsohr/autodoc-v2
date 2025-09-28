"""Repository abstractions wrapping Beanie documents.

Provides common CRUD helpers with UUID normalisation so existing services can
continue working with simple dict payloads while we transition to Beanie.
"""

from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)
from uuid import UUID

from beanie import Document
from motor.motor_asyncio import AsyncIOMotorCollection

TDocument = TypeVar("TDocument", bound=Document)


class BaseRepository(Generic[TDocument]):
    """Generic repository exposing dict-friendly helpers for Beanie documents."""

    def __init__(self, document: Type[TDocument]):
        self.document = document

    @property
    def collection(self) -> AsyncIOMotorCollection:
        """Return the underlying Motor collection."""

        return self.document.get_pymongo_collection()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------
    def _normalize_value(self, value: Any) -> Any:
        """Convert UUIDs (and nested structures) into Mongo-compatible values."""

        if isinstance(value, UUID):
            # Don't convert UUIDs to strings anymore since we configured UUID representation
            return value
        if isinstance(value, datetime):
            return value
        if isinstance(value, list):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._normalize_value(item) for item in value)
        if isinstance(value, dict):
            return {key: self._normalize_value(val) for key, val in value.items()}
        return value

    def _prepare_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._normalize_value(value) for key, value in query.items()}

    def _prepare_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        return {key: self._normalize_value(value) for key, value in updates.items()}

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    async def insert(self, payload: TDocument | Dict[str, Any]) -> TDocument:
        document = (
            payload if isinstance(payload, self.document) else self.document(**payload)
        )
        await document.insert()
        return document

    async def find_one(self, query: Dict[str, Any]) -> Optional[TDocument]:
        prepared = self._prepare_query(query)
        return await self.document.find_one(prepared)

    async def find_many(
        self,
        query: Dict[str, Any],
        *,
        limit: int = 100,
        offset: int = 0,
        sort: Optional[Sequence[Tuple[str, int]]] = None,
    ) -> List[TDocument]:
        prepared = self._prepare_query(query)
        cursor = self.document.find(prepared)
        if sort:
            cursor = cursor.sort(list(sort))
        if offset:
            cursor = cursor.skip(offset)
        if limit:
            cursor = cursor.limit(limit)
        return await cursor.to_list()

    async def update_one(self, query: Dict[str, Any], updates: Dict[str, Any]) -> bool:
        result = await self.collection.update_one(
            self._prepare_query(query), {"$set": self._prepare_updates(updates)}
        )
        return result.modified_count > 0

    async def delete_one(self, query: Dict[str, Any]) -> bool:
        result = await self.collection.delete_one(self._prepare_query(query))
        return result.deleted_count > 0

    async def delete_many(self, query: Dict[str, Any]) -> int:
        result = await self.collection.delete_many(self._prepare_query(query))
        return result.deleted_count

    async def count(self, query: Dict[str, Any]) -> int:
        return await self.collection.count_documents(self._prepare_query(query))

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def serialize(self, document: TDocument) -> Dict[str, Any]:
        data = document.model_dump(mode="python")
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
        return data

    def serialize_many(self, documents: Iterable[TDocument]) -> List[Dict[str, Any]]:
        return [self.serialize(doc) for doc in documents]
