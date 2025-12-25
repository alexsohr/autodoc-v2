"""Repository for code documents with vector and hybrid search helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from pymongo.errors import OperationFailure

from ..models.code_document import CodeDocument
from .base import BaseRepository


class CodeDocumentRepository(BaseRepository[CodeDocument]):
    """Repository of code documents with vector-search helpers."""

    async def store_embedding(self, document_id: str, embedding: List[float]) -> None:
        await self.update_one(
            {"_id": document_id},
            {"embedding": embedding, "updated_at": datetime.now(timezone.utc)},
        )

    async def vector_search(
        self,
        query_embedding: List[float],
        *,
        repository_id: Optional[UUID] = None,
        language_filter: Optional[str] = None,
        k: int = 10,
        score_threshold: float = 0.7,
    ) -> List[Dict[str, object]]:
        pipeline: List[Dict[str, object]] = []
        vector_stage: Dict[str, object] = {
            "$vectorSearch": {
                "index": "code_embeddings_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": k * 10,
                "limit": k,
            }
        }

        filters: Dict[str, object] = {}
        if repository_id:
            filters["repository_id"] = str(repository_id)
        if language_filter:
            filters["language"] = language_filter
        if filters:
            vector_stage["$vectorSearch"]["filter"] = filters

        pipeline.append(vector_stage)
        pipeline.append(
            {"$addFields": {"similarity_score": {"$meta": "vectorSearchScore"}}}
        )

        if score_threshold > 0:
            pipeline.append({"$match": {"similarity_score": {"$gte": score_threshold}}})

        results: List[Dict[str, object]] = []
        try:
            async for raw in self.collection.aggregate(pipeline):
                raw_id = raw.get("_id")
                if raw_id is not None:
                    raw["id"] = raw_id

                raw.pop("embedding", None)
                document = self.document(**raw)
                results.append(
                    {
                        "document": document,
                        "score": raw.get("similarity_score", 0.0),
                    }
                )
        except OperationFailure:
            return await self._fallback_text_search(
                query_embedding=query_embedding,
                repository_id=repository_id,
                language_filter=language_filter,
                k=k,
            )

        return results

    async def _fallback_text_search(
        self,
        query_embedding: List[float],
        *,
        repository_id: Optional[UUID],
        language_filter: Optional[str],
        k: int,
    ) -> List[Dict[str, object]]:
        query: Dict[str, object] = {}
        if repository_id:
            query["repository_id"] = str(repository_id)
        if language_filter:
            query["language"] = language_filter
        if query_embedding:
            query["$text"] = {"$search": "function class method"}

        cursor = self.collection.find(query).limit(k)
        results: List[Dict[str, object]] = []
        async for raw in cursor:
            raw.pop("embedding", None)
            raw_id = raw.get("_id")
            if raw_id is not None:
                raw["id"] = raw_id
            document = self.document(**raw)
            results.append({"document": document, "score": 0.5})
        return results

    async def hybrid_search(
        self,
        *,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        repository_id: Optional[UUID] = None,
        language_filter: Optional[str] = None,
        k: int = 10,
    ) -> List[Dict[str, object]]:
        base_filter: Dict[str, object] = {}
        if repository_id:
            base_filter["repository_id"] = str(repository_id)
        if language_filter:
            base_filter["language"] = language_filter

        text_results: List[Dict[str, object]] = []
        if query_text:
            text_query = {**base_filter, "$text": {"$search": query_text}}
            cursor = (
                self.collection.find(text_query, {"score": {"$meta": "textScore"}})
                .sort([("score", {"$meta": "textScore"})])
                .limit(k)
            )
            async for raw in cursor:
                raw.pop("embedding", None)
                raw_id = raw.get("_id")
                if raw_id is not None:
                    raw["id"] = raw_id
                document = self.document(**raw)
                text_results.append(
                    {
                        "document": document,
                        "score": raw.get("score", 0.0),
                        "source": "text",
                    }
                )

        vector_results: List[Dict[str, object]] = []
        if query_embedding:
            vector_results = await self.vector_search(
                query_embedding,
                repository_id=repository_id,
                language_filter=language_filter,
                k=k,
            )
            for result in vector_results:
                result["source"] = "vector"

        merged: Dict[str, Dict[str, object]] = {}
        for item in text_results + vector_results:
            doc_id = item["document"].id
            if doc_id in merged:
                existing = merged[doc_id]
                existing["score"] = (existing["score"] + item["score"]) / 2
                existing["source"] = "hybrid"
            else:
                merged[doc_id] = item

        final = list(merged.values())
        final.sort(key=lambda entry: entry["score"], reverse=True)
        return final[:k]

    async def get_language_statistics(self, query: Dict[str, object]) -> Dict[str, int]:
        """Get language statistics for documents matching the query.

        Args:
            query: Query filter

        Returns:
            Dictionary mapping language to count
        """
        pipeline = [
            {"$match": self._prepare_query(query)},
            {"$group": {"_id": "$language", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
        ]
        language_stats = {}
        async for doc in self.collection.aggregate(pipeline):
            language_stats[doc["_id"]] = doc["count"]
        return language_stats
