"""Repository data access helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from pymongo import DESCENDING

from ..models.repository import Repository
from .base import BaseRepository


class RepositoryRepository(BaseRepository[Repository]):
    """Repository access helpers with pagination and webhook safeguards."""

    async def create(self, repository: Repository) -> Repository:
        repository.created_at = repository.created_at or datetime.now(timezone.utc)
        repository.updated_at = datetime.now(timezone.utc)
        repository.ensure_webhook_configuration()
        return await self.insert(repository)

    async def get(self, repository_id: UUID) -> Optional[Repository]:
        return await self.find_one({"_id": repository_id})

    async def get_by_url(self, url: str) -> Optional[Repository]:
        return await self.find_one({"url": url})

    async def update(self, repository_id: UUID, updates: Dict[str, Any]) -> bool:
        updates.setdefault("updated_at", datetime.now(timezone.utc))
        if updates.get("webhook_configured") and not updates.get("webhook_secret"):
            raise ValueError("Webhook secret must be provided when enabling webhook")
        return await self.update_one({"_id": repository_id}, updates)

    async def delete(self, repository_id: UUID) -> bool:
        return await self.delete_one({"_id": repository_id})

    async def list_paginated(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> Tuple[List[Repository], int]:
        query: Dict[str, Any] = {}
        if status_filter:
            query["analysis_status"] = status_filter

        documents = await self.find_many(
            query,
            limit=limit,
            offset=offset,
            sort=[("created_at", DESCENDING)],
        )
        total = await self.count(query)
        return documents, total

    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics including counts by status and provider.

        Returns:
            Dictionary with statistics including total count, status breakdown,
            provider breakdown, and recent repositories
        """
        from ..models.repository import AnalysisStatus, RepositoryProvider

        total = await self.count({})

        status_counts = {}
        for status in AnalysisStatus:
            status_counts[status.value] = await self.count({"analysis_status": status.value})

        provider_counts = {}
        for provider in RepositoryProvider:
            provider_counts[provider.value] = await self.count({"provider": provider.value})

        recent = await self.find_many({}, limit=5, sort=[("updated_at", DESCENDING)])

        return {
            "total": total,
            "by_status": status_counts,
            "by_provider": provider_counts,
            "recent": recent
        }
