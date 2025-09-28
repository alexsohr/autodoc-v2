"""Database initialization and connection management.

Provides database connection management and Beanie initialization
without the data access layer abstraction.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from beanie import init_beanie
from bson.binary import UuidRepresentation
from bson.codec_options import CodecOptions
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ServerSelectionTimeoutError

from ..models.chat import Answer, ChatSession, Question
from ..models.code_document import CodeDocument
from ..models.repository import Repository
from ..models.user import UserDocument
from ..models.wiki import WikiStructure
from ..utils.config_loader import get_settings


_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None
_initialized = False
_lock = asyncio.Lock()


async def _initialize() -> None:
    """Initialize database connection and Beanie ODM."""
    global _client, _database, _initialized

    if _initialized:
        return

    async with _lock:
        if _initialized:
            return

        settings = get_settings()

        try:
            client = AsyncIOMotorClient(
                settings.mongodb_url,
                maxPoolSize=settings.mongodb_max_connections,
                minPoolSize=settings.mongodb_min_connections,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
                uuidRepresentation="standard",  # Configure UUID representation
            )
            await client.admin.command("ping")
        except (
            ServerSelectionTimeoutError
        ) as exc:  # pragma: no cover - connection issues
            raise ConnectionError("Unable to connect to MongoDB server") from exc

        # Configure database with UUID codec options
        codec_options = CodecOptions(uuid_representation=UuidRepresentation.STANDARD)
        database = client[settings.mongodb_database].with_options(
            codec_options=codec_options
        )

        await init_beanie(
            database=database,
            document_models=[
                Repository,
                CodeDocument,
                WikiStructure,
                ChatSession,
                Question,
                Answer,
                UserDocument,
            ],
            allow_index_dropping=True,
        )

        _client = client
        _database = database
        _initialized = True


async def get_database() -> AsyncIOMotorDatabase:
    """Get the initialized database instance."""
    await _initialize()
    if _database is None:
        raise RuntimeError("Database not initialized")
    return _database


async def close_database() -> None:
    """Close the database connection."""
    global _client, _database, _initialized

    async with _lock:
        if _client:
            _client.close()
        _client = None
        _database = None
        _initialized = False


async def health_check() -> dict[str, object]:
    """Return database health information."""
    try:
        database = await get_database()
        stats = await database.command("dbStats")
        return {
            "status": "healthy",
            "database": database.name,
            "collections": stats.get("collections", 0),
            "data_size": stats.get("dataSize", 0),
            "index_size": stats.get("indexSize", 0),
        }
    except Exception as exc:  # pragma: no cover - defensive catch
        return {"status": "unhealthy", "error": str(exc)}


# Compatibility functions for existing API initialization
async def init_mongodb() -> None:
    """Initialize MongoDB connection (compatibility function)."""
    await _initialize()


async def close_mongodb() -> None:
    """Close MongoDB connection (compatibility function)."""
    await close_database()