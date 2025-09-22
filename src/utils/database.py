"""MongoDB connection and initialization utilities"""

import asyncio
import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, TEXT, IndexModel
from pymongo.errors import CollectionInvalid

from .config_loader import get_settings

logger = logging.getLogger(__name__)


class DatabaseManager:
    """MongoDB database manager with connection pooling and initialization"""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.settings = get_settings()

    async def connect(self) -> None:
        """Connect to MongoDB and initialize database"""
        try:
            # Create MongoDB client
            self.client = AsyncIOMotorClient(
                self.settings.mongodb_url,
                maxPoolSize=self.settings.mongodb_max_connections,
                minPoolSize=self.settings.mongodb_min_connections,
                serverSelectionTimeoutMS=5000,  # 5 second timeout
                connectTimeoutMS=10000,  # 10 second timeout
            )

            # Test connection
            await self.client.admin.command("ping")
            logger.info(f"Connected to MongoDB at {self.settings.mongodb_url}")

            # Get database
            self.database = self.client[self.settings.mongodb_database]

            # Initialize database schema
            await self.initialize_database()

        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def initialize_database(self) -> None:
        """Initialize database collections and indexes"""
        if self.database is None:
            raise RuntimeError("Database not connected")

        try:
            # Create collections and indexes
            await self._create_repositories_collection()
            await self._create_code_documents_collection()
            await self._create_wiki_structures_collection()
            await self._create_chat_sessions_collection()
            await self._create_questions_collection()
            await self._create_answers_collection()

            logger.info("Database initialization completed")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def _create_repositories_collection(self) -> None:
        """Create repositories collection with indexes"""
        collection_name = "repositories"

        try:
            # Create collection
            await self.database.create_collection(collection_name)
        except CollectionInvalid:
            # Collection already exists
            pass

        collection = self.database[collection_name]

        # Create indexes
        indexes = [
            IndexModel([("url", ASCENDING)], unique=True),
            IndexModel(
                [("provider", ASCENDING), ("org", ASCENDING), ("name", ASCENDING)]
            ),
            IndexModel([("analysis_status", ASCENDING)]),
            IndexModel([("last_analyzed", ASCENDING)]),
            IndexModel([("webhook_configured", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        logger.info(f"Created {collection_name} collection with indexes")

    async def _create_code_documents_collection(self) -> None:
        """Create code_documents collection with indexes"""
        collection_name = "code_documents"

        try:
            await self.database.create_collection(collection_name)
        except CollectionInvalid:
            pass

        collection = self.database[collection_name]

        indexes = [
            IndexModel(
                [("repository_id", ASCENDING), ("file_path", ASCENDING)], unique=True
            ),
            IndexModel([("repository_id", ASCENDING)]),
            IndexModel([("language", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)]),
            IndexModel([("updated_at", ASCENDING)]),
            # Text index for content search
            IndexModel([("processed_content", TEXT)]),
        ]

        await collection.create_indexes(indexes)
        logger.info(f"Created {collection_name} collection with indexes")

    async def _create_wiki_structures_collection(self) -> None:
        """Create wiki_structures collection with indexes"""
        collection_name = "wiki_structures"

        try:
            await self.database.create_collection(collection_name)
        except CollectionInvalid:
            pass

        collection = self.database[collection_name]

        indexes = [
            IndexModel([("repository_id", ASCENDING)], unique=True),
            IndexModel([("title", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        logger.info(f"Created {collection_name} collection with indexes")

    async def _create_chat_sessions_collection(self) -> None:
        """Create chat_sessions collection with indexes"""
        collection_name = "chat_sessions"

        try:
            await self.database.create_collection(collection_name)
        except CollectionInvalid:
            pass

        collection = self.database[collection_name]

        indexes = [
            IndexModel([("repository_id", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("last_activity", ASCENDING)]),
            IndexModel([("created_at", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        logger.info(f"Created {collection_name} collection with indexes")

    async def _create_questions_collection(self) -> None:
        """Create questions collection with indexes"""
        collection_name = "questions"

        try:
            await self.database.create_collection(collection_name)
        except CollectionInvalid:
            pass

        collection = self.database[collection_name]

        indexes = [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", ASCENDING)]),
            # Text index for question content search
            IndexModel([("content", TEXT)]),
        ]

        await collection.create_indexes(indexes)
        logger.info(f"Created {collection_name} collection with indexes")

    async def _create_answers_collection(self) -> None:
        """Create answers collection with indexes"""
        collection_name = "answers"

        try:
            await self.database.create_collection(collection_name)
        except CollectionInvalid:
            pass

        collection = self.database[collection_name]

        indexes = [
            IndexModel([("question_id", ASCENDING)], unique=True),
            IndexModel([("timestamp", ASCENDING)]),
            IndexModel([("confidence_score", ASCENDING)]),
        ]

        await collection.create_indexes(indexes)
        logger.info(f"Created {collection_name} collection with indexes")

    async def health_check(self) -> bool:
        """Check database health"""
        try:
            if not self.client:
                return False

            # Ping the database
            await self.client.admin.command("ping")
            return True

        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def get_collection(self, collection_name: str):
        """Get a collection by name"""
        if self.database is None:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> AsyncIOMotorDatabase:
    """Get database instance"""
    if db_manager.database is None:
        await db_manager.connect()
    return db_manager.database


async def init_database() -> None:
    """Initialize database connection and schema"""
    await db_manager.connect()


async def close_database() -> None:
    """Close database connection"""
    await db_manager.disconnect()


# Convenience function for getting collections
async def get_collection(collection_name: str):
    """Get a collection by name"""
    database = await get_database()
    return database[collection_name]
