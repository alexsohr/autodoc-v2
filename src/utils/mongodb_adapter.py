"""MongoDB adapter with vector search capabilities

This module provides MongoDB operations with vector search support
for semantic search and RAG functionality in AutoDoc v2.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from uuid import UUID
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, OperationFailure, ServerSelectionTimeoutError
import numpy as np

from .config_loader import get_settings
from ..models.repository import Repository
from ..models.code_document import CodeDocument
from ..models.wiki import WikiStructure
from ..models.chat import ChatSession, Question, Answer

logger = logging.getLogger(__name__)


class MongoDBAdapter:
    """MongoDB adapter with vector search capabilities
    
    Provides async MongoDB operations with support for vector similarity search
    using MongoDB's vector search capabilities for semantic search and RAG.
    """
    
    def __init__(self, database_url: Optional[str] = None, database_name: Optional[str] = None):
        """Initialize MongoDB adapter
        
        Args:
            database_url: MongoDB connection URL (defaults to settings)
            database_name: Database name (defaults to settings)
        """
        self.settings = get_settings()
        self.database_url = database_url or self.settings.mongodb_url
        self.database_name = database_name or self.settings.mongodb_database
        
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize MongoDB connection and setup collections
        
        Raises:
            ConnectionError: If unable to connect to MongoDB
        """
        try:
            # Create MongoDB client
            self.client = AsyncIOMotorClient(
                self.database_url,
                maxPoolSize=self.settings.mongodb_max_connections,
                minPoolSize=self.settings.mongodb_min_connections,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000,
            )
            
            # Test connection
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB at {self.database_url}")
            
            # Get database
            self.database = self.client[self.database_name]
            
            # Setup collections and indexes
            await self._setup_collections()
            
            self._initialized = True
            
        except ServerSelectionTimeoutError:
            raise ConnectionError("Unable to connect to MongoDB server")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise ConnectionError(f"MongoDB initialization failed: {e}")
    
    async def cleanup(self) -> None:
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
        self._initialized = False
    
    async def _setup_collections(self) -> None:
        """Setup collections and indexes"""
        if not self.database:
            raise RuntimeError("Database not initialized")
        
        # Setup each collection
        await self._setup_repositories_collection()
        await self._setup_code_documents_collection()
        await self._setup_wiki_structures_collection()
        await self._setup_chat_sessions_collection()
        await self._setup_questions_collection()
        await self._setup_answers_collection()
        
        logger.info("MongoDB collections and indexes setup completed")
    
    async def _setup_repositories_collection(self) -> None:
        """Setup repositories collection with indexes"""
        collection = self.database["repositories"]
        
        indexes = [
            IndexModel([("url", ASCENDING)], unique=True),
            IndexModel([("provider", ASCENDING), ("org", ASCENDING), ("name", ASCENDING)]),
            IndexModel([("analysis_status", ASCENDING)]),
            IndexModel([("last_analyzed", DESCENDING)]),
            IndexModel([("webhook_configured", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ]
        
        await collection.create_indexes(indexes)
    
    async def _setup_code_documents_collection(self) -> None:
        """Setup code_documents collection with indexes and vector search"""
        collection = self.database["code_documents"]
        
        # Regular indexes
        indexes = [
            IndexModel([("repository_id", ASCENDING), ("file_path", ASCENDING)], unique=True),
            IndexModel([("repository_id", ASCENDING)]),
            IndexModel([("language", ASCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("updated_at", DESCENDING)]),
            # Text index for content search
            IndexModel([("processed_content", TEXT)]),
        ]
        
        await collection.create_indexes(indexes)
        
        # Setup vector search index (MongoDB Atlas Vector Search)
        try:
            # Note: Vector search indexes need to be created via MongoDB Atlas UI or mongosh
            # This is a placeholder for the vector search index configuration
            vector_index_spec = {
                "name": "code_embeddings_index",
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "embedding",
                            "numDimensions": 384,  # Common embedding dimension
                            "similarity": "cosine"
                        },
                        {
                            "type": "filter",
                            "path": "repository_id"
                        },
                        {
                            "type": "filter", 
                            "path": "language"
                        }
                    ]
                }
            }
            
            # Vector indexes need to be created manually in Atlas or via mongosh
            logger.info("Vector search index configuration ready: code_embeddings_index")
            
        except Exception as e:
            logger.warning(f"Could not setup vector search index: {e}")
    
    async def _setup_wiki_structures_collection(self) -> None:
        """Setup wiki_structures collection with indexes"""
        collection = self.database["wiki_structures"]
        
        indexes = [
            IndexModel([("repository_id", ASCENDING)], unique=True),
            IndexModel([("title", TEXT)]),
            IndexModel([("description", TEXT)]),
        ]
        
        await collection.create_indexes(indexes)
    
    async def _setup_chat_sessions_collection(self) -> None:
        """Setup chat_sessions collection with indexes"""
        collection = self.database["chat_sessions"]
        
        indexes = [
            IndexModel([("repository_id", ASCENDING)]),
            IndexModel([("status", ASCENDING)]),
            IndexModel([("last_activity", DESCENDING)]),
            IndexModel([("created_at", DESCENDING)]),
        ]
        
        await collection.create_indexes(indexes)
    
    async def _setup_questions_collection(self) -> None:
        """Setup questions collection with indexes"""
        collection = self.database["questions"]
        
        indexes = [
            IndexModel([("session_id", ASCENDING)]),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("content", TEXT)]),
        ]
        
        await collection.create_indexes(indexes)
    
    async def _setup_answers_collection(self) -> None:
        """Setup answers collection with indexes"""
        collection = self.database["answers"]
        
        indexes = [
            IndexModel([("question_id", ASCENDING)], unique=True),
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel([("confidence_score", DESCENDING)]),
        ]
        
        await collection.create_indexes(indexes)
    
    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection by name
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            MongoDB collection instance
            
        Raises:
            RuntimeError: If database not initialized
        """
        if not self.database:
            raise RuntimeError("Database not initialized")
        return self.database[collection_name]
    
    # Repository operations
    
    async def create_repository(self, repository: Repository) -> Repository:
        """Create a new repository
        
        Args:
            repository: Repository model instance
            
        Returns:
            Created repository with assigned ID
            
        Raises:
            DuplicateKeyError: If repository URL already exists
        """
        collection = self.get_collection("repositories")
        
        try:
            # Convert to dict and handle UUID/datetime serialization
            repo_dict = repository.model_dump()
            repo_dict["id"] = str(repository.id)
            
            result = await collection.insert_one(repo_dict)
            repository.id = UUID(repo_dict["id"])
            
            return repository
            
        except DuplicateKeyError:
            raise ValueError(f"Repository with URL {repository.url} already exists")
    
    async def get_repository(self, repository_id: UUID) -> Optional[Repository]:
        """Get repository by ID
        
        Args:
            repository_id: Repository UUID
            
        Returns:
            Repository instance or None if not found
        """
        collection = self.get_collection("repositories")
        
        doc = await collection.find_one({"id": str(repository_id)})
        if doc:
            # Convert back to Repository model
            doc["id"] = UUID(doc["id"])
            return Repository(**doc)
        
        return None
    
    async def get_repository_by_url(self, url: str) -> Optional[Repository]:
        """Get repository by URL
        
        Args:
            url: Repository URL
            
        Returns:
            Repository instance or None if not found
        """
        collection = self.get_collection("repositories")
        
        doc = await collection.find_one({"url": url})
        if doc:
            doc["id"] = UUID(doc["id"])
            return Repository(**doc)
        
        return None
    
    async def update_repository(self, repository_id: UUID, updates: Dict[str, Any]) -> bool:
        """Update repository
        
        Args:
            repository_id: Repository UUID
            updates: Fields to update
            
        Returns:
            True if repository was updated, False if not found
        """
        collection = self.get_collection("repositories")
        
        # Add updated_at timestamp
        updates["updated_at"] = datetime.now()
        
        result = await collection.update_one(
            {"id": str(repository_id)},
            {"$set": updates}
        )
        
        return result.modified_count > 0
    
    async def delete_repository(self, repository_id: UUID) -> bool:
        """Delete repository and all related data
        
        Args:
            repository_id: Repository UUID
            
        Returns:
            True if repository was deleted, False if not found
        """
        # Delete in transaction to ensure consistency
        async with await self.client.start_session() as session:
            async with session.start_transaction():
                # Delete repository
                repo_result = await self.get_collection("repositories").delete_one(
                    {"id": str(repository_id)}, session=session
                )
                
                if repo_result.deleted_count == 0:
                    return False
                
                # Delete related data
                await self.get_collection("code_documents").delete_many(
                    {"repository_id": str(repository_id)}, session=session
                )
                await self.get_collection("wiki_structures").delete_many(
                    {"repository_id": str(repository_id)}, session=session
                )
                await self.get_collection("chat_sessions").delete_many(
                    {"repository_id": str(repository_id)}, session=session
                )
                
                return True
    
    async def list_repositories(self, limit: int = 50, offset: int = 0, status_filter: Optional[str] = None) -> Tuple[List[Repository], int]:
        """List repositories with pagination
        
        Args:
            limit: Maximum number of repositories to return
            offset: Number of repositories to skip
            status_filter: Optional status filter
            
        Returns:
            Tuple of (repositories list, total count)
        """
        collection = self.get_collection("repositories")
        
        # Build query filter
        query_filter = {}
        if status_filter:
            query_filter["analysis_status"] = status_filter
        
        # Get total count
        total_count = await collection.count_documents(query_filter)
        
        # Get repositories with pagination
        cursor = collection.find(query_filter).sort("created_at", DESCENDING).skip(offset).limit(limit)
        
        repositories = []
        async for doc in cursor:
            doc["id"] = UUID(doc["id"])
            repositories.append(Repository(**doc))
        
        return repositories, total_count
    
    # Vector search operations
    
    async def store_document_embedding(self, document_id: str, embedding: List[float]) -> None:
        """Store document embedding for vector search
        
        Args:
            document_id: Document identifier
            embedding: Vector embedding
        """
        collection = self.get_collection("code_documents")
        
        await collection.update_one(
            {"id": document_id},
            {"$set": {"embedding": embedding, "updated_at": datetime.now()}}
        )
    
    async def vector_search(
        self, 
        query_embedding: List[float], 
        repository_id: Optional[UUID] = None,
        language_filter: Optional[str] = None,
        k: int = 10,
        score_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search
        
        Args:
            query_embedding: Query vector embedding
            repository_id: Optional repository filter
            language_filter: Optional language filter
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with documents and scores
        """
        collection = self.get_collection("code_documents")
        
        try:
            # Build aggregation pipeline for vector search
            pipeline = []
            
            # Vector search stage (MongoDB Atlas Vector Search)
            vector_search_stage = {
                "$vectorSearch": {
                    "index": "code_embeddings_index",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,  # Search more candidates for better results
                    "limit": k
                }
            }
            
            # Add filters if specified
            if repository_id or language_filter:
                filter_conditions = {}
                if repository_id:
                    filter_conditions["repository_id"] = str(repository_id)
                if language_filter:
                    filter_conditions["language"] = language_filter
                
                vector_search_stage["$vectorSearch"]["filter"] = filter_conditions
            
            pipeline.append(vector_search_stage)
            
            # Add score projection
            pipeline.append({
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            })
            
            # Filter by score threshold
            if score_threshold > 0:
                pipeline.append({
                    "$match": {
                        "similarity_score": {"$gte": score_threshold}
                    }
                })
            
            # Execute search
            results = []
            async for doc in collection.aggregate(pipeline):
                # Convert to CodeDocument model
                doc["id"] = doc["_id"] if "_id" in doc else doc["id"]
                doc["repository_id"] = UUID(doc["repository_id"])
                
                # Remove embedding from response (too large)
                doc.pop("embedding", None)
                
                code_doc = CodeDocument(**doc)
                
                results.append({
                    "document": code_doc,
                    "score": doc.get("similarity_score", 0.0)
                })
            
            return results
            
        except OperationFailure as e:
            # Fallback to text search if vector search not available
            logger.warning(f"Vector search failed, falling back to text search: {e}")
            return await self._fallback_text_search(query_embedding, repository_id, language_filter, k)
    
    async def _fallback_text_search(
        self,
        query_embedding: List[float],
        repository_id: Optional[UUID] = None,
        language_filter: Optional[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Fallback text search when vector search is not available
        
        This is a simplified fallback that uses text search instead of vector similarity.
        """
        collection = self.get_collection("code_documents")
        
        # Build query
        query = {}
        if repository_id:
            query["repository_id"] = str(repository_id)
        if language_filter:
            query["language"] = language_filter
        
        # Use text search on processed_content
        if query_embedding:
            # Simple keyword extraction from embedding (placeholder)
            # In a real implementation, you'd reverse-engineer keywords from embedding
            query["$text"] = {"$search": "function class method"}
        
        # Execute search
        cursor = collection.find(query).limit(k)
        
        results = []
        async for doc in cursor:
            doc["repository_id"] = UUID(doc["repository_id"])
            doc.pop("embedding", None)
            
            code_doc = CodeDocument(**doc)
            
            results.append({
                "document": code_doc,
                "score": 0.5  # Default score for text search
            })
        
        return results
    
    async def hybrid_search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        repository_id: Optional[UUID] = None,
        language_filter: Optional[str] = None,
        k: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining text and vector search
        
        Args:
            query_text: Text query for keyword search
            query_embedding: Optional vector embedding for semantic search
            repository_id: Optional repository filter
            language_filter: Optional language filter
            k: Number of results to return
            
        Returns:
            List of search results with documents and scores
        """
        collection = self.get_collection("code_documents")
        
        # Build base query
        base_query = {}
        if repository_id:
            base_query["repository_id"] = str(repository_id)
        if language_filter:
            base_query["language"] = language_filter
        
        # Text search results
        text_results = []
        if query_text:
            text_query = {**base_query, "$text": {"$search": query_text}}
            cursor = collection.find(
                text_query,
                {"score": {"$meta": "textScore"}}
            ).sort([("score", {"$meta": "textScore"})]).limit(k)
            
            async for doc in cursor:
                doc["repository_id"] = UUID(doc["repository_id"])
                doc.pop("embedding", None)
                
                code_doc = CodeDocument(**doc)
                text_results.append({
                    "document": code_doc,
                    "score": doc.get("score", 0.0),
                    "source": "text"
                })
        
        # Vector search results
        vector_results = []
        if query_embedding:
            vector_results = await self.vector_search(
                query_embedding, repository_id, language_filter, k
            )
            for result in vector_results:
                result["source"] = "vector"
        
        # Combine and rank results
        all_results = text_results + vector_results
        
        # Remove duplicates and combine scores
        seen_docs = {}
        for result in all_results:
            doc_id = result["document"].id
            if doc_id in seen_docs:
                # Combine scores (weighted average)
                existing = seen_docs[doc_id]
                combined_score = (existing["score"] + result["score"]) / 2
                existing["score"] = combined_score
                existing["source"] = "hybrid"
            else:
                seen_docs[doc_id] = result
        
        # Sort by combined score and return top k
        final_results = list(seen_docs.values())
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        return final_results[:k]
    
    # Generic CRUD operations
    
    async def insert_document(self, collection_name: str, document: Dict[str, Any]) -> str:
        """Insert document into collection
        
        Args:
            collection_name: Collection name
            document: Document to insert
            
        Returns:
            Inserted document ID
        """
        collection = self.get_collection(collection_name)
        
        # Handle UUID and datetime serialization
        serialized_doc = self._serialize_document(document)
        
        result = await collection.insert_one(serialized_doc)
        return str(result.inserted_id)
    
    async def find_document(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find single document
        
        Args:
            collection_name: Collection name
            query: Query filter
            
        Returns:
            Document dict or None if not found
        """
        collection = self.get_collection(collection_name)
        
        doc = await collection.find_one(self._serialize_query(query))
        if doc:
            return self._deserialize_document(doc)
        
        return None
    
    async def find_documents(
        self, 
        collection_name: str, 
        query: Dict[str, Any], 
        limit: int = 100, 
        offset: int = 0,
        sort_field: Optional[str] = None,
        sort_direction: int = DESCENDING
    ) -> List[Dict[str, Any]]:
        """Find multiple documents
        
        Args:
            collection_name: Collection name
            query: Query filter
            limit: Maximum number of documents
            offset: Number of documents to skip
            sort_field: Field to sort by
            sort_direction: Sort direction (ASCENDING or DESCENDING)
            
        Returns:
            List of document dicts
        """
        collection = self.get_collection(collection_name)
        
        cursor = collection.find(self._serialize_query(query))
        
        if sort_field:
            cursor = cursor.sort(sort_field, sort_direction)
        
        cursor = cursor.skip(offset).limit(limit)
        
        documents = []
        async for doc in cursor:
            documents.append(self._deserialize_document(doc))
        
        return documents
    
    async def update_document(self, collection_name: str, query: Dict[str, Any], updates: Dict[str, Any]) -> bool:
        """Update document
        
        Args:
            collection_name: Collection name
            query: Query filter to find document
            updates: Fields to update
            
        Returns:
            True if document was updated, False if not found
        """
        collection = self.get_collection(collection_name)
        
        # Add updated_at timestamp
        updates["updated_at"] = datetime.now()
        
        result = await collection.update_one(
            self._serialize_query(query),
            {"$set": self._serialize_document(updates)}
        )
        
        return result.modified_count > 0
    
    async def delete_document(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """Delete document
        
        Args:
            collection_name: Collection name
            query: Query filter to find document
            
        Returns:
            True if document was deleted, False if not found
        """
        collection = self.get_collection(collection_name)
        
        result = await collection.delete_one(self._serialize_query(query))
        return result.deleted_count > 0
    
    async def count_documents(self, collection_name: str, query: Dict[str, Any]) -> int:
        """Count documents matching query
        
        Args:
            collection_name: Collection name
            query: Query filter
            
        Returns:
            Number of matching documents
        """
        collection = self.get_collection(collection_name)
        return await collection.count_documents(self._serialize_query(query))
    
    def _serialize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize document for MongoDB storage"""
        serialized = {}
        
        for key, value in doc.items():
            if isinstance(value, UUID):
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value
            elif isinstance(value, (list, dict)):
                serialized[key] = value
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize query for MongoDB"""
        return self._serialize_document(query)
    
    def _deserialize_document(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize document from MongoDB"""
        if "_id" in doc:
            doc.pop("_id")  # Remove MongoDB ObjectId
        
        return doc
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on MongoDB connection
        
        Returns:
            Dictionary with health check results
        """
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Client not initialized"
                }
            
            # Ping database
            await self.client.admin.command('ping')
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            return {
                "status": "healthy",
                "database": self.database_name,
                "collections": stats.get("collections", 0),
                "data_size": stats.get("dataSize", 0),
                "index_size": stats.get("indexSize", 0),
                "initialized": self._initialized
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "database": self.database_name,
                "error": str(e),
                "initialized": self._initialized
            }


# Global MongoDB adapter instance
mongodb_adapter = MongoDBAdapter()


async def get_mongodb_adapter() -> MongoDBAdapter:
    """Get initialized MongoDB adapter instance"""
    if not mongodb_adapter._initialized:
        await mongodb_adapter.initialize()
    return mongodb_adapter


async def init_mongodb() -> None:
    """Initialize MongoDB adapter"""
    await mongodb_adapter.initialize()


async def close_mongodb() -> None:
    """Close MongoDB adapter"""
    await mongodb_adapter.cleanup()
