"""Embedding tool for LangGraph workflows

This module implements the embedding tool for generating and storing
vector embeddings for semantic search capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
import numpy as np

from langchain_core.tools import BaseTool
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from pydantic import BaseModel, Field

from ..models.code_document import CodeDocument
from ..models.config import LLMProvider
from ..utils.config_loader import get_settings
from ..utils.mongodb_adapter import get_mongodb_adapter

logger = logging.getLogger(__name__)


class EmbeddingGenerateInput(BaseModel):
    """Input schema for embedding generation"""
    texts: List[str] = Field(description="List of texts to generate embeddings for")
    provider: Optional[str] = Field(default=None, description="Embedding provider (defaults to configured)")
    batch_size: Optional[int] = Field(default=None, description="Batch size for processing")


class EmbeddingStoreInput(BaseModel):
    """Input schema for embedding storage"""
    document_id: str = Field(description="Document ID")
    embedding: List[float] = Field(description="Embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class EmbeddingSearchInput(BaseModel):
    """Input schema for embedding search"""
    query_embedding: List[float] = Field(description="Query embedding vector")
    repository_id: Optional[str] = Field(default=None, description="Repository filter")
    language_filter: Optional[str] = Field(default=None, description="Language filter")
    k: Optional[int] = Field(default=10, description="Number of results")
    score_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score")


class EmbeddingTool(BaseTool):
    """LangGraph tool for embedding operations
    
    Provides embedding generation, storage, and retrieval capabilities
    for semantic search and RAG functionality.
    """
    
    name: str = "embedding_tool"
    description: str = "Tool for generating, storing, and searching vector embeddings"
    
    def __init__(self):
        super().__init__()
        # Initialize settings and configuration
        settings = get_settings()
        self._batch_size = settings.embedding_batch_size
        
        # Initialize embedding providers
        self._embedding_providers = {}
        self._setup_embedding_providers(settings)
    
    def _setup_embedding_providers(self, settings) -> None:
        """Setup embedding providers based on configuration"""
        try:
            # OpenAI embeddings
            if settings.openai_api_key:
                self._embedding_providers[LLMProvider.OPENAI] = OpenAIEmbeddings(
                    api_key=settings.openai_api_key,
                    model="text-embedding-3-small",  # More cost-effective
                    dimensions=384  # Smaller dimension for faster search
                )
            
            # Google Gemini embeddings
            if settings.google_api_key:
                self._embedding_providers[LLMProvider.GEMINI] = GoogleGenerativeAIEmbeddings(
                    google_api_key=settings.google_api_key,
                    model="models/embedding-001"
                )
            
            # Ollama embeddings (for local development)
            if settings.ollama_base_url:
                self._embedding_providers[LLMProvider.OLLAMA] = OllamaEmbeddings(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model
                )
            
            logger.info(f"Initialized {len(self._embedding_providers)} embedding providers")
            
        except Exception as e:
            logger.error(f"Failed to setup embedding providers: {e}")
    
    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Async run method for LangGraph"""
        if operation == "generate":
            return await self.generate_embeddings(**kwargs)
        elif operation == "store":
            return await self.store_embedding(**kwargs)
        elif operation == "search":
            return await self.search_embeddings(**kwargs)
        elif operation == "generate_and_store":
            return await self.generate_and_store_embeddings(**kwargs)
        else:
            raise ValueError(f"Unknown embedding operation: {operation}")
    
    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Sync run method (not used in async workflows)"""
        raise NotImplementedError("Embedding tool only supports async operations")
    
    async def generate_embeddings(
        self,
        texts: List[str],
        provider: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to generate embeddings for
            provider: Embedding provider (defaults to first available)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with embeddings and metadata
        """
        try:
            if not texts:
                return {
                    "status": "success",
                    "embeddings": [],
                    "count": 0
                }
            
            # Get embedding provider
            embedding_provider = self._get_embedding_provider(provider)
            if not embedding_provider:
                raise ValueError("No embedding provider available")
            
            batch_size = batch_size or self._batch_size
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                try:
                    # Generate embeddings for batch
                    batch_embeddings = await embedding_provider.aembed_documents(batch)
                    embeddings.extend(batch_embeddings)
                    
                    logger.debug(f"Generated embeddings for batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                    
                except Exception as e:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {e}")
                    # Add None for failed embeddings
                    embeddings.extend([None] * len(batch))
            
            # Filter out None embeddings
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            return {
                "status": "success",
                "embeddings": valid_embeddings,
                "count": len(valid_embeddings),
                "total_requested": len(texts),
                "failed_count": len(texts) - len(valid_embeddings),
                "provider": provider or "default",
                "dimension": len(valid_embeddings[0]) if valid_embeddings else 0,
                "generation_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "count": 0
            }
    
    async def store_embedding(
        self,
        document_id: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Store embedding in vector database
        
        Args:
            document_id: Document identifier
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            Dictionary with storage results
        """
        try:
            # Get MongoDB adapter
            mongodb = await get_mongodb_adapter()
            
            # Store embedding
            await mongodb.store_document_embedding(document_id, embedding)
            
            # Store additional metadata if provided
            if metadata:
                await mongodb.update_document(
                    "code_documents",
                    {"id": document_id},
                    {"embedding_metadata": metadata}
                )
            
            return {
                "status": "success",
                "document_id": document_id,
                "embedding_dimension": len(embedding),
                "storage_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding storage failed: {e}")
            return {
                "status": "error",
                "document_id": document_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def search_embeddings(
        self,
        query_embedding: List[float],
        repository_id: Optional[str] = None,
        language_filter: Optional[str] = None,
        k: int = 10,
        score_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            repository_id: Optional repository filter
            language_filter: Optional language filter
            k: Number of results to return
            score_threshold: Minimum similarity score
            
        Returns:
            Dictionary with search results
        """
        try:
            # Get MongoDB adapter
            mongodb = await get_mongodb_adapter()
            
            # Perform vector search
            from uuid import UUID
            repo_uuid = UUID(repository_id) if repository_id else None
            
            results = await mongodb.vector_search(
                query_embedding=query_embedding,
                repository_id=repo_uuid,
                language_filter=language_filter,
                k=k,
                score_threshold=score_threshold
            )
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "document_id": result["document"].id,
                    "file_path": result["document"].file_path,
                    "language": result["document"].language,
                    "similarity_score": result["score"],
                    "metadata": result["document"].metadata
                })
            
            return {
                "status": "success",
                "results": formatted_results,
                "count": len(formatted_results),
                "query_dimension": len(query_embedding),
                "search_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "count": 0
            }
    
    async def generate_and_store_embeddings(
        self,
        documents: List[Dict[str, Any]],
        provider: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate and store embeddings for multiple documents
        
        Args:
            documents: List of document dicts with id, content, and metadata
            provider: Embedding provider
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with processing results
        """
        try:
            if not documents:
                return {
                    "status": "success",
                    "processed_count": 0,
                    "failed_count": 0
                }
            
            # Extract texts for embedding generation
            texts = []
            document_ids = []
            
            for doc in documents:
                content = doc.get("processed_content") or doc.get("content", "")
                if content and content.strip():
                    texts.append(content.strip())
                    document_ids.append(doc["id"])
            
            if not texts:
                return {
                    "status": "warning",
                    "message": "No valid content found for embedding generation",
                    "processed_count": 0,
                    "failed_count": len(documents)
                }
            
            # Generate embeddings
            embedding_result = await self.generate_embeddings(
                texts=texts,
                provider=provider,
                batch_size=batch_size
            )
            
            if embedding_result["status"] != "success":
                return embedding_result
            
            embeddings = embedding_result["embeddings"]
            
            # Store embeddings
            stored_count = 0
            failed_count = 0
            
            for i, (doc_id, embedding) in enumerate(zip(document_ids, embeddings)):
                if embedding is not None:
                    store_result = await self.store_embedding(
                        document_id=doc_id,
                        embedding=embedding,
                        metadata={
                            "provider": provider or "default",
                            "dimension": len(embedding),
                            "generated_at": datetime.now(timezone.utc).isoformat()
                        }
                    )
                    
                    if store_result["status"] == "success":
                        stored_count += 1
                    else:
                        failed_count += 1
                        logger.error(f"Failed to store embedding for document {doc_id}: {store_result.get('error')}")
                else:
                    failed_count += 1
            
            return {
                "status": "success",
                "processed_count": stored_count,
                "failed_count": failed_count,
                "total_documents": len(documents),
                "embedding_dimension": embedding_result.get("dimension", 0),
                "provider": provider or "default",
                "processing_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Generate and store embeddings failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "processed_count": 0,
                "failed_count": len(documents) if documents else 0
            }
    
    def _get_embedding_provider(self, provider: Optional[str] = None):
        """Get embedding provider instance
        
        Args:
            provider: Provider name (defaults to first available)
            
        Returns:
            Embedding provider instance or None
        """
        if provider:
            return self._embedding_providers.get(LLMProvider(provider))
        
        # Return first available provider
        if self._embedding_providers:
            return next(iter(self._embedding_providers.values()))
        
        return None
    
    async def get_embedding_for_query(self, query: str, provider: Optional[str] = None) -> Optional[List[float]]:
        """Generate embedding for a single query
        
        Args:
            query: Query text
            provider: Embedding provider
            
        Returns:
            Embedding vector or None if failed
        """
        try:
            embedding_provider = self._get_embedding_provider(provider)
            if not embedding_provider:
                return None
            
            # Generate embedding for query
            embedding = await embedding_provider.aembed_query(query)
            return embedding
            
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            return None
    
    async def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        try:
            # Convert to numpy arrays for efficient computation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure result is in [0, 1] range
            return max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
        except Exception as e:
            logger.error(f"Similarity calculation failed: {e}")
            return 0.0
    
    async def batch_process_documents(
        self,
        documents: List[CodeDocument],
        provider: Optional[str] = None,
        batch_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process multiple documents for embedding generation and storage
        
        Args:
            documents: List of CodeDocument instances
            provider: Embedding provider
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with batch processing results
        """
        try:
            if not documents:
                return {
                    "status": "success",
                    "processed_count": 0,
                    "failed_count": 0,
                    "skipped_count": 0
                }
            
            # Prepare documents for processing
            docs_to_process = []
            skipped_count = 0
            
            for doc in documents:
                # Skip documents that already have embeddings
                if doc.has_embedding():
                    skipped_count += 1
                    continue
                
                # Skip documents without content
                if not doc.processed_content or not doc.processed_content.strip():
                    skipped_count += 1
                    continue
                
                docs_to_process.append({
                    "id": doc.id,
                    "processed_content": doc.processed_content,
                    "file_path": doc.file_path,
                    "language": doc.language
                })
            
            if not docs_to_process:
                return {
                    "status": "success",
                    "processed_count": 0,
                    "failed_count": 0,
                    "skipped_count": skipped_count,
                    "message": "No documents need embedding generation"
                }
            
            # Generate and store embeddings
            result = await self.generate_and_store_embeddings(
                documents=docs_to_process,
                provider=provider,
                batch_size=batch_size
            )
            
            # Add skipped count to result
            result["skipped_count"] = skipped_count
            result["total_documents"] = len(documents)
            
            return result
            
        except Exception as e:
            logger.error(f"Batch document processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "processed_count": 0,
                "failed_count": len(documents) if documents else 0,
                "skipped_count": 0
            }
    
    async def reprocess_embeddings(
        self,
        repository_id: str,
        provider: Optional[str] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """Reprocess embeddings for all documents in a repository
        
        Args:
            repository_id: Repository ID
            provider: Embedding provider
            force: Force reprocessing even if embeddings exist
            
        Returns:
            Dictionary with reprocessing results
        """
        try:
            # Get MongoDB adapter
            mongodb = await get_mongodb_adapter()
            
            # Find all documents for repository
            query = {"repository_id": repository_id}
            if not force:
                # Only process documents without embeddings
                query["embedding"] = {"$exists": False}
            
            documents = await mongodb.find_documents("code_documents", query)
            
            if not documents:
                return {
                    "status": "success",
                    "message": "No documents need reprocessing",
                    "processed_count": 0,
                    "failed_count": 0
                }
            
            # Convert to CodeDocument instances
            code_documents = []
            for doc in documents:
                try:
                    from uuid import UUID
                    doc["repository_id"] = UUID(doc["repository_id"])
                    code_documents.append(CodeDocument(**doc))
                except Exception as e:
                    logger.warning(f"Could not convert document {doc.get('id')}: {e}")
            
            # Process documents
            result = await self.batch_process_documents(
                documents=code_documents,
                provider=provider
            )
            
            result["repository_id"] = repository_id
            result["force_reprocess"] = force
            
            return result
            
        except Exception as e:
            logger.error(f"Embedding reprocessing failed: {e}")
            return {
                "status": "error",
                "repository_id": repository_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def get_embedding_stats(self, repository_id: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding statistics
        
        Args:
            repository_id: Optional repository filter
            
        Returns:
            Dictionary with embedding statistics
        """
        try:
            mongodb = await get_mongodb_adapter()
            
            # Build query
            query = {}
            if repository_id:
                query["repository_id"] = repository_id
            
            # Count total documents
            total_docs = await mongodb.count_documents("code_documents", query)
            
            # Count documents with embeddings
            query_with_embedding = {**query, "embedding": {"$exists": True, "$ne": None}}
            docs_with_embeddings = await mongodb.count_documents("code_documents", query_with_embedding)
            
            # Get language breakdown
            pipeline = [
                {"$match": query_with_embedding},
                {"$group": {
                    "_id": "$language",
                    "count": {"$sum": 1},
                    "avg_embedding_size": {"$avg": {"$size": "$embedding"}}
                }}
            ]
            
            language_stats = {}
            collection = mongodb.get_collection("code_documents")
            async for doc in collection.aggregate(pipeline):
                language_stats[doc["_id"]] = {
                    "count": doc["count"],
                    "avg_dimension": int(doc.get("avg_embedding_size", 0))
                }
            
            return {
                "status": "success",
                "total_documents": total_docs,
                "documents_with_embeddings": docs_with_embeddings,
                "embedding_coverage": (docs_with_embeddings / total_docs * 100) if total_docs > 0 else 0,
                "language_stats": language_stats,
                "repository_id": repository_id,
                "stats_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Embedding stats retrieval failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available embedding providers
        
        Returns:
            List of available provider names
        """
        return [provider.value for provider in self._embedding_providers.keys()]
    
    def get_provider_info(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """Get information about embedding provider
        
        Args:
            provider: Provider name (defaults to first available)
            
        Returns:
            Dictionary with provider information
        """
        embedding_provider = self._get_embedding_provider(provider)
        if not embedding_provider:
            return {
                "status": "error",
                "error": "No embedding provider available"
            }
        
        provider_info = {
            "provider_class": embedding_provider.__class__.__name__,
            "available": True
        }
        
        # Add provider-specific info
        if hasattr(embedding_provider, 'model'):
            provider_info["model"] = embedding_provider.model
        
        if hasattr(embedding_provider, 'dimensions'):
            provider_info["dimensions"] = embedding_provider.dimensions
        
        return {
            "status": "success",
            "provider_info": provider_info
        }


# Tool instance for LangGraph
embedding_tool = EmbeddingTool()
