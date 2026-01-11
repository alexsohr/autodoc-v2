"""Document processing service for content analysis and embedding generation

This module provides document processing services including content analysis,
file processing, embedding generation, and semantic search capabilities.
"""

import asyncio
import structlog
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..agents.document_agent import DocumentProcessingAgent
from ..models.code_document import (
    CodeDocument,
    CodeDocumentCreate,
    CodeDocumentResponse,
)
from ..repository.code_document_repository import CodeDocumentRepository
from ..tools.context_tool import ContextTool
from ..tools.embedding_tool import EmbeddingTool
from ..utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


class DocumentProcessingService:
    """Document processing service for repository content analysis

    Provides comprehensive document processing including file analysis,
    content cleaning, embedding generation, and semantic search.
    """

    def __init__(
        self,
        code_document_repo: CodeDocumentRepository,
        document_agent: DocumentProcessingAgent,
        context_tool: ContextTool,
        embedding_tool: EmbeddingTool,
    ):
        """Initialize document processing service with dependency injection.
        
        Args:
            code_document_repo: CodeDocumentRepository instance (injected via DI).
            document_agent: DocumentProcessingAgent instance (injected via DI).
            context_tool: ContextTool instance (injected via DI).
            embedding_tool: EmbeddingTool instance (injected via DI).
        """
        self.settings = get_settings()
        self.max_file_size = self.settings.max_file_size_mb * 1024 * 1024
        self.supported_languages = set(self.settings.supported_languages)
        self.batch_size = self.settings.embedding_batch_size
        self._code_document_repo = code_document_repo
        self._document_agent = document_agent
        self._context_tool = context_tool
        self._embedding_tool = embedding_tool

    async def process_repository_documents(
        self,
        repository_id: UUID,
        repository_url: str,
        branch: Optional[str] = None,
        force_reprocess: bool = False,
    ) -> Dict[str, Any]:
        """Process all documents in a repository

        Args:
            repository_id: Repository UUID
            repository_url: Repository URL
            branch: Specific branch to process
            force_reprocess: Force reprocessing even if already done

        Returns:
            Dictionary with processing results
        """
        try:
            # Check if already processed
            if not force_reprocess:
                existing_docs = await self._code_document_repo.count(
                    {"repository_id": str(repository_id)}
                )

                if existing_docs > 0:
                    return {
                        "status": "success",
                        "message": "Repository already processed",
                        "documents_found": existing_docs,
                        "reprocessed": False,
                    }

            # Process repository using document agent
            processing_result = await self._document_agent.process_repository(
                repository_id=str(repository_id),
                repository_url=repository_url,
                branch=branch,
            )

            return {
                "status": processing_result["status"],
                "repository_id": str(repository_id),
                "documentation_files": processing_result.get("documentation_files", []),
                "file_tree": processing_result.get("file_tree", ""),
                "clone_path": processing_result.get("clone_path"),
                "error_message": processing_result.get("error_message"),
                "reprocessed": force_reprocess,
            }

        except Exception as e:
            logger.error(f"Repository document processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def get_repository_documents(
        self,
        repository_id: UUID,
        language_filter: Optional[str] = None,
        path_pattern: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get processed documents for repository

        Args:
            repository_id: Repository UUID
            language_filter: Optional language filter
            path_pattern: Optional file path pattern filter
            limit: Maximum number of documents to return
            offset: Number of documents to skip

        Returns:
            Dictionary with document list and metadata
        """
        try:
            # Build query filter
            query_filter = {"repository_id": str(repository_id)}
            if language_filter:
                query_filter["language"] = language_filter

            # Get documents
            documents_data = await self._code_document_repo.find_many(
                query_filter,
                limit=limit,
                offset=offset,
                sort=[("file_path", 1)],
            )
            documents_data = self._code_document_repo.serialize_many(documents_data)

            # Apply path pattern filter if specified
            if path_pattern:
                import fnmatch

                documents_data = [
                    doc
                    for doc in documents_data
                    if fnmatch.fnmatch(doc["file_path"], path_pattern)
                ]

            # Convert to response format
            documents = []
            for doc_data in documents_data:
                doc_data["repository_id"] = UUID(doc_data["repository_id"])
                code_doc = CodeDocument(**doc_data)
                documents.append(CodeDocumentResponse.from_code_document(code_doc))

            # Get language statistics
            language_stats = await self._get_language_statistics(
                repository_id, query_filter
            )

            # Get total count
            total_count = await self._code_document_repo.count(query_filter)

            return {
                "status": "success",
                "files": [doc.model_dump() for doc in documents],
                "repository_id": str(repository_id),
                "total": total_count,
                "languages": language_stats,
                "filters": {"language": language_filter, "path_pattern": path_pattern},
                "pagination": {"limit": limit, "offset": offset},
            }

        except Exception as e:
            logger.error(f"Get repository documents failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "files": [],
            }

    async def search_documents(
        self,
        query: str,
        repository_id: Optional[UUID] = None,
        language_filter: Optional[str] = None,
        search_type: str = "hybrid",
        k: int = 10,
    ) -> Dict[str, Any]:
        """Search documents using semantic search

        Args:
            query: Search query
            repository_id: Optional repository filter
            language_filter: Optional language filter
            search_type: Search type (vector, text, hybrid)
            k: Number of results to return

        Returns:
            Dictionary with search results
        """
        try:
            # Perform search using context tool
            search_result = await self._context_tool._arun(
                "search",
                query=query,
                repository_id=str(repository_id) if repository_id else None,
                language_filter=language_filter,
                search_type=search_type,
                k=k,
            )

            if search_result["status"] != "success":
                return search_result

            # Format results
            formatted_results = []
            for result in search_result["results"]:
                formatted_results.append(
                    {
                        "document_id": result["document_id"],
                        "file_path": result["file_path"],
                        "language": result["language"],
                        "similarity_score": result["similarity_score"],
                        "content_preview": result.get("content_preview", ""),
                        "metadata": result.get("metadata", {}),
                    }
                )

            return {
                "status": "success",
                "results": formatted_results,
                "count": len(formatted_results),
                "query": query,
                "search_type": search_type,
                "search_time": search_result.get("search_time"),
            }

        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "results": [],
            }

    async def get_document_content(self, document_id: str) -> Dict[str, Any]:
        """Get full content of a specific document

        Args:
            document_id: Document ID

        Returns:
            Dictionary with document content
        """
        try:
            doc = await self._code_document_repo.find_one({"id": document_id})
            doc_data = self._code_document_repo.serialize(doc) if doc else None

            if not doc_data:
                return {
                    "status": "error",
                    "error": "Document not found",
                    "error_type": "NotFound",
                }

            # Convert to CodeDocument
            doc_data["repository_id"] = UUID(doc_data["repository_id"])
            code_document = CodeDocument(**doc_data)

            return {
                "status": "success",
                "document": {
                    "id": code_document.id,
                    "file_path": code_document.file_path,
                    "language": code_document.language,
                    "content": code_document.content,
                    "processed_content": code_document.processed_content,
                    "metadata": code_document.metadata,
                    "has_embedding": code_document.has_embedding(),
                    "created_at": code_document.created_at.isoformat(),
                    "updated_at": code_document.updated_at.isoformat(),
                },
            }

        except Exception as e:
            logger.error(f"Get document content failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def update_document_content(
        self, document_id: str, new_content: str, regenerate_embedding: bool = True
    ) -> Dict[str, Any]:
        """Update document content and optionally regenerate embedding

        Args:
            document_id: Document ID
            new_content: New content
            regenerate_embedding: Whether to regenerate embedding

        Returns:
            Dictionary with update result
        """
        try:
            # Get existing document
            doc = await self._code_document_repo.find_one({"id": document_id})
            doc_data = self._code_document_repo.serialize(doc) if doc else None
            if not doc_data:
                return {
                    "status": "error",
                    "error": "Document not found",
                    "error_type": "NotFound",
                }

            # Process new content
            processed_content = self._clean_content_for_embedding(
                new_content, doc_data["language"]
            )

            # Update document
            updates = {
                "content": new_content,
                "processed_content": processed_content,
                "updated_at": datetime.now(timezone.utc),
            }

            # Clear existing embedding if regenerating
            if regenerate_embedding:
                updates["embedding"] = None

            success = await self._code_document_repo.update_one(
                {"id": document_id}, updates
            )

            if not success:
                return {
                    "status": "error",
                    "error": "Failed to update document",
                    "error_type": "UpdateFailed",
                }

            # Regenerate embedding if requested
            embedding_result = None
            if regenerate_embedding:
                embedding_result = await self._embedding_tool._arun(
                    "generate_and_store",
                    documents=[
                        {"id": document_id, "processed_content": processed_content}
                    ],
                )

            return {
                "status": "success",
                "document_id": document_id,
                "content_updated": True,
                "embedding_regenerated": regenerate_embedding,
                "embedding_result": embedding_result,
                "message": "Document updated successfully",
            }

        except Exception as e:
            logger.error(f"Document update failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "document_id": document_id,
            }

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document

        Args:
            document_id: Document ID

        Returns:
            Dictionary with deletion result
        """
        try:
            # Check if document exists
            doc = await self._code_document_repo.find_one({"id": document_id})
            doc_data = self._code_document_repo.serialize(doc) if doc else None
            if not doc_data:
                return {
                    "status": "error",
                    "error": "Document not found",
                    "error_type": "NotFound",
                }

            # Delete document
            success = await self._code_document_repo.delete_one({"id": document_id})

            if success:
                return {
                    "status": "success",
                    "message": f"Document {doc_data['file_path']} deleted successfully",
                }
            else:
                return {
                    "status": "error",
                    "error": "Failed to delete document",
                    "error_type": "DeletionFailed",
                }

        except Exception as e:
            logger.error(f"Document deletion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "document_id": document_id,
            }

    async def regenerate_embeddings(
        self,
        repository_id: UUID,
        language_filter: Optional[str] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Regenerate embeddings for repository documents

        Args:
            repository_id: Repository UUID
            language_filter: Optional language filter
            force: Force regeneration even if embeddings exist

        Returns:
            Dictionary with regeneration results
        """
        try:
            # Use embedding tool to reprocess embeddings
            reprocess_result = await self._embedding_tool._arun(
                "reprocess_embeddings", repository_id=str(repository_id), force=force
            )

            return {
                "status": reprocess_result["status"],
                "repository_id": str(repository_id),
                "processed_count": reprocess_result.get("processed_count", 0),
                "failed_count": reprocess_result.get("failed_count", 0),
                "skipped_count": reprocess_result.get("skipped_count", 0),
                "force_reprocess": force,
                "error_message": reprocess_result.get("error"),
            }

        except Exception as e:
            logger.error(f"Embedding regeneration failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def get_embedding_statistics(
        self, repository_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get embedding statistics

        Args:
            repository_id: Optional repository filter

        Returns:
            Dictionary with embedding statistics
        """
        try:
            # Use embedding tool to get statistics
            stats_result = await self._embedding_tool._arun(
                "get_embedding_stats",
                repository_id=str(repository_id) if repository_id else None,
            )

            return stats_result

        except Exception as e:
            logger.error(f"Get embedding statistics failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}

    async def analyze_document_quality(self, repository_id: UUID) -> Dict[str, Any]:
        """Analyze document quality metrics for repository

        Args:
            repository_id: Repository UUID

        Returns:
            Dictionary with quality analysis
        """
        try:
            # Get all documents for repository
            documents = await self._code_document_repo.find_many(
                {"repository_id": str(repository_id)}, limit=1000
            )
            documents_data = self._code_document_repo.serialize_many(documents)

            if not documents_data:
                return {
                    "status": "error",
                    "error": "No documents found for repository",
                    "error_type": "NoDocuments",
                }

            # Analyze quality metrics
            quality_metrics = {
                "total_documents": len(documents_data),
                "documents_with_embeddings": 0,
                "average_file_size": 0,
                "language_distribution": {},
                "content_quality_scores": [],
                "embedding_coverage": 0.0,
            }

            total_size = 0
            language_counts = {}

            for doc_data in documents_data:
                # Count embeddings
                if doc_data.get("embedding"):
                    quality_metrics["documents_with_embeddings"] += 1

                # Calculate size statistics
                content_size = len(doc_data.get("content", ""))
                total_size += content_size

                # Language distribution
                language = doc_data.get("language", "unknown")
                language_counts[language] = language_counts.get(language, 0) + 1

                # Content quality score (simplified)
                quality_score = self._calculate_content_quality_score(
                    doc_data.get("content", ""),
                    doc_data.get("processed_content", ""),
                    language,
                )
                quality_metrics["content_quality_scores"].append(quality_score)

            # Calculate final metrics
            quality_metrics["average_file_size"] = (
                total_size // len(documents_data) if documents_data else 0
            )
            quality_metrics["language_distribution"] = language_counts
            quality_metrics["embedding_coverage"] = (
                quality_metrics["documents_with_embeddings"]
                / quality_metrics["total_documents"]
                * 100
                if quality_metrics["total_documents"] > 0
                else 0
            )

            # Calculate average quality score
            avg_quality = (
                sum(quality_metrics["content_quality_scores"])
                / len(quality_metrics["content_quality_scores"])
                if quality_metrics["content_quality_scores"]
                else 0
            )
            quality_metrics["average_quality_score"] = avg_quality

            return {
                "status": "success",
                "repository_id": str(repository_id),
                "quality_metrics": quality_metrics,
                "analysis_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Document quality analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def process_file_changes(
        self, repository_id: UUID, changed_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process file changes for incremental updates

        Args:
            repository_id: Repository UUID
            changed_files: List of changed file information

        Returns:
            Dictionary with processing results
        """
        try:
            processed_files = 0
            added_files = 0
            modified_files = 0
            removed_files = 0

            for file_change in changed_files:
                file_path = file_change.get("path", "")
                change_type = file_change.get("status", "unknown")

                try:
                    if change_type == "added":
                        # Process new file
                        await self._process_new_file(repository_id, file_change)
                        added_files += 1
                        processed_files += 1

                    elif change_type == "modified":
                        # Update existing file
                        await self._process_modified_file(repository_id, file_change)
                        modified_files += 1
                        processed_files += 1

                    elif change_type == "removed":
                        # Remove file
                        await self._process_removed_file(repository_id, file_path)
                        removed_files += 1
                        processed_files += 1

                except Exception as e:
                    logger.warning(f"Failed to process file change {file_path}: {e}")
                    continue

            return {
                "status": "success",
                "repository_id": str(repository_id),
                "processed_files": processed_files,
                "added_files": added_files,
                "modified_files": modified_files,
                "removed_files": removed_files,
                "total_changes": len(changed_files),
            }

        except Exception as e:
            logger.error(f"File changes processing failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "repository_id": str(repository_id),
            }

    async def _get_language_statistics(
        self, repository_id: UUID, base_query: Dict[str, Any]
    ) -> Dict[str, int]:
        """Get language statistics for repository

        Args:
            repository_id: Repository UUID
            base_query: Base query filter

        Returns:
            Dictionary mapping language to count
        """
        try:
            return await self._code_document_repo.get_language_statistics(base_query)
        except Exception as e:
            logger.error(f"Language statistics failed: {e}")
            return {}

    def _calculate_content_quality_score(
        self, content: str, processed_content: str, language: str
    ) -> float:
        """Calculate content quality score

        Args:
            content: Original content
            processed_content: Processed content
            language: Programming language

        Returns:
            Quality score (0.0 to 1.0)
        """
        try:
            if not content:
                return 0.0

            score = 0.5  # Base score

            # Length score (reasonable file size)
            content_length = len(content)
            if 100 <= content_length <= 10000:  # Reasonable size
                score += 0.2
            elif content_length > 50:  # At least some content
                score += 0.1

            # Comment score (for code files)
            if language in ["python", "javascript", "typescript", "java", "go"]:
                comment_ratio = self._calculate_comment_ratio(content, language)
                score += comment_ratio * 0.2

            # Structure score (for code files)
            if language in self.supported_languages:
                structure_score = self._calculate_structure_score(content, language)
                score += structure_score * 0.1

            return min(1.0, max(0.0, score))

        except Exception as e:
            logger.debug(f"Quality score calculation failed: {e}")
            return 0.5  # Default score

    def _calculate_comment_ratio(self, content: str, language: str) -> float:
        """Calculate ratio of comments to code

        Args:
            content: File content
            language: Programming language

        Returns:
            Comment ratio (0.0 to 1.0)
        """
        try:
            lines = content.split("\n")
            total_lines = len([line for line in lines if line.strip()])

            if total_lines == 0:
                return 0.0

            comment_lines = 0

            if language == "python":
                comment_lines = len(
                    [line for line in lines if line.strip().startswith("#")]
                )
            elif language in ["javascript", "typescript", "java", "go"]:
                comment_lines = len(
                    [line for line in lines if line.strip().startswith("//")]
                )

            return min(1.0, comment_lines / total_lines)

        except Exception:
            return 0.0

    def _calculate_structure_score(self, content: str, language: str) -> float:
        """Calculate code structure score

        Args:
            content: File content
            language: Programming language

        Returns:
            Structure score (0.0 to 1.0)
        """
        try:
            score = 0.0

            if language == "python":
                # Look for functions and classes
                if re.search(r"^\s*def\s+\w+", content, re.MULTILINE):
                    score += 0.3
                if re.search(r"^\s*class\s+\w+", content, re.MULTILINE):
                    score += 0.3
                if re.search(r"^\s*import\s+", content, re.MULTILINE):
                    score += 0.2
                if '"""' in content or "'''" in content:  # Docstrings
                    score += 0.2

            elif language in ["javascript", "typescript"]:
                # Look for functions and classes
                if re.search(r"function\s+\w+|const\s+\w+\s*=.*=>", content):
                    score += 0.3
                if re.search(r"class\s+\w+", content):
                    score += 0.3
                if re.search(r"import\s+.*from|require\(", content):
                    score += 0.2
                if "/**" in content:  # JSDoc comments
                    score += 0.2

            return min(1.0, score)

        except Exception:
            return 0.5

    def _clean_content_for_embedding(self, content: str, language: str) -> str:
        """Clean content for embedding generation

        Args:
            content: Raw content
            language: Programming language

        Returns:
            Cleaned content
        """
        try:
            # Remove excessive whitespace
            cleaned = re.sub(r"\s+", " ", content.strip())

            # Language-specific cleaning
            if language == "python":
                # Preserve docstrings but clean quotes
                cleaned = re.sub(r'"""', " ", cleaned)
                cleaned = re.sub(r"'''", " ", cleaned)

            elif language in ["javascript", "typescript"]:
                # Remove single-line comments
                cleaned = re.sub(r"//.*$", "", cleaned, flags=re.MULTILINE)
                # Remove multi-line comments
                cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

            # Remove empty lines and normalize
            lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
            cleaned = "\n".join(lines)

            # Truncate if too long
            max_length = 8000
            if len(cleaned) > max_length:
                cleaned = cleaned[:max_length] + "..."

            return cleaned

        except Exception as e:
            logger.warning(f"Content cleaning failed: {e}")
            return content[:8000]  # Fallback

    async def _process_new_file(
        self, repository_id: UUID, file_info: Dict[str, Any]
    ) -> None:
        """Process a new file

        Args:
            repository_id: Repository UUID
            file_info: File information
        """
        # Implementation would process new file and create document
        # For now, this is a placeholder
        pass

    async def _process_modified_file(
        self, repository_id: UUID, file_info: Dict[str, Any]
    ) -> None:
        """Process a modified file

        Args:
            repository_id: Repository UUID
            file_info: File information
        """
        # Implementation would update existing document
        # For now, this is a placeholder
        pass

    async def _process_removed_file(self, repository_id: UUID, file_path: str) -> None:
        """Process a removed file

        Args:
            repository_id: Repository UUID
            file_path: Path of removed file
        """
        try:
            await self._code_document_repo.delete_one(
                {"repository_id": str(repository_id), "file_path": file_path}
            )
        except Exception as e:
            logger.warning(f"Failed to remove file {file_path}: {e}")


# Global document processing service instance
# Deprecated: Module-level singleton removed
# Use get_document_service() from src.dependencies with FastAPI's Depends() instead
# from ..dependencies import get_document_service
# document_service = get_document_service()  # REMOVED - use dependency injection
