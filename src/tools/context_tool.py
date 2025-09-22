"""Context retrieval tool for LangGraph workflows

This module implements the context retrieval tool for semantic search
and context gathering for RAG (Retrieval-Augmented Generation) workflows.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timezone
import re

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ..models.code_document import CodeDocument
from ..utils.mongodb_adapter import get_mongodb_adapter
from ..tools.embedding_tool import embedding_tool

logger = logging.getLogger(__name__)


class ContextSearchInput(BaseModel):
    """Input schema for context search"""
    query: str = Field(description="Search query text")
    repository_id: Optional[str] = Field(default=None, description="Repository filter")
    language_filter: Optional[str] = Field(default=None, description="Programming language filter")
    file_path_filter: Optional[str] = Field(default=None, description="File path pattern filter")
    k: Optional[int] = Field(default=10, description="Number of results to return")
    score_threshold: Optional[float] = Field(default=0.7, description="Minimum similarity score")
    search_type: Optional[str] = Field(default="hybrid", description="Search type: vector, text, or hybrid")


class ContextRankInput(BaseModel):
    """Input schema for context ranking"""
    contexts: List[Dict[str, Any]] = Field(description="List of context candidates")
    query: str = Field(description="Original query")
    max_contexts: Optional[int] = Field(default=5, description="Maximum contexts to return")
    ranking_strategy: Optional[str] = Field(default="relevance", description="Ranking strategy")


class ContextTool(BaseTool):
    """LangGraph tool for context retrieval and semantic search
    
    Provides semantic search capabilities for retrieving relevant code context
    for RAG workflows and question answering.
    """
    
    name: str = "context_tool"
    description: str = "Tool for semantic search and context retrieval from code repositories"
    
    def __init__(self):
        super().__init__()
        # Initialize configuration
        self._max_context_length = 8000  # Maximum context length in characters
        self._relevance_weights = {
            "semantic_similarity": 0.4,
            "text_relevance": 0.3,
            "recency": 0.1,
            "file_importance": 0.1,
            "code_quality": 0.1
        }
    
    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Async run method for LangGraph"""
        if operation == "search":
            return await self.search_context(**kwargs)
        elif operation == "rank":
            return await self.rank_contexts(**kwargs)
        elif operation == "extract":
            return await self.extract_code_snippets(**kwargs)
        elif operation == "hybrid_search":
            return await self.hybrid_search(**kwargs)
        else:
            raise ValueError(f"Unknown context operation: {operation}")
    
    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Sync run method (not used in async workflows)"""
        raise NotImplementedError("Context tool only supports async operations")
    
    async def search_context(
        self,
        query: str,
        repository_id: Optional[str] = None,
        language_filter: Optional[str] = None,
        file_path_filter: Optional[str] = None,
        k: int = 10,
        score_threshold: float = 0.7,
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """Search for relevant context using semantic search
        
        Args:
            query: Search query text
            repository_id: Repository filter
            language_filter: Programming language filter
            file_path_filter: File path pattern filter
            k: Number of results to return
            score_threshold: Minimum similarity score
            search_type: Search type (vector, text, hybrid)
            
        Returns:
            Dictionary with search results and metadata
        """
        try:
            # Get MongoDB adapter
            mongodb = await get_mongodb_adapter()
            
            # Generate query embedding for vector search
            query_embedding = None
            if search_type in ["vector", "hybrid"]:
                query_embedding = await embedding_tool.get_embedding_for_query(query)
                if not query_embedding:
                    logger.warning("Could not generate query embedding, falling back to text search")
                    search_type = "text"
            
            # Perform search based on type
            if search_type == "vector" and query_embedding:
                results = await self._vector_search(
                    query_embedding, repository_id, language_filter, k, score_threshold
                )
            elif search_type == "text":
                results = await self._text_search(
                    query, repository_id, language_filter, k
                )
            elif search_type == "hybrid":
                results = await self._hybrid_search(
                    query, query_embedding, repository_id, language_filter, k
                )
            else:
                raise ValueError(f"Invalid search type: {search_type}")
            
            # Apply file path filter if specified
            if file_path_filter:
                results = self._apply_file_path_filter(results, file_path_filter)
            
            # Extract and format context
            formatted_results = []
            for result in results[:k]:
                context_info = await self._extract_context_info(result)
                formatted_results.append(context_info)
            
            return {
                "status": "success",
                "results": formatted_results,
                "count": len(formatted_results),
                "search_type": search_type,
                "query": query,
                "repository_id": repository_id,
                "language_filter": language_filter,
                "search_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "count": 0
            }
    
    async def _vector_search(
        self,
        query_embedding: List[float],
        repository_id: Optional[str],
        language_filter: Optional[str],
        k: int,
        score_threshold: float
    ) -> List[Dict[str, Any]]:
        """Perform vector similarity search"""
        mongodb = await get_mongodb_adapter()
        
        from uuid import UUID
        repo_uuid = UUID(repository_id) if repository_id else None
        
        return await mongodb.vector_search(
            query_embedding=query_embedding,
            repository_id=repo_uuid,
            language_filter=language_filter,
            k=k,
            score_threshold=score_threshold
        )
    
    async def _text_search(
        self,
        query: str,
        repository_id: Optional[str],
        language_filter: Optional[str],
        k: int
    ) -> List[Dict[str, Any]]:
        """Perform text-based search"""
        mongodb = await get_mongodb_adapter()
        
        # Build text search query
        search_query = {"$text": {"$search": query}}
        
        if repository_id:
            search_query["repository_id"] = repository_id
        if language_filter:
            search_query["language"] = language_filter
        
        # Execute search
        documents = await mongodb.find_documents(
            "code_documents",
            search_query,
            limit=k,
            sort_field="score",
            sort_direction=1  # Text score descending
        )
        
        # Convert to expected format
        results = []
        for doc in documents:
            from uuid import UUID
            doc["repository_id"] = UUID(doc["repository_id"])
            code_doc = CodeDocument(**doc)
            
            results.append({
                "document": code_doc,
                "score": doc.get("score", 0.5),  # Default text search score
                "source": "text"
            })
        
        return results
    
    async def _hybrid_search(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        repository_id: Optional[str],
        language_filter: Optional[str],
        k: int
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining vector and text search"""
        mongodb = await get_mongodb_adapter()
        
        from uuid import UUID
        repo_uuid = UUID(repository_id) if repository_id else None
        
        return await mongodb.hybrid_search(
            query_text=query,
            query_embedding=query_embedding,
            repository_id=repo_uuid,
            language_filter=language_filter,
            k=k
        )
    
    def _apply_file_path_filter(self, results: List[Dict[str, Any]], file_path_filter: str) -> List[Dict[str, Any]]:
        """Apply file path pattern filter to search results
        
        Args:
            results: Search results
            file_path_filter: File path pattern (glob style)
            
        Returns:
            Filtered results
        """
        import fnmatch
        
        filtered_results = []
        for result in results:
            file_path = result["document"].file_path
            if fnmatch.fnmatch(file_path, file_path_filter):
                filtered_results.append(result)
        
        return filtered_results
    
    async def _extract_context_info(self, search_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from search result
        
        Args:
            search_result: Search result with document and score
            
        Returns:
            Formatted context information
        """
        document = search_result["document"]
        score = search_result["score"]
        
        # Extract key information
        context_info = {
            "document_id": document.id,
            "file_path": document.file_path,
            "language": document.language,
            "similarity_score": score,
            "content_preview": self._generate_content_preview(document.processed_content),
            "metadata": document.metadata,
            "file_size": document.get_content_size(),
            "last_modified": document.updated_at.isoformat()
        }
        
        # Add code structure information
        if document.language in ["python", "javascript", "typescript", "java", "go"]:
            structure_info = self._analyze_code_structure(document.content, document.language)
            context_info["code_structure"] = structure_info
        
        return context_info
    
    def _generate_content_preview(self, content: str, max_length: int = 200) -> str:
        """Generate content preview for context
        
        Args:
            content: Full content
            max_length: Maximum preview length
            
        Returns:
            Content preview
        """
        if not content:
            return ""
        
        # Clean content
        cleaned = re.sub(r'\s+', ' ', content.strip())
        
        if len(cleaned) <= max_length:
            return cleaned
        
        # Try to break at sentence boundary
        preview = cleaned[:max_length]
        last_sentence = preview.rfind('.')
        if last_sentence > max_length * 0.7:  # If sentence break is reasonably close
            preview = preview[:last_sentence + 1]
        else:
            preview = preview[:max_length] + "..."
        
        return preview
    
    def _analyze_code_structure(self, content: str, language: str) -> Dict[str, Any]:
        """Analyze code structure for better context
        
        Args:
            content: Code content
            language: Programming language
            
        Returns:
            Dictionary with code structure information
        """
        structure = {
            "functions": [],
            "classes": [],
            "imports": [],
            "comments": [],
            "complexity_estimate": "low"
        }
        
        try:
            lines = content.split('\n')
            
            if language == "python":
                structure.update(self._analyze_python_structure(lines))
            elif language in ["javascript", "typescript"]:
                structure.update(self._analyze_js_structure(lines))
            elif language == "java":
                structure.update(self._analyze_java_structure(lines))
            elif language == "go":
                structure.update(self._analyze_go_structure(lines))
            
            # Estimate complexity
            total_lines = len([line for line in lines if line.strip()])
            if total_lines > 200:
                structure["complexity_estimate"] = "high"
            elif total_lines > 50:
                structure["complexity_estimate"] = "medium"
            
        except Exception as e:
            logger.debug(f"Code structure analysis failed: {e}")
        
        return structure
    
    def _analyze_python_structure(self, lines: List[str]) -> Dict[str, List[str]]:
        """Analyze Python code structure"""
        functions = []
        classes = []
        imports = []
        
        for line in lines:
            stripped = line.strip()
            
            if stripped.startswith('def '):
                match = re.match(r'def\s+(\w+)', stripped)
                if match:
                    functions.append(match.group(1))
            
            elif stripped.startswith('class '):
                match = re.match(r'class\s+(\w+)', stripped)
                if match:
                    classes.append(match.group(1))
            
            elif stripped.startswith(('import ', 'from ')):
                imports.append(stripped)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports[:10]  # Limit imports
        }
    
    def _analyze_js_structure(self, lines: List[str]) -> Dict[str, List[str]]:
        """Analyze JavaScript/TypeScript code structure"""
        functions = []
        classes = []
        imports = []
        
        for line in lines:
            stripped = line.strip()
            
            # Function declarations
            if 'function ' in stripped:
                match = re.search(r'function\s+(\w+)', stripped)
                if match:
                    functions.append(match.group(1))
            
            # Arrow functions
            elif '=>' in stripped:
                match = re.search(r'(\w+)\s*=.*=>', stripped)
                if match:
                    functions.append(match.group(1))
            
            # Class declarations
            elif stripped.startswith('class '):
                match = re.match(r'class\s+(\w+)', stripped)
                if match:
                    classes.append(match.group(1))
            
            # Imports
            elif stripped.startswith(('import ', 'const ', 'let ', 'var ')) and 'require(' in stripped:
                imports.append(stripped)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports[:10]
        }
    
    def _analyze_java_structure(self, lines: List[str]) -> Dict[str, List[str]]:
        """Analyze Java code structure"""
        functions = []
        classes = []
        imports = []
        
        for line in lines:
            stripped = line.strip()
            
            # Method declarations
            if re.search(r'(public|private|protected|static).*\w+\s*\(', stripped):
                match = re.search(r'\w+\s+(\w+)\s*\(', stripped)
                if match:
                    functions.append(match.group(1))
            
            # Class declarations
            elif stripped.startswith('public class ') or stripped.startswith('class '):
                match = re.search(r'class\s+(\w+)', stripped)
                if match:
                    classes.append(match.group(1))
            
            # Imports
            elif stripped.startswith('import '):
                imports.append(stripped)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports[:10]
        }
    
    def _analyze_go_structure(self, lines: List[str]) -> Dict[str, List[str]]:
        """Analyze Go code structure"""
        functions = []
        classes = []  # Go doesn't have classes, but has structs
        imports = []
        
        for line in lines:
            stripped = line.strip()
            
            # Function declarations
            if stripped.startswith('func '):
                match = re.match(r'func\s+(\w+)', stripped)
                if match:
                    functions.append(match.group(1))
            
            # Struct declarations (similar to classes)
            elif stripped.startswith('type ') and ' struct' in stripped:
                match = re.match(r'type\s+(\w+)\s+struct', stripped)
                if match:
                    classes.append(match.group(1))
            
            # Imports
            elif stripped.startswith('import ') or (stripped.startswith('"') and stripped.endswith('"')):
                imports.append(stripped)
        
        return {
            "functions": functions,
            "classes": classes,
            "imports": imports[:10]
        }
    
    async def rank_contexts(
        self,
        contexts: List[Dict[str, Any]],
        query: str,
        max_contexts: int = 5,
        ranking_strategy: str = "relevance"
    ) -> Dict[str, Any]:
        """Rank and filter context candidates
        
        Args:
            contexts: List of context candidates
            query: Original query for relevance scoring
            max_contexts: Maximum contexts to return
            ranking_strategy: Ranking strategy to use
            
        Returns:
            Dictionary with ranked contexts
        """
        try:
            if not contexts:
                return {
                    "status": "success",
                    "ranked_contexts": [],
                    "count": 0
                }
            
            # Apply ranking strategy
            if ranking_strategy == "relevance":
                ranked_contexts = await self._rank_by_relevance(contexts, query)
            elif ranking_strategy == "recency":
                ranked_contexts = self._rank_by_recency(contexts)
            elif ranking_strategy == "importance":
                ranked_contexts = self._rank_by_importance(contexts)
            elif ranking_strategy == "diversity":
                ranked_contexts = await self._rank_by_diversity(contexts, max_contexts)
            else:
                # Default to relevance
                ranked_contexts = await self._rank_by_relevance(contexts, query)
            
            # Limit to max_contexts
            final_contexts = ranked_contexts[:max_contexts]
            
            return {
                "status": "success",
                "ranked_contexts": final_contexts,
                "count": len(final_contexts),
                "original_count": len(contexts),
                "ranking_strategy": ranking_strategy,
                "ranking_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Context ranking failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "count": 0
            }
    
    async def _rank_by_relevance(self, contexts: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank contexts by relevance to query"""
        scored_contexts = []
        
        for context in contexts:
            relevance_score = await self._calculate_relevance_score(context, query)
            context_copy = context.copy()
            context_copy["relevance_score"] = relevance_score
            scored_contexts.append(context_copy)
        
        # Sort by relevance score
        return sorted(scored_contexts, key=lambda x: x["relevance_score"], reverse=True)
    
    def _rank_by_recency(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank contexts by recency (last modified time)"""
        return sorted(
            contexts,
            key=lambda x: x.get("last_modified", "1970-01-01T00:00:00"),
            reverse=True
        )
    
    def _rank_by_importance(self, contexts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank contexts by file importance (main files, documentation, etc.)"""
        def importance_score(context):
            file_path = context.get("file_path", "").lower()
            
            # Higher scores for important files
            if "readme" in file_path or "main" in file_path:
                return 10
            elif "doc" in file_path:
                return 8
            elif "test" in file_path:
                return 3
            elif file_path.startswith("src/"):
                return 7
            elif file_path.startswith("lib/"):
                return 6
            else:
                return 5
        
        return sorted(contexts, key=importance_score, reverse=True)
    
    async def _rank_by_diversity(self, contexts: List[Dict[str, Any]], max_contexts: int) -> List[Dict[str, Any]]:
        """Rank contexts to maximize diversity (different files, languages, etc.)"""
        if len(contexts) <= max_contexts:
            return contexts
        
        selected_contexts = []
        seen_files = set()
        seen_languages = set()
        
        # Sort by similarity score first
        sorted_contexts = sorted(contexts, key=lambda x: x.get("similarity_score", 0), reverse=True)
        
        for context in sorted_contexts:
            if len(selected_contexts) >= max_contexts:
                break
            
            file_path = context.get("file_path", "")
            language = context.get("language", "")
            
            # Prefer diverse files and languages
            diversity_bonus = 0
            if file_path not in seen_files:
                diversity_bonus += 0.2
            if language not in seen_languages:
                diversity_bonus += 0.1
            
            # Add diversity bonus to score
            context_copy = context.copy()
            original_score = context.get("similarity_score", 0)
            context_copy["diversity_score"] = original_score + diversity_bonus
            
            selected_contexts.append(context_copy)
            seen_files.add(file_path)
            seen_languages.add(language)
        
        return selected_contexts
    
    async def _calculate_relevance_score(self, context: Dict[str, Any], query: str) -> float:
        """Calculate relevance score for context
        
        Args:
            context: Context information
            query: Search query
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        base_score = context.get("similarity_score", 0.0)
        
        # Text relevance (keyword matching)
        content_preview = context.get("content_preview", "").lower()
        query_words = set(query.lower().split())
        content_words = set(content_preview.split())
        
        word_overlap = len(query_words & content_words)
        text_relevance = word_overlap / len(query_words) if query_words else 0
        
        # File importance
        file_path = context.get("file_path", "").lower()
        importance_score = 0.5  # Default
        
        if "readme" in file_path or "main" in file_path:
            importance_score = 1.0
        elif "doc" in file_path:
            importance_score = 0.9
        elif file_path.startswith("src/"):
            importance_score = 0.8
        elif "test" in file_path:
            importance_score = 0.3
        
        # Recency (newer files get slight boost)
        recency_score = 0.5  # Default for missing timestamp
        try:
            last_modified = context.get("last_modified")
            if last_modified:
                from datetime import datetime
                mod_time = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                days_old = (now - mod_time).days
                
                # Boost for files modified in last 30 days
                if days_old <= 30:
                    recency_score = 1.0
                elif days_old <= 90:
                    recency_score = 0.8
                elif days_old <= 365:
                    recency_score = 0.6
        except Exception:
            pass
        
        # Code quality (based on structure analysis)
        quality_score = 0.5  # Default
        code_structure = context.get("code_structure", {})
        if code_structure:
            # Files with good structure (functions, classes, comments) get higher scores
            function_count = len(code_structure.get("functions", []))
            class_count = len(code_structure.get("classes", []))
            
            if function_count > 0 or class_count > 0:
                quality_score = min(1.0, 0.5 + (function_count + class_count) * 0.1)
        
        # Combine scores using weights
        weights = self._relevance_weights
        final_score = (
            base_score * weights["semantic_similarity"] +
            text_relevance * weights["text_relevance"] +
            recency_score * weights["recency"] +
            importance_score * weights["file_importance"] +
            quality_score * weights["code_quality"]
        )
        
        return min(1.0, max(0.0, final_score))
    
    async def extract_code_snippets(
        self,
        contexts: List[Dict[str, Any]],
        query: str,
        max_snippet_length: int = 500
    ) -> Dict[str, Any]:
        """Extract relevant code snippets from contexts
        
        Args:
            contexts: List of context documents
            query: Search query for relevance
            max_snippet_length: Maximum length of each snippet
            
        Returns:
            Dictionary with extracted snippets
        """
        try:
            snippets = []
            
            for context in contexts:
                document = context.get("document")
                if not document:
                    continue
                
                # Extract relevant snippet
                snippet_info = await self._extract_relevant_snippet(
                    document.content,
                    query,
                    document.file_path,
                    document.language,
                    max_snippet_length
                )
                
                if snippet_info:
                    snippets.append(snippet_info)
            
            return {
                "status": "success",
                "snippets": snippets,
                "count": len(snippets),
                "extraction_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Code snippet extraction failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "count": 0
            }
    
    async def _extract_relevant_snippet(
        self,
        content: str,
        query: str,
        file_path: str,
        language: str,
        max_length: int
    ) -> Optional[Dict[str, Any]]:
        """Extract most relevant snippet from content
        
        Args:
            content: Full file content
            query: Search query
            file_path: File path
            language: Programming language
            max_length: Maximum snippet length
            
        Returns:
            Snippet information or None
        """
        try:
            lines = content.split('\n')
            query_words = set(query.lower().split())
            
            # Find lines with query keywords
            relevant_lines = []
            for i, line in enumerate(lines):
                line_words = set(re.findall(r'\w+', line.lower()))
                if query_words & line_words:
                    relevant_lines.append((i, line, len(query_words & line_words)))
            
            if not relevant_lines:
                # No direct matches, return first non-empty lines
                non_empty_lines = [(i, line) for i, line in enumerate(lines) if line.strip()]
                if non_empty_lines:
                    start_line = non_empty_lines[0][0]
                    snippet_lines = lines[start_line:start_line + 10]
                    snippet_text = '\n'.join(snippet_lines)
                    
                    if len(snippet_text) <= max_length:
                        return {
                            "file_path": file_path,
                            "language": language,
                            "snippet": snippet_text,
                            "line_start": start_line + 1,
                            "line_end": start_line + len(snippet_lines),
                            "relevance": "structural"
                        }
                
                return None
            
            # Sort by relevance (number of matching words)
            relevant_lines.sort(key=lambda x: x[2], reverse=True)
            
            # Find best snippet around most relevant line
            best_line_num = relevant_lines[0][0]
            
            # Expand context around the relevant line
            context_size = 5  # Lines before and after
            start_line = max(0, best_line_num - context_size)
            end_line = min(len(lines), best_line_num + context_size + 1)
            
            snippet_lines = lines[start_line:end_line]
            snippet_text = '\n'.join(snippet_lines)
            
            # Trim if too long
            if len(snippet_text) > max_length:
                # Try to keep complete lines
                trimmed_lines = []
                current_length = 0
                
                for line in snippet_lines:
                    if current_length + len(line) + 1 <= max_length:
                        trimmed_lines.append(line)
                        current_length += len(line) + 1
                    else:
                        break
                
                snippet_text = '\n'.join(trimmed_lines)
                end_line = start_line + len(trimmed_lines)
            
            return {
                "file_path": file_path,
                "language": language,
                "snippet": snippet_text,
                "line_start": start_line + 1,
                "line_end": end_line,
                "relevance": "keyword_match",
                "matching_words": list(query_words & set(re.findall(r'\w+', snippet_text.lower())))
            }
            
        except Exception as e:
            logger.debug(f"Snippet extraction failed for {file_path}: {e}")
            return None
    
    async def hybrid_search(
        self,
        query: str,
        repository_id: Optional[str] = None,
        language_filter: Optional[str] = None,
        k: int = 10
    ) -> Dict[str, Any]:
        """Perform hybrid search with automatic context ranking
        
        Args:
            query: Search query
            repository_id: Repository filter
            language_filter: Language filter
            k: Number of results
            
        Returns:
            Dictionary with hybrid search results
        """
        try:
            # First, perform context search
            search_result = await self.search_context(
                query=query,
                repository_id=repository_id,
                language_filter=language_filter,
                k=k * 2,  # Get more candidates for ranking
                search_type="hybrid"
            )
            
            if search_result["status"] != "success":
                return search_result
            
            # Then rank the results
            rank_result = await self.rank_contexts(
                contexts=search_result["results"],
                query=query,
                max_contexts=k,
                ranking_strategy="relevance"
            )
            
            if rank_result["status"] != "success":
                return rank_result
            
            # Extract code snippets from top results
            snippet_result = await self.extract_code_snippets(
                contexts=rank_result["ranked_contexts"],
                query=query
            )
            
            return {
                "status": "success",
                "contexts": rank_result["ranked_contexts"],
                "snippets": snippet_result.get("snippets", []),
                "count": rank_result["count"],
                "query": query,
                "search_metadata": {
                    "search_type": "hybrid",
                    "repository_id": repository_id,
                    "language_filter": language_filter,
                    "candidates_found": search_result["count"],
                    "final_results": rank_result["count"]
                },
                "processing_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "count": 0
            }


# Tool instance for LangGraph
context_tool = ContextTool()
