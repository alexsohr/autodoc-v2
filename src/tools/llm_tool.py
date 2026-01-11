"""LLM provider tool for LangGraph workflows

This module implements the LLM provider tool for multi-provider
language model operations including text generation, chat, and streaming.
"""

import asyncio
import json
import structlog
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from ..models.config import LLMConfig, LLMProvider
from ..utils.config_loader import get_settings

logger = structlog.get_logger(__name__)


class LLMGenerateInput(BaseModel):
    """Input schema for LLM text generation"""

    prompt: str = Field(description="Text prompt for generation")
    provider: Optional[str] = Field(default=None, description="LLM provider to use")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, description="Generation temperature"
    )
    system_message: Optional[str] = Field(
        default=None, description="System message for context"
    )


class LLMChatInput(BaseModel):
    """Input schema for LLM chat"""

    messages: List[Dict[str, str]] = Field(
        description="List of chat messages with role and content"
    )
    provider: Optional[str] = Field(default=None, description="LLM provider to use")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, description="Generation temperature"
    )


class LLMStreamInput(BaseModel):
    """Input schema for LLM streaming"""

    prompt: str = Field(description="Text prompt for streaming generation")
    provider: Optional[str] = Field(default=None, description="LLM provider to use")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None, description="Generation temperature"
    )


class LLMTool(BaseTool):
    """LangGraph tool for LLM operations

    Provides multi-provider LLM capabilities including text generation,
    chat completion, and streaming responses for various AI workflows.
    """

    name: str = "llm_tool"
    description: str = (
        "Tool for multi-provider LLM text generation, chat, and streaming"
    )

    def __init__(self):
        """Initialize LLMTool.
        
        No dependencies to inject - this tool manages its own LLM providers.
        Uses lazy initialization - providers are created on first use.
        """
        super().__init__()
        # Store settings for lazy initialization
        self._settings = get_settings()

        # Lazy initialization - providers created on first use
        self._llm_providers: Dict[LLMProvider, BaseLanguageModel] = {}
        self._providers_initialized = False

        # Default generation parameters
        self._default_max_tokens = 4000
        self._default_temperature = 0.1

    def _setup_llm_providers(self, settings) -> None:
        """Setup LLM providers based on configuration"""
        try:
            # OpenAI provider
            if settings.openai_api_key:
                self._llm_providers[LLMProvider.OPENAI] = ChatOpenAI(
                    api_key=settings.openai_api_key,
                    model=settings.openai_model,
                    max_tokens=settings.openai_max_tokens,
                    temperature=settings.openai_temperature,
                    timeout=120,
                )

            # Google Gemini provider
            if settings.google_api_key:
                self._llm_providers[LLMProvider.GEMINI] = ChatGoogleGenerativeAI(
                    google_api_key=settings.google_api_key,
                    model=settings.gemini_model,
                    max_tokens=settings.gemini_max_tokens,
                    temperature=settings.gemini_temperature,
                )

            # Ollama provider (for local development)
            if settings.ollama_base_url:
                self._llm_providers[LLMProvider.OLLAMA] = ChatOllama(
                    base_url=settings.ollama_base_url,
                    model=settings.ollama_model,
                    temperature=settings.openai_temperature,  # Use OpenAI temp as default
                )

            logger.info(f"Initialized {len(self._llm_providers)} LLM providers")

        except Exception as e:
            logger.error(f"Failed to setup LLM providers: {e}")

    async def _arun(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Async run method for LangGraph"""
        if operation == "generate":
            return await self.generate_text(**kwargs)
        elif operation == "generate_structured":
            return await self.generate_structured(**kwargs)
        elif operation == "chat":
            return await self.chat_completion(**kwargs)
        elif operation == "stream":
            return await self.stream_generation(**kwargs)
        elif operation == "analyze_code":
            return await self.analyze_code(**kwargs)
        elif operation == "generate_documentation":
            return await self.generate_documentation(**kwargs)
        else:
            raise ValueError(f"Unknown LLM operation: {operation}")

    def _run(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Sync run method (not used in async workflows)"""
        raise NotImplementedError("LLM tool only supports async operations")

    async def generate_text(
        self,
        prompt: str,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate text using LLM

        Args:
            prompt: Text prompt for generation
            provider: LLM provider to use
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature
            system_message: System message for context

        Returns:
            Dictionary with generated text and metadata
        """
        try:
            # Get LLM provider
            llm = self._get_llm_provider(provider)
            if not llm:
                raise ValueError("No LLM provider available")

            # Prepare messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))

            # Override model parameters if specified
            generation_kwargs = {}
            if max_tokens:
                generation_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                generation_kwargs["temperature"] = temperature

            # Generate response
            start_time = datetime.now(timezone.utc)

            if generation_kwargs:
                # Create new LLM instance with custom parameters
                llm = llm.bind(**generation_kwargs)

            response = await llm.ainvoke(messages)

            end_time = datetime.now(timezone.utc)
            generation_time = (end_time - start_time).total_seconds()

            # Extract response content
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            return {
                "status": "success",
                "generated_text": response_content,
                "provider": provider or "default",
                "generation_time": generation_time,
                "token_count": len(response_content.split()),  # Approximate
                "timestamp": end_time.isoformat(),
                "model_info": {
                    "model_class": llm.__class__.__name__,
                    "max_tokens": max_tokens or self._default_max_tokens,
                    "temperature": temperature or self._default_temperature,
                },
            }

        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "provider": provider or "unknown",
            }

    async def generate_structured(
        self,
        prompt: str,
        schema: Any,
        provider: Optional[str] = None,
        system_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate structured output using Pydantic schema

        Args:
            prompt: Text prompt for generation
            schema: Pydantic model class for structured output
            provider: LLM provider to use
            system_message: System message for context

        Returns:
            Dictionary with structured output and metadata
        """
        try:
            # Get LLM provider
            llm = self._get_llm_provider(provider)
            if not llm:
                raise ValueError("No LLM provider available")

            # Configure LLM for structured output
            structured_llm = llm.with_structured_output(schema)

            # Prepare messages
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))

            # Generate structured response
            start_time = datetime.now(timezone.utc)
            response = await structured_llm.ainvoke(messages)
            end_time = datetime.now(timezone.utc)

            generation_time = (end_time - start_time).total_seconds()

            # Convert response to dict if it's a Pydantic model
            if hasattr(response, "model_dump"):
                structured_data = response.model_dump()
            else:
                structured_data = response

            return {
                "status": "success",
                "structured_output": structured_data,
                "schema_type": (
                    schema.__name__ if hasattr(schema, "__name__") else str(schema)
                ),
                "provider": provider or "default",
                "generation_time": generation_time,
                "timestamp": end_time.isoformat(),
            }

        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "provider": provider or "unknown",
                "schema_type": (
                    schema.__name__ if hasattr(schema, "__name__") else str(schema)
                ),
            }

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Complete chat conversation using LLM

        Args:
            messages: List of chat messages with role and content
            provider: LLM provider to use
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Returns:
            Dictionary with chat completion and metadata
        """
        try:
            # Get LLM provider
            llm = self._get_llm_provider(provider)
            if not llm:
                raise ValueError("No LLM provider available")

            # Convert messages to LangChain format
            langchain_messages = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))
                else:  # user or any other role
                    langchain_messages.append(HumanMessage(content=content))

            # Override model parameters if specified
            generation_kwargs = {}
            if max_tokens:
                generation_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                generation_kwargs["temperature"] = temperature

            # Generate response
            start_time = datetime.now(timezone.utc)

            if generation_kwargs:
                llm = llm.bind(**generation_kwargs)

            response = await llm.ainvoke(langchain_messages)

            end_time = datetime.now(timezone.utc)
            generation_time = (end_time - start_time).total_seconds()

            # Extract response content
            response_content = (
                response.content if hasattr(response, "content") else str(response)
            )

            return {
                "status": "success",
                "response": {"role": "assistant", "content": response_content},
                "provider": provider or "default",
                "generation_time": generation_time,
                "token_count": len(response_content.split()),
                "timestamp": end_time.isoformat(),
                "conversation_length": len(messages),
            }

        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "provider": provider or "unknown",
            }

    async def stream_generation(
        self,
        prompt: str,
        provider: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Stream text generation from LLM

        Args:
            prompt: Text prompt for generation
            provider: LLM provider to use
            max_tokens: Maximum tokens to generate
            temperature: Generation temperature

        Yields:
            Dictionary with streaming chunks and metadata
        """
        try:
            # Get LLM provider
            llm = self._get_llm_provider(provider)
            if not llm:
                yield {
                    "status": "error",
                    "error": "No LLM provider available",
                    "chunk": "",
                    "finished": True,
                }
                return

            # Prepare message
            messages = [HumanMessage(content=prompt)]

            # Override model parameters if specified
            generation_kwargs = {}
            if max_tokens:
                generation_kwargs["max_tokens"] = max_tokens
            if temperature is not None:
                generation_kwargs["temperature"] = temperature

            if generation_kwargs:
                llm = llm.bind(**generation_kwargs)

            # Stream response
            start_time = datetime.now(timezone.utc)
            total_content = ""
            chunk_count = 0

            try:
                async for chunk in llm.astream(messages):
                    chunk_content = (
                        chunk.content if hasattr(chunk, "content") else str(chunk)
                    )
                    total_content += chunk_content
                    chunk_count += 1

                    yield {
                        "status": "streaming",
                        "chunk": chunk_content,
                        "chunk_number": chunk_count,
                        "total_content": total_content,
                        "finished": False,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }

                # Final chunk with summary
                end_time = datetime.now(timezone.utc)
                generation_time = (end_time - start_time).total_seconds()

                yield {
                    "status": "completed",
                    "chunk": "",
                    "total_content": total_content,
                    "finished": True,
                    "generation_time": generation_time,
                    "total_chunks": chunk_count,
                    "token_count": len(total_content.split()),
                    "provider": provider or "default",
                    "timestamp": end_time.isoformat(),
                }

            except Exception as e:
                yield {
                    "status": "error",
                    "error": str(e),
                    "chunk": "",
                    "finished": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "chunk": "",
                "finished": True,
            }

    async def analyze_code(
        self,
        code_content: str,
        language: str,
        analysis_type: str = "comprehensive",
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze code using LLM

        Args:
            code_content: Code to analyze
            language: Programming language
            analysis_type: Type of analysis (comprehensive, security, performance, etc.)
            provider: LLM provider to use

        Returns:
            Dictionary with code analysis results
        """
        try:
            # Prepare analysis prompt based on type
            analysis_prompts = {
                "comprehensive": f"""Analyze this {language} code comprehensively. Provide:
1. Purpose and functionality
2. Key components (functions, classes, modules)
3. Dependencies and imports
4. Code quality assessment
5. Potential improvements
6. Documentation quality

Code:
```{language}
{code_content}
```""",
                "security": f"""Perform a security analysis of this {language} code. Identify:
1. Security vulnerabilities
2. Input validation issues
3. Authentication/authorization concerns
4. Data exposure risks
5. Recommended security improvements

Code:
```{language}
{code_content}
```""",
                "performance": f"""Analyze the performance characteristics of this {language} code:
1. Time complexity analysis
2. Memory usage patterns
3. Bottlenecks and optimization opportunities
4. Scalability considerations
5. Performance improvement recommendations

Code:
```{language}
{code_content}
```""",
                "documentation": f"""Generate comprehensive documentation for this {language} code:
1. High-level purpose and functionality
2. API documentation (functions, classes, parameters)
3. Usage examples
4. Integration guidelines
5. Configuration options

Code:
```{language}
{code_content}
```""",
            }

            prompt = analysis_prompts.get(
                analysis_type, analysis_prompts["comprehensive"]
            )

            # Generate analysis
            result = await self.generate_text(
                prompt=prompt,
                provider=provider,
                system_message="You are an expert code analyst. Provide detailed, accurate, and actionable analysis.",
            )

            if result["status"] != "success":
                return result

            # Parse and structure the analysis
            analysis_text = result["generated_text"]
            structured_analysis = self._parse_analysis_response(
                analysis_text, analysis_type
            )

            return {
                "status": "success",
                "analysis_type": analysis_type,
                "language": language,
                "analysis": structured_analysis,
                "raw_analysis": analysis_text,
                "generation_time": result["generation_time"],
                "provider": result["provider"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Code analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "analysis_type": analysis_type,
                "language": language,
            }

    async def generate_documentation(
        self,
        code_files: List[Dict[str, Any]],
        documentation_type: str = "api_reference",
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate documentation for code files

        Args:
            code_files: List of code file information
            documentation_type: Type of documentation to generate
            provider: LLM provider to use

        Returns:
            Dictionary with generated documentation
        """
        try:
            if not code_files:
                return {"status": "success", "documentation": "", "sections": []}

            # Prepare documentation prompt
            doc_prompts = {
                "api_reference": """Generate API reference documentation for the following code files. Include:
1. Overview of the API
2. Detailed function/method documentation
3. Parameter descriptions and types
4. Return value descriptions
5. Usage examples
6. Error handling information""",
                "user_guide": """Generate a user guide for the following code. Include:
1. Getting started instructions
2. Basic usage examples
3. Common use cases
4. Configuration options
5. Troubleshooting tips
6. Best practices""",
                "developer_guide": """Generate a developer guide for the following code. Include:
1. Architecture overview
2. Code organization and structure
3. Development setup instructions
4. Contribution guidelines
5. Testing procedures
6. Deployment information""",
                "readme": """Generate a comprehensive README for this project. Include:
1. Project description and purpose
2. Installation instructions
3. Quick start guide
4. Usage examples
5. API documentation
6. Contributing guidelines
7. License information""",
            }

            base_prompt = doc_prompts.get(
                documentation_type, doc_prompts["api_reference"]
            )

            # Build context from code files
            code_context = "\n\n".join(
                [
                    f"File: {file_info['file_path']}\n"
                    f"Language: {file_info['language']}\n"
                    f"```{file_info['language']}\n{file_info.get('content', '')[:2000]}...\n```"
                    for file_info in code_files[:10]  # Limit to avoid token overflow
                ]
            )

            full_prompt = f"{base_prompt}\n\nCode Files:\n{code_context}"

            # Generate documentation
            result = await self.generate_text(
                prompt=full_prompt,
                provider=provider,
                system_message="You are an expert technical writer. Generate clear, comprehensive, and well-structured documentation.",
                max_tokens=max_tokens or 6000,  # Longer for documentation
            )

            if result["status"] != "success":
                return result

            documentation_text = result["generated_text"]

            # Parse documentation into sections
            sections = self._parse_documentation_sections(documentation_text)

            return {
                "status": "success",
                "documentation_type": documentation_type,
                "documentation": documentation_text,
                "sections": sections,
                "files_processed": len(code_files),
                "generation_time": result["generation_time"],
                "provider": result["provider"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "documentation_type": documentation_type,
            }

    def _get_llm_provider(
        self, provider: Optional[str] = None
    ) -> Optional[BaseLanguageModel]:
        """Get LLM provider instance (lazy initialization)

        Args:
            provider: Provider name (defaults to first available)

        Returns:
            LLM provider instance or None
        """
        # Lazy initialize providers on first access
        if not self._providers_initialized:
            self._setup_llm_providers(self._settings)
            self._providers_initialized = True

        if provider:
            try:
                provider_enum = LLMProvider(provider)
                return self._llm_providers.get(provider_enum)
            except ValueError:
                logger.warning(f"Unknown provider: {provider}")
                return None

        # Return first available provider
        if self._llm_providers:
            return next(iter(self._llm_providers.values()))

        return None

    def _parse_analysis_response(
        self, analysis_text: str, analysis_type: str
    ) -> Dict[str, Any]:
        """Parse LLM analysis response into structured format

        Args:
            analysis_text: Raw analysis text from LLM
            analysis_type: Type of analysis performed

        Returns:
            Structured analysis data
        """
        try:
            # Split into sections based on numbered lists or headers
            sections = {}
            current_section = "overview"
            current_content = []

            lines = analysis_text.split("\n")

            for line in lines:
                stripped = line.strip()

                # Detect section headers (numbered or markdown style)
                if re.match(r"^\d+\.", stripped) or stripped.startswith("#"):
                    # Save previous section
                    if current_content:
                        sections[current_section] = "\n".join(current_content).strip()

                    # Start new section
                    section_title = re.sub(r"^\d+\.\s*|^#+\s*", "", stripped).lower()
                    section_title = re.sub(r"[^\w\s]", "", section_title).replace(
                        " ", "_"
                    )
                    current_section = section_title or "section"
                    current_content = []
                else:
                    current_content.append(line)

            # Save last section
            if current_content:
                sections[current_section] = "\n".join(current_content).strip()

            return {
                "sections": sections,
                "raw_text": analysis_text,
                "analysis_type": analysis_type,
            }

        except Exception as e:
            logger.debug(f"Could not parse analysis response: {e}")
            return {
                "sections": {"overview": analysis_text},
                "raw_text": analysis_text,
                "analysis_type": analysis_type,
            }

    def _parse_documentation_sections(
        self, documentation_text: str
    ) -> List[Dict[str, str]]:
        """Parse documentation into sections

        Args:
            documentation_text: Raw documentation text

        Returns:
            List of documentation sections
        """
        try:
            sections = []
            current_title = "Introduction"
            current_content = []

            lines = documentation_text.split("\n")

            for line in lines:
                stripped = line.strip()

                # Detect markdown headers
                if stripped.startswith("#"):
                    # Save previous section
                    if current_content:
                        sections.append(
                            {
                                "title": current_title,
                                "content": "\n".join(current_content).strip(),
                            }
                        )

                    # Start new section
                    current_title = re.sub(r"^#+\s*", "", stripped)
                    current_content = []
                else:
                    current_content.append(line)

            # Save last section
            if current_content:
                sections.append(
                    {
                        "title": current_title,
                        "content": "\n".join(current_content).strip(),
                    }
                )

            return sections

        except Exception as e:
            logger.debug(f"Could not parse documentation sections: {e}")
            return [{"title": "Documentation", "content": documentation_text}]

    async def answer_question(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        provider: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Answer question using provided context documents

        Args:
            question: User question
            context_documents: Relevant context documents
            provider: LLM provider to use

        Returns:
            Dictionary with answer and citations
        """
        try:
            if not context_documents:
                return await self.generate_text(
                    prompt=f"Answer this question: {question}",
                    provider=provider,
                    system_message="You are a helpful assistant. Answer based on your general knowledge.",
                )

            # Build context from documents
            context_parts = []
            citations = []

            for i, doc_info in enumerate(context_documents[:5]):  # Limit context
                file_path = doc_info.get("file_path", f"document_{i}")
                content = doc_info.get("content_preview") or doc_info.get("snippet", "")
                language = doc_info.get("language", "text")

                context_parts.append(
                    f"Source {i+1}: {file_path} ({language})\n{content}"
                )

                citations.append(
                    {
                        "source_number": i + 1,
                        "file_path": file_path,
                        "language": language,
                        "similarity_score": doc_info.get("similarity_score", 0.0),
                    }
                )

            context_text = "\n\n".join(context_parts)

            # Create RAG prompt
            rag_prompt = f"""Based on the following code context, answer the user's question accurately and comprehensively.

Context:
{context_text}

Question: {question}

Instructions:
1. Answer based primarily on the provided context
2. Reference specific source files when relevant
3. Provide code examples if helpful
4. If the context doesn't contain enough information, say so clearly
5. Be precise and technical when appropriate"""

            # Generate answer
            result = await self.generate_text(
                prompt=rag_prompt,
                provider=provider,
                system_message="You are an expert code assistant. Answer questions accurately based on the provided context.",
                max_tokens=2000,
            )

            if result["status"] != "success":
                return result

            # Add citations to result
            result["citations"] = citations
            result["context_used"] = len(context_documents)
            result["question"] = question

            return result

        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "error_type": type(e).__name__,
                "question": question,
            }

    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers

        Returns:
            List of available provider names
        """
        # Ensure providers are initialized
        if not self._providers_initialized:
            self._setup_llm_providers(self._settings)
            self._providers_initialized = True
            
        return [provider.value for provider in self._llm_providers.keys()]

    def get_provider_capabilities(
        self, provider: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get capabilities of LLM provider

        Args:
            provider: Provider name

        Returns:
            Dictionary with provider capabilities
        """
        llm = self._get_llm_provider(provider)
        if not llm:
            return {"status": "error", "error": "Provider not available"}

        capabilities = {
            "text_generation": True,
            "chat_completion": True,
            "streaming": hasattr(llm, "astream"),
            "provider_class": llm.__class__.__name__,
        }

        # Add model-specific info
        if hasattr(llm, "model_name"):
            capabilities["model_name"] = llm.model_name
        elif hasattr(llm, "model"):
            capabilities["model_name"] = llm.model

        if hasattr(llm, "max_tokens"):
            capabilities["max_tokens"] = llm.max_tokens

        if hasattr(llm, "temperature"):
            capabilities["temperature"] = llm.temperature

        return {
            "status": "success",
            "provider": provider or "default",
            "capabilities": capabilities,
        }

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on LLM providers

        Returns:
            Dictionary with health check results
        """
        try:
            provider_health = {}

            for provider_enum, llm in self._llm_providers.items():
                provider_name = provider_enum.value

                try:
                    # Test with simple prompt
                    test_result = await self.generate_text(
                        prompt="Say 'OK' if you can respond.",
                        provider=provider_name,
                        max_tokens=10,
                    )

                    provider_health[provider_name] = {
                        "status": (
                            "healthy"
                            if test_result["status"] == "success"
                            else "unhealthy"
                        ),
                        "response_time": test_result.get("generation_time", 0),
                        "model_class": llm.__class__.__name__,
                    }

                except Exception as e:
                    provider_health[provider_name] = {
                        "status": "unhealthy",
                        "error": str(e),
                        "model_class": llm.__class__.__name__,
                    }

            healthy_count = sum(
                1 for p in provider_health.values() if p["status"] == "healthy"
            )

            return {
                "status": "success",
                "providers": provider_health,
                "total_providers": len(self._llm_providers),
                "healthy_providers": healthy_count,
                "health_check_time": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return {"status": "error", "error": str(e), "error_type": type(e).__name__}


# Tool instance for LangGraph
# Deprecated: Module-level singleton removed
# Use get_llm_tool() from src.dependencies with FastAPI's Depends() instead
# llm_tool = LLMTool()  # REMOVED - use dependency injection
