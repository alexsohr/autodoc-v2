"""AutoDoc v2 FastAPI Application Entry Point"""

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.api.middleware.error_handler import (
    error_handling_middleware,
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from src.api.middleware.logging import request_logging_middleware
from src.api.routes import chat, health, repositories, webhooks, wiki
from src.repository.database import close_mongodb, init_mongodb
from src.utils.config_loader import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    # Startup
    print("AutoDoc v2 starting up...")
    try:
        # Configure LangSmith tracing
        settings = get_settings()
        settings.configure_langsmith()
        if settings.is_langsmith_enabled:
            print(f"LangSmith tracing enabled for project: {settings.langsmith_project}")
        else:
            print("LangSmith tracing disabled (no API key provided)")
        
        # Initialize data access layer (MongoDB/Beanie)
        await init_mongodb()
        print("Database initialized successfully")
        # TODO: Initialize storage adapters
        # TODO: Load LLM configurations
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise

    yield

    # Shutdown
    print("AutoDoc v2 shutting down...")
    try:
        # Close database connections
        await close_mongodb()
        print("Database connections closed")
        # TODO: Cleanup resources
    except Exception as e:
        print(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    settings = get_settings()

    # Enhanced OpenAPI metadata
    app = FastAPI(
        title="AutoDoc v2",
        description="""
## Intelligent Automated Documentation Partner

AutoDoc v2 is an AI-powered documentation system that automatically analyzes repositories, 
generates comprehensive documentation, and provides intelligent chat-based queries about your codebase.

### Features

* **Repository Analysis**: Automatically analyze code repositories and extract documentation
* **Wiki Generation**: Generate comprehensive wiki documentation from your codebase
* **Intelligent Chat**: Ask questions about your code and get AI-powered answers
* **Webhook Integration**: Real-time updates when your repository changes
* **Multi-provider Support**: Works with GitHub, GitLab, and other Git providers

### Getting Started

1. Register a repository using the `/api/v2/repositories` endpoint
2. Wait for analysis to complete (check status with `/api/v2/repositories/{id}/status`)
3. Generate wiki documentation or start chatting about your code
        """,
        version="2.0.0",
        contact={
            "name": "AutoDoc Team",
            "email": "support@autodoc.dev",
            "url": "https://autodoc.dev",
        },
        license_info={
            "name": "MIT License",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
        # Enhanced Swagger UI configuration
        swagger_ui_parameters={
            "deepLinking": True,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "operationsSorter": "method",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True,
            "tryItOutEnabled": True,
            "persistAuthorization": True,
            "displayOperationId": False,
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "syntaxHighlight.theme": "nord",
        },
        openapi_tags=[
            {
                "name": "health",
                "description": "Health check and system status endpoints for monitoring service availability.",
            },
            {
                "name": "repositories",
                "description": "Repository management endpoints for registering, analyzing, and configuring code repositories.",
            },
            {
                "name": "chat",
                "description": "Interactive chat endpoints for asking questions about repository codebases using AI.",
            },
            {
                "name": "wiki",
                "description": "Documentation generation endpoints for creating and managing wiki-style documentation.",
            },
            {
                "name": "webhooks",
                "description": "Webhook endpoints for receiving real-time updates from Git providers.",
            },
        ],
    )

    # Add custom middleware
    app.middleware("http")(request_logging_middleware)
    app.middleware("http")(error_handling_middleware)

    # Add exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers with API prefix
    api_prefix = settings.api_prefix

    # Health checks (no prefix)
    app.include_router(health.router)

    # API routes with prefix
    app.include_router(repositories.router, prefix=api_prefix)
    app.include_router(wiki.router, prefix=api_prefix)
    app.include_router(chat.router, prefix=api_prefix)
    app.include_router(webhooks.router)  # Webhooks don't use API prefix

    # Root endpoint
    @app.get("/", status_code=status.HTTP_200_OK)
    async def root():
        """Root endpoint"""
        return {
            "message": "AutoDoc v2 - Intelligent Automated Documentation Partner",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    # Define custom OpenAPI function as closure
    def custom_openapi():
        """Generate custom OpenAPI schema with security schemes"""
        if app.openapi_schema:
            return app.openapi_schema

        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=app.openapi_tags,
            contact=app.contact,
            license_info=app.license_info,
            openapi_version="3.1.0",
        )

        # Add security schemes (ensure components section exists)
        if "components" not in openapi_schema:
            openapi_schema["components"] = {}

        openapi_schema["components"]["securitySchemes"] = {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT Bearer token authentication. Obtain token via login endpoint.",
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key authentication for service-to-service communication.",
            },
        }

        # Add global security requirement (can be overridden per endpoint)
        openapi_schema["security"] = [{"BearerAuth": []}, {"ApiKeyAuth": []}]

        # Add servers information
        openapi_schema["servers"] = [
            {"url": "/", "description": "Current server"},
            {"url": "http://localhost:8000", "description": "Local development server"},
            {"url": "https://api.autodoc.dev", "description": "Production server"},
        ]

        # Add additional metadata
        openapi_schema["info"]["termsOfService"] = "https://autodoc.dev/terms"
        openapi_schema["info"]["x-logo"] = {
            "url": "https://autodoc.dev/logo.png",
            "altText": "AutoDoc v2 Logo",
        }

        app.openapi_schema = openapi_schema
        return app.openapi_schema

    # Set custom OpenAPI schema
    app.openapi = custom_openapi

    return app


# Create app instance
app = create_app()


def main():
    """Main entry point for running the application"""
    uvicorn.run(
        "src.api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=os.getenv("RELOAD", "true").lower() == "true",
        workers=int(os.getenv("WORKERS", "1")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )


if __name__ == "__main__":
    main()
