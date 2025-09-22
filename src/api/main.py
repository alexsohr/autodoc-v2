"""AutoDoc v2 FastAPI Application Entry Point"""

import os
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.api.routes import health
from src.api.routes import repositories, wiki, chat, webhooks
from src.api.middleware.logging import request_logging_middleware
from src.api.middleware.error_handler import (
    error_handling_middleware,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler
)
from src.utils.database import init_database, close_database
from src.utils.config_loader import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan management"""
    # Startup
    print("AutoDoc v2 starting up...")
    try:
        # Initialize database connections
        await init_database()
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
        await close_database()
        print("Database connections closed")
        # TODO: Cleanup resources
    except Exception as e:
        print(f"Error during shutdown: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    settings = get_settings()
    
    app = FastAPI(
        title="AutoDoc v2",
        description="Intelligent Automated Documentation Partner",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
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
        allow_origins=settings.cors_origins,
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
            "health": "/health"
        }
    
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
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )


if __name__ == "__main__":
    main()
