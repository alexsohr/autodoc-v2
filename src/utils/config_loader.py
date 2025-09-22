"""Configuration loader with environment-specific settings"""

import os
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DOCKER = "docker"


class StorageType(str, Enum):
    """Storage adapter types"""
    LOCAL = "local"
    S3 = "s3"


class LLMProvider(str, Enum):
    """LLM provider types"""
    OPENAI = "openai"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Application settings with environment-specific configurations"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application settings
    app_name: str = Field(default="AutoDoc v2", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=True, description="Debug mode")
    log_level: str = Field(default="INFO", description="Log level")
    
    # API settings
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_prefix: str = Field(default="/api/v2", description="API prefix")
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="CORS origins"
    )
    
    # Database settings
    mongodb_url: str = Field(default="mongodb://localhost:27017", description="MongoDB URL")
    mongodb_database: str = Field(default="autodoc_dev", description="MongoDB database")
    mongodb_max_connections: int = Field(default=10, description="MongoDB max connections")
    mongodb_min_connections: int = Field(default=1, description="MongoDB min connections")
    
    # Storage settings
    storage_type: StorageType = Field(default=StorageType.LOCAL, description="Storage type")
    storage_base_path: str = Field(default="./data", description="Storage base path")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    aws_s3_bucket: Optional[str] = Field(default=None, description="AWS S3 bucket")
    aws_access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    aws_secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    
    # LLM provider settings
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4-turbo-preview", description="OpenAI model")
    openai_max_tokens: int = Field(default=4000, description="OpenAI max tokens")
    openai_temperature: float = Field(default=0.1, description="OpenAI temperature")
    
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    gemini_model: str = Field(default="gemini-pro", description="Gemini model")
    gemini_max_tokens: int = Field(default=4000, description="Gemini max tokens")
    gemini_temperature: float = Field(default=0.1, description="Gemini temperature")
    
    aws_bedrock_region: str = Field(default="us-east-1", description="AWS Bedrock region")
    bedrock_model_id: str = Field(
        default="anthropic.claude-3-sonnet-20240229-v1:0",
        description="Bedrock model ID"
    )
    
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    ollama_model: str = Field(default="llama2", description="Ollama model")
    
    # Security settings
    secret_key: str = Field(
        default="your-super-secret-key-change-this-in-production",
        description="Secret key"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration minutes"
    )
    
    # Webhook settings
    webhook_secret_key: str = Field(
        default="your-webhook-secret-key",
        description="Webhook secret key"
    )
    github_webhook_secret: Optional[str] = Field(
        default=None,
        description="GitHub webhook secret"
    )
    bitbucket_webhook_secret: Optional[str] = Field(
        default=None,
        description="Bitbucket webhook secret"
    )
    
    # Performance settings
    max_concurrent_analyses: int = Field(
        default=3,
        description="Maximum concurrent analyses"
    )
    analysis_timeout_minutes: int = Field(
        default=30,
        description="Analysis timeout in minutes"
    )
    embedding_batch_size: int = Field(
        default=100,
        description="Embedding batch size"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    
    # Repository settings
    clone_timeout_seconds: int = Field(
        default=300,
        description="Repository clone timeout in seconds"
    )
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum file size in MB"
    )
    supported_languages: List[str] = Field(
        default=[
            "python", "javascript", "typescript", "java", "go", "rust",
            "cpp", "c", "csharp", "php", "ruby"
        ],
        description="Supported programming languages"
    )
    
    # Monitoring settings
    enable_metrics: bool = Field(default=True, description="Enable metrics")
    metrics_port: int = Field(default=8001, description="Metrics port")
    enable_tracing: bool = Field(default=False, description="Enable tracing")
    jaeger_endpoint: Optional[str] = Field(
        default=None,
        description="Jaeger endpoint"
    )
    
    # Development settings
    reload: bool = Field(default=True, description="Auto-reload on changes")
    workers: int = Field(default=1, description="Number of workers")
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("supported_languages", mode="before")
    @classmethod
    def parse_supported_languages(cls, v):
        """Parse supported languages from string or list"""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment == Environment.DEVELOPMENT
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode"""
        return self.environment == Environment.TESTING
    
    @property
    def database_url(self) -> str:
        """Get the full database URL"""
        return f"{self.mongodb_url}/{self.mongodb_database}"
    
    def get_llm_config(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get LLM configuration for a specific provider"""
        configs = {
            LLMProvider.OPENAI: {
                "api_key": self.openai_api_key,
                "model": self.openai_model,
                "max_tokens": self.openai_max_tokens,
                "temperature": self.openai_temperature,
            },
            LLMProvider.GEMINI: {
                "api_key": self.google_api_key,
                "model": self.gemini_model,
                "max_tokens": self.gemini_max_tokens,
                "temperature": self.gemini_temperature,
            },
            LLMProvider.BEDROCK: {
                "region": self.aws_bedrock_region,
                "model_id": self.bedrock_model_id,
                "aws_access_key_id": self.aws_access_key_id,
                "aws_secret_access_key": self.aws_secret_access_key,
            },
            LLMProvider.OLLAMA: {
                "base_url": self.ollama_base_url,
                "model": self.ollama_model,
            },
        }
        return configs.get(provider, {})


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
