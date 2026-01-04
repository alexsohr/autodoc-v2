"""Configuration data models

This module defines configuration-related Pydantic models including
LLMConfig and StorageConfig based on data-model.md specifications.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_serializer,
    field_validator,
)


class LLMProvider(str, Enum):
    """LLM provider types"""

    OPENAI = "openai"
    GEMINI = "gemini"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class StorageType(str, Enum):
    """Storage adapter types"""

    LOCAL = "local"
    S3 = "s3"


class LLMConfig(BaseModel):
    """LLM provider configuration

    Contains configuration settings for different LLM providers
    including API keys, model parameters, and connection settings.
    """

    # Provider identification
    provider: LLMProvider = Field(description="Provider name")
    model_name: str = Field(description="Specific model identifier")

    # Authentication
    api_key: SecretStr = Field(description="Provider API key (encrypted)")
    endpoint_url: Optional[str] = Field(
        default=None, description="Custom endpoint (optional)"
    )

    # Model parameters
    max_tokens: int = Field(default=4000, description="Maximum response tokens")
    temperature: float = Field(
        default=0.1, description="Generation temperature (0.0-1.0)"
    )

    # Connection settings
    timeout: int = Field(default=30, description="Request timeout (seconds)")

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    @field_serializer("api_key")
    def serialize_api_key(self, value: SecretStr) -> str | None:
        """Serialize SecretStr for JSON output"""
        return value.get_secret_value() if value else None

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty"""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max tokens is positive and reasonable"""
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        if v > 100000:  # Reasonable upper limit
            raise ValueError("Max tokens seems too high (>100k)")
        return v

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is in valid range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, v: int) -> int:
        """Validate timeout is positive and reasonable"""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        if v > 300:  # 5 minutes max
            raise ValueError("Timeout seems too high (>5 minutes)")
        return v

    @field_validator("endpoint_url")
    @classmethod
    def validate_endpoint_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate endpoint URL format"""
        if v is None or not v.strip():
            return None

        # Basic URL validation
        import re

        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain
            r"localhost|"  # localhost
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # IP
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(v.strip()):
            raise ValueError("Invalid endpoint URL format")

        return v.strip()

    def get_api_key(self) -> str:
        """Get the API key as string"""
        return self.api_key.get_secret_value()

    def is_custom_endpoint(self) -> bool:
        """Check if using custom endpoint"""
        return self.endpoint_url is not None

    def get_provider_defaults(self) -> Dict[str, Any]:
        """Get provider-specific default configurations"""
        defaults = {
            LLMProvider.OPENAI: {
                "endpoint_url": "https://api.openai.com/v1",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 30,
            },
            LLMProvider.GEMINI: {
                "endpoint_url": "https://generativelanguage.googleapis.com/v1",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 30,
            },
            LLMProvider.BEDROCK: {
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 60,  # AWS Bedrock might need more time
            },
            LLMProvider.OLLAMA: {
                "endpoint_url": "http://localhost:11434",
                "max_tokens": 4000,
                "temperature": 0.1,
                "timeout": 60,  # Local models might be slower
            },
        }

        return defaults.get(self.provider, {})

    def __str__(self) -> str:
        return f"{self.provider}:{self.model_name}"

    def __repr__(self) -> str:
        return f"LLMConfig(provider={self.provider}, model={self.model_name})"


class StorageConfig(BaseModel):
    """Storage adapter configuration

    Contains configuration settings for different storage backends
    including local filesystem and cloud storage options.
    """

    # Storage type
    type: StorageType = Field(description="Storage type")
    base_path: str = Field(description="Base directory/bucket path")

    # Provider-specific parameters
    connection_params: Dict[str, Any] = Field(
        default_factory=dict, description="Provider-specific parameters"
    )

    # Backup and retention
    backup_enabled: bool = Field(default=False, description="Backup configuration flag")
    retention_days: int = Field(default=30, description="Data retention period")

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    @field_validator("base_path")
    @classmethod
    def validate_base_path(cls, v: str) -> str:
        """Validate base path is not empty"""
        if not v or not v.strip():
            raise ValueError("Base path cannot be empty")
        return v.strip()

    @field_validator("retention_days")
    @classmethod
    def validate_retention_days(cls, v: int) -> int:
        """Validate retention days is positive"""
        if v <= 0:
            raise ValueError("Retention days must be positive")
        if v > 3650:  # 10 years max
            raise ValueError("Retention days seems too high (>10 years)")
        return v

    def get_connection_param(self, key: str, default: Any = None) -> Any:
        """Get a connection parameter"""
        return self.connection_params.get(key, default)

    def set_connection_param(self, key: str, value: Any) -> None:
        """Set a connection parameter"""
        if not self.connection_params:
            self.connection_params = {}
        self.connection_params[key] = value

    def get_local_defaults(self) -> Dict[str, Any]:
        """Get default configuration for local storage"""
        return {
            "base_path": "./data",
            "backup_enabled": False,
            "retention_days": 30,
            "connection_params": {"create_dirs": True, "permissions": "0755"},
        }

    def get_s3_defaults(self) -> Dict[str, Any]:
        """Get default configuration for S3 storage"""
        return {
            "backup_enabled": True,
            "retention_days": 90,
            "connection_params": {
                "region": "us-east-1",
                "storage_class": "STANDARD",
                "server_side_encryption": "AES256",
                "versioning_enabled": True,
            },
        }

    def is_local_storage(self) -> bool:
        """Check if using local storage"""
        return self.type == StorageType.LOCAL

    def is_cloud_storage(self) -> bool:
        """Check if using cloud storage"""
        return self.type == StorageType.S3

    def __str__(self) -> str:
        return f"{self.type}:{self.base_path}"

    def __repr__(self) -> str:
        return f"StorageConfig(type={self.type}, base_path={self.base_path})"


# Configuration management models


class AppConfig(BaseModel):
    """Application configuration

    Complete application configuration including LLM and storage settings.
    """

    # LLM configurations (multiple providers supported)
    llm_configs: Dict[str, LLMConfig] = Field(
        default_factory=dict, description="LLM provider configurations"
    )
    default_llm_provider: Optional[str] = Field(
        default=None, description="Default LLM provider name"
    )

    # Storage configuration
    storage_config: StorageConfig = Field(description="Storage configuration")

    # Application settings
    app_name: str = Field(default="AutoDoc v2", description="Application name")
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")

    # Performance settings
    max_concurrent_analyses: int = Field(
        default=3, description="Maximum concurrent repository analyses"
    )
    analysis_timeout_minutes: int = Field(
        default=30, description="Analysis timeout in minutes"
    )
    embedding_batch_size: int = Field(
        default=100, description="Embedding generation batch size"
    )

    model_config = ConfigDict(validate_assignment=True)

    @field_validator("max_concurrent_analyses")
    @classmethod
    def validate_max_concurrent_analyses(cls, v: int) -> int:
        """Validate max concurrent analyses"""
        if v <= 0:
            raise ValueError("Max concurrent analyses must be positive")
        if v > 20:  # Reasonable upper limit
            raise ValueError("Max concurrent analyses seems too high (>20)")
        return v

    @field_validator("analysis_timeout_minutes")
    @classmethod
    def validate_analysis_timeout(cls, v: int) -> int:
        """Validate analysis timeout"""
        if v <= 0:
            raise ValueError("Analysis timeout must be positive")
        if v > 480:  # 8 hours max
            raise ValueError("Analysis timeout seems too high (>8 hours)")
        return v

    @field_validator("embedding_batch_size")
    @classmethod
    def validate_embedding_batch_size(cls, v: int) -> int:
        """Validate embedding batch size"""
        if v <= 0:
            raise ValueError("Embedding batch size must be positive")
        if v > 1000:  # Reasonable upper limit
            raise ValueError("Embedding batch size seems too high (>1000)")
        return v

    def add_llm_config(self, name: str, config: LLMConfig) -> None:
        """Add an LLM configuration"""
        self.llm_configs[name] = config

        # Set as default if it's the first one
        if self.default_llm_provider is None:
            self.default_llm_provider = name

    def get_llm_config(
        self, provider_name: Optional[str] = None
    ) -> Optional[LLMConfig]:
        """Get LLM configuration by name or default"""
        if provider_name is None:
            provider_name = self.default_llm_provider

        if provider_name is None:
            return None

        return self.llm_configs.get(provider_name)

    def get_available_llm_providers(self) -> List[str]:
        """Get list of available LLM providers"""
        return list(self.llm_configs.keys())

    def set_default_llm_provider(self, provider_name: str) -> None:
        """Set default LLM provider"""
        if provider_name not in self.llm_configs:
            raise ValueError(f"LLM provider '{provider_name}' not configured")
        self.default_llm_provider = provider_name

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    def __str__(self) -> str:
        return f"AppConfig({self.app_name}, {self.environment})"

    def __repr__(self) -> str:
        return f"AppConfig(app_name={self.app_name}, environment={self.environment}, llm_providers={len(self.llm_configs)})"


# API Models for configuration


class LLMConfigCreate(BaseModel):
    """LLM configuration creation model"""

    provider: LLMProvider = Field(description="LLM provider")
    model_name: str = Field(description="Model identifier")
    api_key: str = Field(description="API key")
    endpoint_url: Optional[str] = Field(default=None, description="Custom endpoint")
    max_tokens: Optional[int] = Field(default=4000, description="Max tokens")
    temperature: Optional[float] = Field(default=0.1, description="Temperature")
    timeout: Optional[int] = Field(default=30, description="Timeout in seconds")


class LLMConfigResponse(BaseModel):
    """LLM configuration response model (without sensitive data)"""

    provider: LLMProvider = Field(description="LLM provider")
    model_name: str = Field(description="Model identifier")
    endpoint_url: Optional[str] = Field(description="Custom endpoint")
    max_tokens: int = Field(description="Max tokens")
    temperature: float = Field(description="Temperature")
    timeout: int = Field(description="Timeout in seconds")
    has_api_key: bool = Field(description="Whether API key is configured")

    model_config = ConfigDict(use_enum_values=True)


class StorageConfigCreate(BaseModel):
    """Storage configuration creation model"""

    type: StorageType = Field(description="Storage type")
    base_path: str = Field(description="Base path")
    connection_params: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Connection parameters"
    )
    backup_enabled: Optional[bool] = Field(default=False, description="Backup enabled")
    retention_days: Optional[int] = Field(default=30, description="Retention days")


class StorageConfigResponse(BaseModel):
    """Storage configuration response model"""

    type: StorageType = Field(description="Storage type")
    base_path: str = Field(description="Base path")
    backup_enabled: bool = Field(description="Backup enabled")
    retention_days: int = Field(description="Retention days")
    connection_params: Dict[str, Any] = Field(description="Connection parameters")

    model_config = ConfigDict(use_enum_values=True)
