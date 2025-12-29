"""
Configuration management for MemOpt SaaS
Validates required env vars, enforces production safety
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Environment
    ENV: str = Field(default="development", description="Environment: development, staging, production")

    # API
    API_TITLE: str = "MemOpt SaaS API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)

    # Database
    DATABASE_URL: str = Field(..., description="PostgreSQL connection URL")
    DB_POOL_SIZE: int = Field(default=20)
    DB_MAX_OVERFLOW: int = Field(default=10)

    # Redis (optional - rate limiting disabled if not provided)
    REDIS_URL: Optional[str] = Field(default=None, description="Redis connection URL")
    REDIS_MAX_CONNECTIONS: int = Field(default=50)

    # Admin authentication
    ADMIN_API_KEY: str = Field(..., description="Admin API key for privileged operations")

    # Rate limiting (requests per minute)
    RATE_LIMIT_REVENUE_SHARE: int = Field(default=60, description="RPM for revenue_share tier")
    RATE_LIMIT_ENTERPRISE: int = Field(default=10000, description="RPM for enterprise tier")

    # Token limits (per minute)
    TOKEN_LIMIT_REVENUE_SHARE_PER_MIN: int = Field(default=100000, description="Tokens/min for revenue_share")
    TOKEN_LIMIT_ENTERPRISE_PER_MIN: int = Field(default=10000000, description="Tokens/min for enterprise")

    # Token limits (per month)
    TOKEN_LIMIT_REVENUE_SHARE_PER_MONTH: int = Field(default=100000000, description="100M tokens/month for revenue_share")
    TOKEN_LIMIT_ENTERPRISE_PER_MONTH: int = Field(default=999999999999, description="Unlimited for enterprise")

    # Concurrent requests
    MAX_CONCURRENT_REVENUE_SHARE: int = Field(default=5)
    MAX_CONCURRENT_ENTERPRISE: int = Field(default=100)

    # Billing
    PRICE_PER_1K_TOKENS: float = Field(default=0.002, description="$0.002 per 1k tokens for revenue_share")
    ENTERPRISE_FLAT_MONTHLY: float = Field(default=66666.67, description="$800k/year = $66,666.67/month")

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_JSON: bool = Field(default=True, description="Use JSON logging in production")

    # CORS
    CORS_ORIGINS: str = Field(default="*", description="Comma-separated CORS origins")

    @validator("ENV")
    def validate_env(cls, v):
        """Validate environment"""
        if v not in ["development", "staging", "production"]:
            raise ValueError("ENV must be development, staging, or production")
        return v

    @validator("ADMIN_API_KEY")
    def validate_admin_key(cls, v, values):
        """Ensure ADMIN_API_KEY is set in production"""
        env = values.get("ENV", "development")
        if env == "production":
            if not v or v == "changeme":
                raise ValueError("ADMIN_API_KEY must be set to a secure value in production")
            if len(v) < 32:
                raise ValueError("ADMIN_API_KEY must be at least 32 characters in production")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENV == "production"

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_settings() -> Settings:
    """Get validated settings instance"""
    try:
        settings = Settings()
        return settings
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        print("\n📋 Required environment variables:")
        print("  - DATABASE_URL (e.g., postgresql://user:pass@localhost/memopt)")
        print("  - ADMIN_API_KEY (secure random string, 32+ chars)")
        print("\n📋 Optional environment variables:")
        print("  - REDIS_URL (e.g., redis://localhost:6379) - for rate limiting")
        print("\n💡 Copy .env.example to .env and configure values")
        raise SystemExit(1)


# Global settings instance
settings = get_settings()
