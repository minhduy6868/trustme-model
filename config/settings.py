"""
Configuration management
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Service info
    VERSION: str = "2.0.0"
    TITLE: str = "TrustScore Verification Service"
    
    # External services
    CRAWLER_API_URL: str = "http://localhost:8000"
    CRAWLER_TIMEOUT: int = 180
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_TTL_JOBS: int = 86400
    REDIS_TTL_CACHE: int = 3600
    
    # Processing
    MAX_TEXT_LENGTH: int = 10000
    MAX_CONCURRENT_JOBS: int = 50
    JOB_TIMEOUT: int = 300
    
    # CORS
    CORS_ORIGINS: str = "*"
    
    # External API keys (optional)
    FACT_CHECK_API_KEY: str = ""
    NEWS_API_KEY: str = ""
    SEARCH_API_KEY: str = ""
    SEARCH_ENGINE_ID: str = ""
    IMAGE_SEARCH_API_KEY: str = ""
    IMAGE_SEARCH_API_URL: str = ""
    NEWSDATA_API_KEY: str = ""
    OPENPAGERANK_API_KEY: str = ""
    
    # Debug
    DEBUG: bool = False
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from string"""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]


def get_settings() -> Settings:
    """Get settings instance"""
    return Settings()

