from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuration sourced from environment variables."""

    fact_check_api_key: str | None = None
    news_api_key: str | None = None
    search_api_key: str | None = None
    search_engine_id: str | None = None
    image_search_api_key: str | None = None
    image_search_api_url: str | None = None
    newsdata_api_key: str | None = None
    openpagerank_api_key: str | None = None
    semantic_model_name: str = "multi-qa-MiniLM-L6-cos-v1"
    trusted_domains: list[str] = Field(default_factory=list)
    suspicious_domains: list[str] = Field(default_factory=list)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "env_prefix": "",
        "case_sensitive": False,
    }


@lru_cache(1)
def get_settings() -> Settings:
    return Settings()
