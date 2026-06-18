from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    mode: Literal["paper", "demo"] = "paper"
    kalshi_demo_api_base: str = "https://demo-api.kalshi.co/trade-api/v2"
    kalshi_demo_key_id: str | None = None
    kalshi_demo_private_key_path: Path | None = None
    kalshi_prod_api_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    kalshi_prod_key_id: str | None = None
    kalshi_prod_private_key_path: Path | None = None
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _require_key_for_demo(self) -> Settings:
        if self.mode == "demo" and not self.kalshi_demo_key_id:
            raise ValueError("kalshi_demo_key_id required when mode is 'demo'")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
