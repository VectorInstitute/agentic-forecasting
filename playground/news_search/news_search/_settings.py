"""Environment-variable settings loaded via pydantic-settings.

Reads from the .env at the repo root so the playground shares credentials
with the rest of the project.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# parents: [0]=news_search pkg, [1]=playground/news_search,
#          [2]=playground, [3]=repo root
_REPO_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"


class Settings(BaseSettings):
    """API keys and service endpoints."""

    # Gemini / Google - the repo uses GEMINI_API_KEY; ADK expects GOOGLE_API_KEY.
    gemini_api_key: str | None = None
    google_api_key: str | None = None

    # Langfuse
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://us.cloud.langfuse.com"

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT_ENV),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def _export_google_api_key(self) -> "Settings":
        """ADK reads GOOGLE_API_KEY from the process environment."""
        key = self.google_api_key or self.gemini_api_key
        if key and not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = key
        return self

    @property
    def has_langfuse(self) -> bool:
        """True when Langfuse credentials are present."""
        return bool(self.langfuse_public_key and self.langfuse_secret_key)
