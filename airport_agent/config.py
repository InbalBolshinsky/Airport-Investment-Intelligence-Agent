"""Settings for the whole project, read once from the project's .env file.

Other modules import `settings` from here instead of reading environment
variables directly, so every setting has one name and one default.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPORTED_PROVIDERS = ("gemini", "groq")

# Load .env from the project folder, so commands work from any directory.
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """A required setting is missing or invalid. The message says how to fix it."""


@dataclass(frozen=True)
class Settings:
    """Read-only project settings.

    API keys are excluded from repr so printing settings never leaks a key.
    """

    llm_provider: str
    gemini_model: str
    groq_model: str
    db_path: Path
    google_api_key: str = field(repr=False)
    groq_api_key: str = field(repr=False)

    @property
    def llm_model(self) -> str:
        """Model name for the selected provider."""
        return self.gemini_model if self.llm_provider == "gemini" else self.groq_model

    def llm_api_key(self) -> str:
        """Return the API key for the selected provider.

        Checked here rather than at import time, so code that doesn't call the
        LLM (e.g. the data ingest scripts) runs without a key.

        Raises:
            ConfigError: if the key for the selected provider is empty.
        """
        key_name = "GOOGLE_API_KEY" if self.llm_provider == "gemini" else "GROQ_API_KEY"
        key = self.google_api_key if self.llm_provider == "gemini" else self.groq_api_key
        if not key:
            raise ConfigError(
                f"{key_name} is missing — copy .env.example to .env and add your key "
                f"(LLM_PROVIDER is '{self.llm_provider}')."
            )
        return key


def load_settings() -> Settings:
    """Build Settings from environment variables (already loaded from .env).

    Raises:
        ConfigError: if LLM_PROVIDER is not one of SUPPORTED_PROVIDERS.
    """
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"LLM_PROVIDER must be one of {SUPPORTED_PROVIDERS}, got '{provider}'."
        )

    db_path = Path(os.getenv("DB_PATH", "data/airports.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    return Settings(
        llm_provider=provider,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        db_path=db_path,
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
    )


settings = load_settings()
