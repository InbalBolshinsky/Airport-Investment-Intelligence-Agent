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
GEMINI_THINKING_LEVELS = ("low", "medium", "high")
API_KEY_NAMES = {"gemini": "GOOGLE_API_KEY", "groq": "GROQ_API_KEY"}

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
    gemini_thinking_level: str
    groq_model: str
    fallback_provider: str  # "" = no fallback
    fallback_model: str
    db_path: Path
    google_api_key: str = field(repr=False)
    groq_api_key: str = field(repr=False)

    @property
    def llm_model(self) -> str:
        """Main model name for the selected provider."""
        return self.gemini_model if self.llm_provider == "gemini" else self.groq_model

    def has_api_key(self, provider: str) -> bool:
        """True if a key is set for this provider."""
        return bool(self.google_api_key if provider == "gemini" else self.groq_api_key)

    def api_key(self, provider: str) -> str:
        """Return the API key for a provider.

        Checked here rather than at import time, so code that doesn't call the
        LLM (e.g. the data ingest scripts) runs without a key.

        Raises:
            ConfigError: if the key for that provider is empty.
        """
        if not self.has_api_key(provider):
            raise ConfigError(
                f"{API_KEY_NAMES[provider]} is missing — copy .env.example to .env and add your key "
                f"(provider '{provider}')."
            )
        return self.google_api_key if provider == "gemini" else self.groq_api_key

    def llm_api_key(self) -> str:
        """Return the API key for the main provider (LLM_PROVIDER)."""
        return self.api_key(self.llm_provider)

    @property
    def fallback_enabled(self) -> bool:
        """A fallback is used only if it is configured and its provider has a key."""
        return bool(self.fallback_provider and self.fallback_model and self.has_api_key(self.fallback_provider))


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    """Read an environment variable that must be one of `allowed` (case-insensitive)."""
    value = os.getenv(name, default).strip().lower()
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {allowed}, got '{value}'.")
    return value


def load_settings() -> Settings:
    """Build Settings from environment variables (already loaded from .env).

    Raises:
        ConfigError: if LLM_PROVIDER, FALLBACK_PROVIDER or GEMINI_THINKING_LEVEL
            has an unsupported value.
    """
    db_path = Path(os.getenv("DB_PATH", "data/airports.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path

    return Settings(
        llm_provider=_choice("LLM_PROVIDER", "gemini", SUPPORTED_PROVIDERS),
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip(),
        gemini_thinking_level=_choice("GEMINI_THINKING_LEVEL", "low", GEMINI_THINKING_LEVELS),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip(),
        fallback_provider=_choice("FALLBACK_PROVIDER", "groq", ("", *SUPPORTED_PROVIDERS)),
        fallback_model=os.getenv("FALLBACK_MODEL", "openai/gpt-oss-120b").strip(),
        db_path=db_path,
        google_api_key=os.getenv("GOOGLE_API_KEY", "").strip(),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
    )


settings = load_settings()
