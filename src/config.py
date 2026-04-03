"""
Centralized configuration for the multimodal RAG system.

All environment-variable lookups are consolidated here.
Modules import `cfg` and use typed, validated fields instead of
scattered os.getenv() calls.

Supports two LLM providers: "google" (Gemini) and "openai".
Set LLM_PROVIDER=google and GOOGLE_API_KEY to use Gemini.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes")


# ── Domain prompt profiles ────────────────────────────────────────

PROMPT_PROFILES: dict[str, dict[str, str]] = {
    "sports": {
        "system": (
            "You are a sports commentator and analyst. "
            "Use the provided context to answer the question."
        ),
        "caption": (
            "You are an expert sports commentator. "
            "Describe what is happening in this frame in 2-3 sentences. "
            "Focus on: the action taking place, player positions and movements, "
            "tactical situation, and any notable events (goals, fouls, saves, etc.). "
            "Be specific and vivid, as if you are commentating live. "
            "Respond with plain text only."
        ),
    },
    "general": {
        "system": (
            "You are a helpful AI assistant. "
            "Use the provided context to answer the question."
        ),
        "caption": (
            "Describe what is happening in this image in 2-3 sentences. "
            "Be specific and detailed. Respond with plain text only."
        ),
    },
}

# ── Provider-specific model defaults ──────────────────────────────

_PROVIDER_DEFAULTS = {
    "google": {
        "llm_model": "gemini-2.0-flash",
        "eval_model": "gemini-2.0-flash",
        "caption_model": "gemini-2.0-flash",
        "embedding_model": "models/text-embedding-004",
        "embedding_dimension": 768,
    },
    "openai": {
        "llm_model": "gpt-5-mini",
        "eval_model": "gpt-4o-mini",
        "caption_model": "gpt-5-mini",
        "embedding_model": "text-embedding-3-small",
        "embedding_dimension": 1536,
    },
}


@dataclass(frozen=True)
class Config:
    """Immutable, validated application configuration."""

    # ── Provider ──────────────────────────────────────────────────
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "google"))

    # ── API Keys ──────────────────────────────────────────────────
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    google_api_key: str = field(default_factory=lambda: _env("GOOGLE_API_KEY"))
    pinecone_api_key: str = field(default_factory=lambda: _env("PINECONE_API_KEY"))

    # ── Langfuse (optional) ───────────────────────────────────────
    langfuse_public_key: str = field(default_factory=lambda: _env("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: str = field(default_factory=lambda: _env("LANGFUSE_SECRET_KEY"))
    langfuse_host: str = field(default_factory=lambda: _env("LANGFUSE_HOST", "https://cloud.langfuse.com"))

    # ── LLM (overridable, otherwise uses provider defaults) ───────
    _llm_model: str = field(default_factory=lambda: _env("LLM_MODEL"))
    _eval_model: str = field(default_factory=lambda: _env("EVAL_MODEL"))
    _caption_model: str = field(default_factory=lambda: _env("CAPTION_MODEL"))
    _embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL"))

    # ── CLIP ──────────────────────────────────────────────────────
    clip_model_name: str = field(default_factory=lambda: _env("CLIP_MODEL", "openai/clip-vit-large-patch14"))

    # ── Pinecone ──────────────────────────────────────────────────
    pinecone_cloud: str = field(default_factory=lambda: _env("PINECONE_CLOUD", "aws"))
    pinecone_region: str = field(default_factory=lambda: _env("PINECONE_REGION", "us-east-1"))
    pinecone_index_text: str = field(default_factory=lambda: _env("PINECONE_INDEX_TEXT", "multimodal-rag-text"))
    pinecone_index_audio: str = field(default_factory=lambda: _env("PINECONE_INDEX_AUDIO", "multimodal-rag-audio"))
    pinecone_index_video: str = field(default_factory=lambda: _env("PINECONE_INDEX_VIDEO", "multimodal-rag-video"))

    # ── Ingestion ─────────────────────────────────────────────────
    max_ingest_workers: int = field(default_factory=lambda: _env_int("MAX_INGEST_WORKERS", 2))
    video_frame_interval: float = field(default_factory=lambda: _env_float("VIDEO_FRAME_INTERVAL_SECONDS", 3.0))
    caption_max_tokens: int = field(default_factory=lambda: _env_int("CAPTION_MAX_TOKENS", 500))
    enable_captions: bool = field(default_factory=lambda: _env_bool("ENABLE_CAPTIONS", True))

    # ── Text chunking ─────────────────────────────────────────────
    text_chunk_size: int = field(default_factory=lambda: _env_int("TEXT_CHUNK_SIZE", 1000))
    text_chunk_overlap: int = field(default_factory=lambda: _env_int("TEXT_CHUNK_OVERLAP", 200))

    # ── Retrieval ─────────────────────────────────────────────────
    video_top_k: int = field(default_factory=lambda: _env_int("VIDEO_TOP_K", 20))
    text_top_k: int = field(default_factory=lambda: _env_int("TEXT_TOP_K", 5))
    max_context_chars: int = field(default_factory=lambda: _env_int("MAX_CONTEXT_CHARS", 12000))

    # ── Domain ────────────────────────────────────────────────────
    domain_profile: str = field(default_factory=lambda: _env("DOMAIN_PROFILE", "sports"))

    # ── Data ──────────────────────────────────────────────────────
    data_dir: str = field(default_factory=lambda: _env("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")))
    tracking_file: str = field(default_factory=lambda: _env("TRACKING_FILE", "processed_files.json"))

    # ── Provider-aware model properties ───────────────────────────

    @property
    def _defaults(self) -> dict:
        return _PROVIDER_DEFAULTS.get(self.llm_provider, _PROVIDER_DEFAULTS["google"])

    @property
    def llm_model(self) -> str:
        return self._llm_model or self._defaults["llm_model"]

    @property
    def eval_model(self) -> str:
        return self._eval_model or self._defaults["eval_model"]

    @property
    def caption_model(self) -> str:
        return self._caption_model or self._defaults["caption_model"]

    @property
    def embedding_model(self) -> str:
        return self._embedding_model or self._defaults["embedding_model"]

    @property
    def embedding_dimension(self) -> int:
        return self._defaults["embedding_dimension"]

    @property
    def is_google(self) -> bool:
        return self.llm_provider == "google"

    @property
    def active_api_key(self) -> str:
        """Return the API key for the active provider."""
        if self.is_google:
            return self.google_api_key
        return self.openai_api_key

    # ── Prompt helpers ────────────────────────────────────────────
    @property
    def system_prompt(self) -> str:
        return PROMPT_PROFILES.get(self.domain_profile, PROMPT_PROFILES["general"])["system"]

    @property
    def caption_prompt(self) -> str:
        return PROMPT_PROFILES.get(self.domain_profile, PROMPT_PROFILES["general"])["caption"]

    def validate(self) -> list[str]:
        """Return a list of configuration warnings (not fatal)."""
        warnings = []
        if self.is_google and not self.google_api_key:
            warnings.append("GOOGLE_API_KEY is not set (provider=google)")
        if not self.is_google and not self.openai_api_key:
            warnings.append("OPENAI_API_KEY is not set (provider=openai)")
        if not self.pinecone_api_key:
            warnings.append("PINECONE_API_KEY is not set")
        return warnings


# Module-level singleton — import as `from src.config import cfg`
cfg = Config()
