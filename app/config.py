"""Runtime configuration, read from environment variables (see .env.example).

Every setting carries the DOCQA_ prefix (DOCQA_TOP_K, DOCQA_API_KEY, ...) so it cannot collide
with other tools on the same machine. The LLM SDKs keep their own standard names
(ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "DOCQA_"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BACKENDS = ("auto", "sentence-transformers", "hashing")
LLM_PROVIDERS = ("auto", "anthropic", "openai", "extractive")


def _env(name: str, default: str | None = None) -> str | None:
    """Read DOCQA_<name>; empty or whitespace-only counts as unset."""
    raw = os.getenv(ENV_PREFIX + name, "").strip()
    return raw or default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be a whole number, got {raw!r}") from None


def _env_float(name: str, default: float | None) -> float | None:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from None


@dataclass(frozen=True)
class Settings:
    data_dir: Path | None = Path("data")  # None = keep everything in memory
    embedding_backend: str = "auto"       # auto | sentence-transformers | hashing
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    llm_provider: str = "auto"            # auto | anthropic | openai | extractive
    llm_model: str | None = None          # None = provider default (anthropic only)
    llm_timeout: float = 60.0             # seconds to wait for the language model
    chunk_size: int = 180                 # max words per chunk
    chunk_overlap: int = 30               # words repeated between neighbouring chunks
    top_k: int = 4                        # chunks retrieved per question
    min_score: float | None = None        # None = use the embedder's recommended cut-off
    max_upload_mb: int = 10
    api_key: str | None = None            # when set, clients must send an X-API-Key header

    def __post_init__(self) -> None:
        if self.embedding_backend not in EMBEDDING_BACKENDS:
            raise ValueError(f"{ENV_PREFIX}EMBEDDING_BACKEND must be one of {EMBEDDING_BACKENDS}")
        if self.llm_provider not in LLM_PROVIDERS:
            raise ValueError(f"{ENV_PREFIX}LLM_PROVIDER must be one of {LLM_PROVIDERS}")
        if self.llm_timeout <= 0:
            raise ValueError(f"{ENV_PREFIX}LLM_TIMEOUT must be greater than 0 seconds")
        if self.chunk_size < 20:
            raise ValueError(f"{ENV_PREFIX}CHUNK_SIZE must be at least 20 words")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError(f"{ENV_PREFIX}CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        if not 1 <= self.top_k <= 20:
            raise ValueError(f"{ENV_PREFIX}TOP_K must be between 1 and 20")
        if self.max_upload_mb < 1:
            raise ValueError(f"{ENV_PREFIX}MAX_UPLOAD_MB must be at least 1")

    @classmethod
    def from_env(cls) -> Settings:
        data_dir = _env("DATA_DIR", "data")
        return cls(
            data_dir=Path(data_dir) if data_dir else None,
            embedding_backend=(_env("EMBEDDING_BACKEND", "auto") or "auto").lower(),
            embedding_model=_env("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL,
            llm_provider=(_env("LLM_PROVIDER", "auto") or "auto").lower(),
            llm_model=_env("LLM_MODEL"),
            llm_timeout=_env_float("LLM_TIMEOUT", 60.0) or 60.0,
            chunk_size=_env_int("CHUNK_SIZE", 180),
            chunk_overlap=_env_int("CHUNK_OVERLAP", 30),
            top_k=_env_int("TOP_K", 4),
            min_score=_env_float("MIN_SCORE", None),
            max_upload_mb=_env_int("MAX_UPLOAD_MB", 10),
            api_key=_env("API_KEY"),
        )
