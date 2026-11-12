"""Central configuration for the RehlaAI Hybrid RAG pipeline.

All values are read from environment variables (optionally a `.env` file) so the
system can be reconfigured without touching application code. See
`.env.example` for the full list of supported variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - dotenv is optional at runtime
    from dotenv import load_dotenv

    # Load a project-root .env if present; never raises if missing.
    _ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(_ROOT / ".env")
except ImportError:  # pragma: no cover
    pass


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() != "" else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None and raw.strip() != "" else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    try:
        return float(raw) if raw is not None and raw.strip() != "" else default
    except ValueError:
        return default


def _project_root() -> Path:
    """Project root (the folder containing the `rag` package)."""
    return Path(__file__).resolve().parents[1]


class RAGConfig:
    """Container for every configurable knob of the RAG pipeline."""

    def __init__(self) -> None:
        self.project_root = _project_root()

        # ------------------------- Knowledge base -------------------------
        self.kb_path: Path = Path(
            _env_str("KB_PATH", str(self.project_root / "data" / "egypt_landmarks_kb"))
        )
        # Optional override pointing at a single enhanced JSON knowledge base.
        # Defaults to the enhanced RehlaAI file in data/ when it exists, so the
        # enhanced KB is the primary source without any manual configuration.
        self.kb_single_file: Optional[Path] = None
        raw_single = _env_str("KB_SINGLE_FILE", "")
        if raw_single:
            self.kb_single_file = Path(raw_single)
        else:
            default_enhanced = self.project_root / "data" / "rehlaai_rag_data_enhanced.json"
            if default_enhanced.is_file():
                self.kb_single_file = default_enhanced

        # ------------------------- Embeddings -------------------------
        # Primary model: intfloat/multilingual-e5-large (Arabic + English).
        # Any sentence-transformers compatible model name is accepted, which
        # also lets tests use a small model without downloading 2GB.
        self.embedding_model: str = _env_str(
            "EMBEDDING_MODEL", "intfloat/multilingual-e5-large"
        )
        self.embedding_device: str = _env_str("EMBEDDING_DEVICE", "auto")
        self.embedding_batch_size: int = _env_int("EMBEDDING_BATCH_SIZE", 16)
        self.embedding_normalize: bool = _env_str("EMBEDDING_NORMALIZE", "true") in (
            "1",
            "true",
            "True",
            "yes",
        )

        # ------------------------- ChromaDB -------------------------
        self.chroma_persist_directory: Path = Path(
            _env_str(
                "CHROMA_PERSIST_DIRECTORY",
                str(self.project_root / "data" / "chroma"),
            )
        )
        self.chroma_collection_name: str = _env_str(
            "CHROMA_COLLECTION_NAME", "rehlai_knowledge"
        )

        # ------------------------- Chunking -------------------------
        self.chunk_size: int = _env_int("CHUNK_SIZE", 300)
        self.chunk_overlap: int = _env_int("CHUNK_OVERLAP", 45)

        # ------------------------- Retrieval -------------------------
        self.top_k: int = _env_int("TOP_K", 4)
        self.dense_top_k: int = _env_int("DENSE_TOP_K", 8)
        self.sparse_top_k: int = _env_int("SPARSE_TOP_K", 8)
        # Reciprocal Rank Fusion constant and per-method weights.
        self.rrf_k: int = _env_int("RRF_K", 60)
        self.dense_weight: float = _env_float("DENSE_WEIGHT", 1.0)
        self.sparse_weight: float = _env_float("SPARSE_WEIGHT", 1.0)
        self.structured_weight: float = _env_float("STRUCTURED_WEIGHT", 1.5)
        # Minimum similarity accepted for a dense hit to be reported as
        # relevant. NOTE: this is a *ranking guardrail*, not a factual
        # confidence score. Tune against real queries (see docs).
        self.min_relevance_threshold: float = _env_float("MIN_RELEVANCE_THRESHOLD", 0.0)

        # ------------------------- Context construction -------------------------
        self.max_context_chars: int = _env_int("MAX_CONTEXT_CHARS", 6000)
        self.max_context_docs: int = _env_int("MAX_CONTEXT_DOCS", 8)

        # ------------------------- Groq generation -------------------------
        # Read from environment; never hardcoded to a specific provider key.
        self.groq_api_key_env: str = _env_str("GROQ_API_KEY_ENV", "GROQ_API_KEY")
        self.groq_api_key: Optional[str] = os.getenv(self.groq_api_key_env) or None
        # Default model chosen from Groq's public model list (checked 2026-11;
        # llama-3.3-70b-versatile was deprecated/shut down on 2026-08-16).
        # Override with GROQ_MODEL after checking
        # https://console.groq.com/docs/models.
        self.groq_model: str = _env_str("GROQ_MODEL", "openai/gpt-oss-120b")
        self.groq_temperature: float = _env_float("GROQ_TEMPERATURE", 0.2)
        self.groq_top_p: float = _env_float("GROQ_TOP_P", 0.9)
        self.groq_max_output_tokens: int = _env_int("GROQ_MAX_OUTPUT_TOKENS", 1024)
        self.groq_timeout_seconds: float = _env_float("GROQ_TIMEOUT_SECONDS", 30.0)
        self.groq_max_retries: int = _env_int("GROQ_MAX_RETRIES", 2)

        # ------------------------- Ingestion -------------------------
        self.ingest_manifest_name: str = _env_str(
            "INGEST_MANIFEST_NAME", "ingest_manifest.json"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @property
    def has_groq_key(self) -> bool:
        """True when a non-placeholder Groq API key is available."""
        key = self.groq_api_key
        if not key:
            return False
        key = key.strip()
        return bool(key) and "your_groq_api_key_here" not in key


_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Return a lazily-created shared configuration instance.

    Tests should construct their own :class:`RAGConfig` instead of mutating
    this shared one.
    """
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reset_config() -> None:
    """Drop the shared config so the next ``get_config`` re-reads the env."""
    global _config
    _config = None
