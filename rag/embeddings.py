"""Local embedding model wrapper.

Primary model: ``intfloat/multilingual-e5-large`` (Arabic + English), executed
locally through ``sentence-transformers``. No cloud embedding API is used.

E5 formatting contract:
  - documents/chunks  ->  ``"passage: <text>"``
  - queries           ->  ``"query: <text>"``

The model is loaded once per process (singleton) and automatically falls back
from CUDA to CPU on out-of-memory errors or when no GPU is available.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional, Sequence

import numpy as np

from rag.config import RAGConfig

logger = logging.getLogger(__name__)

_E5_PREFIX_PATTERN = re.compile(r"^(query|passage):\s*", re.IGNORECASE)


def detect_best_device(preferred: str = "auto") -> str:
    """Pick the execution device: cuda if available, else cpu."""
    try:
        import torch
    except ImportError:  # pragma: no cover
        return "cpu"
    if preferred in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
    return "cpu" if preferred == "auto" else preferred


class LocalEmbedder:
    """Thread-safe local sentence-transformers embedder with CPU fallback."""

    _instance: Optional["LocalEmbedder"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str = "auto",
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.requested_device = device
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self._device: Optional[str] = None
        self._model = None
        self._lock = threading.Lock()
        self._is_e5 = "e5" in model_name.lower()

    # ------------------------------------------------------------------
    @classmethod
    def get(
        cls,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
        config: Optional[RAGConfig] = None,
    ) -> "LocalEmbedder":
        """Return a process-wide shared embedder (model loaded only once)."""
        cfg = config or RAGConfig()
        key = (
            model_name or cfg.embedding_model,
            device or cfg.embedding_device,
            batch_size or cfg.embedding_batch_size,
            cfg.embedding_normalize if normalize is None else normalize,
        )
        with cls._instance_lock:
            inst = cls._instance
            if inst is not None and inst.signature() == key:
                return inst
            new_inst = cls(
                model_name=key[0], device=key[1], batch_size=key[2], normalize=key[3]
            )
            cls._instance = new_inst
            return new_inst

    def signature(self) -> tuple:
        return (
            self.model_name,
            self.requested_device,
            self.batch_size,
            self.normalize,
        )

    # ------------------------------------------------------------------
    def _load_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            from sentence_transformers import SentenceTransformer

            device = detect_best_device(self.requested_device)
            logger.info(
                "Loading embedding model '%s' on device '%s' "
                "(first call downloads it from Hugging Face)",
                self.model_name,
                device,
            )
            try:
                self._model = SentenceTransformer(
                    self.model_name, device=device
                )
                self._device = device
            except Exception as exc:  # OOM / CUDA unavailable / download issue
                if device == "cuda":
                    logger.warning(
                        "CUDA embedding failed (%s). Falling back to CPU.", exc
                    )
                    self._model = SentenceTransformer(self.model_name, device="cpu")
                    self._device = "cpu"
                else:
                    raise
        return self._model

    # ------------------------------------------------------------------
    @property
    def device(self) -> str:
        self._load_model()
        return self._device or "cpu"

    @property
    def dimension(self) -> int:
        model = self._load_model()
        return int(model.get_sentence_embedding_dimension())

    # ------------------------------------------------------------------
    @staticmethod
    def _apply_e5_prefix(texts: Sequence[str], prefix: str) -> List[str]:
        out = []
        for t in texts:
            cleaned = _E5_PREFIX_PATTERN.sub("", str(t).strip())
            out.append(f"{prefix}: {cleaned}" if cleaned else "")
        return out

    def _encode(self, texts: Sequence[str], prefix: str) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._load_model()
        prepared = (
            self._apply_e5_prefix(texts, prefix) if self._is_e5 else [str(t) for t in texts]
        )
        try:
            vectors = model.encode(
                prepared,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(vectors, dtype=np.float32)
        except Exception as exc:  # e.g. CUDA OOM mid-batch
            if self._device == "cuda":
                logger.warning("Embedding on CUDA failed (%s); retrying on CPU", exc)
                try:
                    model = model.to("cpu")
                    self._device = "cpu"
                    vectors = model.encode(
                        prepared,
                        batch_size=max(1, self.batch_size // 2),
                        normalize_embeddings=self.normalize,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                    )
                    return np.asarray(vectors, dtype=np.float32)
                except Exception:
                    raise
            raise

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed knowledge-base chunks with the e5 ``passage`` prefix."""
        return self._encode(texts, "passage")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a user query with the e5 ``query`` prefix."""
        return self._encode([text], "query")

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, "query")


def get_embedder(config: Optional[RAGConfig] = None) -> LocalEmbedder:
    """Convenience accessor for the shared embedder."""
    cfg = config or RAGConfig()
    return LocalEmbedder.get(config=cfg)
