"""RAGPipeline — the stable entry point of the RehlaAI RAG system.

Public interface (consumed by the API layer and the Agentic AI layer):

    pipeline.retrieve(query, filters=None, top_k=4)  -> RetrievalResult
    pipeline.answer(query, language=None, filters=None) -> RAGResponse

All retrieval runs locally (embeddings, ChromaDB, BM25, structured lookups);
Groq is used exclusively for generation and degrades gracefully to a
rule-based evidence response when unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from rag.config import RAGConfig
from rag.context_builder import ContextBuilder
from rag.embeddings import get_embedder
from rag.generation.fallback import (
    FallbackResponder,
    _NO_INFO_AR,
    _NO_INFO_EN,
    asserts_missing_info,
)
from rag.generation.groq_client import GroqClient
from rag.generation.prompts import build_messages
from rag.generation.validator import OUTCOME_PARTIAL
from rag.ingestion.indexer import KBIndexer
from rag.ingestion.loader import LoadedKB, load_kb
from rag.query_processing import QueryProcessor, detect_language
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.sparse import SparseIndex
from rag.retrieval.structured import StructuredRetriever
from rag.schemas import RAGResponse, RetrievedContext, RetrievalResult

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Facade that wires ingestion, retrieval, context, and generation."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        kb: Optional[LoadedKB] = None,
        groq_client: Optional[GroqClient] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.kb = kb if kb is not None else load_kb(
            self.config.kb_path, self.config.kb_single_file
        )
        self.structured = StructuredRetriever(self.kb)
        self.query_processor = QueryProcessor(self.kb, self.structured)
        self.context_builder = ContextBuilder(self.structured, self.config)
        self.groq = groq_client or GroqClient(self.config)
        self.fallback = FallbackResponder(self.structured)

        # retrieval components (built lazily so they can be injected in tests)
        self.store = None
        self.embedder = None
        self.sparse: Optional[SparseIndex] = None
        self.dense: Optional[DenseRetriever] = None
        self.hybrid: Optional[HybridRetriever] = None

    # ------------------------------------------------------------------
    # Initialization / index management
    # ------------------------------------------------------------------
    def initialize(
        self,
        rebuild: bool = False,
        load_sparse: bool = True,
    ) -> Optional[Any]:
        """Load (or lazily build) the local indexes.

        Returns ingest stats when a build was required; ``None`` when the
        persisted index already exists.
        """
        from rag.vector_store import ChromaStore

        if self.store is None:
            self.store = ChromaStore(self.config)
        stats = None
        if self.store.count() == 0:
            indexer = KBIndexer(self.config)
            stats = indexer.build(self.kb, rebuild=True)
            # reuse the freshly built store to avoid stale collection handles
            self.store = indexer.store
        if self.embedder is None:
            self.embedder = get_embedder(self.config)
        if self.dense is None:
            self.dense = DenseRetriever(self.store, self.embedder, self.config)
        if load_sparse and self.sparse is None:
            self.sparse = SparseIndex.load(
                Path_bm25(self.config)
            )
        if self.sparse is None:
            self.sparse = SparseIndex()
        if self.hybrid is None:
            self.hybrid = HybridRetriever(
                self.dense, self.sparse, self.structured,
                self.query_processor, self.config,
            )
        return stats

    def rebuild_index(self, load_sparse: bool = True) -> Any:
        """Force a full rebuild of ChromaDB + BM25."""
        self.store = ChromaStore(self.config) if self.store is None else self.store
        indexer = KBIndexer(self.config)
        self.embedder = self.embedder or get_embedder(self.config)
        stats = indexer.build(self.kb, rebuild=True)
        self.store = indexer.store
        self.dense = DenseRetriever(self.store, self.embedder, self.config)
        if load_sparse:
            self.sparse = indexer.load_sparse_index()
        self.hybrid = HybridRetriever(
            self.dense, self.sparse, self.structured,
            self.query_processor, self.config,
        )
        return stats

    # ------------------------------------------------------------------
    # Retrieval interface
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 4,
    ) -> RetrievalResult:
        """Hybrid retrieval (dense + sparse + structured, RRF-fused).

        Returns structured evidence — documents, place ids, verification
        statuses, warnings — never plain text only.
        """
        self.initialize()
        started = time.time()
        result = self.hybrid.retrieve(query, filters=filters, top_k=top_k)
        result.latency_ms = (time.time() - started) * 1000.0
        return result

    def get_context(self, query: str, **kwargs) -> RetrievedContext:
        """Convenience: retrieve + build the LLM-ready context block."""
        result = self.retrieve(query, **kwargs)
        return self.context_builder.build(result)

    # ------------------------------------------------------------------
    # Generation interface
    # ------------------------------------------------------------------
    def answer(
        self,
        query: str,
        language: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResponse:
        """End-to-end: retrieve locally, then generate with Groq.

        Falls back to deterministic template-based generation on provider
        failures. Answers are validated with graded outcomes — a partially
        supported answer is qualified, not discarded; only empty/contradictory
        answers are replaced by the fallback.
        """
        self.initialize()
        started = time.time()
        retrieval = self.retrieve(query, filters=filters, top_k=top_k or self.config.top_k)
        context = self.context_builder.build(retrieval)
        lang = language or detect_language(query)

        # The deterministic responder short-circuits to a "not found" message
        # exactly when the requested information does not exist in the KB.
        # That decision is generation-mode independent: whether the final text
        # comes from Groq or the fallback, a missing-info request must report
        # status "unavailable" (never "success" with empty sources).
        fallback_probe = self.fallback.respond(context, lang, retrieval=retrieval)
        request_unsatisfiable = fallback_probe.strip() in (_NO_INFO_EN, _NO_INFO_AR)

        warnings: List[str] = []
        status = "success"
        generated_by = "groq"

        if not self.groq.is_available:
            answer_text = fallback_probe
            generated_by = "fallback"
            status = "unavailable" if request_unsatisfiable else "success"
            return self._finalize(
                query, answer_text, lang, generated_by, status,
                warnings, retrieval, context, started,
            )

        try:
            messages = build_messages(context, query, lang)
            answer_text = self.groq.generate(messages)
        except Exception as exc:  # noqa: BLE001 - any generation failure falls back
            logger.error("Generation failed (%s); using fallback", exc)
            answer_text = fallback_probe
            status = "unavailable" if request_unsatisfiable else "success"
            return self._finalize(
                query, answer_text, lang, "fallback", status, warnings,
                retrieval, context, started, error=str(exc),
            )

        # graded validation — qualify, don't discard
        validation = self.fallback.validator.validate(answer_text, context, retrieval, lang)
        warnings.extend(validation.warnings)
        if not validation.is_usable:
            # empty, leaky or fabricated-key-facts answer -> replace with fallback
            logger.warning(
                "Generated answer rejected (%s); using fallback", validation.outcome
            )
            answer_text = fallback_probe
            generated_by = "fallback"
            if validation.warnings and any(
                w.startswith("unsupported_numbers") for w in validation.warnings
            ):
                status = "partial"
        elif validation.outcome == OUTCOME_PARTIAL:
            status = "partial"

        if status == "success" and asserts_missing_info(answer_text):
            # the generated answer honestly reports that the requested
            # information is not in the knowledge base — label the request
            # unavailable (missing-info contract is generation-mode
            # independent). The data-free guard inside asserts_missing_info
            # keeps genuine partial answers (price known, hours missing)
            # labelled partial/success instead.
            status = "unavailable"

        return self._finalize(
            query, answer_text, lang, generated_by, status,
            warnings, retrieval, context, started,
        )

    def _finalize(
        self,
        query: str,
        answer_text: str,
        lang: str,
        generated_by: str,
        status: str,
        warnings: List[str],
        retrieval: RetrievalResult,
        context: RetrievedContext,
        started: float,
        error: Optional[str] = None,
    ) -> RAGResponse:
        _ = started  # latency is measured in retrieve(); kept for signature stability
        guardrail_codes: List[str] = []
        if generated_by == "groq":
            # numeric/grounding guardrails are already applied via the validator
            # above; keep the legacy warning list for internal diagnostics only
            guardrail_codes = [
                w for w in warnings if w.startswith(("unsupported_numbers", "no_evidence"))
            ]
        return RAGResponse(
            query=query,
            answer=answer_text,
            language=lang,
            generated_by=generated_by,
            status=status,
            warnings=list(dict.fromkeys(warnings)),
            context=context,
            retrieval=retrieval,
            sources=context.sources,
            verification_flags=guardrail_codes,
            error=error,
        )

    # ------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        """Quick status report for /rag/health."""
        try:
            chroma_count = self.store.count() if self.store else None
        except Exception:
            chroma_count = None
        return {
            "kb_loaded": self.kb.num_places > 0,
            "kb_name": self.kb.kb_name,
            "kb_places": self.kb.num_places,
            "kb_faqs": self.kb.num_faqs,
            "kb_cities": len(getattr(self.kb, "cities", [])),
            "chroma_docs": chroma_count,
            "bm25_docs": len(self.sparse) if self.sparse else None,
            "embedding_model": self.config.embedding_model,
            "embedding_device": self.config.embedding_device,
            "groq_configured": self.groq.is_available,
            "groq_model": self.config.groq_model,
        }


def Path_bm25(config: RAGConfig):
    """Helper kept for clarity: path of the persisted BM25 corpus."""
    from pathlib import Path

    return Path(config.chroma_persist_directory) / "bm25_corpus.json"
