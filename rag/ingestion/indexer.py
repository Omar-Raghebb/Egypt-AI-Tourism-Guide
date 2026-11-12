"""Indexing orchestrator: knowledge base -> embeddings + ChromaDB + BM25 corpus.

Features:
  - duplicate prevention via deterministic doc IDs and content fingerprints
  - incremental updates: unchanged documents are not re-embedded
  - safe full rebuild behind an explicit ``rebuild=True``
  - the original JSON files are only ever read, never written
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.ingestion.chunker import KBChunker
from rag.ingestion.loader import LoadedKB, load_kb
from rag.retrieval.sparse import SparseIndex, persist_corpus
from rag.vector_store import ChromaStore, content_fingerprint

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    total_docs: int = 0
    embedded_docs: int = 0
    skipped_unchanged: int = 0
    chroma_docs: int = 0
    elapsed_s: float = 0.0


class KBIndexer:
    """Builds and maintains the local retrieval indexes."""

    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        embedder: Optional[LocalEmbedder] = None,
    ) -> None:
        self.config = config or RAGConfig()
        self.embedder = embedder
        self.store = ChromaStore(self.config)
        self.bm25_path = Path(self.config.chroma_persist_directory) / "bm25_corpus.json"

    # ------------------------------------------------------------------
    def build(self, kb: Optional[LoadedKB] = None, rebuild: bool = False) -> IngestStats:
        """Index the knowledge base into ChromaDB and the BM25 corpus.

        ``rebuild=True`` wipes the existing collection first; otherwise the
        run is incremental — documents whose content fingerprint already
        appears in the manifest are skipped without re-embedding.
        """
        started = time.time()
        if kb is None:
            kb = load_kb(self.config.kb_path, self.config.kb_single_file)

        chunker = KBChunker(self.config.chunk_size, self.config.chunk_overlap)
        documents = chunker.chunk_kb(kb)

        if rebuild:
            self.store.reset()
            self.store.save_manifest({})
            fingerprints: Dict[str, str] = {}
        else:
            fingerprints = self.store.load_manifest()
        chroma_count = self.store.count() if not rebuild else 0

        to_embed: List[int] = []
        for i, doc in enumerate(documents):
            fp = content_fingerprint(doc.content)
            previous = fingerprints.get(doc.doc_id)
            if previous == fp and chroma_count > 0:
                fingerprints[doc.doc_id] = fp  # unchanged, keep as-is
                continue
            fingerprints[doc.doc_id] = fp
            to_embed.append(i)

        stats = IngestStats(total_docs=len(documents))
        stats.skipped_unchanged = len(documents) - len(to_embed)

        if to_embed:
            if self.embedder is None:
                from rag.embeddings import get_embedder

                self.embedder = get_embedder(self.config)
            texts = [documents[i].content for i in to_embed]
            logger.info("Embedding %d / %d documents", len(texts), len(documents))
            vectors = self.embedder.embed_documents(texts)
            self.store.upsert(
                doc_ids=[documents[i].doc_id for i in to_embed],
                documents=texts,
                metadatas=[documents[i].metadata for i in to_embed],
                embeddings=vectors,
            )
            stats.embedded_docs = len(texts)
        elif rebuild and documents:
            # Rebuild with an empty collection still requires embedding.
            logger.warning(
                "Rebuild requested but no new documents embedded; "
                "collection may be stale"
            )

        self.store.save_manifest(fingerprints)
        persist_corpus(documents, self.bm25_path)
        stats.chroma_docs = self.store.count()
        stats.elapsed_s = time.time() - started
        logger.info(
            "Indexing done: %d docs (%d embedded, %d unchanged) in %.1fs",
            stats.total_docs,
            stats.embedded_docs,
            stats.skipped_unchanged,
            stats.elapsed_s,
        )
        return stats

    # ------------------------------------------------------------------
    def load_sparse_index(self) -> SparseIndex:
        """Load the BM25 index from the persisted corpus."""
        return SparseIndex.load(self.bm25_path)
