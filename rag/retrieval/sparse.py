"""Sparse (keyword) retrieval with BM25Okapi.

The corpus is loaded from the persisted document store built by the indexer
(``data/bm25/corpus.json``). All indexing and searching run locally.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

from rag.retrieval.normalize import normalize_text, tokenize

logger = logging.getLogger(__name__)


class SparseIndex:
    """BM25 index over the KB corpus with Arabic-aware tokenization."""

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._docs: List[Dict[str, Any]] = []
        self._tokenized: List[List[str]] = []

    # ------------------------------------------------------------------
    @classmethod
    def from_corpus(cls, corpus: List[Dict[str, Any]]) -> "SparseIndex":
        index = cls()
        index.rebuild(corpus)
        return index

    @classmethod
    def load(cls, path: Path) -> "SparseIndex":
        """Load the corpus JSON previously written by the indexer."""
        path = Path(path)
        if not path.is_file():
            logger.info("No BM25 corpus at %s; index starts empty", path)
            return cls()
        data = json.loads(path.read_text("utf-8"))
        return cls.from_corpus(data)

    # ------------------------------------------------------------------
    def rebuild(self, corpus: List[Dict[str, Any]]) -> None:
        """(Re)build the BM25 index from a list of corpus documents."""
        seen = set()
        docs: List[Dict[str, Any]] = []
        for doc in corpus:
            doc_id = doc.get("doc_id")
            if not doc_id or doc_id in seen or not doc.get("content"):
                continue
            seen.add(doc_id)
            docs.append(doc)
        self._docs = docs
        self._tokenized = [tokenize(d["content"]) for d in docs]
        if docs:
            self._bm25 = BM25Okapi(self._tokenized)
        else:
            self._bm25 = None
        logger.info("BM25 index built over %d documents", len(docs))

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Keyword search. Returns doc_id/content/metadata/score hits."""
        if self._bm25 is None or not query or not query.strip():
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(range(len(self._docs)), scores), key=lambda pair: pair[1], reverse=True
        )
        hits: List[Dict[str, Any]] = []
        for idx, score in ranked[: max(0, top_k)]:
            if score <= 0:
                break
            doc = self._docs[idx]
            hits.append(
                {
                    "doc_id": doc["doc_id"],
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "score": float(score),
                }
            )
        return hits

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._docs)

    def get_doc(self, doc_id: str) -> Optional[Dict[str, Any]]:
        for doc in self._docs:
            if doc["doc_id"] == doc_id:
                return doc
        return None


def persist_corpus(documents: List[Any], path: Path) -> None:
    """Write the indexed corpus to disk so BM25 survives process restarts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = []
    for doc in documents:
        payload.append(
            {
                "doc_id": doc.doc_id,
                "content": doc.content,
                "metadata": dict(doc.metadata),
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    logger.info("Persisted BM25 corpus (%d docs) to %s", len(payload), path)
