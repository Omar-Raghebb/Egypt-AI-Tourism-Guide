"""Dense (semantic) retrieval over the local ChromaDB collection."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from rag.config import RAGConfig
from rag.schemas import RetrievedDocument, SourceMetadata, SourceType, VerificationMetadata, VerificationStatus
from rag.vector_store import ChromaStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Semantic search using locally generated query embeddings."""

    def __init__(
        self,
        store: ChromaStore,
        embedder,
        config: Optional[RAGConfig] = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.config = config or RAGConfig()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedDocument]:
        """Embed the query locally and search ChromaDB.

        ``filters`` accepts a simple key/value mapping (e.g. ``{"city": "Giza"}``)
        which is translated into a Chroma ``where`` clause. Unknown keys are
        ignored with a warning rather than failing the query.
        """
        if not query or not query.strip():
            return []
        try:
            embedding = self.embedder.embed_query(query)
        except Exception:
            logger.exception("Query embedding failed")
            return []

        where = self._build_where(filters)
        try:
            hits = self.store.query(
                query_embedding=embedding,
                top_k=top_k or self.config.top_k,
                where=where,
            )
        except Exception:
            logger.exception("Chroma query failed")
            return []

        documents: List[RetrievedDocument] = []
        for rank, hit in enumerate(hits, start=1):
            meta = hit.get("metadata") or {}
            similarity = hit.get("similarity")
            if similarity is not None and (
                np.isnan(similarity) or similarity < self.config.min_relevance_threshold
            ):
                continue
            documents.append(
                RetrievedDocument(
                    doc_id=hit["doc_id"],
                    content=hit.get("content", ""),
                    place_id=meta.get("place_id"),
                    collection=str(meta.get("collection", "document")),
                    section=meta.get("section"),
                    metadata=dict(meta),
                    score=similarity,
                    distance=hit.get("distance"),
                    rank=rank,
                    source_type=SourceType.DENSE,
                    verification=VerificationMetadata(
                        status=self._status_from(meta.get("verification_status")),
                        last_verified=meta.get("last_verified"),
                        notes=meta.get("kb_disclaimer"),
                    ),
                    source=SourceMetadata(
                        source_file=meta.get("source_file"),
                        kb_version=meta.get("kb_version"),
                        kb_last_updated=meta.get("kb_last_updated"),
                    ),
                )
            )
        return documents

    @staticmethod
    def _status_from(value: Optional[str]) -> VerificationStatus:
        try:
            return VerificationStatus(value) if value else VerificationStatus.UNVERIFIED
        except ValueError:
            return VerificationStatus.UNVERIFIED

    @staticmethod
    def _build_where(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not filters:
            return None
        allowed = {}
        for key, value in filters.items():
            if value in (None, "", [], {}):
                continue
            if isinstance(value, (str, int, float, bool)):
                allowed[key] = value
            elif isinstance(value, list):
                allowed[key] = {"$in": value}
        if not allowed:
            return None
        if len(allowed) == 1:
            return next(iter(allowed.values())) if isinstance(
                next(iter(allowed.values())), dict
            ) else allowed
        return {"$and": [{k: v} for k, v in allowed.items()]}
