"""ChromaDB vector store wrapper (local, persistent).

Responsibilities:
  - persistent local storage of documents + embeddings + metadata
  - duplicate prevention (deterministic IDs + manifest of ingested content)
  - safe rebuild and incremental upsert
  - similarity search with optional metadata filtering
  - collection/dimension validation
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from rag.config import RAGConfig

logger = logging.getLogger(__name__)


def stable_doc_id(*parts: Any) -> str:
    """Deterministic ID so re-ingesting unchanged content is a no-op."""
    raw = "||".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class VectorStoreError(Exception):
    pass


class ChromaStore:
    """Thin, explicit wrapper around a persistent Chroma collection."""

    def __init__(self, config: RAGConfig) -> None:
        self.config = config
        self.persist_directory = Path(config.chroma_persist_directory)
        self.collection_name = config.chroma_collection_name
        self._client = None
        self._collection = None
        self._dimension: Optional[int] = None

    # ------------------------------------------------------------------
    def _get_client(self):
        if self._client is None:
            import chromadb

            key = str(self.persist_directory.resolve())
            if key not in _CLIENT_CACHE:
                self.persist_directory.mkdir(parents=True, exist_ok=True)
                _CLIENT_CACHE[key] = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
            self._client = _CLIENT_CACHE[key]
        return self._client

    def get_collection(self):
        if self._collection is not None:
            return self._collection
        client = self._get_client()
        try:
            self._collection = client.get_collection(self.collection_name)
        except Exception:
            self._collection = client.get_or_create_collection(
                self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def _invalidate(self) -> None:
        """Drop the cached collection handle (e.g. after a rebuild)."""
        self._collection = None

    # ------------------------------------------------------------------
    def count(self) -> int:
        try:
            return int(self.get_collection().count())
        except Exception:
            # stale handle (collection was rebuilt elsewhere) -> re-fetch once
            self._invalidate()
            try:
                return int(self.get_collection().count())
            except Exception:
                return 0

    def get_dimension(self) -> Optional[int]:
        return self._dimension

    def validate_dimension(self, dimension: int) -> None:
        """Ensure the collection's embedding dim matches the model's."""
        if self._dimension is not None and self._dimension != dimension:
            raise VectorStoreError(
                f"Embedding dimension mismatch: collection={self._dimension}, "
                f"model={dimension}. Rebuild the index or change EMBEDDING_MODEL."
            )

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Delete the collection (used by explicit rebuilds)."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None
        manifest = self.persist_directory / self.config.ingest_manifest_name
        if manifest.is_file():
            manifest.unlink()
        logger.info("Chroma collection '%s' reset", self.collection_name)

    # ------------------------------------------------------------------
    def upsert(
        self,
        doc_ids: List[str],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: np.ndarray,
    ) -> int:
        """Add or update documents; returns the number stored."""
        if not doc_ids:
            return 0
        if embeddings.shape[0] != len(doc_ids):
            raise VectorStoreError("Embeddings/doc_ids length mismatch")
        if self._dimension is None:
            self._dimension = int(embeddings.shape[1])
        self.validate_dimension(int(embeddings.shape[1]))

        clean_meta = []
        for m in metadatas:
            cleaned = {k: v for k, v in m.items() if isinstance(v, (str, int, float, bool))}
            if not cleaned:  # Chroma rejects empty metadata dicts
                cleaned = {"collection": "document"}
            clean_meta.append(cleaned)

        collection = self.get_collection()
        batch = 128
        stored = 0
        for start in range(0, len(doc_ids), batch):
            end = min(start + batch, len(doc_ids))
            collection.upsert(
                ids=doc_ids[start:end],
                documents=documents[start:end],
                metadatas=clean_meta[start:end],
                embeddings=embeddings[start:end].tolist(),
            )
            stored = end
        logger.info("Upserted %d documents into '%s'", stored, self.collection_name)
        return stored

    # ------------------------------------------------------------------
    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Similarity search. Returns list of dicts with content/metadata/distance."""
        collection = self.get_collection()
        if collection.count() == 0:
            return []
        query_vec = (
            query_embedding.tolist()
            if hasattr(query_embedding, "tolist")
            else list(query_embedding)
        )
        try:
            result = collection.query(
                query_embeddings=query_vec,
                n_results=max(1, min(top_k, collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            # stale collection handle (rebuilt elsewhere) -> re-fetch once
            self._invalidate()
            collection = self.get_collection()
            if collection.count() == 0:
                return []
            result = collection.query(
                query_embeddings=query_vec,
                n_results=max(1, min(top_k, collection.count())),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        hits: List[Dict[str, Any]] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else None
            # cosine distance -> cosine similarity
            similarity = 1.0 - distance if distance is not None else None
            hits.append(
                {
                    "doc_id": doc_id,
                    "content": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": distance,
                    "similarity": similarity,
                }
            )
        return hits

    # ------------------------------------------------------------------
    def get_by_ids(self, ids: List[str]) -> List[Dict[str, Any]]:
        if not ids:
            return []
        collection = self.get_collection()
        result = collection.get(ids=ids, include=["documents", "metadatas"])
        out = []
        for i, doc_id in enumerate(result.get("ids", [])):
            out.append(
                {
                    "doc_id": doc_id,
                    "content": result["documents"][i],
                    "metadata": result["metadatas"][i],
                }
            )
        return out

    # ------------------------------------------------------------------
    def save_manifest(self, fingerprints: Dict[str, str]) -> None:
        manifest = self.persist_directory / self.config.ingest_manifest_name
        manifest.write_text(json.dumps(fingerprints, ensure_ascii=False, indent=2), "utf-8")

    def load_manifest(self) -> Dict[str, str]:
        manifest = self.persist_directory / self.config.ingest_manifest_name
        if manifest.is_file():
            try:
                return json.loads(manifest.read_text("utf-8"))
            except json.JSONDecodeError:
                logger.warning("Corrupt ingest manifest ignored")
        return {}


# Process-wide cache of PersistentClients (keyed by resolved path) so that
# multiple ChromaStore instances never hold conflicting handles to the same
# local database.
_CLIENT_CACHE: Dict[str, Any] = {}


def content_fingerprint(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8")).hexdigest()
