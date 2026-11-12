"""Tests for local embeddings, ChromaDB ingestion, dense + sparse retrieval,
structured retrieval, and the indexer (with model mocking where practical).
"""

import numpy as np
import pytest

from rag.embeddings import LocalEmbedder, detect_best_device
from rag.ingestion.indexer import KBIndexer
from rag.ingestion.loader import load_kb
from rag.query_processing import QueryProcessor
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.normalize import normalize_arabic, tokenize
from rag.retrieval.sparse import SparseIndex
from rag.retrieval.structured import StructuredRetriever
from rag.vector_store import ChromaStore, content_fingerprint

pytestmark = pytest.mark.model  # requires the small test embedding model download


# ----------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------
class TestEmbeddings:
    def test_device_detection(self):
        device = detect_best_device("auto")
        assert device in ("cpu", "cuda")

    def test_embed_query_and_document(self):
        embedder = LocalEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        vec = embedder.embed_query("What is the ticket price of the pyramids?")
        assert vec.shape == (1, 384)
        doc_vecs = embedder.embed_documents(["passage text one", "passage text two"])
        assert doc_vecs.shape == (2, 384)
        # normalized vectors
        assert np.allclose(np.linalg.norm(vec[0]), 1.0, atol=1e-3)

    def test_model_loaded_once(self):
        e1 = LocalEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        m1 = e1._load_model()
        m2 = e1._load_model()
        assert m1 is m2


# ----------------------------------------------------------------------
# Normalization / tokenization
# ----------------------------------------------------------------------
class TestNormalization:
    def test_arabic_normalization(self):
        assert normalize_arabic("أَهْرَامَاتُ") == "اهرامات"
        assert normalize_arabic("القاهرة") == "القاهره"
        assert normalize_arabic("مُتاح") == "متاح"

    def test_tokenize_bilingual(self):
        tokens = tokenize("Pyramids of Giza أهرامات الجيزة!")
        assert "pyramids" in tokens and "giza" in tokens
        assert "اهرامات" in tokens and "الجيزه" in tokens
        assert "!" not in tokens


# ----------------------------------------------------------------------
# ChromaDB
# ----------------------------------------------------------------------
class TestChromaStore:
    def test_upsert_and_count(self, test_config, mini_kb):
        store = ChromaStore(test_config)
        vectors = np.random.rand(3, 4).astype("float32")
        store.upsert(
            doc_ids=["a", "b", "c"],
            documents=["doc a", "doc b", "doc c"],
            metadatas=[{"place_id": "p1"}, {"place_id": "p1"}, {"place_id": "p2"}],
            embeddings=vectors,
        )
        assert store.count() == 3

    def test_query_returns_hits(self, test_config):
        store = ChromaStore(test_config)
        # make one document clearly closest to the unit query vector
        store.upsert(
            doc_ids=["near", "far"],
            documents=["near doc", "far doc"],
            metadatas=[{"place_id": "p1"}, {"place_id": "p2"}],
            embeddings=np.array(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype="float32"
            ),
        )
        hits = store.query(np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"), top_k=2)
        assert hits[0]["doc_id"] == "near"
        assert hits[0]["similarity"] is not None

    def test_dimension_mismatch_raises(self, test_config):
        store = ChromaStore(test_config)
        store.upsert(
            doc_ids=["a"], documents=["a"], metadatas=[{}],
            embeddings=np.zeros((1, 4), dtype="float32"),
        )
        from rag.vector_store import VectorStoreError

        with pytest.raises(VectorStoreError):
            store.upsert(
                doc_ids=["b"], documents=["b"], metadatas=[{}],
                embeddings=np.zeros((1, 7), dtype="float32"),
            )

    def test_reset(self, test_config):
        store = ChromaStore(test_config)
        store.upsert(
            doc_ids=["a"], documents=["a"], metadatas=[{}],
            embeddings=np.zeros((1, 4), dtype="float32"),
        )
        store.reset()
        assert store.count() == 0


# ----------------------------------------------------------------------
# Indexer (real embedding model, small KB)
# ----------------------------------------------------------------------
class TestIndexer:
    def test_build_and_retrieve(self, test_config, mini_kb):
        embedder = LocalEmbedder(
            model_name=test_config.embedding_model, device="cpu"
        )
        indexer = KBIndexer(test_config, embedder=embedder)
        stats = indexer.build(mini_kb, rebuild=True)
        assert stats.total_docs == stats.embedded_docs
        assert stats.chroma_docs == stats.total_docs

        store = ChromaStore(test_config)
        dense = DenseRetriever(store, embedder, test_config)
        docs = dense.retrieve("ticket price for the Great Pyramid", top_k=3)
        assert docs, "dense retrieval should return hits"
        assert all(d.doc_id and d.content for d in docs)
        assert all(d.rank == i + 1 for i, d in enumerate(docs))
        assert all(d.source_type.value == "dense_retrieval" for d in docs)

    def test_incremental_skips_unchanged(self, test_config, mini_kb):
        embedder = LocalEmbedder(
            model_name=test_config.embedding_model, device="cpu"
        )
        indexer = KBIndexer(test_config, embedder=embedder)
        indexer.build(mini_kb, rebuild=True)
        stats2 = indexer.build(mini_kb, rebuild=False)
        assert stats2.embedded_docs == 0
        assert stats2.skipped_unchanged == stats2.total_docs

    def test_metadata_filtering(self, test_config, mini_kb):
        embedder = LocalEmbedder(
            model_name=test_config.embedding_model, device="cpu"
        )
        indexer = KBIndexer(test_config, embedder=embedder)
        indexer.build(mini_kb, rebuild=True)
        store = ChromaStore(test_config)
        dense = DenseRetriever(store, embedder, test_config)
        docs = dense.retrieve("ancient temple", top_k=5, filters={"place_id": "luxor_temple"})
        assert docs
        assert all(d.metadata["place_id"] == "luxor_temple" for d in docs)


# ----------------------------------------------------------------------
# BM25 sparse retrieval
# ----------------------------------------------------------------------
class TestSparseIndex:
    def _corpus(self, chunker, mini_kb):
        return [
            {"doc_id": d.doc_id, "content": d.content, "metadata": dict(d.metadata)}
            for d in chunker.chunk_kb(mini_kb)
        ]

    def test_bilingual_keyword_match(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        hits = index.search("ticket price pyramids", top_k=5)
        assert hits
        assert any("giza_pyramids" in h["doc_id"] for h in hits)
        assert hits[0]["score"] > 0

    def test_arabic_query(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        # "أهرامات الجيزة" appears in the KB (name_arabic); BM25 finds it
        # exactly after Arabic normalization.
        hits = index.search("أهرامات الجيزة", top_k=5)
        assert hits
        assert any("giza_pyramids" in h["doc_id"] for h in hits)

    def test_arabic_no_match_is_empty_not_error(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        # tokens that do not occur anywhere in the corpus -> no hits, no crash
        assert index.search("سعر تذكرة الهرم", top_k=5) == []

    def test_exact_name_match(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        hits = index.search("Luxor Temple", top_k=5)
        assert hits
        assert any("luxor_temple" in h["doc_id"] for h in hits)

    def test_no_match_returns_empty(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        assert index.search("zzzq xqzz", top_k=5) == []

    def test_empty_query(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        assert index.search("", top_k=5) == []

    def test_metadata_preserved(self, chunker, mini_kb):
        index = SparseIndex.from_corpus(self._corpus(chunker, mini_kb))
        hits = index.search("ticket price pyramids", top_k=3)
        for hit in hits:
            assert "place_id" in hit["metadata"]
            assert "verification_status" in hit["metadata"]


# ----------------------------------------------------------------------
# Structured retrieval
# ----------------------------------------------------------------------
class TestStructuredRetriever:
    def test_find_by_id(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        place = sr.find_by_id("giza_pyramids")
        assert place and place.place_id == "giza_pyramids"

    def test_find_by_name_english(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        assert sr.find_by_name("Pyramids of Giza").place_id == "giza_pyramids"
        assert sr.find_by_name("pyramids of giza").place_id == "giza_pyramids"

    def test_find_by_name_arabic(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        assert sr.find_by_name("أهرامات الجيزة").place_id == "giza_pyramids"
        assert sr.find_by_name("معبد الأقصر").place_id == "luxor_temple"

    def test_find_by_alternate_name(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        # "Giza Necropolis" is an alternate name of giza_pyramids
        assert sr.find_by_name("Giza Necropolis").place_id == "giza_pyramids"

    def test_find_by_city(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        luxor_places = sr.find_places_by_city("Luxor")
        assert {p.place_id for p in luxor_places} == {"luxor_temple"}

    def test_find_by_category(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        temples = sr.find_places_by_category("temple")
        assert "luxor_temple" in {p.place_id for p in temples}

    def test_exact_price_lookup(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        lookup = sr.get_field("giza_pyramids", "ticket_price")
        assert lookup.status.value == "verified"
        assert lookup.value["general_plateau_entry_foreign_adult"] == 700
        assert lookup.value["currency"] == "EGP"

    def test_exact_hours_lookup(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        lookup = sr.get_field("giza_pyramids", "opening_hours")
        assert "07:00" in lookup.value["summer_may_to_september"]

    def test_coordinates_lookup(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        lookup = sr.get_field("giza_pyramids", "location.coordinates")
        assert lookup.value["latitude"] == 29.9792

    def test_missing_vs_unavailable(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        missing = sr.get_field("giza_pyramids", "this_field_does_not_exist")
        assert missing.status.value == "missing"
        # unknown place id
        unknown = sr.get_field("nonexistent", "ticket_price")
        assert unknown.status.value == "missing"

    def test_no_guessing(self, mini_kb):
        """Structured lookup must never invent values."""
        sr = StructuredRetriever(mini_kb)
        lookup = sr.get_field("giza_pyramids", "age_suitability")
        assert lookup.value is None
        assert lookup.status in ("missing", "unavailable")

    def test_verification_preserved(self, mini_kb):
        sr = StructuredRetriever(mini_kb)
        place = sr.find_by_id("giza_pyramids")
        assert place.verification.last_verified == "2026-07"
        assert place.source.source_file == "giza_pyramids/info.json"
