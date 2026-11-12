"""Tests for query processing, hybrid retrieval + RRF, and context building."""

import pytest

from rag.context_builder import ContextBuilder
from rag.embeddings import LocalEmbedder
from rag.ingestion.indexer import KBIndexer
from rag.ingestion.loader import load_kb
from rag.query_processing import ProcessedQuery, QueryProcessor, detect_language, normalize_query
from rag.retrieval.dense import DenseRetriever
from rag.retrieval.hybrid import HybridRetriever
from rag.retrieval.sparse import SparseIndex
from rag.retrieval.structured import StructuredRetriever
from rag.vector_store import ChromaStore


@pytest.fixture(scope="module")
def built(test_config_session):
    """A fully built pipeline over the session mini KB (small model)."""
    config = test_config_session
    embedder = LocalEmbedder(model_name=config.embedding_model, device="cpu")
    indexer = KBIndexer(config, embedder=embedder)
    kb = load_kb(config.kb_path, config.kb_single_file)
    indexer.build(kb, rebuild=True)
    store = ChromaStore(config)
    dense = DenseRetriever(store, embedder, config)
    sparse = indexer.load_sparse_index()
    structured = StructuredRetriever(kb)
    qp = QueryProcessor(kb, structured)
    hybrid = HybridRetriever(dense, sparse, structured, qp, config)
    context_builder = ContextBuilder(structured, config)
    return {
        "config": config, "kb": kb, "structured": structured,
        "qp": qp, "hybrid": hybrid, "context_builder": context_builder,
    }


@pytest.fixture(scope="module")
def test_config_session(mini_kb_dir, tmp_path_factory):
    from rag.config import RAGConfig

    config = RAGConfig()
    config.kb_path = mini_kb_dir
    config.kb_single_file = None
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_hybrid")
    config.chroma_collection_name = "test_hybrid"
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.max_context_chars = 4000
    config.max_context_docs = 8
    return config


# ----------------------------------------------------------------------
# Query processing
# ----------------------------------------------------------------------
class TestQueryProcessing:
    def test_price_intent_english(self, built):
        pq = built["qp"].process("What is the ticket price of the Egyptian Museum?")
        assert pq.intent == "price"
        assert "prices" in pq.structured_fields

    def test_price_intent_arabic(self, built):
        pq = built["qp"].process("سعر تذكرة الأهرامات؟")
        assert pq.intent == "price"
        assert pq.language == "ar"

    def test_hours_intent_bilingual(self, built):
        assert built["qp"].process("What are the opening hours of the Pyramids?").intent == "hours"
        assert built["qp"].process("مواعيد فتح معبد الكرنك؟").intent == "hours"

    def test_place_detection_english(self, built):
        pq = built["qp"].process("What is the ticket price of the Pyramids of Giza?")
        assert "giza_pyramids" in pq.detected_places

    def test_place_detection_arabic(self, built):
        pq = built["qp"].process("ما هي مواعيد معبد الأقصر؟")
        assert "luxor_temple" in pq.detected_places

    def test_city_detection(self, built):
        pq = built["qp"].process("What are the best places to visit in Giza?")
        assert pq.detected_city == "Giza"

    def test_arabic_city_alias(self, built):
        pq = built["qp"].process("ما هي الأماكن في الأسوان؟")
        assert pq.detected_city == "Aswan"

    def test_not_hardcoded_examples(self, built):
        """Novel phrasings must still route."""
        pq = built["qp"].process("How much would entry fees be for a school trip?")
        assert pq.intent == "price"
        pq2 = built["qp"].process("Is the site reachable by public transport?")
        assert pq2.intent == "transportation"
        assert "nearest_airport" in pq2.structured_fields or pq2.structured_fields

    def test_semantic_default(self, built):
        pq = built["qp"].process("Tell me about the ancient wonder of the world")
        assert pq.intent in ("semantic", "recommendation", "definition")

    def test_language_detection(self):
        assert detect_language("What is the price?") == "en"
        assert detect_language("ما هو السعر؟") == "ar"

    def test_normalization(self):
        assert normalize_query("أَهْرَامَات") == "اهرامات"


# ----------------------------------------------------------------------
# Hybrid retrieval / RRF
# ----------------------------------------------------------------------
class TestHybridRetrieval:
    def test_basic_retrieval(self, built):
        result = built["hybrid"].retrieve("What is the ticket price of the Pyramids of Giza?")
        assert result.documents
        assert result.structured_records
        assert any(p.place_id == "giza_pyramids" for p in result.structured_records)
        assert result.intent == "price"
        assert set(result.methods_used) & {"dense", "sparse", "structured"}

    def test_rrf_ranks_dedup(self, built):
        result = built["hybrid"].retrieve("Great Pyramid of Khufu history")
        doc_ids = [d.doc_id for d in result.documents]
        assert len(doc_ids) == len(set(doc_ids)), "documents must be deduplicated"
        ranks = [d.rank for d in result.documents]
        assert ranks == sorted(ranks), "documents must be sorted by RRF score"
        assert all(d.rank is not None and d.score is not None for d in result.documents)

    def test_arabic_query_retrieval(self, built):
        result = built["hybrid"].retrieve("سعر تذكرة أهرامات الجيزة")
        assert result.documents or result.structured_records
        assert any(p.place_id == "giza_pyramids" for p in result.structured_records)

    def test_structured_records_not_discarded(self, built):
        """Exact structured results survive even when similarity is low."""
        result = built["hybrid"].retrieve("ساعات فتح الأقصر معبد الأقصر بالتحديد")
        assert result.structured_records
        assert any(p.place_id == "luxor_temple" for p in result.structured_records)

    def test_city_query(self, built):
        result = built["hybrid"].retrieve("What are the best places to visit in Luxor?")
        assert any(p.place_id == "luxor_temple" for p in result.structured_records)

    def test_irrelevant_query_warns(self, built):
        result = built["hybrid"].retrieve("how to cook italian pasta")
        assert result.warnings  # something like no_relevant_results
        assert result.is_empty or all(
            d.score is not None for d in result.documents
        )

    def test_empty_query(self, built):
        result = built["hybrid"].retrieve("")
        assert result.is_empty

    def test_filters_applied(self, built):
        result = built["hybrid"].retrieve(
            "ancient temple", filters={"place_id": "luxor_temple"}
        )
        for doc in result.documents:
            assert doc.metadata.get("place_id") == "luxor_temple"

    def test_verification_metadata_propagated(self, built):
        result = built["hybrid"].retrieve("ticket price of Pyramids of Giza")
        giza_docs = [d for d in result.documents if d.place_id == "giza_pyramids"]
        assert giza_docs
        statuses = {d.verification.status.value for d in giza_docs}
        assert statuses  # verification metadata present on matched docs

    def test_top_k_respected(self, built):
        result = built["hybrid"].retrieve("pyramids", top_k=2)
        assert len(result.documents) <= 2


# ----------------------------------------------------------------------
# Context construction
# ----------------------------------------------------------------------
class TestContextBuilder:
    def test_context_contains_evidence_and_attribution(self, built):
        result = built["hybrid"].retrieve("What is the ticket price of the Pyramids of Giza?")
        context = built["context_builder"].build(result)
        assert context.text
        assert "place_id" in context.text
        assert context.sources
        assert all(":" in s for s in context.sources)

    def test_exact_values_preserved(self, built):
        result = built["hybrid"].retrieve("ticket price giza pyramids foreign adult")
        context = built["context_builder"].build(result)
        # the exact numeric price appears in the context
        assert "700" in context.text

    def test_verification_flags_collected(self, built):
        result = built["hybrid"].retrieve("What are the opening hours of Luxor Temple?")
        context = built["context_builder"].build(result)
        assert isinstance(context.missing_info, list)
        assert isinstance(context.unverified_info, list)
        assert isinstance(context.estimated_info, list)

    def test_deduplication(self, built):
        result = built["hybrid"].retrieve("ticket price giza pyramids")
        context = built["context_builder"].build(result)
        doc_ids = [d.doc_id for d in context.documents]
        assert len(doc_ids) == len(set(doc_ids))
        contents = [d.content for d in context.documents]
        assert len(contents) == len(set(contents))

    def test_size_limit_respected(self, built):
        result = built["hybrid"].retrieve("pyramids giza luxor temple history facts")
        built["context_builder"].config.max_context_chars = 1500
        context = built["context_builder"].build(result)
        assert len(context.text) <= 1500
        built["context_builder"].config.max_context_chars = 4000

    def test_source_never_removed(self, built):
        result = built["hybrid"].retrieve("pyramids history")
        built["context_builder"].config.max_context_chars = 1200
        context = built["context_builder"].build(result)
        # even when truncated, evidence keeps its source line
        if context.documents:
            assert "source:" in context.text
        built["context_builder"].config.max_context_chars = 4000
