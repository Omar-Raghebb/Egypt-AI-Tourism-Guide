"""Integration tests: full pipeline with mocked Groq + FastAPI endpoints."""

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.ingestion.indexer import KBIndexer
from rag.ingestion.loader import load_kb
from rag.pipeline import RAGPipeline
from rag.generation.groq_client import GroqClient
from rag.api import create_rag_router


@pytest.fixture(scope="module")
def pipeline_module(test_config_module):
    """A built pipeline wired to a mock Groq client."""
    config = test_config_module
    kb = load_kb(config.kb_path, config.kb_single_file)
    embedder = LocalEmbedder(model_name=config.embedding_model, device="cpu")
    # patch the embedder factory so initialize() reuses the small test model
    import rag.embeddings as embeddings_module

    original_get = embeddings_module.get_embedder
    embeddings_module.get_embedder = lambda cfg=None, config=config: embedder
    try:
        pipeline = RAGPipeline(config=config, kb=kb, groq_client=_mock_groq())
        pipeline.initialize()
        yield pipeline
    finally:
        embeddings_module.get_embedder = original_get


def _mock_groq():
    class MockGroq:
        is_available = True

        def generate(self, messages, **kwargs):
            return (
                "Based on the retrieved evidence, here is the requested information "
                "[giza_pyramids]. (Mocked Groq response for testing.)"
            )

    return MockGroq()


@pytest.fixture(scope="module")
def test_config_module(mini_kb_dir, tmp_path_factory):
    config = RAGConfig()
    config.kb_path = mini_kb_dir
    config.kb_single_file = None
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_pipe")
    config.chroma_collection_name = "test_pipeline"
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    return config


class TestPipelineIntegration:
    def test_retrieve_interface(self, pipeline_module):
        result = pipeline_module.retrieve("What is the ticket price of the Pyramids?")
        assert result.documents or result.structured_records
        assert result.intent == "price"
        # structured evidence carries place ids and verification
        if result.structured_records:
            assert result.structured_records[0].place_id
            assert result.structured_records[0].verification.status is not None

    def test_retrieve_interface_signature(self, pipeline_module):
        """The documented signature works with defaults."""
        result = pipeline_module.retrieve("pyramids history")
        assert hasattr(result, "documents")
        assert hasattr(result, "structured_records")

    def test_answer_with_mocked_groq(self, pipeline_module):
        response = pipeline_module.answer("What is the ticket price of the Pyramids?")
        assert response.generated_by == "groq"
        assert "retrieved evidence" in response.answer
        assert response.context is not None
        assert response.context.sources
        assert response.retrieval is not None

    def test_answer_language_detection(self, pipeline_module):
        response = pipeline_module.answer("سعر تذكرة الأهرامات؟")
        assert response.language == "ar"
        assert response.generated_by == "groq"

    def test_answer_fallback_without_key(self, pipeline_module):
        # temporarily disable the key -> fallback must engage, not crash
        real_available = pipeline_module.groq.is_available
        pipeline_module.groq.is_available = False
        try:
            response = pipeline_module.answer("What are the opening hours?")
            assert response.generated_by == "fallback"
            assert "temporarily unavailable" in response.answer
            assert response.context is not None
        finally:
            pipeline_module.groq.is_available = real_available

    def test_health(self, pipeline_module):
        health = pipeline_module.health()
        assert health["kb_loaded"] is True
        assert health["kb_places"] >= 2
        assert health["chroma_docs"] and health["chroma_docs"] > 0
        assert health["bm25_docs"] and health["bm25_docs"] > 0
        assert health["groq_configured"] is True


class TestAgenticInterface:
    """The Agentic AI layer's required access points."""

    def test_agent_can_read_structured_values(self, pipeline_module):
        result = pipeline_module.retrieve("ticket price of Pyramids of Giza")
        for place in result.structured_records:
            assert place.data  # original fields accessible
            assert place.verification.status.value in (
                "verified", "unverified", "estimated", "unavailable", "missing"
            )
            assert place.source.source_file

    def test_agent_gets_place_ids(self, pipeline_module):
        result = pipeline_module.retrieve("pyramids")
        ids = {d.place_id for d in result.documents if d.place_id}
        ids |= {p.place_id for p in result.structured_records}
        assert ids

    def test_agent_gets_missing_info_indicators(self, pipeline_module):
        response = pipeline_module.answer("Is there a pharmacy inside the pyramids?")
        assert response.context is not None
        assert isinstance(response.context.missing_info, list)


class TestFastAPI:
    def test_endpoints(self, pipeline_module):
        app = FastAPI(title="RAG test")
        app.include_router(create_rag_router(pipeline_module))
        client = TestClient(app)

        # health
        resp = client.get("/rag/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["kb_places"] >= 2

        # retrieve
        resp = client.post(
            "/rag/retrieve",
            json={"query": "What is the ticket price of the Pyramids?"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["intent"] == "price"
        assert body["documents"] or body["structured_records"]

        # answer (mocked Groq)
        resp = client.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Pyramids?", "language": "en"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["generation_mode"] == "groq"
        assert body["status"] in ("success", "partial")
        assert body["sources"]
        assert "verification" in body and "status" in body["verification"]
        assert isinstance(body["warnings"], list)

        # validation errors
        resp = client.post("/rag/retrieve", json={"query": ""})
        assert resp.status_code == 422
        resp = client.post("/rag/answer", json={"query": "x", "language": "fr"})
        assert resp.status_code == 422
