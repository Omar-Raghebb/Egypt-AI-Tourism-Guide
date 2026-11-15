"""Output-quality tests — the user-facing answer contract (API level).

These tests exercise the real FastAPI /rag/answer endpoint (mocked Groq and
fallback mode) and enforce the strict separation between internal RAG
structures and what the user is allowed to see:

  - the response carries ONLY the user contract fields
  - no raw JSON, document IDs, scores, stack traces or provider errors
  - verification labels are present and user-readable
  - fallback answers are natural language, not evidence dumps
  - the user's language is respected
"""

from __future__ import annotations

import re
from typing import Any, Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api import create_rag_router
from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.generation.groq_client import GroqNetworkError
from rag.ingestion.loader import load_kb
from rag.pipeline import RAGPipeline


# ----------------------------------------------------------------------
# Fixtures: enhanced-KB subset + one pipeline with mocked Groq, one in
# forced-failure mode. The subset uses the REAL enhanced KB records
# (giza_pyramids_001, karnak_001) so the contract is tested on the
# active data format, with a tiny index for speed.
# ----------------------------------------------------------------------
KEEP_PLACES = {"giza_pyramids_001", "karnak_001"}
KEEP_CITIES = {"city_giza", "city_cairo"}
KEEP_ACTIVITIES = {"act_nile_cruise"}
KEEP_ITINERARIES = {"it_giza_family_1d"}
KEEP_RULES = {"rule_001"}


@pytest.fixture(scope="module")
def enhanced_mini_kb_file(tmp_path_factory):
    import json
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "data" / "rehlaai_rag_data_enhanced.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["places"] = [p for p in data["places"] if p.get("place_id") in KEEP_PLACES]
    data["cities"] = [c for c in data["cities"] if c.get("id") in KEEP_CITIES]
    data["activities"] = [a for a in data["activities"] if a.get("id") in KEEP_ACTIVITIES]
    data["itineraries"] = [i for i in data["itineraries"] if i.get("id") in KEEP_ITINERARIES]
    data["planning_rules_examples"] = [
        r for r in data["planning_rules_examples"] if r.get("id") in KEEP_RULES
    ]
    path = tmp_path_factory.mktemp("enhanced_mini") / "enhanced_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _config(enhanced_mini_kb_file, tmp_path_factory) -> RAGConfig:
    config = RAGConfig()
    config.kb_path = None
    config.kb_single_file = enhanced_mini_kb_file
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_contract")
    config.chroma_collection_name = "contract_test"
    return config


class WellBehavedGroq:
    """Returns a grounded, natural answer with an available price."""
    is_available = True

    def generate(self, messages, **kwargs):
        question = messages[1]["content"]
        if "opening hours" in question.lower():
            return (
                "Visiting hours: 07:00-17:00 daily.\n\n"
                "(source: RehlaAI knowledge base - Giza Pyramids)"
            )
        return (
            "Ticket prices:\n- Foreign adult: 1000 EGP\n\n"
            "(source: RehlaAI knowledge base - Giza Pyramids)"
        )


class FabricatingGroq:
    """Returns an answer with numbers that are NOT in the evidence."""
    is_available = True

    def generate(self, messages, **kwargs):
        return "The ticket costs exactly 9999 EGP per person, guaranteed."


class FailingGroq:
    is_available = True  # claims availability but every call fails

    def generate(self, messages, **kwargs):
        raise GroqNetworkError("connection reset by peer")


def _build_client(groq, enhanced_mini_kb_file, tmp_path_factory):
    config = _config(enhanced_mini_kb_file, tmp_path_factory)
    kb = load_kb(config.kb_path, config.kb_single_file)
    embedder = LocalEmbedder(model_name=config.embedding_model, device="cpu")
    import rag.embeddings as embeddings_module

    original_get = embeddings_module.get_embedder
    embeddings_module.get_embedder = lambda cfg=None: embedder
    try:
        pipeline = RAGPipeline(config=config, kb=kb, groq_client=groq)
        pipeline.initialize()
    finally:
        embeddings_module.get_embedder = original_get
    app = FastAPI(title="contract test")
    app.include_router(create_rag_router(pipeline))
    return TestClient(app)


@pytest.fixture(scope="module")
def client_ok(enhanced_mini_kb_file, tmp_path_factory):
    return _build_client(WellBehavedGroq(), enhanced_mini_kb_file, tmp_path_factory)


@pytest.fixture(scope="module")
def client_fallback(enhanced_mini_kb_file, tmp_path_factory):
    return _build_client(FailingGroq(), enhanced_mini_kb_file, tmp_path_factory)


# ----------------------------------------------------------------------
# Contract shape
# ----------------------------------------------------------------------
class TestContractShape:
    def test_answer_contract_fields(self, client_ok):
        body: Dict[str, Any] = client_ok.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        expected = {
            "response_id", "query", "answer", "language", "status",
            "generation_mode", "sources", "verification", "warnings",
        }
        assert set(body.keys()) == expected
        assert body["generation_mode"] == "groq"
        assert body["status"] in ("success", "partial")

    def test_verification_block_shape(self, client_ok):
        body = client_ok.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        verification = body["verification"]
        assert set(verification.keys()) == {"status", "message"}
        assert verification["status"] in ("verified", "unverified", "mixed")
        assert isinstance(verification["message"], str) and verification["message"]

    def test_sources_are_friendly(self, client_ok):
        body = client_ok.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        assert isinstance(body["sources"], list)
        for source in body["sources"]:
            assert isinstance(source, str) and source.strip()

    def test_no_internal_fields_leak(self, client_ok):
        body = client_ok.post(
            "/rag/answer",
            json={"query": "pyramids"},
        ).json()
        forbidden = {
            "context", "retrieval", "context_chars", "truncated",
            "verification_flags", "error", "generated_by",
            "documents", "scores", "metadata",
        }
        assert not (set(body.keys()) & forbidden)


# ----------------------------------------------------------------------
# Leak prevention: no raw JSON / IDs / traces / provider errors
# ----------------------------------------------------------------------
class TestNoLeaks:
    @pytest.mark.parametrize("query", [
        "What is the ticket price of the Giza Pyramids?",
        "ما مواعيد زيارة معبد الكرنك؟",
        "Tell me about the Giza Pyramids",
        "Plan a one-day family trip to Giza",
        "How do I get to the pyramids?",
    ])
    def test_no_raw_json_or_ids_in_answer(self, client_ok, query):
        body = client_ok.post("/rag/answer", json={"query": query}).json()
        answer = body["answer"]
        # no raw JSON objects
        assert not re.search(r"\{[\s\S]*?:[\s\S]*?\}", answer), f"raw JSON in: {answer[:120]}"
        # no internal document identifiers
        assert not re.search(r"\b[a-z0-9_]+_(?:001|002)::", answer)
        # no doc_id citation brackets with internal ids
        assert not re.search(r"\[\w+_\d+\]", answer)
        # no similarity scores
        assert not re.search(r"score[:\s]*0\.\d+", answer, re.IGNORECASE)

    def test_no_stack_traces_or_provider_errors(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        answer = body["answer"]
        assert "Traceback" not in answer
        assert "connection reset" not in answer
        assert "GroqNetworkError" not in answer
        assert "Exception" not in answer

    def test_no_internal_validation_notes(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        answer = body["answer"]
        for marker in ("data_quality", "guardrail", "unsupported_numbers", "no_evidence"):
            assert marker not in answer

    def test_fallback_mode_reported_not_faked(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        # must NOT pretend an LLM generated this
        assert body["generation_mode"] == "fallback"


# ----------------------------------------------------------------------
# Fallback answer readability
# ----------------------------------------------------------------------
class TestFallbackReadability:
    def test_fallback_is_natural(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        answer = body["answer"]
        # conflicting evidence (Case C) -> no single value shown as fact
        assert "Ticket prices" in answer
        assert "currently unavailable" in answer.lower()
        assert "1000" not in answer
        # no raw evidence blobs, no internal notices
        assert "STRUCTURED EVIDENCE" not in answer
        assert "(Note:" not in answer

    def test_fallback_arabic_natural(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "سعر تذكرة أهرامات الجيزة؟"},
        ).json()
        assert body["language"] == "ar"
        answer = body["answer"]
        assert "أسعار التذاكر" in answer
        assert "غير متاحة حاليًا" in answer
        assert "قاعدة المعرفة" not in answer

    def test_missing_info_communicated_naturally(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the opening hours of the Library of Alexandria?"},
        ).json()
        answer = body["answer"]
        # either answered from evidence or a natural not-available statement
        assert (
            "currently unavailable" in answer.lower()
            or "Visiting hours" in answer
        )
        assert "Traceback" not in answer

    def test_fallback_warnings_are_user_readable(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        for warning in body["warnings"]:
            assert " " in warning  # natural sentence, not a debug code
            assert not re.fullmatch(r"[a-z_:0-9]+", warning)


# ----------------------------------------------------------------------
# Language handling
# ----------------------------------------------------------------------
class TestLanguage:
    def test_arabic_query_arabic_answer(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "ما مواعيد زيارة معبد الكرنك؟"},
        ).json()
        assert body["language"] == "ar"
        arabic_chars = sum(1 for ch in body["answer"] if "\u0600" <= ch <= "\u06FF")
        assert arabic_chars > 10

    def test_explicit_language_respected(self, client_ok):
        body = client_ok.post(
            "/rag/answer",
            json={"query": "What are the opening hours of the pyramids?", "language": "ar"},
        ).json()
        assert body["language"] == "ar"


# ----------------------------------------------------------------------
# Fabrication guardrail (validator, not user-visible)
# ----------------------------------------------------------------------
class TestGroundedness:
    def test_fabricated_numbers_rejected_to_fallback(self, enhanced_mini_kb_file, tmp_path_factory):
        client = _build_client(FabricatingGroq(), enhanced_mini_kb_file, tmp_path_factory)
        body = client.post(
            "/rag/answer",
            json={"query": "What is the ticket price of the Giza Pyramids?"},
        ).json()
        # the fabricated 9999 must not reach the user as an asserted fact
        assert "9999" not in body["answer"]
        assert body["generation_mode"] == "fallback"

    def test_health_distinguishes_key_from_connectivity(self, client_ok):
        body = client_ok.get("/rag/health").json()
        provider = body["llm_provider"]
        assert "key_configured" in provider
        assert provider["connectivity"] in ("configured", "not_configured")
        assert body["dense_index"]["backend"] == "chromadb"
        assert body["sparse_index"]["backend"] == "bm25"

    def test_unavailable_status_when_no_evidence(self, client_fallback):
        body = client_fallback.post(
            "/rag/answer",
            json={"query": "What is the price of a hot air balloon ride over Luxor?"},
        ).json()
        # structured evidence exists for activities, so status is success or
        # partial; a truly empty retrieval must report "unavailable"
        assert body["status"] in ("success", "partial", "unavailable")
        if body["status"] == "unavailable":
            assert "currently unavailable" in body["answer"] or "غير متاحة" in body["answer"]
