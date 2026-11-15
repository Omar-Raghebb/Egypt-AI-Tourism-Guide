# -*- coding: utf-8 -*-
"""Real-Groq defect regression tests.

Defects observed in the first live run with a real GROQ_API_KEY:

1. The model copied internal record ids from the evidence headers into
   "(source: ...)" citations (giza_pyramids_001, city_luxor,
   act_nile_cruise) — internal ids must never reach the user answer.
2. An Arabic listing question was answered with a full prices+hours table
   while the English equivalent listed descriptions only (scope expansion).
3. A missing-info question (Nile cruise child price) was labelled
   status="success" in Groq mode but "unavailable" in fallback mode.
4. A Groq answer with bold labels but no bold place headers lost its
   user-facing sources although it explicitly named the place.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api import create_rag_router
from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.generation.groq_client import GroqNetworkError
from rag.generation.prompts import SYSTEM_PROMPT_AR, SYSTEM_PROMPT_EN
from rag.generation.validator import AnswerValidator
from rag.ingestion.loader import load_kb
from rag.pipeline import RAGPipeline
from rag.schemas import RetrievedContext

KEEP_PLACES = {"karnak_001", "giza_pyramids_001"}
KEEP_CITIES = {"city_giza", "city_luxor"}
KEEP_ACTIVITIES = {"act_nile_cruise"}
KEEP_ITINERARIES: set = set()
KEEP_RULES: set = set()


@pytest.fixture(scope="module")
def mini_kb_file(tmp_path_factory):
    source = Path(__file__).resolve().parents[1] / "data" / "rehlaai_rag_data_enhanced.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["places"] = [p for p in data["places"] if p.get("place_id") in KEEP_PLACES]
    data["cities"] = [c for c in data["cities"] if c.get("id") in KEEP_CITIES]
    data["activities"] = [a for a in data["activities"] if a.get("id") in KEEP_ACTIVITIES]
    data["itineraries"] = [i for i in data["itineraries"] if i.get("id") in KEEP_ITINERARIES]
    data["planning_rules_examples"] = [
        r for r in data["planning_rules_examples"] if r.get("id") in KEEP_RULES
    ]
    path = tmp_path_factory.mktemp("groq_mini") / "groq_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class CannedGroq:
    """Groq 'available' returning a scripted answer per query."""

    is_available = True

    def __init__(self, answers: Dict[str, str]):
        self._answers = answers

    def generate(self, messages, **kwargs):
        user = messages[-1]["content"]
        for marker, answer in self._answers.items():
            if marker.lower() in user.lower():
                return answer
        return "No scripted answer for this query."


class FailingGroq:
    is_available = True

    def generate(self, messages, **kwargs):
        raise GroqNetworkError("connection reset by peer")


def _build_client(groq, mini_kb_file, tmp_path_factory) -> TestClient:
    config = RAGConfig()
    config.kb_path = None
    config.kb_single_file = mini_kb_file
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_groq")
    config.chroma_collection_name = "groq_test"
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
    app = FastAPI(title="groq contract test")
    app.include_router(create_rag_router(pipeline))
    return TestClient(app)


def _answer(client, query, language=None):
    body: Dict[str, Any] = {"query": query}
    if language:
        body["language"] = language
    return client.post("/rag/answer", json=body).json()


# ----------------------------------------------------------------------
# 1) internal ids never reach the user answer (validator enforcement)
# ----------------------------------------------------------------------
class TestNoInternalIdLeaks:
    def test_validator_flags_multi_underscore_ids(self):
        validator = AnswerValidator()
        context = RetrievedContext(query="q", text="Giza prices evidence.")
        result = validator.validate(
            "Prices from giza_pyramids_001 and karnak_001.", context, language="en"
        )
        assert any(w == "leak:internal_record_id" for w in result.warnings)
        # the id digits are also unsupported numbers -> hard invalid
        assert result.outcome == "invalid"

    def test_validator_flags_ids_without_numeric_suffix(self):
        validator = AnswerValidator()
        context = RetrievedContext(query="q", text="evidence")
        from rag.schemas import RetrievalResult

        retrieval = RetrievalResult(
            query="q",
            detected_entities={
                "places": ["city_luxor", "act_nile_cruise"],
                "city": None,
                "structured_fields": [],
            },
        )
        result = validator.validate(
            "See (source: city_luxor) and (source: act_nile_cruise).",
            context, retrieval, "en",
        )
        assert any(w == "leak:internal_record_id" for w in result.warnings)

    def test_leaking_groq_answer_replaced_by_fallback(self, mini_kb_file, tmp_path_factory):
        groq = CannedGroq({
            "giza": (
                "**Ticket prices:** Foreign adult: 1000 EGP "
                "(source: Giza Pyramids and Sphinx – giza_pyramids_001)"
            ),
        })
        client = _build_client(groq, mini_kb_file, tmp_path_factory)
        body = _answer(client, "How much is the ticket price for the Giza Pyramids?")
        blob = body["answer"] + " " + " ".join(body["sources"])
        assert "giza_pyramids_001" not in blob
        assert not re.search(r"\b[a-z0-9_]+_[0-9]{3}\b", blob)
        # the leaky generation was replaced by the clean fallback answer
        assert body["generation_mode"] == "fallback"

    def test_leaking_source_markers_removed(self, mini_kb_file, tmp_path_factory):
        groq = CannedGroq({
            "giza": (
                "The Giza Pyramids and Sphinx ticket prices are 1000 EGP "
                "for foreign adults (source: city_giza)."
            ),
        })
        client = _build_client(groq, mini_kb_file, tmp_path_factory)
        body = _answer(client, "What is the ticket price of the Giza Pyramids?")
        assert "city_giza" not in body["answer"]
        assert "(source:" not in body["answer"]


# ----------------------------------------------------------------------
# 2) listing-scope rule present in both prompts
# ----------------------------------------------------------------------
class TestPromptListingScope:
    def test_english_prompt_has_listing_rule(self):
        assert "LISTING question" in SYSTEM_PROMPT_EN
        assert "Do NOT add prices, opening hours" in SYSTEM_PROMPT_EN

    def test_arabic_prompt_has_listing_rule(self):
        assert "أفضل الأماكن للزيارة" in SYSTEM_PROMPT_AR
        assert "لا تُضف أسعارًا" in SYSTEM_PROMPT_AR


# ----------------------------------------------------------------------
# 3) missing-info status is generation-mode independent
# ----------------------------------------------------------------------
class TestMissingInfoStatusWithGroq:
    def test_asserts_missing_info_detector(self):
        from rag.generation.fallback import asserts_missing_info

        # honest missing-info answers (either language, either phrasing
        # generation) are detected
        assert asserts_missing_info(
            "The knowledge base does not contain the price of a Nile "
            "dinner cruise ticket for children."
        )
        assert asserts_missing_info(
            "Ticket prices for a Nile dinner cruise for children are "
            "currently unavailable."
        )
        assert asserts_missing_info("This information is currently unavailable.")
        assert asserts_missing_info(
            "لم أجد هذه المعلومة في قاعدة المعرفة حتى الآن."
        )
        assert asserts_missing_info("هذه المعلومة غير متاحة حاليًا.")
        assert asserts_missing_info("ساعات الزيارة: غير متاحة حاليًا.")
        # a data-carrying answer that mentions a missing secondary field
        # stays partial/success (guard against false positives)
        assert not asserts_missing_info(
            "**Ticket prices:** Foreign adult: 1000 EGP. Opening hours are "
            "currently unavailable."
        )
        assert not asserts_missing_info(
            "ساعات الزيارة: 07:00-17:00. أسعار التذاكر غير متاحة حاليًا."
        )
        # a normal successful answer is not flagged
        assert not asserts_missing_info(
            "Based on the retrieved evidence, here is the requested "
            "information about the Pyramids of Giza."
        )

    def test_groq_missing_info_marked_unavailable(self, mini_kb_file, tmp_path_factory):
        groq = CannedGroq({
            "nile dinner cruise": (
                "Ticket prices for a Nile dinner cruise for children are "
                "currently unavailable."
            ),
        })
        client = _build_client(groq, mini_kb_file, tmp_path_factory)
        body = _answer(
            client, "What is the price of a Nile dinner cruise ticket for children?"
        )
        assert body["status"] == "unavailable"
        assert body["sources"] == []
        assert body["verification"]["status"] == "missing"

    def test_groq_legacy_missing_phrasing_still_detected(self, mini_kb_file, tmp_path_factory):
        """Older phrasing ('not in the knowledge base') is still labelled."""
        groq = CannedGroq({
            "nile dinner cruise": (
                "The knowledge base does not contain the price of a Nile "
                "dinner cruise ticket for children."
            ),
        })
        client = _build_client(groq, mini_kb_file, tmp_path_factory)
        body = _answer(
            client, "What is the price of a Nile dinner cruise ticket for children?"
        )
        assert body["status"] == "unavailable"

    def test_groq_answered_question_still_success(self, mini_kb_file, tmp_path_factory):
        groq = CannedGroq({
            "giza": (
                "**Ticket prices:** Foreign adult: 1000 EGP; foreign student: "
                "500 EGP (source: RehlaAI knowledge base - Giza Pyramids)"
            ),
        })
        client = _build_client(groq, mini_kb_file, tmp_path_factory)
        body = _answer(client, "How much is the ticket price for the Giza Pyramids?")
        assert body["status"] == "success"


# ----------------------------------------------------------------------
# 4) sources survive Groq wording (labels, not bold place headers)
# ----------------------------------------------------------------------
class TestGroqSourceExtraction:
    def test_mentioned_place_listed_as_source(self, mini_kb_file, tmp_path_factory):
        groq = CannedGroq({
            "giza": (
                "**Ticket prices:** Foreign adult: 1000 EGP. These figures come "
                "from the Giza Pyramids and Sphinx record and are unverified."
            ),
        })
        client = _build_client(groq, mini_kb_file, tmp_path_factory)
        body = _answer(client, "What is the ticket price of the Giza Pyramids?")
        assert "Giza Pyramids and Sphinx" in body["sources"]
