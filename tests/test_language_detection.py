"""Automatic language detection and response-consistency regression tests.

The user's LATEST question determines the response language by default:
  - explicit API ``language`` override is respected when provided
  - otherwise the language is auto-detected from the question itself
  - the KB language, model default, and previous turns never override it
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
from rag.query_processing import detect_language


# ----------------------------------------------------------------------
# Unit tests: detection rules
# ----------------------------------------------------------------------
class TestDetectLanguage:
    def test_pure_arabic(self):
        assert detect_language("ما هي ساعات زيارة معبد الكرنك؟") == "ar"

    def test_pure_english(self):
        assert detect_language("What are the opening hours of Karnak Temple?") == "en"

    def test_egyptian_arabic(self):
        assert detect_language("عاوز أعرف مواعيد زيارة معبد الكرنك") == "ar"

    def test_mixed_arabic_dominant(self):
        # a few English technical terms inside an Arabic sentence
        assert detect_language("عاوز أعرف مواعيد زيارة Karnak Temple") == "ar"

    def test_mixed_english_dominant(self):
        # a few Arabic words inside an English question
        assert detect_language("What are the opening hours of معبد الكرنك?") == "en"

    def test_no_letters_defaults_to_english(self):
        assert detect_language("12345 ?") == "en"


# ----------------------------------------------------------------------
# API-level tests: detected/override language drives the whole contract
# ----------------------------------------------------------------------
KEEP_PLACES = {"karnak_001", "giza_pyramids_001", "hatshepsut_001", "philae_001"}
KEEP_CITIES = {"city_giza", "city_cairo", "city_luxor"}
KEEP_ACTIVITIES = {"act_nile_cruise"}
KEEP_ITINERARIES = {"it_giza_family_1d"}
KEEP_RULES = {"rule_001", "rule_002"}


@pytest.fixture(scope="module")
def mini_kb_file(tmp_path_factory):
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
    path = tmp_path_factory.mktemp("lang_mini") / "lang_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class FailingGroq:
    is_available = True

    def generate(self, messages, **kwargs):
        raise GroqNetworkError("connection reset by peer")


class LanguageTrackingGroq:
    """Records the language passed into the prompt; returns a fixed answer."""

    is_available = True
    seen_languages: list = []

    def generate(self, messages, **kwargs):
        system = messages[0]["content"]
        arabic_prompt = bool(re.search(r"[\u0600-\u06FF]", system))
        self.seen_languages.append("ar" if arabic_prompt else "en")
        return "Ticket prices: 1000 EGP (source: RehlaAI knowledge base - Giza Pyramids)"


def _build_client(groq, mini_kb_file, tmp_path_factory) -> TestClient:
    config = RAGConfig()
    config.kb_path = None
    config.kb_single_file = mini_kb_file
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_lang")
    config.chroma_collection_name = "lang_test"
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
    app = FastAPI(title="language test")
    app.include_router(create_rag_router(pipeline))
    return TestClient(app)


@pytest.fixture(scope="module")
def client_fallback(mini_kb_file, tmp_path_factory):
    return _build_client(FailingGroq(), mini_kb_file, tmp_path_factory)


@pytest.fixture(scope="module")
def client_groq(mini_kb_file, tmp_path_factory):
    LanguageTrackingGroq.seen_languages = []
    return _build_client(LanguageTrackingGroq(), mini_kb_file, tmp_path_factory)


def _answer(client, query, language=None):
    body: Dict[str, Any] = {"query": query}
    if language:
        body["language"] = language
    return client.post("/rag/answer", json=body).json()


AR_CHARS = re.compile(r"[\u0600-\u06FF]")


class TestAutomaticDetection:
    def test_11_arabic_question_arabic_answer(self, client_fallback):
        body = _answer(client_fallback, "ما هي ساعات زيارة معبد الكرنك؟")
        assert body["language"] == "ar"
        assert "المكان" in body["answer"]
        assert "معبد الكرنك" in body["answer"]

    def test_12_english_question_english_answer(self, client_fallback):
        body = _answer(client_fallback, "What are the opening hours of Karnak Temple?")
        assert body["language"] == "en"
        assert "Visiting hours" in body["answer"]
        assert "**Place:" in body["answer"]

    def test_13_egyptian_arabic_arabic_answer(self, client_fallback):
        body = _answer(
            client_fallback, "عاوز أعرف مواعيد زيارة معبد الكرنك فين؟"
        )
        assert body["language"] == "ar"
        assert AR_CHARS.search(body["answer"])

    def test_14_mixed_arabic_dominant_arabic_answer(self, client_fallback):
        body = _answer(client_fallback, "عاوز أعرف مواعيد زيارة Karnak Temple")
        assert body["language"] == "ar"

    def test_14b_mixed_english_dominant_english_answer(self, client_fallback):
        body = _answer(client_fallback, "What are the opening hours of معبد الكرنك?")
        assert body["language"] == "en"

    def test_15_arabic_unavailable_arabic_message(self, client_fallback):
        body = _answer(
            client_fallback, "كم سعر تذكرة رحلة العشاء النيلية للأطفال؟"
        )
        assert body["status"] == "unavailable"
        assert body["language"] == "ar"
        assert body["sources"] == []
        assert AR_CHARS.search(body["answer"])
        assert "لم أجد" in body["answer"] or "قاعدة المعرفة" in body["answer"]

    def test_16_english_unavailable_english_message(self, client_fallback):
        body = _answer(
            client_fallback, "What is the price of a Nile dinner cruise for children?"
        )
        assert body["status"] == "unavailable"
        assert body["language"] == "en"
        assert "couldn't find this information" in body["answer"]

    def test_17_arabic_verification_arabic(self, client_fallback):
        body = _answer(client_fallback, "ما هي ساعات زيارة معبد الكرنك؟")
        assert AR_CHARS.search(body["verification"]["message"])

    def test_18_english_verification_english(self, client_fallback):
        body = _answer(client_fallback, "What are the opening hours of Karnak Temple?")
        message = body["verification"]["message"]
        assert not AR_CHARS.search(message)
        assert message.strip()

    def test_19_fallback_follows_detected_language(self, client_fallback):
        ar = _answer(client_fallback, "ما هي ساعات زيارة معبد الكرنك؟")
        en = _answer(client_fallback, "What are the opening hours of Karnak Temple?")
        assert ar["generation_mode"] == "fallback" and ar["language"] == "ar"
        assert en["generation_mode"] == "fallback" and en["language"] == "en"
        # labels follow the language in both fallback answers
        assert "المكان" in ar["answer"]
        assert "Place:" in en["answer"]

    def test_20_groq_follows_detected_language(self, client_groq):
        _answer(client_groq, "How much is the ticket price for the Giza Pyramids?")
        _answer(client_groq, "كم سعر تذكرة أهرامات الجيزة؟")
        seen = LanguageTrackingGroq.seen_languages
        assert seen == ["en", "ar"]

    def test_21_explicit_override_respected(self, client_fallback):
        # English question with an explicit Arabic override
        body = _answer(
            client_fallback, "What are the opening hours of Karnak Temple?",
            language="ar",
        )
        assert body["language"] == "ar"
        assert "المكان" in body["answer"]
        # and the reverse: Arabic question with an explicit English override
        body = _answer(
            client_fallback, "ما هي ساعات زيارة معبد الكرنك؟", language="en"
        )
        assert body["language"] == "en"
        assert "Visiting hours" in body["answer"]

    def test_22_previous_turn_language_does_not_override(self, client_fallback):
        # first turn: Arabic; second turn: English — each answer follows
        # its OWN question, not the previous turn
        first = _answer(client_fallback, "ما هي ساعات زيارة معبد الكرنك؟")
        second = _answer(client_fallback, "What are the opening hours of Karnak Temple?")
        third = _answer(client_fallback, "ما هي ساعات زيارة معبد الكرنك؟")
        assert first["language"] == "ar"
        assert second["language"] == "en"
        assert third["language"] == "ar"
        assert "Visiting hours" in second["answer"]
        assert "ساعات الزيارة" in third["answer"]
