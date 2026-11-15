# -*- coding: utf-8 -*-
"""Task 5 regression suite: conversation context + multi-destination itineraries.

Covers the two confirmed real-test failures ONLY:

  Bug A — destination lost in follow-ups: the /rag/answer API is stateless
  (documented in test_user_contract.py Group F), so the Streamlit client —
  the conversation owner — resolves the follow-up destination from the chat
  history and appends it to the query TEXT (no server/API contract change).

  Bug B — multi-destination itineraries: one city was picked
  nondeterministically from a set, so the other destination's evidence was
  never retrieved; now ALL mentioned cities are detected (in mention order),
  their evidence retrieved, and the best exact-city KB plans merged into one
  deduplicated itinerary record (no new itinerary engine).

Groq is exercised through deterministic stubs (FailingGroq -> fallback mode)
exactly like the Task 4 suite; live API checks are executed separately and
never reported as mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api import create_rag_router
from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.generation.groq_client import GroqNetworkError
from rag.generation.validator import find_day_headings
from rag.ingestion.loader import load_kb
from rag.pipeline import RAGPipeline

# ----------------------------------------------------------------------
# Mini KB — identical subset to test_itinerary_structure.py (Task 4
# baseline protection): all itineraries, the cities and places they visit.
# ----------------------------------------------------------------------
KEEP_PLACES = {
    "giza_pyramids_001", "gem_001", "saqqara_001", "dahshur_001",
    "citadel_001", "sultan_hassan_001", "moez_street_001",
    "khan_khalili_001",
    "karnak_001", "luxor_temple_001", "valley_kings_001",
    "hatshepsut_001", "colossi_memnon_001",
    "philae_001", "high_dam_001", "nubian_village_001", "abu_simbel_001",
    "bibliotheca_001", "qaitbay_001", "kom_shoqafa_001",
}
KEEP_CITIES = {
    "city_cairo", "city_giza", "city_luxor", "city_aswan",
    "city_alexandria",
}
KEEP_ACTIVITIES = {
    "act_museum_visits", "act_historical_tours", "act_walking_tours",
}


class FailingGroq:
    """Groq configured but always failing -> deterministic fallback mode."""

    is_available = True

    def generate(self, messages, **kwargs):
        raise GroqNetworkError("connection reset by peer")


def _build_pipeline(mini_kb_file, tmp_path_factory) -> RAGPipeline:
    config = RAGConfig()
    config.kb_path = None
    config.kb_single_file = mini_kb_file
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_ctx")
    config.chroma_collection_name = "context_test"
    kb = load_kb(config.kb_path, config.kb_single_file)
    embedder = LocalEmbedder(model_name=config.embedding_model, device="cpu")
    import rag.embeddings as embeddings_module

    original_get = embeddings_module.get_embedder
    embeddings_module.get_embedder = lambda cfg=None: embedder
    try:
        pipeline = RAGPipeline(config=config, kb=kb, groq_client=FailingGroq())
        pipeline.initialize()
    finally:
        embeddings_module.get_embedder = original_get
    return pipeline


@pytest.fixture(scope="module")
def mini_kb_file(tmp_path_factory):
    source = (
        Path(__file__).resolve().parents[1] / "data" / "rehlaai_rag_data_enhanced.json"
    )
    data = json.loads(source.read_text(encoding="utf-8"))
    data["places"] = [p for p in data["places"] if p.get("place_id") in KEEP_PLACES]
    data["cities"] = [c for c in data["cities"] if c.get("id") in KEEP_CITIES]
    data["activities"] = [
        a for a in data["activities"] if a.get("id") in KEEP_ACTIVITIES
    ]
    path = tmp_path_factory.mktemp("context_mini") / "context_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fallback_pipeline(mini_kb_file, tmp_path_factory):
    return _build_pipeline(mini_kb_file, tmp_path_factory)


@pytest.fixture(scope="module")
def client(fallback_pipeline):
    app = FastAPI(title="conversation context test")
    app.include_router(create_rag_router(fallback_pipeline))
    return TestClient(app)


# ======================================================================
# Bug A — follow-up destination resolver (unit tests, client side)
# ======================================================================
class TestFollowupResolution:
    def _user(self, text: str) -> Dict[str, Any]:
        return {"role": "user", "content": text}

    def test_arabic_aswan_attractions_then_4day_followup(self):
        import streamlit_app

        history = [self._user("أفضل الأنشطة في أسوان")]
        result = streamlit_app.resolve_followup_destination(
            "عاوز خطة 4 أيام", history
        )
        assert result["mode"] == "resolved"
        assert result["destination"]["name_en"] == "Aswan"

    def test_arabic_cairo_attractions_then_3day_followup(self):
        import streamlit_app

        history = [self._user("الأماكن السياحية في القاهرة")]
        result = streamlit_app.resolve_followup_destination(
            "اعملي برنامج 3 أيام", history
        )
        assert result["mode"] == "resolved"
        assert result["destination"]["name_en"] == "Cairo"

    def test_arabic_luxor_then_2day_followup(self):
        import streamlit_app

        history = [self._user("اقترح أماكن في الأقصر")]
        result = streamlit_app.resolve_followup_destination(
            "خليها يومين", history
        )
        assert result["mode"] == "resolved"
        assert result["destination"]["name_en"] == "Luxor"

    @pytest.mark.parametrize("first,followup,expected", [
        ("Best places to visit in Aswan", "Plan a 4-day trip", "Aswan"),
        ("Tourist attractions in Cairo", "Make it a 3-day itinerary", "Cairo"),
        ("Things to do in Luxor", "Make it a 2-day plan", "Luxor"),
    ])
    def test_english_followups(self, first, followup, expected):
        import streamlit_app

        history = [self._user(first)]
        result = streamlit_app.resolve_followup_destination(followup, history)
        assert result["mode"] == "resolved"
        assert result["destination"]["name_en"] == expected

    def test_explicit_new_destination_overrides_previous(self):
        import streamlit_app

        history = [self._user("أفضل الأنشطة في أسوان")]
        result = streamlit_app.resolve_followup_destination(
            "خليها يومين في الأقصر", history
        )
        assert result["mode"] == "explicit"
        assert result["destination"]["name_en"] == "Luxor"

    def test_most_recent_destination_wins_older_does_not_leak(self):
        import streamlit_app

        history = [
            self._user("أفضل الأنشطة في أسوان"),
            {"role": "assistant", "payload": {"answer": "أسوان ..."}},
            self._user("الأماكن السياحية في القاهرة"),
            {"role": "assistant", "payload": {"answer": "القاهرة ..."}},
        ]
        result = streamlit_app.resolve_followup_destination("خليها يومين", history)
        assert result["mode"] == "resolved"
        assert result["destination"]["name_en"] == "Cairo"

    def test_unrelated_place_question_does_not_inherit_destination(self):
        import streamlit_app

        history = [self._user("أفضل الأنشطة في أسوان")]
        result = streamlit_app.resolve_followup_destination(
            "ما هي مواعيد معبد الكرنك؟", history
        )
        # not a plan follow-up: sent unchanged, nothing inherited
        assert result["mode"] == "none"

    def test_ambiguous_context_asks_instead_of_guessing(self):
        import streamlit_app

        history = [self._user("أفضل الأنشطة في الجيزة والقاهرة")]
        result = streamlit_app.resolve_followup_destination(
            "اعمل برنامج 3 أيام", history
        )
        assert result["mode"] == "ambiguous"
        names = {d["name_en"] for d in result["destinations"]}
        assert names == {"Giza", "Cairo"}

    def test_assistant_messages_are_never_scanned(self):
        import streamlit_app

        # the assistant answer mentions Aswan; the user never did
        history = [
            {"role": "assistant", "payload": {"answer": "أنشطة في أسوان رائعة"}}
        ]
        result = streamlit_app.resolve_followup_destination(
            "عاوز خطة 4 أيام", history
        )
        assert result["mode"] == "none"

    def test_no_history_sends_query_unchanged(self):
        import streamlit_app

        result = streamlit_app.resolve_followup_destination(
            "Plan a 4-day trip", []
        )
        assert result["mode"] == "none"

    def test_augmented_arabic_query_text(self):
        import streamlit_app

        assert streamlit_app._is_rtl("عاوز خطة 4 أيام")
        augmented = "عاوز خطة 4 أيام في أسوان"
        assert streamlit_app.find_destinations(augmented) == [
            {"name_en": "Aswan", "name_ar": "أسوان"}
        ]

    def test_augmented_english_query_text(self):
        import streamlit_app

        assert not streamlit_app._is_rtl("Plan a 4-day trip")
        augmented = "Plan a 4-day trip in Aswan"
        assert streamlit_app.find_destinations(augmented) == [
            {"name_en": "Aswan", "name_ar": "أسوان"}
        ]


# ======================================================================
# Bug A — end-to-end: the RESOLVED query produces the pure-city plan
# ======================================================================
class TestResolvedQueriesProducePurePlans:
    def test_arabic_4day_aswan_followup_is_pure_aswan(self, fallback_pipeline):
        resp = fallback_pipeline.answer("عاوز خطة 4 أيام في أسوان")
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        # the Luxor->Aswan plan must NOT leak in (Bug A)
        assert "الكرنك" not in resp.answer
        assert "وادي الملوك" not in resp.answer
        assert "فيله" in resp.answer

    def test_english_4day_aswan_followup_is_pure_aswan(self, fallback_pipeline):
        resp = fallback_pipeline.answer("Plan a 4-day trip in Aswan")
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        assert "Karnak" not in resp.answer
        assert "Philae" in resp.answer

    def test_arabic_3day_cairo_followup(self, fallback_pipeline):
        resp = fallback_pipeline.answer("اعملي برنامج 3 أيام في القاهرة")
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2, 3}

    def test_arabic_2day_luxor_followup(self, fallback_pipeline):
        resp = fallback_pipeline.answer("خليها يومين في الأقصر")
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2}

    def test_api_keeps_the_resolved_query_contract(self, client):
        resp = client.post(
            "/rag/answer",
            json={"query": "عاوز خطة 4 أيام في أسوان", "language": "ar"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert find_day_headings(payload["answer"]) == {1, 2, 3, 4}
        assert "الكرنك" not in payload["answer"]


# ======================================================================
# Bug B — all mentioned cities are detected (deterministic order)
# ======================================================================
class TestMultiCityDetection:
    def test_arabic_giza_and_cairo_order(self, fallback_pipeline):
        pq = fallback_pipeline.query_processor.process(
            "عاوز خطة 4 أيام في الجيزة والقاهرة"
        )
        assert pq.detected_cities == ["Giza", "Cairo"]
        assert pq.detected_city == "Giza"
        assert pq.days == 4

    def test_english_cairo_and_giza_order(self, fallback_pipeline):
        pq = fallback_pipeline.query_processor.process(
            "Create a 4-day itinerary covering Cairo and Giza."
        )
        assert pq.detected_cities == ["Cairo", "Giza"]
        assert pq.detected_city == "Cairo"

    def test_arabic_luxor_and_aswan_order(self, fallback_pipeline):
        pq = fallback_pipeline.query_processor.process(
            "خطة 4 أيام في الأقصر وأسوان"
        )
        assert pq.detected_cities == ["Luxor", "Aswan"]

    def test_single_city_detection_unchanged(self, fallback_pipeline):
        pq = fallback_pipeline.query_processor.process(
            "What are the best places to visit in Giza?"
        )
        assert pq.detected_city == "Giza"
        assert pq.detected_cities == ["Giza"]

    def test_retrieval_exposes_all_cities(self, fallback_pipeline):
        result = fallback_pipeline.retrieve(
            "عاوز خطة 4 أيام في الجيزة والقاهرة"
        )
        entities = result.detected_entities or {}
        assert entities["cities"] == ["Giza", "Cairo"]


# ======================================================================
# Bug B — multi-destination itinerary selection (merged, deduplicated)
# ======================================================================
class TestMultiDestinationItineraries:
    def _itinerary_extras(self, fallback_pipeline, query):
        result = fallback_pipeline.retrieve(query)
        return result, [
            e for e in result.structured_extras if e.record_type == "itinerary"
        ]

    def test_giza_cairo_4d_one_merged_plan(self, fallback_pipeline):
        _, its = self._itinerary_extras(
            fallback_pipeline, "عاوز خطة 4 أيام في الجيزة والقاهرة"
        )
        assert len(its) == 1
        assert its[0].record_id == "itinerary_combined"
        data = its[0].data or {}
        days = data.get("days") or []
        assert [d["day"] for d in days] == [1, 2, 3]  # renumbered, no gaps
        place_ids = [s["place_id"] for d in days for s in d.get("stops") or []]
        # no duplicate attractions across the merged days
        assert len(place_ids) == len(set(place_ids))
        # day 1 is Giza's own plan; Cairo's duplicate first day was dropped
        assert place_ids[:2] == ["giza_pyramids_001", "gem_001"]
        assert "philae_001" not in place_ids  # wrong-destination leak guard

    def test_giza_cairo_4d_answer_bilingual(self, fallback_pipeline):
        resp = fallback_pipeline.answer("عاوز خطة 4 أيام في الجيزة والقاهرة")
        assert resp.generated_by == "fallback"
        # exact requested number of day sections (4), none empty-by-default
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        # both requested destinations are present
        assert "الجيزة" in resp.answer
        assert "القاهرة" in resp.answer
        assert resp.answer.count("الخطة المقترحة") == 1
        # the Giza attraction is not repeated (dedupe), Aswan never leaks
        assert resp.answer.count("الوجهة: أهرامات الجيزة") == 1
        assert "فيله" not in resp.answer
        # day 4 has no KB evidence: honest limitation, no fabrication
        assert "لا تتوفر تفاصيل كافية" in resp.answer

    def test_english_cairo_giza_4d_answer(self, fallback_pipeline):
        resp = fallback_pipeline.answer(
            "Create a 4-day itinerary covering Cairo and Giza."
        )
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        assert "Cairo" in resp.answer
        assert "Giza" in resp.answer
        assert resp.answer.count("Suggested plan") == 1
        assert resp.answer.count("Destination: Giza Pyramids") == 1
        assert "Philae" not in resp.answer

    def test_luxor_aswan_uses_the_single_combined_kb_plan(self, fallback_pipeline):
        _, its = self._itinerary_extras(fallback_pipeline, "خطة 4 أيام في الأقصر وأسوان")
        # one KB plan already covers both cities — reuse it as-is
        assert len(its) == 1
        assert its[0].record_id == "it_luxor_aswan_5d"
        resp = fallback_pipeline.answer("خطة 4 أيام في الأقصر وأسوان")
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        assert "الكرنك" in resp.answer
        assert "فيله" in resp.answer

    def test_no_fabrication_in_merged_plan(self, fallback_pipeline):
        # every stop in the merged record comes verbatim from a KB plan
        result, its = self._itinerary_extras(
            fallback_pipeline, "Create a 4-day itinerary covering Cairo and Giza."
        )
        kb_plans = {}
        for it in result.structured_extras:
            if it.record_type == "itinerary" and it.record_id != "itinerary_combined":
                continue
        kb = fallback_pipeline.structured.kb
        kb_stops = {
            s["place_id"]
            for plan in kb.itineraries
            for day in (plan.get("days") or [])
            for s in (day.get("stops") or [])
        }
        data = its[0].data or {}
        for day in data.get("days") or []:
            for stop in day.get("stops") or []:
                assert stop["place_id"] in kb_stops


# ======================================================================
# Bug A — Streamlit sends the augmented query / asks for clarification
# ======================================================================
class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class TestStreamlitFollowup:
    def _run_app(self, monkeypatch, posts):
        import streamlit_app

        canned = {
            "response_id": "r5",
            "query": "Q",
            "answer": "إجابة تجريبية من قاعدة المعرفة",
            "language": "ar", "status": "success",
            "generation_mode": "fallback",
            "sources": ["city_aswan"],
            "verification": {"status": "unverified", "message": "ok"},
            "warnings": [],
        }

        def fake_post(url, json=None, timeout=None):
            posts.append({"url": url, "json": json})
            return _FakeResponse(canned)

        monkeypatch.setattr(streamlit_app.requests, "post", fake_post)
        monkeypatch.setattr(
            streamlit_app.requests, "get", lambda *a, **k: _FakeResponse({})
        )
        from streamlit.testing.v1 import AppTest
        from pathlib import Path

        app_path = Path(streamlit_app.__file__)
        at = AppTest.from_file(str(app_path), default_timeout=60)
        at.run()
        assert not at.exception
        return at

    def test_followup_sends_destination_augmented_query(self, monkeypatch):
        posts: List[Dict[str, Any]] = []
        at = self._run_app(monkeypatch, posts)
        at.chat_input[0].set_value("أفضل الأنشطة في أسوان")
        at.run()
        assert not at.exception
        at.chat_input[0].set_value("عاوز خطة 4 أيام")
        at.run()
        assert not at.exception
        assert len(posts) == 2
        # the ORIGINAL text is stored in history, the DESTINATION is sent
        assert posts[1]["json"]["query"] == "عاوز خطة 4 أيام في أسوان"

    def test_english_followup_augmentation(self, monkeypatch):
        posts: List[Dict[str, Any]] = []
        at = self._run_app(monkeypatch, posts)
        at.chat_input[0].set_value("Best places to visit in Aswan")
        at.run()
        at.chat_input[0].set_value("Plan a 4-day trip")
        at.run()
        assert len(posts) == 2
        assert posts[1]["json"]["query"] == "Plan a 4-day trip in Aswan"

    def test_explicit_destination_sent_unchanged(self, monkeypatch):
        posts: List[Dict[str, Any]] = []
        at = self._run_app(monkeypatch, posts)
        at.chat_input[0].set_value("أفضل الأنشطة في أسوان")
        at.run()
        at.chat_input[0].set_value("خليها يومين في الأقصر")
        at.run()
        assert len(posts) == 2
        assert posts[1]["json"]["query"] == "خليها يومين في الأقصر"

    def test_ambiguous_followup_asks_no_api_call(self, monkeypatch):
        posts: List[Dict[str, Any]] = []
        at = self._run_app(monkeypatch, posts)
        at.chat_input[0].set_value("أفضل الأنشطة في الجيزة والقاهرة")
        at.run()
        assert len(posts) == 1
        at.chat_input[0].set_value("اعمل برنامج 3 أيام")
        at.run()
        assert not at.exception
        # ambiguity -> concise clarification, NOT a guessed API call
        assert len(posts) == 1
        markdown = "\n".join(str(m.value) for m in at.markdown)
        assert "لأي مدينة البرنامج؟" in markdown
        assert "الجيزة" in markdown and "القاهرة" in markdown
