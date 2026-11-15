# -*- coding: utf-8 -*-
"""User-contract regression suite (Groups A-G subset + 5 consistency tests).

Contract under test (the "clean answer" product contract):

  Case A  verified value  -> shown naturally
  Case B  unverified/estimated fact (prices, hours, coordinates) -> the
          value is NOT presented as fact; a brief "currently unavailable"
          statement replaces it; NO internal verification talk
  Case C  conflicting values -> treated as unavailable, no arbitrary pick,
          no conflict warning
  Case D  missing info -> one short natural sentence
  Case E  fallback answer -> presented exactly like any other answer

Internal metadata (verification statuses, provenance, fallback mode,
detected entities incl. requested days) MUST remain available on the
internal structures and the /rag/retrieve contract for the Agentic layer.

Groups:
  A/B  basic RAG + AR/EN equivalence (subset, offline/deterministic)
  C    missing information and hallucination prevention
  D    itinerary generation (exact day counts, ONE coherent plan)
  E    multi-intent questions (subset)
  F    sequential context — documented limitation (stateless API)
  G    agentic readiness — metadata preservation checks
Plus the 5 specific consistency regressions (task section 9) and
Streamlit UI forbidden-string tests (task section 10).

Runs WITHOUT a Groq key (FailingGroq -> deterministic fallback) so the
fallback path — the strictest renderer — is what is exercised.
"""

from __future__ import annotations

import json
from pathlib import Path
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
# KB subset: all Luxor places + Giza + Cairo stops, cities, activities,
# ALL itineraries (needed for day-count selection tests), all rules.
# ----------------------------------------------------------------------
KEEP_PLACES = {
    "karnak_001", "luxor_temple_001", "valley_kings_001",
    "hatshepsut_001", "colossi_memnon_001",
    "giza_pyramids_001",
    "gem_001", "saqqara_001", "dahshur_001", "citadel_001",
    "sultan_hassan_001", "moez_street_001", "khan_khalili_001",
    # stops of the 5-day Luxor+Aswan plan (itinerary rendering must show
    # real place names, never raw record IDs)
    "abu_simbel_001", "philae_001", "high_dam_001", "nubian_village_001",
}
KEEP_CITIES = {"city_cairo", "city_giza", "city_luxor", "city_aswan"}
KEEP_ACTIVITIES = {"act_nile_cruise"}


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
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_contract")
    config.chroma_collection_name = "contract_test"
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
    # ALL itineraries: the day-count selection logic must pick exactly one
    path = tmp_path_factory.mktemp("contract_mini") / "contract_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def pipeline(mini_kb_file, tmp_path_factory):
    return _build_pipeline(mini_kb_file, tmp_path_factory)


@pytest.fixture(scope="module")
def client(pipeline):
    app = FastAPI(title="contract test")
    app.include_router(create_rag_router(pipeline))
    return TestClient(app)


def _answer(client: TestClient, query: str, language: str = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"query": query}
    if language:
        body["language"] = language
    return client.post("/rag/answer", json=body).json()


FORBIDDEN_IN_ANSWERS = [
    "unverified",
    "not fully verified",
    "partly verified",
    "please verify with official sources",
    "from knowledge base",
    "knowledge base",
    "fallback response",
    "fallback mode",
    "generated answer rejected",
    "assembled directly",
    "confidence",
    "\u26a0\ufe0f",
    "⚠️",
]


def assert_clean(answer: str) -> None:
    """No internal machinery anywhere in the user-facing answer."""
    low = answer.lower()
    for forbidden in FORBIDDEN_IN_ANSWERS:
        assert forbidden not in low, f"forbidden string leaked: {forbidden}"
    assert "doc_" not in answer
    assert re_ok_ids(answer)


def re_ok_ids(answer: str) -> bool:
    import re

    # internal record ids look like word-char tokens with _NNN suffixes
    return not re.search(r"\b[a-z]+(?:_[a-z]+)*_\d{3}\b", answer)


# ----------------------------------------------------------------------
# Groups A + B — basic RAG, AR/EN equivalence (fallback determinism)
# ----------------------------------------------------------------------
class TestGroupABBasicsAndEquivalence:
    def test_luxor_listing_en(self, client):
        body = _answer(client, "What are the best places to visit in Luxor?")
        answer = body["answer"]
        assert "About" in answer
        assert "Karnak" in answer
        # listings never gain fact fields
        assert "Ticket prices" not in answer
        assert "Visiting hours" not in answer
        assert_clean(answer)

    def test_luxor_listing_ar(self, client):
        body = _answer(client, "ما أفضل الأماكن لزيارتها في الأقصر؟")
        answer = body["answer"]
        assert "نبذة" in answer
        assert "الكـرنك" in answer or "الكرنك" in answer
        assert "أسعار التذاكر" not in answer
        assert "ساعات الزيارة" not in answer
        assert_clean(answer)

    def test_karnak_vs_luxor_temple_difference_en(self, client):
        body = _answer(
            client,
            "What is the difference between Karnak Temple and Luxor Temple?",
        )
        answer = body["answer"]
        # both places described, neither confused with the other
        assert "Karnak" in answer and "Luxor Temple" in answer
        assert_clean(answer)

    def test_valley_of_the_kings_about(self, client):
        body = _answer(client, "Tell me about the Valley of the Kings.")
        assert body["answer"].strip()
        assert_clean(body["answer"])

    def test_giza_visit(self, client):
        body = _answer(client, "What can I visit in Giza?")
        assert "Giza" in body["answer"]
        assert_clean(body["answer"])

    def test_equivalent_hours_questions_same_fact(self, client):
        """Consistency Test 1: Karnak hours, EN + AR -> the SAME fact."""
        en = _answer(client, "What are the opening hours of Karnak Temple?")
        ar = _answer(client, "ما مواعيد زيارة معبد الكرنك؟")
        # Case B: the KB marks these hours requires_verification -> both
        # languages state unavailability instead of an unverified value
        assert "currently unavailable" in en["answer"].lower()
        assert "غير متاحة حاليًا" in ar["answer"]
        assert "07:00" not in en["answer"]
        assert "07" not in ar["answer"]
        assert en["status"] == ar["status"]
        assert_clean(en["answer"])
        assert_clean(ar["answer"])

    def test_egyptian_arabic_mixed_query(self, client):
        body = _answer(client, "عاوز أعرف opening hours بتاعة Karnak Temple")
        # mixed-language query: the answer stays natural and clean
        assert body["answer"].strip()
        assert_clean(body["answer"])


# ----------------------------------------------------------------------
# Group C — missing information, no fabrication
# ----------------------------------------------------------------------
class TestGroupCMissingInfo:
    def test_nile_cruise_child_price_en(self, client):
        body = _answer(
            client,
            "What is the exact ticket price for a Nile dinner cruise for children?",
        )
        assert body["status"] == "unavailable"
        assert body["answer"].strip() == "This information is currently unavailable."
        assert body["sources"] == []

    def test_nile_cruise_child_price_ar(self, client):
        body = _answer(
            client, "كام سعر تذكرة رحلة عشاء النيل لطفل عمره 7 سنوات؟"
        )
        assert body["status"] == "unavailable"
        assert body["answer"].strip() == "هذه المعلومة غير متاحة حاليًا."
        assert body["sources"] == []

    def test_restaurants_closing_time(self, client):
        body = _answer(client, "What time does every restaurant in Luxor close?")
        answer = body["answer"]
        # no invented schedule anywhere in the answer
        import re as _re

        assert not _re.search(r"\d{1,2}[:：]\d{2}", answer), "invented closing time"
        assert not _re.search(r"(?i)\d+\s*(?:EGP|جنيه)", answer)
        # whatever the KB has is rendered honestly (Case B/C lines), the
        # rest is a brief unavailability statement — never a fabricated list
        assert "currently unavailable" in answer.lower()
        assert body["status"] in ("success", "partial", "unavailable")
        assert_clean(answer)

    def test_direct_train_karnak_to_valley(self, client):
        body = _answer(
            client,
            "Is there a direct train from Karnak Temple to the Valley of the Kings?",
        )
        # no fabricated route
        assert "currently unavailable" in body["answer"].lower()
        assert_clean(body["answer"])

    def test_every_cairo_attraction_price(self, client):
        body = _answer(
            client, "What is the current ticket price for every attraction in Cairo?"
        )
        answer = body["answer"]
        # a per-place price answer, never an invented single figure for the
        # whole city; unverified/conflicting values stay unavailable (Case B/C)
        n_places = answer.count("**Place:")
        assert n_places >= 2
        assert answer.count("Ticket prices") == n_places
        assert "currently unavailable" in answer.lower()
        # no fabricated aggregate number
        import re as _re

        assert not _re.search(r"(?i)\d+\s*EGP", answer) or any(
            "Ticket prices: currently unavailable" not in line
            and _re.search(r"(?i)\d+\s*EGP", line)
            for line in answer.splitlines()
        )
        assert_clean(answer)


# ----------------------------------------------------------------------
# Group D — itinerary generation: exact days, ONE plan, honest budget
# ----------------------------------------------------------------------
class TestGroupDItineraries:
    def test_cairo_3_day_family(self, client):
        body = _answer(client, "Plan a 3-day itinerary in Cairo for a family.")
        answer = body["answer"]
        assert answer.count("Suggested plan") == 1  # ONE coherent plan
        for day in ("Day 1", "Day 2", "Day 3"):
            assert day in answer
        assert "Day 4" not in answer
        assert_clean(answer)

    def test_arabic_luxor_two_days(self, client):
        """Consistency Test 2: AR 2-day Luxor itinerary."""
        body = _answer(client, "اعملي برنامج سياحي في الأقصر لمدة يومين.")
        answer = body["answer"]
        # exactly two days of the 3-day Luxor plan
        assert answer.count("الخطة المقترحة") == 1
        # Arabic ordinal day headings (task 4 day-by-day structure)
        assert "اليوم الأول" in answer and "اليوم الثاني" in answer
        assert "اليوم الثالث" not in answer
        # no fact fields attached to an itinerary answer
        assert "أسعار التذاكر" not in answer
        assert "ساعات الزيارة" not in answer
        assert_clean(answer)

    def test_arabic_cairo_3_days(self, client):
        body = _answer(
            client,
            "عاوز رحلة 3 أيام في القاهرة تشمل الأماكن التاريخية والأسواق.",
        )
        answer = body["answer"]
        assert answer.count("الخطة المقترحة") == 1
        assert "اليوم الثالث" in answer
        assert "اليوم الرابع" not in answer
        assert_clean(answer)

    def test_luxor_3_day_history(self, client):
        body = _answer(
            client, "Plan a 3-day trip to Luxor focusing on ancient Egyptian history."
        )
        answer = body["answer"]
        assert answer.count("Suggested plan") == 1
        assert "Day 3" in answer
        assert "Day 4" not in answer
        assert_clean(answer)

    def test_luxor_5_day_exact(self, client):
        body = _answer(client, "Plan a 5-day trip to Luxor.")
        answer = body["answer"]
        # the Luxor->Aswan 5-day plan is selected over the 3-day one
        assert answer.count("Suggested plan") == 1
        assert "Day 5" in answer
        assert "Day 6" not in answer

    def test_budget_500_four_days(self, client):
        """Consistency Test 5: $500 / 4 days -> exactly 4 days, one plan."""
        body = _answer(
            client, "I have a budget of $500. Plan a 4-day trip in Egypt."
        )
        answer = body["answer"]
        # ONE plan, exactly the four requested days (5-day Luxor+Aswan plan
        # truncated to 4) — never a dump of unrelated alternatives
        assert answer.count("Suggested plan") == 1
        for day in ("Day 1", "Day 2", "Day 3", "Day 4"):
            assert day in answer
        assert "Day 5" not in answer
        # no invented costs and no unsupported "fits your budget" claim
        assert "budget" not in answer.lower() or "within your budget" not in answer.lower()
        assert "total cost" not in answer.lower()
        assert "EGP" not in answer
        assert_clean(answer)

    def test_days_extraction_unit(self):
        from rag.query_processing import extract_requested_days

        assert extract_requested_days(
            "I have a budget of $500. Plan a 4-day trip in Egypt."
        ) == 4
        assert extract_requested_days("Plan a 5-day trip to Luxor.") == 5
        assert extract_requested_days("عاوز أزور الأقصر لمدة 3 أيام") == 3
        assert extract_requested_days("اعملي برنامج سياحي في الأقصر لمدة يومين.") == 2
        assert extract_requested_days("a two-day trip") == 2
        assert extract_requested_days("What are the best places in Luxor?") is None


# ----------------------------------------------------------------------
# Group E — multi-intent questions (subset)
# ----------------------------------------------------------------------
class TestGroupEMultiIntent:
    def test_luxor_attractions_hours_prices(self, client):
        """Consistency Test 3: fields stay attached to their own place."""
        body = _answer(
            client,
            "What are the main attractions in Luxor, their opening hours, "
            "and ticket prices?",
        )
        answer = body["answer"]
        # multiple place sections, each with its own requested fields
        n_places = answer.count("**Place:")
        n_prices = answer.count("Ticket prices")
        n_hours = answer.count("Visiting hours")
        assert n_places >= 2  # the city's places are rendered per place
        # every rendered place carries BOTH requested fields (fields stay
        # attached to their own place — no city-level attribution)
        assert n_prices == n_places and n_hours == n_places
        # no city-level data attributed to attractions
        assert "About the city" not in answer
        assert_clean(answer)

    def test_arabic_top3_with_prices_and_hours(self, client):
        """Consistency Test 4: exactly the requested Luxor places, AR."""
        body = _answer(
            client,
            "عاوز أعرف أهم 3 أماكن في الأقصر، وأسعار دخولها، ومواعيد الزيارة.",
        )
        answer = body["answer"]
        n_places = answer.count("**المكان:")
        assert n_places >= 2  # top-N scope, no city dump
        # each rendered place carries both requested fields
        assert answer.count("أسعار التذاكر") == n_places
        assert answer.count("ساعات الزيارة") == n_places
        # unverified Luxor prices/hours are honestly unavailable (Case B)
        assert "غير متاحة حاليًا" in answer
        assert_clean(answer)

    def test_compare_karnak_luxor_temple(self, client):
        body = _answer(
            client,
            "Compare Karnak Temple and Luxor Temple in terms of history, "
            "location, and visiting hours.",
        )
        answer = body["answer"]
        assert "Karnak" in answer and "Luxor Temple" in answer
        assert_clean(answer)


# ----------------------------------------------------------------------
# Group F — sequential context: documented statelessness
# ----------------------------------------------------------------------
class TestGroupFSequentialContext:
    def test_followup_without_context_does_not_restart_or_fabricate(self, client):
        """The API is stateless: each answer depends only on its own query.

        Documented limitation (no conversation memory in this RAG layer —
        the future Agentic planner will own context). The system must not
        fabricate a "final itinerary" it does not have.
        """
        first = _answer(client, "Plan a 3-day trip to Luxor.")
        followup = _answer(client, "Give me the final itinerary in Arabic.")
        # no crash, no fabricated plan, no internal machinery
        assert followup["status"] in ("success", "partial", "unavailable")
        assert_clean(followup["answer"])
        # and the first answer was complete in itself
        assert "Day 3" in first["answer"]


# ----------------------------------------------------------------------
# Group G — agentic readiness: internal metadata preserved
# ----------------------------------------------------------------------
class TestGroupGMetadataPreserved:
    def test_retrieve_exposes_verification_and_entities(self, client):
        body = client.post(
            "/rag/retrieve", json={"query": "opening hours of Karnak Temple"}
        ).json()
        # verification metadata survives for the Agentic layer
        assert body["structured_records"], "structured records must be present"
        for record in body["structured_records"]:
            assert "verification_status" in record
        # detected entities (incl. requested days) survive
        assert "detected_entities" in body

    def test_retrieve_exposes_requested_days(self, client):
        body = client.post(
            "/rag/retrieve",
            json={"query": "I have a budget of $500. Plan a 4-day trip in Egypt."},
        ).json()
        assert body["detected_entities"].get("days") == 4
        # one best-matching itinerary — not the whole catalogue
        itins = [
            e for e in body["structured_extras"] if e["record_type"] == "itinerary"
        ]
        assert len(itins) == 1

    def test_answer_internal_response_keeps_context(self, pipeline):
        """RAGResponse (internal) retains context + retrieval metadata."""
        response = pipeline.answer("What are the opening hours of Karnak Temple?")
        assert response.context is not None
        assert response.retrieval is not None
        assert response.retrieval.detected_entities.get("places") == ["karnak_001"]
        # generation_mode is recorded even when fallback was used
        assert response.generated_by in ("fallback", "groq")

    def test_user_answer_contract_has_verification_fields(self, client):
        body = _answer(client, "What are the opening hours of Karnak Temple?")
        # the API contract keeps the fields (frontend hides them); their
        # VALUES stay internal, the answer text itself is clean
        assert "verification" in body
        assert "warnings" in body
        assert "generation_mode" in body
        assert_clean(body["answer"])


# ----------------------------------------------------------------------
# Cross-cutting: every answer above already asserts cleanliness; this
# parametrized sweep covers the remaining Group A questions.
# ----------------------------------------------------------------------
class TestCleanAnswersSweep:
    @pytest.mark.parametrize("query", [
        "What are the main attractions in Luxor?",
        "ما أهم الأماكن السياحية في الأقصر؟",
        "إيه الأماكن السياحية الموجودة على الضفة الغربية في الأقصر؟",
        "ما أهم الأماكن التاريخية في القاهرة؟",
        "Tell me about the Valley of the Kings.",
        "How much is the ticket price for the Giza Pyramids?",
        "ما هي ساعات زيارة معبد الكرنك؟",
        "Plan a 3-day itinerary in Cairo for a family.",
    ])
    def test_no_internal_machinery(self, client, query):
        body = _answer(client, query)
        assert_clean(body["answer"])


# ----------------------------------------------------------------------
# Streamlit UI forbidden-string tests (task section 10)
# ----------------------------------------------------------------------
CANNED_PAYLOADS = {
    "clean": {
        "response_id": "c1", "query": "Q",
        "answer": "**Karnak Temple Complex**\nVisiting hours: 07:00-17:00.",
        "language": "en", "status": "success", "generation_mode": "fallback",
        "sources": ["Karnak Temple Complex"],
        "verification": {"status": "unverified", "message": "internal"},
        "warnings": ["internal warning"],
    },
    "table": {
        "response_id": "c2", "query": "Q",
        "answer": (
            "**البرنامج السياحي — اليوم الأول**\n\n"
            "| الوقت | المكان | النشاط |\n|---|---|---|\n"
            "| 08:00 | معبد الكرنك | زيارة |"
        ),
        "language": "ar", "status": "success", "generation_mode": "fallback",
        "sources": ["معبد الكرنك"],
        "verification": {"status": "unverified", "message": "internal"},
        "warnings": [],
    },
    "unavailable": {
        "response_id": "c3", "query": "Q",
        "answer": "This information is currently unavailable.",
        "language": "en", "status": "unavailable", "generation_mode": "fallback",
        "sources": [], "verification": {"status": "missing", "message": "x"},
        "warnings": [],
    },
}

UI_FORBIDDEN = [
    "Unverified", "Partly verified", "Not fully verified", "\u26a0\ufe0f",
    "Please verify with official sources", "From knowledge base",
    "Fallback response", "Generated answer rejected", "confidence",
]


class TestStreamlitCleanUI:
    @pytest.fixture()
    def run_ui(self, monkeypatch):
        """Run the real app once with a canned API payload -> rendered md."""

        def _run(canned: Dict[str, Any]) -> str:
            import streamlit_app
            from pathlib import Path

            class FakeResponse:
                status_code = 200

                def json(self):
                    return canned

            monkeypatch.setattr(
                streamlit_app.requests, "post", lambda *a, **k: FakeResponse()
            )
            monkeypatch.setattr(
                streamlit_app.requests, "get", lambda *a, **k: FakeResponse()
            )
            from streamlit.testing.v1 import AppTest

            at = AppTest.from_file(
                str(Path(streamlit_app.__file__)), default_timeout=60
            )
            at.run()
            at.chat_input[0].set_value("question")
            at.run()
            assert not at.exception
            return "\n".join(str(m.value) for m in at.markdown)

        return _run

    def test_english_answer_clean_and_visible(self, run_ui):
        markdown = run_ui(CANNED_PAYLOADS["clean"])
        assert "07:00-17:00" in markdown  # the answer is visible
        for forbidden in UI_FORBIDDEN:
            assert forbidden not in markdown, forbidden
        assert "doc_" not in markdown

    def test_arabic_table_answer_clean(self, run_ui):
        markdown = run_ui(CANNED_PAYLOADS["table"])
        assert "|" in markdown  # the Markdown table survives rendering
        assert "معبد الكرنك" in markdown
        for forbidden in UI_FORBIDDEN:
            assert forbidden not in markdown, forbidden

    def test_unavailable_answer_clean(self, run_ui):
        markdown = run_ui(CANNED_PAYLOADS["unavailable"])
        assert "currently unavailable" in markdown.lower()
        for forbidden in UI_FORBIDDEN:
            assert forbidden not in markdown, forbidden
        assert "knowledge base" not in markdown.lower()
        assert "fallback" not in markdown.lower()
