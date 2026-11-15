# -*- coding: utf-8 -*-
"""Itinerary day-by-day structure regression suite (task 4).

Covers:
  1. Duration extraction (AR/EN, numerals, words, hyphenated forms,
     non-duration numbers) — pure unit tests, independent of the pipeline.
  2. Intent detection / primary-intent priority for itinerary requests.
  3. Structured itinerary selection (destination + duration matching).
  4. Day-by-day structure in the deterministic fallback (AR + EN matrix).
  5. Groq-mode structure preservation and validator-driven repair
     (CannedGroq: good / flattened / leaky answers).
  6. Structural validator unit tests (bilingual day headings).
  7. /rag/answer + /rag/retrieve contract (requested days preserved).
  8. Streamlit rendering of day headings.

The Groq tests use canned responses (explicitly internal-behavior tests);
live API verification is executed separately and never reported as mocked.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api import create_rag_router
from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.generation.groq_client import GroqNetworkError
from rag.generation.prompts import build_messages
from rag.generation.validator import (
    AnswerValidator,
    find_day_headings,
    itinerary_structure_warnings,
)
from rag.ingestion.loader import load_kb
from rag.pipeline import RAGPipeline
from rag.query_processing import extract_requested_days

# ----------------------------------------------------------------------
# KB subset: Cairo + Luxor + Aswan + Giza + Alexandria places, ALL
# itineraries (day-count selection), their cities, Alexandria activities.
# ----------------------------------------------------------------------
KEEP_PLACES = {
    # Cairo itinerary stops
    "giza_pyramids_001", "gem_001", "saqqara_001", "dahshur_001",
    "citadel_001", "sultan_hassan_001", "moez_street_001",
    "khan_khalili_001",
    # Luxor itinerary stops
    "karnak_001", "luxor_temple_001", "valley_kings_001",
    "hatshepsut_001", "colossi_memnon_001",
    # Aswan itinerary stops
    "philae_001", "high_dam_001", "nubian_village_001", "abu_simbel_001",
    # Alexandria itinerary stops
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


class CannedGroq:
    """Groq stub returning a canned answer (records its prompt)."""

    is_available = True

    def __init__(self, response: str):
        self.response = response
        self.last_messages: Optional[List[Dict[str, str]]] = None

    def generate(self, messages, **kwargs):
        self.last_messages = messages
        return self.response


def _build_pipeline(mini_kb_file, tmp_path_factory, groq) -> RAGPipeline:
    config = RAGConfig()
    config.kb_path = None
    config.kb_single_file = mini_kb_file
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_itin")
    config.chroma_collection_name = "itinerary_test"
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
    path = tmp_path_factory.mktemp("itinerary_mini") / "itinerary_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def fallback_pipeline(mini_kb_file, tmp_path_factory):
    return _build_pipeline(mini_kb_file, tmp_path_factory, FailingGroq())


@pytest.fixture(scope="module")
def client(fallback_pipeline):
    app = FastAPI(title="itinerary structure test")
    app.include_router(create_rag_router(fallback_pipeline))
    return TestClient(app)


# ======================================================================
# 1) Duration extraction (pure unit tests)
# ======================================================================
class TestDurationExtraction:
    @pytest.mark.parametrize("query,expected", [
        # Arabic — Western numerals
        ("رحلة لمدة 3 أيام", 3),
        ("رحلة 3 أيام", 3),
        ("برنامج سياحي لمدة 4 أيام", 4),
        ("خطة لمدة 5 أيام في الغردقة", 5),
        # Arabic — Arabic-Indic numerals
        ("رحلة لمدة ٣ أيام", 3),
        ("برنامج لمدة ٤ أيام في الإسكندرية", 4),
        ("عندي ٥ أيام إجازة اعمل بيهم ايه", 5),
        # Arabic — number words
        ("خططلي رحلة يومين", 2),
        ("رحلة لمدة يوم واحد", 1),
        ("رحلة عائلية لمدة خمسة أيام", 5),
        ("برنامج لمدة ستة أيام", 6),
        ("خطط لمدة سبعة أيام", 7),
        ("اعمل برنامج لمدة ثلاثة أيام", 3),
        ("اربعة ايام في الشرما", 4),
        # English — numeric
        ("Plan a 3-day trip to Cairo", 3),
        ("I want to spend 3 days in Luxor", 3),
        ("for 3 days", 3),
        ("Create a 4-day itinerary for Alexandria", 4),
        ("plan a trip lasting 4 days", 4),
        # English — number words (singular/hyphenated/plural)
        ("a one-day trip", 1),
        ("a trip for two days", 2),
        ("a five-day family vacation", 5),
        ("a seven-day tour", 7),
    ])
    def test_extracts_expected_days(self, query, expected):
        assert extract_requested_days(query) == expected

    @pytest.mark.parametrize("query", [
        # budgets / prices / counts / years must NOT become a duration
        "I have a budget of $500",
        "كم سعر تذكرة الأهرامات؟",
        "التذكرة بـ 200 جنيه",
        "Karnak Temple was built around 2000 BC",
        "In 1954 the museum moved",
        "Give me the 3 best museums",
        "أفضل 5 مطاعم في القاهرة",
        "a 99-day trip",  # out of supported range -> graceful None
        "",
    ])
    def test_non_duration_numbers_ignored(self, query):
        assert extract_requested_days(query) is None


# ======================================================================
# 2) Intent detection and primary-intent priority
# ======================================================================
class TestItineraryIntent:
    @pytest.mark.parametrize("query", [
        "اقترح رحلة عائلية لمدة 3 أيام في القاهرة",
        "خططلي رحلة لمدة يومين في الأقصر",
        "اعمل برنامج سياحي لمدة 4 أيام في الإسكندرية",
        "Plan a 3-day family trip to Cairo",
        "Create a 2-day itinerary for Luxor",
        "Build a 4-day travel plan for Alexandria",
    ])
    def test_multiday_request_is_primary_itinerary(self, fallback_pipeline, query):
        pq = fallback_pipeline.query_processor.process(query)
        assert pq.intent == "itinerary"
        assert pq.days is not None and pq.days >= 2
        # per-place fact fields must not hijack a plan request
        assert pq.structured_fields == []

    @pytest.mark.parametrize("query", [
        "اقترح أشهر الأماكن السياحية في القاهرة",
        "What are the most famous tourist attractions in Cairo?",
        "Suggest family-friendly activities in Alexandria",
        "اقترح أنشطة مناسبة للعائلات في الإسكندرية",
    ])
    def test_generic_requests_stay_generic(self, fallback_pipeline, query):
        pq = fallback_pipeline.query_processor.process(query)
        assert pq.intent != "itinerary"
        assert pq.days is None

    def test_dinner_cruise_is_not_itinerary(self, fallback_pipeline):
        # "رحلة عشاء النيلية" is a dinner CRUISE activity, not a trip plan
        pq = fallback_pipeline.query_processor.process(
            "عاوز رحلة عشاء النيلية"
        )
        assert not (pq.intent == "itinerary" and pq.days is not None)

    def test_single_day_itinerary_is_single(self, fallback_pipeline):
        pq = fallback_pipeline.query_processor.process(
            "Plan a one-day family trip to Giza"
        )
        assert pq.intent == "itinerary"
        assert pq.days == 1


# ======================================================================
# 3) Structured itinerary selection
# ======================================================================
class TestItinerarySelection:
    def _itinerary_extras(self, fallback_pipeline, query):
        result = fallback_pipeline.retrieve(query)
        return result, [
            e for e in result.structured_extras if e.record_type == "itinerary"
        ]

    def test_cairo_3d_selects_cairo_plan(self, fallback_pipeline):
        _, its = self._itinerary_extras(
            fallback_pipeline, "Plan a 3-day family trip to Cairo"
        )
        assert len(its) == 1
        assert its[0].record_id == "it_cairo_history_3d"
        assert int((its[0].data or {}).get("days_count")) == 3

    def test_luxor_2d_selects_single_closest_plan(self, fallback_pipeline):
        _, its = self._itinerary_extras(
            fallback_pipeline, "اعملي برنامج سياحي في الأقصر لمدة يومين."
        )
        assert len(its) == 1
        assert its[0].record_id == "it_luxor_3d"

    def test_luxor_5d_selects_5d_plan(self, fallback_pipeline):
        _, its = self._itinerary_extras(
            fallback_pipeline, "Plan a 5-day trip to Luxor"
        )
        assert len(its) == 1
        assert its[0].record_id == "it_luxor_aswan_5d"

    def test_alexandria_4d_keeps_its_only_plan(self, fallback_pipeline):
        _, its = self._itinerary_extras(
            fallback_pipeline, "Build a 4-day travel plan for Alexandria"
        )
        assert len(its) == 1
        assert its[0].record_id == "it_alexandria_1d"

    def test_detected_days_preserved_in_retrieval(self, fallback_pipeline):
        result, _ = self._itinerary_extras(
            fallback_pipeline, "اقترح رحلة عائلية لمدة 3 أيام في القاهرة"
        )
        assert (result.detected_entities or {}).get("days") == 3
        assert result.intent == "itinerary"


# ======================================================================
# 4) Day-by-day structure in the deterministic fallback (matrix)
# ======================================================================
AR_HEADINGS = [
    "### اليوم الأول", "### اليوم الثاني", "### اليوم الثالث",
    "### اليوم الرابع",
]


class TestFallbackDayStructure:
    @pytest.mark.parametrize("query,lang,n", [
        ("اقترح رحلة عائلية لمدة 3 أيام في القاهرة", "ar", 3),
        ("خططلي رحلة لمدة يومين في الأقصر", "ar", 2),
        ("اعمل برنامج سياحي لمدة 4 أيام في الإسكندرية", "ar", 4),
        ("Plan a 3-day family trip to Cairo", "en", 3),
        ("Create a 2-day itinerary for Luxor", "en", 2),
        ("Build a 4-day travel plan for Alexandria", "en", 4),
        ("اعملي برنامج سياحي في الأقصر لمدة يومين.", "ar", 2),
    ])
    def test_exact_day_headings(self, fallback_pipeline, query, lang, n):
        resp = fallback_pipeline.answer(query)
        assert resp.language == lang
        found = find_day_headings(resp.answer)
        assert found == set(range(1, n + 1)), resp.answer
        assert resp.generated_by == "fallback"
        if lang == "ar":
            for heading in AR_HEADINGS[:n]:
                assert heading in resp.answer
            assert not re.search(r"###\s*Day\s*\d", resp.answer)
        else:
            assert re.search(rf"###\s*Day\s*{n}\b", resp.answer)
            assert "اليوم" not in resp.answer

    def test_no_flat_list_no_id_leak(self, fallback_pipeline):
        resp = fallback_pipeline.answer(
            "اقترح رحلة عائلية لمدة 3 أيام في القاهرة"
        )
        assert "it_cairo_history_3d" not in resp.answer
        assert "### اليوم الأول" in resp.answer
        # attractions are assigned to days, not a flat list
        assert "أهرامات الجيزة" in resp.answer

    def test_shorter_evidence_pads_requested_days(self, fallback_pipeline):
        resp = fallback_pipeline.answer(
            "Build a 4-day travel plan for Alexandria"
        )
        # 4 headings; evidence fills day 1, remaining days keep their
        # heading with an honest natural limitation (no invented places)
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        assert "No further day details are available" in resp.answer
        for invented in ("Montaza", "Stanley"):
            assert invented not in resp.answer

    def test_arabic_padded_days_use_arabic_headings(self, fallback_pipeline):
        resp = fallback_pipeline.answer(
            "اعمل برنامج سياحي لمدة 4 أيام في الإسكندرية"
        )
        assert find_day_headings(resp.answer) == {1, 2, 3, 4}
        assert "### اليوم الرابع" in resp.answer
        assert "تفاصيل كافية" in resp.answer

    def test_truncated_longer_plan_is_disclosed(self, fallback_pipeline):
        resp = fallback_pipeline.answer(
            "Create a 2-day itinerary for Luxor"
        )
        assert find_day_headings(resp.answer) == {1, 2}
        assert "first part of a 3-day itinerary" in resp.answer

    def test_single_day_not_forced_multiday(self, fallback_pipeline):
        resp = fallback_pipeline.answer(
            "Plan a one-day family trip to Giza"
        )
        assert find_day_headings(resp.answer) <= {1}
        assert "Suggested plan" in resp.answer
        assert "Family-friendly" in resp.answer

    @pytest.mark.parametrize("query,expected_text", [
        ("اقترح أشهر الأماكن السياحية في القاهرة", "خان الخليلي"),
        ("What are the most famous tourist attractions in Cairo?",
         "Khan El-Khalili Bazaar"),
    ])
    def test_generic_recommendations_have_no_day_headings(
        self, fallback_pipeline, query, expected_text
    ):
        resp = fallback_pipeline.answer(query)
        assert find_day_headings(resp.answer) == set()
        assert expected_text in resp.answer


# ======================================================================
# 5) Groq mode: structure preservation + validator-driven repair
# ======================================================================
GOOD_3DAY_EN = """Here is your 3-day Cairo plan:

### Day 1 — Giza
- Giza Pyramids and Sphinx in the morning.
- Grand Egyptian Museum in the afternoon.

### Day 2 — Saqqara and Dahshur
- Saqqara and the Step Pyramid of Djoser.
- Dahshur Pyramids (Bent and Red Pyramids).

### Day 3 — Islamic and Coptic Cairo
- Saladin Citadel and Muhammad Ali Mosque.
- Khan El-Khalili Bazaar in the evening."""

FLAT_3DAY_EN = """Giza Pyramids and Sphinx, Grand Egyptian Museum, Saqqara,
Dahshur Pyramids, Saladin Citadel, Khan El-Khalili Bazaar are the top Cairo
sights — 3 days recommended."""

FLAT_2DAY_AR = """معبد الكرنك ومعبد الأقصر في البر الشرقي، ووادي الملوك
ومعبد حتشبسوت وتمثالا ممنون في البر الغربي — البرنامج يوصي بيومين."""

LEAKY_ANSWER = """### Day 1 — Giza
- Giza Pyramids (giza_pyramids_001).

### Day 2 — Saqqara
- Saqqara (doc_4f2a1b).

### Day 3 — Cairo
- Khan El-Khalili Bazaar."""


class TestGroqModeStructure:
    @pytest.fixture(scope="class")
    def good_pipeline(self, mini_kb_file, tmp_path_factory):
        return _build_pipeline(
            mini_kb_file, tmp_path_factory, CannedGroq(GOOD_3DAY_EN)
        )

    @pytest.fixture(scope="class")
    def flat_pipeline(self, mini_kb_file, tmp_path_factory):
        return _build_pipeline(
            mini_kb_file, tmp_path_factory, CannedGroq(FLAT_3DAY_EN)
        )

    @pytest.fixture(scope="class")
    def flat_ar_pipeline(self, mini_kb_file, tmp_path_factory):
        return _build_pipeline(
            mini_kb_file, tmp_path_factory, CannedGroq(FLAT_2DAY_AR)
        )

    @pytest.fixture(scope="class")
    def leaky_pipeline(self, mini_kb_file, tmp_path_factory):
        return _build_pipeline(
            mini_kb_file, tmp_path_factory, CannedGroq(LEAKY_ANSWER)
        )

    def test_prompt_carries_explicit_day_requirement(self, good_pipeline):
        retrieval = good_pipeline.retrieve("Plan a 3-day family trip to Cairo")
        context = good_pipeline.context_builder.build(retrieval)
        messages = build_messages(
            context, "Plan a 3-day family trip to Cairo", "en", requested_days=3
        )
        user_text = re.sub(r"\s+", " ", messages[1]["content"])
        assert "exactly 3 days" in user_text
        assert "day sections" in user_text

    def test_itinerary_evidence_reaches_groq_with_day_structure(
        self, good_pipeline
    ):
        good_pipeline.answer("Plan a 3-day family trip to Cairo")
        system_text = good_pipeline.groq.last_messages[0]["content"]
        user_text = good_pipeline.groq.last_messages[1]["content"]
        # structural instruction is present in the system prompt
        assert "day sections" in re.sub(r"\s+", " ", system_text)
        # the structured itinerary JSON (with per-day stops) is in the evidence
        assert '"days"' in user_text

    def test_good_day_by_day_answer_passes_through(self, good_pipeline):
        resp = good_pipeline.answer("Plan a 3-day family trip to Cairo")
        assert resp.generated_by == "groq"
        assert resp.answer == GOOD_3DAY_EN
        assert find_day_headings(resp.answer) == {1, 2, 3}

    def test_flattened_answer_replaced_by_structured_fallback(self, flat_pipeline):
        resp = flat_pipeline.answer("Plan a 3-day family trip to Cairo")
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2, 3}
        assert "### Day 1 — Giza" in resp.answer

    def test_flattened_arabic_answer_repaired_in_arabic(self, flat_ar_pipeline):
        resp = flat_ar_pipeline.answer(
            "اعملي برنامج سياحي في الأقصر لمدة يومين."
        )
        assert resp.generated_by == "fallback"
        assert find_day_headings(resp.answer) == {1, 2}
        assert "### اليوم الأول" in resp.answer
        assert "### اليوم الثاني" in resp.answer

    def test_structural_failure_recorded_in_warnings(self, flat_pipeline):
        resp = flat_pipeline.answer("Plan a 3-day family trip to Cairo")
        assert any(
            w.startswith("itinerary_structure:") for w in resp.warnings
        )

    def test_leaky_answer_rejected(self, leaky_pipeline):
        resp = leaky_pipeline.answer("Plan a 3-day family trip to Cairo")
        assert resp.generated_by == "fallback"
        assert "giza_pyramids_001" not in resp.answer

    def test_single_day_flat_answer_not_replaced(self, mini_kb_file, tmp_path_factory):
        pipeline = _build_pipeline(
            mini_kb_file, tmp_path_factory,
            CannedGroq(
                "Giza Pyramids in the morning, Grand Egyptian Museum after lunch."
            ),
        )
        resp = pipeline.answer("Plan a one-day family trip to Giza")
        # single-day requests are NOT forced into multi-day structure
        assert resp.generated_by == "groq"
        assert resp.answer.startswith("Giza Pyramids")


# ======================================================================
# 6) Structural validator unit tests
# ======================================================================
class TestItineraryValidator:
    def test_finds_en_headings(self):
        assert find_day_headings("### Day 1 — A\n### Day 2 — B") == {1, 2}

    def test_finds_ar_ordinal_headings(self):
        assert find_day_headings(
            "### اليوم الأول\n### اليوم الثاني"
        ) == {1, 2}

    def test_number_words_in_prose_do_not_count(self):
        # "day one" in prose is NOT a structured heading
        assert find_day_headings("Spend day one at the museum and enjoy.") == set()

    def test_structure_warnings_full(self):
        answer = "### Day 1\ncontent\n### Day 2\ncontent\n### Day 3\ncontent"
        assert itinerary_structure_warnings(answer, 3) == []

    def test_structure_warnings_missing_days(self):
        answer = "### Day 1\ncontent"
        warnings = itinerary_structure_warnings(answer, 3)
        assert warnings and warnings[0].startswith("itinerary_structure")

    def test_structure_warnings_no_headings(self):
        warnings = itinerary_structure_warnings("a flat list", 3)
        assert warnings and warnings[0].startswith("itinerary_structure")

    def test_structure_warnings_arabic(self):
        answer = "### اليوم الأول\n### اليوم الثاني\n### اليوم الثالث"
        assert itinerary_structure_warnings(answer, 3) == []

    def _fake_context(self, text: str):
        class FakeDoc:
            doc_id = "doc_itinerary_evidence"
            content = "itinerary evidence"

        class FakeContext:
            documents = [FakeDoc()]

        FakeContext.text = text
        FakeContext.structured_records = []
        FakeContext.structured_extras = []
        return FakeContext()

    class FakeRetrieval:
        intent = "itinerary"
        detected_entities = {"days": 3}
        intents = ["itinerary"]

    def test_validate_rejects_flat_multiday_itinerary(self):
        validator = AnswerValidator()
        context = self._fake_context(
            "STRUCTURED ITINERARY — Cairo\n"
            '{"days": [{"day": 1, "stops": [{"place_id": "giza_pyramids_001"}]}]}'
        )
        result = validator.validate(
            FLAT_3DAY_EN, context, self.FakeRetrieval(), "en"
        )
        assert not result.is_usable
        assert any(w.startswith("itinerary_structure") for w in result.warnings)

    def test_validate_accepts_wellformed_multiday_itinerary(self):
        validator = AnswerValidator()
        context = self._fake_context(
            "STRUCTURED ITINERARY — Cairo\n"
            '{"days_count": 3, "days": [{"day": 1}, {"day": 2}, {"day": 3}]}'
        )
        result = validator.validate(
            GOOD_3DAY_EN, context, self.FakeRetrieval(), "en"
        )
        assert result.is_usable
        assert not any(
            w.startswith("itinerary_structure") for w in result.warnings
        )


# ======================================================================
# 7) Prompt structure mandates
# ======================================================================
class TestPromptMandates:
    def test_en_prompt_has_day_by_day_rule(self):
        from rag.generation.prompts import SYSTEM_PROMPT_EN

        assert "### Day 1" in SYSTEM_PROMPT_EN
        assert "exactly N day sections" in re.sub(r"\s+", " ", SYSTEM_PROMPT_EN)
        # the EN prompt must stay Arabic-free (existing contract)
        assert not re.search(r"[\u0600-\u06FF]", SYSTEM_PROMPT_EN)

    def test_ar_prompt_has_day_by_day_rule(self):
        from rag.generation.prompts import SYSTEM_PROMPT_AR

        assert "### اليوم الأول" in SYSTEM_PROMPT_AR
        assert "### اليوم الثاني" in SYSTEM_PROMPT_AR


# ======================================================================
# 8) API contract: /rag/retrieve keeps the requested duration available
# ======================================================================
class TestApiContract:
    def test_retrieve_keeps_requested_days_for_agentic_layer(self, client):
        resp = client.post(
            "/rag/retrieve",
            json={"query": "Plan a 3-day family trip to Cairo"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["detected_entities"]["days"] == 3
        assert payload["intent"] == "itinerary"

    def test_answer_contract_structure(self, client):
        resp = client.post(
            "/rag/answer",
            json={
                "query": "Create a 2-day itinerary for Luxor",
                "language": "en",
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert find_day_headings(payload["answer"]) == {1, 2}
        assert payload["generation_mode"] == "fallback"
        assert payload["sources"]


# ======================================================================
# 9) Streamlit renders the day-by-day structure
# ======================================================================
class TestStreamlitDayHeadings:
    def test_day_headings_reach_the_chat_markdown(self, monkeypatch):
        canned = {
            "response_id": "r9",
            "query": "Q",
            "answer": (
                "**Cairo**\n\n**Suggested plan: 3 days in Cairo**\n\n"
                "### Day 1 — Giza\n- Giza Pyramids\n\n"
                "### Day 2 — Saqqara\n- Saqqara\n\n"
                "### Day 3 — Islamic Cairo\n- Khan El-Khalili"
            ),
            "language": "en", "status": "success", "generation_mode": "fallback",
            "sources": ["it_cairo_history_3d"],
            "verification": {"status": "verified", "message": "ok"},
            "warnings": [],
        }

        class FakeResponse:
            status_code = 200

            def json(self):
                return canned

        import streamlit_app
        from pathlib import Path

        app_path = Path(streamlit_app.__file__)
        monkeypatch.setattr(
            streamlit_app.requests, "post", lambda *a, **k: FakeResponse()
        )
        monkeypatch.setattr(
            streamlit_app.requests, "get", lambda *a, **k: FakeResponse()
        )
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(app_path), default_timeout=60)
        at.run()
        assert not at.exception
        at.chat_input[0].set_value("Plan a 3-day family trip to Cairo")
        at.run()
        assert not at.exception
        markdown = "\n".join(str(m.value) for m in at.markdown)
        for heading in ("### Day 1 — Giza", "### Day 2 — Saqqara",
                        "### Day 3 — Islamic Cairo"):
            assert heading in markdown, heading
