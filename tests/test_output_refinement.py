"""Regression tests for the observed output-quality defects.

Each test maps 1:1 to a defect observed in real API / Streamlit outputs:

  1. Karnak hours question pulled unrelated temples (Hatshepsut, Philae)
     into the rendered answer and the sources list.
  2. The Cairo three-day itinerary stopped during day three.
  3. Luxor answers exposed internal record IDs (karnak_001).
  4. The Nile dinner cruise child-price question returned empty sources
     but was labelled "unverified" instead of "missing"/unavailable.
  5. Conflicting Giza ticket prices were collapsed without qualification.
  6. Arabic answers used English labels/names.
  7. English answers contained accidental Arabic labels.
  8. Partial evidence produced a useless answer instead of a qualified one.
  9. Raw JSON / debug text / provider errors could leak into answers.
 10. Streamlit must render the final contract without internal metadata.

Updated for the clean-answer contract (refinement pass): unverified /
estimated / conflicting FACT values are NOT shown as facts — a requested
value that cannot be safely established is stated as "currently
unavailable" (no verification talk, no fallback notices anywhere in the
user-facing answer; see tests/test_user_contract.py for the full suite).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api import create_rag_router
from rag.config import RAGConfig
from rag.embeddings import LocalEmbedder
from rag.generation.groq_client import GroqNetworkError
from rag.generation.validator import AnswerValidator, OUTCOME_PARTIAL
from rag.ingestion.loader import load_kb
from rag.pipeline import RAGPipeline
from rag.schemas import RetrievedContext


# ----------------------------------------------------------------------
# KB subset: real enhanced-KB records only (small index for speed).
# Includes the "distractor" temples so the tests prove they are EXCLUDED.
# ----------------------------------------------------------------------
KEEP_PLACES = {
    "karnak_001", "hatshepsut_001", "philae_001",          # Karnak + distractors
    "luxor_temple_001", "valley_kings_001", "colossi_memnon_001",
    "giza_pyramids_001",
    # Cairo 3-day itinerary stops
    "gem_001", "saqqara_001", "dahshur_001", "citadel_001",
    "sultan_hassan_001", "moez_street_001", "khan_khalili_001",
}
KEEP_CITIES = {"city_giza", "city_cairo", "city_luxor"}
KEEP_ACTIVITIES = {"act_nile_cruise"}
KEEP_ITINERARIES = {"it_cairo_history_3d", "it_giza_family_1d"}
KEEP_RULES = {"rule_001", "rule_002", "rule_003"}


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
    path = tmp_path_factory.mktemp("refinement_mini") / "refinement_mini.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class FailingGroq:
    """Groq configured but always failing -> deterministic fallback mode."""

    is_available = True

    def generate(self, messages, **kwargs):
        raise GroqNetworkError("connection reset by peer")


def _build_client(mini_kb_file, tmp_path_factory) -> TestClient:
    config = RAGConfig()
    config.kb_path = None
    config.kb_single_file = mini_kb_file
    config.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    config.embedding_device = "cpu"
    config.chroma_persist_directory = tmp_path_factory.mktemp("chroma_refinement")
    config.chroma_collection_name = "refinement_test"
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
    app = FastAPI(title="refinement test")
    app.include_router(create_rag_router(pipeline))
    return TestClient(app)


@pytest.fixture(scope="module")
def client(mini_kb_file, tmp_path_factory):
    return _build_client(mini_kb_file, tmp_path_factory)


def _answer(client: TestClient, query: str, language: str = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"query": query}
    if language:
        body["language"] = language
    return client.post("/rag/answer", json=body).json()


# ----------------------------------------------------------------------
# 1) Karnak hours: no unrelated temples in answer or sources
# ----------------------------------------------------------------------
class TestKarnakSourceRelevance:
    def test_karnak_hours_sources_are_karnak_only(self, client):
        body = _answer(client, "What are the opening hours of Karnak Temple?")
        # Case B: the KB marks these hours requires_verification -> the
        # value is not presented; the request may be labelled unavailable
        assert body["status"] in ("success", "unavailable")
        if body["sources"]:
            assert any("Karnak" in s for s in body["sources"])
            joined = " | ".join(body["sources"])
            assert "Philae" not in joined, f"unrelated Philae source: {joined}"
            assert "Hatshepsut" not in joined, f"unrelated Hatshepsut source: {joined}"

    def test_karnak_hours_answer_does_not_render_other_temples(self, client):
        body = _answer(client, "What are the opening hours of Karnak Temple?")
        answer = body["answer"]
        # Case B: the KB marks Karnak's hours "requires_verification" ->
        # the value is NOT presented as a confirmed fact; a brief natural
        # unavailability statement replaces it (no internal verification talk)
        assert "currently unavailable" in answer.lower()
        assert "07:00" not in answer
        # no unrelated temple is rendered as an answer section (the KB rule
        # text may legitimately mention West Bank sights by name)
        assert "Place: Philae" not in answer
        assert "Place: Hatshepsut" not in answer

    def test_karnak_query_does_not_retrieve_distractors(self, mini_kb_file):
        # the entity filter must drop keyword-only ("temple") matches
        from rag.query_processing import QueryProcessor

        kb = load_kb(None, mini_kb_file)
        qp = QueryProcessor(kb)
        pq = qp.process("What are the opening hours of Karnak Temple?")
        assert pq.detected_places == ["karnak_001"]
        pq_giza = qp.process("How much is the ticket price for the Giza Pyramids?")
        assert pq_giza.detected_places == ["giza_pyramids_001"]
        # generic-only queries (no distinctive token) keep all matches
        pq_generic = qp.process("Which temples are in Luxor?")
        assert "karnak_001" in pq_generic.detected_places
        # city listing works through the city path
        pq_city = qp.process("What places are in Luxor?")
        assert pq_city.detected_city == "Luxor"


# ----------------------------------------------------------------------
# 2) Cairo three-day itinerary: complete, all days present
# ----------------------------------------------------------------------
class TestItineraryCompleteness:
    def test_cairo_itinerary_has_all_three_days(self, client):
        body = _answer(client, "Plan a 3-day family itinerary in Cairo.")
        answer = body["answer"]
        assert "Day 1" in answer
        assert "Day 2" in answer
        assert "Day 3" in answer
        # stops render as destinations, not internal IDs
        assert "Giza" in answer

    def test_itinerary_record_listed_in_sources(self, client):
        body = _answer(client, "Plan a 3-day family itinerary in Cairo.")
        # the itinerary record itself (rendered as "Suggested plan: ...")
        # must be listed — a one-word city name must not swallow it
        assert any("3 days in Cairo" in s for s in body["sources"])

    def test_itinerary_day_format(self, client):
        body = _answer(client, "Plan a 3-day family itinerary in Cairo.")
        answer = body["answer"]
        # "Day N — Title" structure
        assert re.search(r"Day 1 — ", answer)
        assert re.search(r"Day 2 — ", answer)
        assert re.search(r"Day 3 — ", answer)

    def test_family_note_only_when_supported(self, client):
        body = _answer(client, "Plan a 3-day family itinerary in Cairo.")
        # it_cairo_history_3d has no "family" marker in title or interests
        assert "Family-friendly" not in body["answer"]

    def test_family_note_present_for_family_itinerary(self, client):
        body = _answer(client, "Plan a one-day family trip to Giza")
        assert "Family-friendly" in body["answer"]


# ----------------------------------------------------------------------
# 3) Luxor attractions: no internal record IDs
# ----------------------------------------------------------------------
class TestNoInternalIds:
    @pytest.mark.parametrize("query", [
        "What are the best places to visit in Luxor?",
        "ما هي أفضل الأماكن في الأقصر؟",
    ])
    def test_luxor_answer_has_no_internal_ids(self, client, query):
        body = _answer(client, query)
        blob = body["answer"] + " " + " ".join(body["sources"])
        assert not re.search(r"\b[a-z][a-z0-9]+_001\b", blob), f"internal ID leaked: {blob[:200]}"
        assert "::" not in blob
        assert "doc_" not in blob

    def test_luxor_answer_renders_place_names(self, client):
        body = _answer(client, "What are the best places to visit in Luxor?")
        assert "Karnak Temple Complex" in body["answer"]


# ----------------------------------------------------------------------
# 4) Nile cruise child price: unavailable, empty sources, missing status
# ----------------------------------------------------------------------
class TestUnavailableAnswer:
    def test_child_price_unavailable_contract(self, client):
        body = _answer(
            client, "What is the ticket price for a Nile dinner cruise for children?"
        )
        assert body["status"] == "unavailable"
        assert body["sources"] == []
        assert body["warnings"] == []
        assert body["verification"]["status"] == "missing"
        assert "currently unavailable" in body["answer"].lower()

    def test_unavailable_verification_message_is_plain(self, client):
        body = _answer(
            client, "What is the ticket price for a Nile dinner cruise for children?"
        )
        assert body["verification"]["message"].strip()


# ----------------------------------------------------------------------
# 5) Conflicting Giza prices are treated as unavailable (Case C): no
#    arbitrary value is chosen, no conflict warning is exposed
# ----------------------------------------------------------------------
class TestConflictingValues:
    def test_giza_conflict_is_unavailable_without_warning(self, client):
        body = _answer(client, "How much is the ticket price for the Giza Pyramids?")
        answer = body["answer"]
        # no single conflicting value is presented as fact
        assert "1000" not in answer
        assert "500" not in answer
        # no internal conflict/verification talk
        assert "conflicting" not in answer.lower()
        assert "unverified" not in answer.lower()
        # honest brief unavailability instead
        assert "currently unavailable" in answer.lower()

    def test_giza_sources_relevant(self, client):
        body = _answer(client, "How much is the ticket price for the Giza Pyramids?")
        joined = " | ".join(body["sources"])
        assert "Giza" in joined
        assert "Dahshur" not in joined  # only mentioned inside rule text


# ----------------------------------------------------------------------
# 6) + 7) Language correctness
# ----------------------------------------------------------------------
class TestLanguageConsistency:
    def test_arabic_answer_uses_arabic_labels_and_names(self, client):
        body = _answer(client, "ما هي ساعات زيارة معبد الكرنك؟", language="ar")
        answer = body["answer"]
        assert "المكان" in answer
        assert "معبد الكرنك" in answer
        assert "Visiting hours" not in answer
        assert "**Place:" not in answer

    def test_arabic_sources_use_arabic_names(self, client):
        body = _answer(client, "ما هي ساعات زيارة معبد الكرنك؟", language="ar")
        assert any(re.search(r"[\u0600-\u06FF]", s) for s in body["sources"])
        assert not any("Karnak" in s for s in body["sources"])

    def test_english_answer_has_no_arabic_labels(self, client):
        body = _answer(client, "What are the opening hours of Karnak Temple?")
        answer = body["answer"]
        assert "المكان" not in answer
        assert "ملاحظات" not in answer
        assert "ساعات" not in answer


# ----------------------------------------------------------------------
# 8) Partial evidence -> useful, qualified answer
# ----------------------------------------------------------------------
class TestPartialEvidence:
    def test_hours_and_accessibility_unverified(self, client):
        body = _answer(
            client, "What are the opening hours and accessibility of Hatshepsut Temple?"
        )
        answer = body["answer"]
        # the KB marks both fields "requires_verification" (Case B): neither
        # value is presented as a confirmed fact; each requested field gets
        # its own brief, honest unavailability line — no verification talk
        low = answer.lower()
        assert low.count("currently unavailable") >= 2
        assert "06:00" not in answer
        assert "unverified" not in low
        assert "Visiting hours" in answer
        assert "Accessibility" in answer
        # never fabricated
        assert body["status"] in ("success", "partial", "unavailable")


# ----------------------------------------------------------------------
# 9) No leaks anywhere
# ----------------------------------------------------------------------
class TestNoLeaksAnywhere:
    @pytest.mark.parametrize("query", [
        "What are the opening hours of Karnak Temple?",
        "Plan a 3-day family itinerary in Cairo.",
        "What are the best places to visit in Luxor?",
        "How much is the ticket price for the Giza Pyramids?",
        "ما هي ساعات زيارة معبد الكرنك؟",
    ])
    def test_no_raw_json_or_internal_markers(self, client, query):
        body = _answer(client, query)
        blob = body["answer"] + " " + " ".join(body["sources"]) + " " + " ".join(
            body["warnings"]
        )
        assert not re.search(r"\{['\"]", blob)
        assert "Traceback" not in blob
        assert "connection reset" not in blob
        assert "GroqNetworkError" not in blob
        assert not re.search(r"\b[a-z][a-z0-9]+_001\b", blob)
        assert not re.search(r"\b[a-z][a-z0-9]+_[0-9]{3}\b", blob)


class TestValidatorStructureChecks:
    def _context(self, text="Some evidence text with numbers 220 and 110."):
        return RetrievedContext(query="q", text=text)

    def test_internal_record_id_flagged(self):
        validator = AnswerValidator()
        result = validator.validate(
            "See karnak_001 for details.", self._context(), language="en"
        )
        assert any(w.startswith("leak:internal_record_id") for w in result.warnings)
        # the leaked ID also introduces an unsupported number -> hard invalid
        assert result.outcome == "invalid"

    def test_internal_record_id_only_still_detected(self):
        # a record ID whose number IS in the evidence: leak flagged, and the
        # outcome downgrades to insufficient_evidence (not wholesale invalid)
        validator = AnswerValidator()
        context = self._context(text="Evidence mentions karnak_002 explicitly.")
        result = validator.validate(
            "See karnak_002 for details.", context, language="en"
        )
        assert any(w.startswith("leak:internal_record_id") for w in result.warnings)
        assert result.outcome == "insufficient_evidence"

    def test_raw_json_flagged(self):
        validator = AnswerValidator()
        result = validator.validate(
            'Here: {"place_id": "x"}', self._context(), language="en"
        )
        assert any(w == "leak:raw_json" for w in result.warnings)

    def test_clean_answer_not_flagged(self):
        validator = AnswerValidator()
        result = validator.validate(
            "Visiting hours: 07:00-17:00 daily.", self._context(), language="en"
        )
        assert not any(w.startswith("leak:") for w in result.warnings)

    def test_language_mismatch_flagged(self):
        validator = AnswerValidator()
        english_text = (
            "The Karnak Temple Complex opens at seven in the morning and "
            "closes at five in the afternoon, and the price for foreign "
            "adults is about two hundred and twenty pounds."
        )
        result = validator.validate(english_text, self._context(), language="ar")
        assert any(w.startswith("language_mismatch") for w in result.warnings)
        assert result.outcome == OUTCOME_PARTIAL

    def test_no_verification_qualification_in_answer(self, mini_kb_file):
        """Case B/E: the fallback answer carries NO verification talk."""
        from rag.generation.fallback import FallbackResponder
        from rag.retrieval.structured import StructuredRetriever
        from rag.schemas import StructuredPlace, VerificationMetadata, VerificationStatus
        from rag.schemas import SourceMetadata

        kb = load_kb(None, mini_kb_file)
        responder = FallbackResponder(StructuredRetriever(kb))
        place = StructuredPlace(
            place_id="x_001", name="X", data={},
            verification=VerificationMetadata(status=VerificationStatus.UNVERIFIED),
            source=SourceMetadata(),
        )
        place2 = StructuredPlace(
            place_id="y_001", name="Y", data={},
            verification=VerificationMetadata(status=VerificationStatus.VERIFIED),
            source=SourceMetadata(),
        )
        context = RetrievedContext(
            query="q", text="evidence",
            structured_records=[place, place2],
        )
        answer = responder.respond(context, "en")
        # both places are rendered as content
        assert "**Place: X**" in answer and "**Place: Y**" in answer
        # no verification qualification, no unverified labels, no notices
        assert "well-sourced" not in answer.lower()
        assert "double-check" not in answer.lower()
        assert "unverified" not in answer.lower()
        assert "knowledge base" not in answer.lower()


# ----------------------------------------------------------------------
# 10) Streamlit renders the final contract (no internal metadata shown)
# ----------------------------------------------------------------------
class TestStreamlitRendering:
    @pytest.fixture()
    def streamlit_run(self, monkeypatch):
        """Run the real streamlit_app with a canned API payload."""
        canned = {
            "response_id": "r1", "query": "Q", "answer": "Visiting hours: 07:00-17:00.",
            "language": "en", "status": "success", "generation_mode": "fallback",
            "sources": ["Karnak Temple Complex"],
            "verification": {"status": "unverified", "message": "not fully verified."},
            "warnings": ["This answer was assembled directly from the knowledge base."],
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
        # submit a question through the real chat input flow
        at.chat_input[0].set_value("What are the opening hours of Karnak Temple?")
        at.run()
        return at, canned

    def test_renders_answer_without_internal_metadata(self, streamlit_run):
        at, canned = streamlit_run
        markdown = "\n".join(str(m.value) for m in at.markdown)
        # the answer itself is visible
        assert canned["answer"] in markdown
        # no internal identifiers
        assert "karnak_001" not in markdown
        assert "doc_" not in markdown
        # no internal metadata surfaces in the UI
        for forbidden in (
            "Not fully verified",
            "Unverified",
            "Verified",
            "From knowledge base",
            "assembled directly",
            "fallback response",
            "Generated answer rejected",
            "\u26a0\ufe0f",
        ):
            assert forbidden.lower() not in markdown.lower(), forbidden

    def test_missing_status_chip(self, monkeypatch):
        canned = {
            "response_id": "r2", "query": "Q",
            "answer": "This information is currently unavailable.",
            "language": "en", "status": "unavailable", "generation_mode": "fallback",
            "sources": [], "verification": {"status": "missing", "message": "not found."},
            "warnings": [],
        }

        class FakeResponse:
            status_code = 200

            def json(self):
                return canned

        import streamlit_app
        from pathlib import Path

        app_path = Path(streamlit_app.__file__)

        monkeypatch.setattr(streamlit_app.requests, "post", lambda *a, **k: FakeResponse())
        monkeypatch.setattr(streamlit_app.requests, "get", lambda *a, **k: FakeResponse())
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(app_path), default_timeout=60)
        at.run()
        at.chat_input[0].set_value("Nile cruise price for children?")
        at.run()
        assert not at.exception
        markdown = "\n".join(str(m.value) for m in at.markdown)
        # the missing-info answer renders as a normal answer: brief, natural,
        # no fallback label, no internal status
        assert "currently unavailable" in markdown.lower()
        assert "knowledge base" not in markdown.lower()
        assert "fallback" not in markdown.lower()


# ----------------------------------------------------------------------
# 11) AR/EN attraction-query equivalence (Phase 4/5 defect: Arabic
#     listings gained prices/hours that English listings did not)
# ----------------------------------------------------------------------
class TestArEnAttractionEquivalence:
    def test_listing_intents_are_language_symmetric(self, mini_kb_file):
        from rag.query_processing import QueryProcessor

        kb = load_kb(None, mini_kb_file)
        qp = QueryProcessor(kb)
        en = qp.process("What are the best places to visit in Luxor?")
        ar = qp.process("ما هي أفضل الأماكن للزيارة في الأقصر؟")
        assert en.intents == ["recommendation"] == ar.intents
        # the Arabic copula prefix "ما هي/ما هى" must not invent intents
        # ("أهم الأماكن" IS a recommendation phrase — AR/EN symmetry)
        assert qp.process("ما أهم الأماكن السياحية في الأقصر؟").intents == [
            "recommendation"
        ]
        # fact questions: "definition" never rides along with another intent
        assert qp.process("ما هي ساعات زيارة معبد الكرنك؟").intents == ["hours"]
        assert qp.process(
            "What is the ticket price of the Giza Pyramids?"
        ).intents == ["price"]

    def test_arabic_listing_has_no_hours_or_prices(self, client):
        body = _answer(client, "ما هي أفضل الأماكن للزيارة في الأقصر؟")
        answer = body["answer"]
        assert "نبذة" in answer                      # descriptions included
        assert "ساعات الزيارة" not in answer          # no auto hours
        assert "أسعار التذاكر" not in answer          # no auto prices

    def test_english_listing_has_no_hours_or_prices(self, client):
        body = _answer(client, "What are the best places to visit in Luxor?")
        answer = body["answer"]
        assert "About" in answer
        assert "Visiting hours" not in answer
        assert "Ticket prices" not in answer

    def test_arabic_listing_equivalent_to_english(self, client):
        en = _answer(client, "What are the best places to visit in Luxor?")
        ar = _answer(client, "ما هي أفضل الأماكن للزيارة في الأقصر؟")
        # same information scope: names + city + description, no fact fields
        for marker_en, marker_ar in (("Visiting hours", "ساعات الزيارة"),
                                     ("Ticket prices", "أسعار التذاكر")):
            assert marker_en not in en["answer"]
            assert marker_ar not in ar["answer"]
        # both list the same set of Luxor places (Arabic names in the AR one)
        assert len(en["sources"]) == len(ar["sources"])

    def test_price_question_scoped_to_prices(self, client):
        body = _answer(client, "What is the ticket price of the Giza Pyramids?")
        answer = body["answer"]
        assert "Ticket prices" in answer
        assert "Visiting hours" not in answer
        assert "About" not in answer

    def test_arabic_hours_question_has_no_description_addition(self, client):
        body = _answer(client, "ما هي ساعات زيارة معبد الكرنك؟")
        answer = body["answer"]
        assert "ساعات الزيارة" in answer
        # the "ما هي" prefix must not turn this into a definition answer
        assert "نبذة" not in answer

    def test_single_entity_definition_omits_unverified_hours(self, client):
        # a genuine one-place definition question keeps its description;
        # hours the KB marks unverified are silently omitted (they were not
        # requested, and they must not be presented as confirmed facts)
        body = _answer(client, "What is the Karnak Temple?")
        assert "About" in body["answer"]
        assert "Visiting hours" not in body["answer"]
        assert "currently unavailable" not in body["answer"].lower()
