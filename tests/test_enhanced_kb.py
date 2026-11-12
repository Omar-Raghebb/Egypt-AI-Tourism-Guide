"""Tests for the enhanced RehlaAI JSON knowledge base adapter.

Covers: schema validation, per-field verification semantics (exact source
status preserved — never upgraded), retrieval aliases, chunking of the
enhanced records (no legacy doc-id markers), and structured retrieval
against the real enhanced KB file.

These tests read the REAL enhanced KB (data/rehlaai_rag_data_enhanced.json);
it is only ever opened read-only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.ingestion.chunker import KBChunker
from rag.ingestion.enhanced_loader import (
    _field_status,
    load_enhanced_kb,
    map_status,
)
from rag.ingestion.loader import KBLoadError
from rag.query_processing import QueryProcessor
from rag.retrieval.structured import StructuredRetriever
from rag.schemas import VerificationStatus

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENHANCED_KB = PROJECT_ROOT / "data" / "rehlaai_rag_data_enhanced.json"


@pytest.fixture(scope="module")
def enhanced_kb():
    return load_enhanced_kb(ENHANCED_KB)


@pytest.fixture(scope="module")
def enhanced_data():
    return json.loads(ENHANCED_KB.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def structured(enhanced_kb):
    return StructuredRetriever(enhanced_kb)


@pytest.fixture(scope="module")
def processor(enhanced_kb, structured):
    return QueryProcessor(enhanced_kb, structured)


# =====================================================================
# Schema / loading / validation
# =====================================================================
class TestEnhancedKBSchema:
    def test_loads_enhanced_kb(self, enhanced_kb):
        assert enhanced_kb is not None

    def test_collection_counts(self, enhanced_kb):
        assert len(enhanced_kb.places) == 28
        assert len(enhanced_kb.cities) == 9
        assert len(enhanced_kb.activities) == 9
        assert len(enhanced_kb.planning_rules) == 3
        assert len(enhanced_kb.itineraries) == 6
        assert len(enhanced_kb.transportation_routes) == 0

    def test_kb_metadata(self, enhanced_kb):
        assert enhanced_kb.kb_name == "RehlaAI"
        assert enhanced_kb.kb_version is not None

    def test_every_place_has_required_ids(self, enhanced_kb):
        for pid, place in enhanced_kb.places.items():
            assert place.place_id == pid
            assert place.name
            assert place.data.get("place_id") == pid

    def test_missing_place_id_rejected(self, tmp_path):
        data = _minimal_enhanced()
        del data["places"][0]["place_id"]
        _write(tmp_path, data)
        with pytest.raises(KBLoadError, match="place_id"):
            load_enhanced_kb(tmp_path / "kb.json")

    def test_duplicate_place_id_rejected(self, tmp_path):
        data = _minimal_enhanced()
        data["places"].append(dict(data["places"][0]))
        _write(tmp_path, data)
        with pytest.raises(KBLoadError, match="duplicate"):
            load_enhanced_kb(tmp_path / "kb.json")

    def test_missing_collection_rejected(self, tmp_path):
        data = _minimal_enhanced()
        del data["cities"]
        _write(tmp_path, data)
        with pytest.raises(KBLoadError, match="cities"):
            load_enhanced_kb(tmp_path / "kb.json")

    def test_place_missing_required_field_rejected(self, tmp_path):
        data = _minimal_enhanced()
        del data["places"][0]["prices"]
        _write(tmp_path, data)
        with pytest.raises(KBLoadError, match="prices"):
            load_enhanced_kb(tmp_path / "kb.json")

    def test_invalid_json_rejected(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(KBLoadError, match="invalid JSON"):
            load_enhanced_kb(path)

    def test_missing_file_rejected(self, tmp_path):
        with pytest.raises(KBLoadError, match="not found"):
            load_enhanced_kb(tmp_path / "missing.json")

    def test_non_object_top_level_rejected(self, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(KBLoadError, match="object"):
            load_enhanced_kb(path)

    def test_transportation_note_preserved(self, enhanced_kb):
        assert enhanced_kb.transportation_note
        assert "transportation" in enhanced_kb.transportation_note.lower()


# =====================================================================
# Verification semantics — statuses are preserved, never upgraded
# =====================================================================
class TestVerificationSemantics:
    def test_record_level_unverified(self, enhanced_kb):
        # All 28 places are marked unverified in the source data.
        for place in enhanced_kb.places.values():
            assert place.verification.status == VerificationStatus.UNVERIFIED

    def test_last_verified_preserved(self, enhanced_kb):
        place = enhanced_kb.places["giza_pyramids_001"]
        assert place.verification.last_verified == "2026-11-11"

    def test_hours_field_unverified_despite_official_confidence(self, enhanced_kb):
        # giza hours carry confidence "official", but the data_quality block
        # says "requires_verification" -> per-field status stays UNVERIFIED.
        assert (
            _field_status(enhanced_kb.places["giza_pyramids_001"].data, "opening_hours")
            == VerificationStatus.UNVERIFIED
        )

    def test_hours_note_carries_requires_verification(self, enhanced_kb):
        dq = enhanced_kb.places["giza_pyramids_001"].data["data_quality"]
        assert dq["opening_hours"]["status"] == "requires_verification"

    def test_prices_conflicting_maps_to_unverified(self, enhanced_kb):
        status = _field_status(
            enhanced_kb.places["giza_pyramids_001"].data, "prices"
        )
        assert status == VerificationStatus.UNVERIFIED
        prices = enhanced_kb.places["giza_pyramids_001"].data["prices"]
        assert prices.get("confidence") == "conflicting"

    def test_coordinates_estimated(self, enhanced_kb):
        status = _field_status(
            enhanced_kb.places["giza_pyramids_001"].data, "coordinates"
        )
        assert status == VerificationStatus.ESTIMATED

    def test_no_place_upgraded_to_verified(self, enhanced_kb):
        verified = [
            p for p in enhanced_kb.places.values()
            if p.verification.status == VerificationStatus.VERIFIED
        ]
        assert verified == []

    def test_map_status_mapping(self):
        assert map_status("official") == VerificationStatus.VERIFIED
        assert map_status("verified") == VerificationStatus.VERIFIED
        assert map_status("estimated") == VerificationStatus.ESTIMATED
        assert map_status("unverified") == VerificationStatus.UNVERIFIED
        assert map_status("requires_verification") == VerificationStatus.UNVERIFIED
        assert map_status("conflicting") == VerificationStatus.UNVERIFIED
        assert map_status("unavailable") == VerificationStatus.UNAVAILABLE
        assert map_status(None) == VerificationStatus.UNVERIFIED
        assert map_status("unknown-nonsense") == VerificationStatus.UNVERIFIED


# =====================================================================
# Retrieval aliases (search_index aliases + keywords, minus generic words)
# =====================================================================
class TestRetrievalAliases:
    def test_common_alias_included(self, enhanced_kb):
        aliases = [a.lower() for a in enhanced_kb.places["giza_pyramids_001"].alternate_names]
        assert "pyramids" in aliases

    def test_short_name_keyword_included(self, enhanced_kb):
        aliases = [a.lower() for a in enhanced_kb.places["abu_simbel_001"].alternate_names]
        assert "abu simbel" in aliases

    def test_arabic_alias_included(self, enhanced_kb):
        aliases = enhanced_kb.places["karnak_001"].alternate_names
        assert any("\u0600" <= ch <= "\u06FF" for ch in "".join(aliases))

    def test_city_names_excluded(self, enhanced_kb):
        aliases = [a.lower() for a in enhanced_kb.places["giza_pyramids_001"].alternate_names]
        assert "giza" not in aliases

    def test_category_words_excluded(self, enhanced_kb, enhanced_data):
        meta = enhanced_data["_meta"]
        categories = {str(c).lower() for c in (meta.get("categories") or [])}
        for place in enhanced_kb.places.values():
            for alias in place.alternate_names:
                assert alias.lower() not in categories


# =====================================================================
# Chunking the enhanced KB — stable IDs, no legacy markers
# =====================================================================
class TestEnhancedChunker:
    @pytest.fixture(scope="class")
    def docs(self, enhanced_kb):
        chunker = KBChunker(chunk_size=300, chunk_overlap=45)
        return chunker.chunk_kb(enhanced_kb)

    def test_doc_count(self, docs):
        assert len(docs) == 263

    def test_doc_id_format(self, docs):
        for doc in docs:
            parts = doc.doc_id.split("::")
            if parts[0].startswith(("city_", "act_", "it_", "rule_", "transportation")):
                continue  # collection records have simple ids
            assert len(parts) == 3, doc.doc_id

    def test_no_legacy_place_markers(self, docs):
        legacy = ("luxor_temple::", "karnak_temple::", "faq_")
        assert not any(doc.doc_id.startswith(legacy) for doc in docs)

    def test_place_sections_present(self, docs):
        giza = [d.doc_id for d in docs if d.doc_id.startswith("giza_pyramids_001")]
        sections = {d.split("::")[1] for d in giza}
        assert {"overview", "facts", "practical", "accessibility"} <= sections

    def test_collection_docs_present(self, docs):
        ids = {d.doc_id.split("::")[0] for d in docs}
        assert "city_cairo" in ids
        assert "act_nile_cruise" in ids
        assert "it_cairo_history_3d" in ids
        assert "rule_001" in ids
        assert "transportation_routes_empty" in ids

    def test_chunks_respect_size(self, docs):
        # overview/facts/accessibility are split to chunk_size+overlap;
        # practical sections stay as one coherent bounded block
        assert all(len(doc.content) <= 2400 for doc in docs)


# =====================================================================
# Structured retrieval against the enhanced KB
# =====================================================================
class TestEnhancedStructuredRetrieval:
    def test_giza_prices_value_preserved(self, structured):
        lookup = structured.get_field("giza_pyramids_001", "prices")
        tickets = lookup.value["tickets"]
        amounts = {t["foreigner_adult"] for t in tickets}
        assert amounts == {1000}

    def test_giza_prices_status_unverified(self, structured):
        lookup = structured.get_field("giza_pyramids_001", "prices")
        assert lookup.status == VerificationStatus.UNVERIFIED

    def test_giza_hours_value(self, structured):
        lookup = structured.get_field("giza_pyramids_001", "hours")
        assert lookup.value["daily_hours"] == "07:00–17:00"

    def test_karnak_prices(self, structured):
        lookup = structured.get_field("karnak_001", "prices")
        main = [t for t in lookup.value["tickets"] if "Karnak Temple" in (t.get("name") or "")]
        assert {t["foreigner_adult"] for t in main} == {220}

    def test_gem_hours(self, structured):
        lookup = structured.get_field("gem_001", "hours")
        assert lookup.value["daily_hours"] == "09:00–18:00"

    def test_coordinates_estimated_status(self, structured):
        lookup = structured.get_field("giza_pyramids_001", "coordinates")
        assert lookup.status == VerificationStatus.ESTIMATED
        assert lookup.value["lat"] == 29.9792

    def test_unknown_place_is_missing(self, structured):
        lookup = structured.get_field("no_such_place", "prices")
        assert lookup.status == VerificationStatus.MISSING

    def test_missing_field_is_missing(self, structured):
        lookup = structured.get_field("giza_pyramids_001", "no_such_field")
        assert lookup.status == VerificationStatus.MISSING

    def test_transportation_routes_unavailable(self, structured):
        lookup = structured.get_transportation_routes()
        assert lookup.status == VerificationStatus.UNAVAILABLE
        assert lookup.note  # the source note is kept, not swallowed

    def test_price_query_detects_giza(self, processor):
        processed = processor.process("What is the ticket price of the Giza Pyramids?")
        assert "price" in processed.intents
        assert processed.detected_places == ["giza_pyramids_001"]

    def test_arabic_hours_query_detects_karnak(self, processor):
        processed = processor.process("ما مواعيد زيارة معبد الكرنك؟")
        assert "hours" in processed.intents
        assert "karnak_001" in processed.detected_places

    def test_itinerary_query_detects_itinerary(self, processor):
        processed = processor.process("Plan a 3-day itinerary in Cairo")
        assert "itinerary" in processed.intents

    def test_day_trip_wording_detects_itinerary(self, processor):
        processed = processor.process("Plan a 3-day trip to Luxor")
        assert "itinerary" in processed.intents

    def test_word_number_day_trip_detects_itinerary(self, processor):
        processed = processor.process("Suggest a one-day family trip to Giza")
        assert "itinerary" in processed.intents

    def test_rule_query_retrieves_rule(self, processor, structured):
        # the rule_001 text mentions Abu Simbel; alias "abu simbel" must
        # resolve to the temple, and the rule applies_to links them
        processed = processor.process("What is the planning rule for visiting Abu Simbel?")
        assert "abu_simbel_001" in processed.detected_places
        rules = structured.get_planning_rules_for_place("abu_simbel_001")
        assert any(r.get("id") == "rule_001" for r in rules)


# =====================================================================
# Helpers
# =====================================================================
def _minimal_enhanced():
    """A minimal valid enhanced-KB-shaped document (1 place, 1 city)."""
    return {
        "_meta": {
            "project": "RehlaAI",
            "version": "test-1.0",
            "categories": ["historical site"],
        },
        "cities": [
            {"id": "city_giza", "name_en": "Giza", "name_ar": "الجيزة", "region": "Greater Cairo"}
        ],
        "places": [
            {
                "id": "giza_pyramids_001",
                "place_id": "giza_pyramids_001",
                "name_en": "Giza Pyramids",
                "name_ar": "أهرامات الجيزة",
                "city": "Giza",
                "category": "historical site",
                "prices": {"currency": "EGP", "tickets": [], "confidence": "unverified"},
                "coordinates": {"lat": 29.9792, "lng": 31.1342},
                "opening_hours": {"daily_hours": "07:00–17:00", "confidence": "official"},
                "data_quality": {
                    "opening_hours": {"status": "requires_verification"},
                    "prices": {"status": "requires_verification"},
                },
                "verification_status": "unverified",
                "last_verified": "2026-11-11",
            }
        ],
        "activities": [],
        "planning_rules_examples": [],
        "itineraries": [],
        "transportation_routes": [],
        "transportation_routes_note": "No route data available.",
        "validation_rules": {"required_place_fields": ["id", "place_id", "name_en", "name_ar", "city", "category", "prices", "coordinates"]},
    }


def _write(tmp_path, data):
    path = tmp_path / "kb.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
