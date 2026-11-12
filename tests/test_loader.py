"""Tests for JSON loading and validation."""

import pytest

from rag.ingestion.loader import (
    KBLoadError,
    LoadedKB,
    load_kb,
    load_kb_folder,
    load_kb_single_file,
    validate_kb,
)


class TestFolderLoading:
    def test_loads_real_kb(self, mini_kb, mini_kb_dir):
        assert isinstance(mini_kb, LoadedKB)
        assert mini_kb.num_places == 2
        assert {"giza_pyramids", "luxor_temple"} <= set(mini_kb.places)
        assert mini_kb.num_faqs > 0
        assert mini_kb.kb_name
        assert mini_kb.kb_version

    def test_place_ids_preserved(self, mini_kb):
        for pid in ("giza_pyramids", "luxor_temple"):
            assert pid in mini_kb.places

    def test_original_data_untouched(self, mini_kb):
        giza = mini_kb.places["giza_pyramids"]
        # original field names and values survive verbatim
        assert giza.data["location"]["coordinates"]["latitude"] == 29.9792
        assert giza.data["ticket_price"]["currency"] == "EGP"
        assert giza.name_arabic == "أهرامات الجيزة"

    def test_missing_index_raises(self, tmp_path):
        with pytest.raises(KBLoadError):
            load_kb_folder(tmp_path)

    def test_invalid_json_raises(self, tmp_path):
        (tmp_path / "index.json").write_text("{invalid", encoding="utf-8")
        with pytest.raises(KBLoadError):
            load_kb_folder(tmp_path)

    def test_original_files_not_modified(self, mini_kb_dir):
        info = mini_kb_dir / "giza_pyramids" / "info.json"
        before = info.read_bytes()
        load_kb_folder(mini_kb_dir)
        assert info.read_bytes() == before


class TestSingleFileLoading:
    def test_enhanced_format(self, tmp_path):
        kb_file = tmp_path / "enhanced_kb.json"
        kb_file.write_text(
            """
            {
              "knowledge_base_name": "Enhanced RehlaAI KB",
              "places": [
                {"place_id": "p1", "name": "Test Place", "name_arabic": "مكان",
                 "location": {"city": "Cairo"}},
                {"id": "p2", "name": "Second Place", "category": "museum"}
              ],
              "faqs": [
                {"place_id": "p1", "question": "Q1?", "answer": "A1"}
              ]
            }
            """,
            encoding="utf-8",
        )
        kb = load_kb_single_file(kb_file)
        assert kb.num_places == 2
        assert set(kb.places) == {"p1", "p2"}
        assert kb.num_faqs == 1
        assert kb.faqs[0].place_id == "p1"

    def test_duplicate_ids_rejected(self, tmp_path):
        kb_file = tmp_path / "bad.json"
        kb_file.write_text(
            '{"places": [{"id": "p1", "name": "A"}, {"id": "p1", "name": "B"}]}',
            encoding="utf-8",
        )
        with pytest.raises(KBLoadError):
            load_kb_single_file(kb_file)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(KBLoadError):
            load_kb_single_file(tmp_path / "nope.json")


class TestValidation:
    def test_validate_flags_issues(self, mini_kb):
        issues = validate_kb(mini_kb)
        # advisory only: should run and return a list of strings
        assert isinstance(issues, list)
        assert all(isinstance(i, str) for i in issues)

    def test_verification_metadata(self, mini_kb):
        giza = mini_kb.places["giza_pyramids"]
        # giza info.json contains ticket_price.last_verified = "2026-07"
        assert giza.verification.status.value == "verified"
        assert giza.verification.last_verified == "2026-07"
