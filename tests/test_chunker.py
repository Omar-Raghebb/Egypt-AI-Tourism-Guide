"""Tests for chunking and document conversion."""

import json


class TestPlaceChunking:
    def test_creates_section_docs(self, chunker, mini_kb):
        docs = chunker.chunk_kb(mini_kb)
        assert len(docs) > 0
        giza_sections = {
            d.metadata["section"] for d in docs if d.metadata["place_id"] == "giza_pyramids"
        }
        assert {"overview", "practical", "facts"} <= giza_sections

    def test_practical_kept_intact(self, chunker, mini_kb):
        """Prices and hours must never be split across chunks."""
        docs = chunker.chunk_kb(mini_kb)
        practical = [
            d for d in docs
            if d.metadata["place_id"] == "giza_pyramids"
            and d.metadata["section"] == "practical"
        ]
        assert len(practical) == 1
        payload = json.loads(practical[0].content)
        assert "opening_hours" in payload
        assert "ticket_price" in payload
        # exact numeric price preserved
        assert payload["ticket_price"]["general_plateau_entry_foreign_adult"] == 700

    def test_metadata_preserved_in_every_chunk(self, chunker, mini_kb):
        docs = chunker.chunk_kb(mini_kb)
        for doc in docs:
            assert doc.metadata["place_id"]
            assert doc.metadata["collection"] in ("place", "faq")
            assert doc.metadata["verification_status"]
            assert doc.metadata["source_file"]
            assert doc.metadata["section"]

    def test_doc_ids_unique(self, chunker, mini_kb):
        docs = chunker.chunk_kb(mini_kb)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids))

    def test_chunks_respect_size(self, chunker, mini_kb):
        docs = chunker.chunk_kb(mini_kb)
        long_free_text = [
            d for d in docs
            if d.metadata["section"] in ("overview", "history")
            and d.metadata["collection"] == "place"
        ]
        for doc in long_free_text:
            # small overage allowed for keep_separator behavior
            assert len(doc.content) <= chunker.chunk_size + 120

    def test_faq_docs_pair_question_and_answer(self, chunker, mini_kb):
        docs = chunker.chunk_kb(mini_kb)
        faq_docs = [d for d in docs if d.metadata["collection"] == "faq"]
        assert faq_docs
        for doc in faq_docs:
            assert doc.content.startswith("Q:")
            assert "\nA:" in doc.content

    def test_arabic_name_in_metadata(self, chunker, mini_kb):
        docs = chunker.chunk_kb(mini_kb)
        giza = [d for d in docs if d.metadata["place_id"] == "giza_pyramids"]
        assert all(d.metadata.get("name_arabic") == "أهرامات الجيزة" for d in giza)
