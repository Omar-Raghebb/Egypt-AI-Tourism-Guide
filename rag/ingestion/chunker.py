"""Document conversion and chunking.

Chunking strategy (per collection):

- **Place records** are split into *semantic sections* first:
    - ``overview``     — name, category, city, coordinates, keywords, historical description
    - ``history``      — construction details, dynasty
    - ``practical``    — opening hours + ticket prices + best visiting time, serialized
      as compact JSON and kept **intact** (never split, so numeric facts never
      lose their context)
    - ``facts``        — interesting facts
    - ``accessibility``— accessibility notes, UNESCO status, nearest airport
  Only long free-text fields (descriptions) go through the
  ``RecursiveCharacterTextSplitter`` (size 300 / overlap 45 by config).

- **FAQ records** become one document per Q&A pair (they are self-contained
  facts, splitting them would separate questions from answers).

Limitations (documented in ``docs/RAG.md``):
  - Section splitting assumes the per-landmark info.json schema; records whose
    fields differ are handled gracefully (missing sections are skipped).
  - Very long practical JSON blocks are kept whole even if they exceed
    ``chunk_size`` — exact prices/hours are considered more important than
    uniform chunk sizes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.ingestion.loader import LoadedKB
from rag.schemas import SourceMetadata, VerificationMetadata, VerificationStatus

logger = logging.getLogger(__name__)


def _flatten_list(values: Iterable[Any], sep: str = "; ") -> str:
    parts = []
    for v in values:
        if v is None or str(v).strip() == "":
            continue
        parts.append(str(v))
    return sep.join(parts)


def _metadata_str(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Chroma metadata must be str/int/float/bool — coerce recursively."""
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out


class KBDocument:
    """A chunk ready for embedding with its metadata."""

    __slots__ = ("doc_id", "content", "metadata")

    def __init__(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> None:
        self.doc_id = doc_id
        self.content = content
        self.metadata = metadata

    def as_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "content": self.content, "metadata": dict(self.metadata)}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"KBDocument({self.doc_id!r}, {len(self.content)} chars)"


class KBChunker:
    """Convert validated knowledge-base records into retrievable documents."""

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 45) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", " — ", " ", ""],
            keep_separator=True,
        )

    # ------------------------------------------------------------------
    def _split_if_needed(self, text: str, max_keep: int = 1) -> List[str]:
        """Split long free text; short text passes through untouched."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size * max_keep:
            return [text]
        return [c.strip() for c in self._splitter.split_text(text) if c.strip()]

    # ------------------------------------------------------------------
    # Place sections
    # ------------------------------------------------------------------
    def _place_section_docs(
        self,
        place: Any,
        verification: VerificationMetadata,
        source: SourceMetadata,
    ) -> List[KBDocument]:
        data = place.data
        docs: List[KBDocument] = []

        def base_meta(section: str) -> Dict[str, Any]:
            meta: Dict[str, Any] = {
                "place_id": place.place_id,
                "place_name": place.name,
                "collection": "place",
                "section": section,
                "source_file": source.source_file,
                "verification_status": verification.status.value,
                "kb_version": source.kb_version,
                "kb_last_updated": source.kb_last_updated,
            }
            if place.name_arabic:
                meta["name_arabic"] = place.name_arabic
            if place.city:
                meta["city"] = place.city
            if place.category:
                meta["category"] = place.category
            if verification.last_verified:
                meta["last_verified"] = verification.last_verified
            if source.disclaimer:
                meta["kb_disclaimer"] = source.disclaimer
            return meta

        pid = place.place_id

        # --- overview -------------------------------------------------
        location = data.get("location") or {}
        coords = location.get("coordinates") if isinstance(location, dict) else None
        overview_lines: List[str] = [
            f"{place.name} ({place.name_arabic})" if place.name_arabic else place.name,
            f"Category: {place.category}" if place.category else "",
            f"City: {place.city}" if place.city else "",
            f"Region: {location.get('region')}" if isinstance(location, dict) and location.get("region") else "",
            f"Country: {location.get('country')}" if isinstance(location, dict) and location.get("country") else "",
            f"Coordinates: latitude {coords.get('latitude')}, longitude {coords.get('longitude')}"
            if isinstance(coords, dict) and coords.get("latitude") is not None
            else "Coordinates: not available",
            f"Overview: {data.get('historical_description')}" if data.get("historical_description") else "",
        ]
        if data.get("keywords"):
            overview_lines.append(f"Keywords: {_flatten_list(data['keywords'])}")
        for i, chunk in enumerate(self._split_if_needed("\n".join(l for l in overview_lines if l))):
            docs.append(KBDocument(f"{pid}::overview::{i}", chunk, base_meta("overview")))

        # --- history --------------------------------------------------
        history_lines: List[str] = []
        if data.get("dynasty"):
            history_lines.append(f"Dynasty/era: {data['dynasty']}")
        construction = data.get("construction_details")
        if isinstance(construction, dict):
            history_lines.append(f"Construction: {json.dumps(construction, ensure_ascii=False)}")
        elif isinstance(construction, str):
            history_lines.append(f"Construction: {construction}")
        if history_lines:
            for i, chunk in enumerate(self._split_if_needed("\n".join(history_lines))):
                docs.append(KBDocument(f"{pid}::history::{i}", chunk, base_meta("history")))

        # --- practical (kept intact, never split) ----------------------
        practical: Dict[str, Any] = {}
        if data.get("opening_hours") is not None:
            practical["opening_hours"] = data["opening_hours"]
        if data.get("ticket_price") is not None:
            practical["ticket_price"] = data["ticket_price"]
        if data.get("best_visiting_time") is not None:
            practical["best_visiting_time"] = data["best_visiting_time"]
        if practical:
            payload = json.dumps(practical, ensure_ascii=False, indent=2)
            doc = KBDocument(f"{pid}::practical::0", payload, base_meta("practical"))
            docs.append(doc)

        # --- facts ------------------------------------------------------
        facts = data.get("interesting_facts")
        if isinstance(facts, list) and facts:
            text = "Interesting facts:\n" + "\n".join(f"- {f}" for f in facts if f)
            for i, chunk in enumerate(self._split_if_needed(text)):
                docs.append(KBDocument(f"{pid}::facts::{i}", chunk, base_meta("facts")))
        elif isinstance(facts, str) and facts.strip():
            for i, chunk in enumerate(self._split_if_needed(facts)):
                docs.append(KBDocument(f"{pid}::facts::{i}", chunk, base_meta("facts")))

        # --- accessibility ---------------------------------------------
        access_lines: List[str] = []
        if data.get("accessibility_notes"):
            access_lines.append(f"Accessibility: {data['accessibility_notes']}")
        if data.get("unesco_status"):
            access_lines.append(f"UNESCO status: {data['unesco_status']}")
        if data.get("nearest_airport"):
            access_lines.append(f"Nearest airport: {data['nearest_airport']}")
        if access_lines:
            for i, chunk in enumerate(self._split_if_needed("\n".join(access_lines))):
                docs.append(KBDocument(f"{pid}::accessibility::{i}", chunk, base_meta("accessibility")))

        return docs

    # ------------------------------------------------------------------
    # FAQ pairs
    # ------------------------------------------------------------------
    def _faq_docs(self, kb: LoadedKB) -> List[KBDocument]:
        docs: List[KBDocument] = []
        for faq in kb.faqs:
            place = kb.places.get(faq.place_id)
            meta: Dict[str, Any] = {
                "place_id": faq.place_id,
                "collection": "faq",
                "section": "faq",
                "faq_id": faq.faq_id,
                "source_file": faq.source_file,
                "verification_status": (
                    kb.places[faq.place_id].verification.status.value
                    if place
                    else "unverified"
                ),
                "kb_version": kb.kb_version,
                "kb_last_updated": kb.kb_last_updated,
            }
            if place:
                meta["place_name"] = place.name
                if place.city:
                    meta["city"] = place.city
                if place.name_arabic:
                    meta["name_arabic"] = place.name_arabic
            content = f"Q: {faq.question}\nA: {faq.answer}"
            if len(content) > self.chunk_size:
                # Split only the answer, always keeping the question attached.
                parts = self._split_if_needed(faq.answer)
                for i, part in enumerate(parts):
                    docs.append(
                        KBDocument(
                            f"{faq.faq_id}::{i}",
                            f"Q: {faq.question}\nA: {part}",
                            meta,
                        )
                    )
            else:
                docs.append(KBDocument(faq.faq_id, content, meta))
        return docs

    # ------------------------------------------------------------------
    def chunk_kb(self, kb: LoadedKB) -> List[KBDocument]:
        """Convert the whole knowledge base into documents (deduplicated)."""
        _KB_HOLDER.kb = kb
        docs: List[KBDocument] = []
        seen: set = set()

        def add(doc: KBDocument) -> None:
            if doc.doc_id in seen:
                logger.debug("Duplicate doc id skipped: %s", doc.doc_id)
                return
            seen.add(doc.doc_id)
            docs.append(doc)

        for place in kb.places.values():
            if kb.is_enhanced:
                for doc in self._enhanced_place_docs(place, place.verification, place.source):
                    add(doc)
            else:
                for doc in self._place_section_docs(place, place.verification, place.source):
                    add(doc)
        for doc in self._faq_docs(kb):
            add(doc)

        # Enhanced-only collections (never mixed with the legacy folder KB)
        if kb.is_enhanced:
            for doc in self._city_docs(kb):
                add(doc)
            for doc in self._activity_docs(kb):
                add(doc)
            for doc in self._rule_docs(kb):
                add(doc)
            for doc in self._itinerary_docs(kb):
                add(doc)
            for doc in self._transportation_docs(kb):
                add(doc)

        logger.info("Chunked KB into %d documents", len(docs))
        return docs

    # ------------------------------------------------------------------
    # Enhanced-RehlaAI collections
    # ------------------------------------------------------------------
    def _collection_meta(
        self, collection: str, record_id: str, title: str, city: Optional[str],
        category: Optional[str], kb: LoadedKB, verification: VerificationMetadata,
        source_file: str,
    ) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "collection": collection,
            "record_id": record_id,
            "title": title,
            "source_file": source_file,
            "verification_status": verification.status.value,
            "kb_version": kb.kb_version,
            "kb_last_updated": kb.kb_last_updated,
        }
        if city:
            meta["city"] = city
        if category:
            meta["category"] = category
        if verification.last_verified:
            meta["last_verified"] = verification.last_verified
        if kb.disclaimer:
            meta["kb_disclaimer"] = kb.disclaimer
        return meta

    def _city_docs(self, kb: LoadedKB) -> List[KBDocument]:
        docs: List[KBDocument] = []
        for city in kb.cities:
            cid = str(city.get("id"))
            name_en, name_ar = city.get("name_en"), city.get("name_ar")
            title = f"{name_en} ({name_ar})" if name_ar else str(name_en)
            verification = VerificationMetadata(
                status=_enhanced_record_status(city),
                last_verified=city.get("last_verified"),
            )
            meta = self._collection_meta(
                "city", cid, title, city.get("region"), None, kb, verification, "rehlaai_rag_data_enhanced.json"
            )
            coords = city.get("coordinates") or {}
            lines = [
                f"City: {title}",
                f"Region: {city.get('region')}" if city.get("region") else "",
                f"Description: {city.get('description_en')}" if city.get("description_en") else "",
                f"الوصف: {city.get('description_ar')}" if city.get("description_ar") else "",
                f"Interests: {_flatten_list(city.get('interests', []))}" if city.get("interests") else "",
                f"Suggested days: {city.get('suggested_days')}" if city.get("suggested_days") else "",
                f"Nearby cities: {_flatten_list(city.get('nearby_cities', []))}" if city.get("nearby_cities") else "",
                f"Coordinates: lat {coords.get('lat')}, lng {coords.get('lng')}"
                if coords.get("lat") is not None else "Coordinates: not available",
            ]
            for i, chunk in enumerate(self._split_if_needed("\n".join(l for l in lines if l))):
                docs.append(KBDocument(f"{cid}::overview::{i}", chunk, meta))
        return docs

    def _activity_docs(self, kb: LoadedKB) -> List[KBDocument]:
        docs: List[KBDocument] = []
        for act in kb.activities:
            aid = str(act.get("id"))
            name_en, name_ar = act.get("name_en"), act.get("name_ar")
            title = f"{name_en} ({name_ar})" if name_ar else str(name_en)
            verification = VerificationMetadata(status=_enhanced_record_status(act))
            meta = self._collection_meta(
                "activity", aid, title, None, act.get("type"), kb, verification,
                "rehlaai_rag_data_enhanced.json",
            )
            lines = [
                f"Activity: {title}",
                f"Type: {act.get('type')}" if act.get("type") else "",
                f"Cities: {_flatten_list(act.get('cities', []))}" if act.get("cities") else "",
                f"Interests: {_flatten_list(act.get('interests', []))}" if act.get("interests") else "",
                f"Typical duration: {act.get('typical_duration')}" if act.get("typical_duration") else "",
                f"Description: {act.get('description_en')}" if act.get("description_en") else "",
                f"الوصف: {act.get('description_ar')}" if act.get("description_ar") else "",
                f"Suitable for: {_flatten_list(act.get('suitable_for', []))}" if act.get("suitable_for") else "",
            ]
            docs.append(KBDocument(aid, "\n".join(l for l in lines if l), meta))
        return docs

    def _rule_docs(self, kb: LoadedKB) -> List[KBDocument]:
        docs: List[KBDocument] = []
        for rule in kb.planning_rules:
            rid = str(rule.get("id"))
            text = rule.get("text_en") or rule.get("text_ar") or ""
            verification = VerificationMetadata(status=_enhanced_record_status(rule))
            meta = self._collection_meta(
                "planning_rule", rid, f"Planning rule {rid}", None, None, kb, verification,
                "rehlaai_rag_data_enhanced.json",
            )
            applies = rule.get("applies_to") or []
            lines = [f"Planning rule: {text}"]
            if applies:
                lines.append(f"Applies to: {_flatten_list(applies)}")
            docs.append(KBDocument(rid, "\n".join(lines), meta))
        return docs

    def _itinerary_docs(self, kb: LoadedKB) -> List[KBDocument]:
        docs: List[KBDocument] = []
        for it in kb.itineraries:
            iid = str(it.get("id"))
            title = f"{it.get('title_en')} ({it.get('title_ar')})" if it.get("title_ar") else str(it.get("title_en"))
            verification = VerificationMetadata(status=_enhanced_record_status(it))
            meta = self._collection_meta(
                "itinerary", iid, title, it.get("city"), None, kb, verification,
                "rehlaai_rag_data_enhanced.json",
            )
            payload = json.dumps(
                {
                    "id": iid,
                    "title_en": it.get("title_en"),
                    "title_ar": it.get("title_ar"),
                    "city": it.get("city"),
                    "days_count": it.get("days_count"),
                    "pace": it.get("pace"),
                    "interests": it.get("interests"),
                    "days": it.get("days"),
                },
                ensure_ascii=False,
            )
            docs.append(KBDocument(iid, payload, meta))
        return docs

    def _transportation_docs(self, kb: LoadedKB) -> List[KBDocument]:
        docs: List[KBDocument] = []
        if kb.transportation_routes:
            for route in kb.transportation_routes:
                rid = str(route.get("route_id") or route.get("id"))
                meta = self._collection_meta(
                    "transportation_route", rid, str(route.get("from_place_id")),
                    None, None, kb,
                    VerificationMetadata(status=_enhanced_record_status(route)),
                    "rehlaai_rag_data_enhanced.json",
                )
                docs.append(
                    KBDocument(rid, json.dumps(route, ensure_ascii=False), meta)
                )
        elif kb.transportation_note:
            # The source explicitly says no routes exist - reflect that, verbatim.
            meta = self._collection_meta(
                "transportation_route", "transportation_routes_empty",
                "Transportation routes", None, None, kb,
                VerificationMetadata(status=VerificationStatus.UNAVAILABLE),
                "rehlaai_rag_data_enhanced.json",
            )
            docs.append(
                KBDocument(
                    "transportation_routes_empty",
                    f"Transportation routes: not available in the knowledge base. "
                    f"Source note: {kb.transportation_note}",
                    meta,
                )
            )
        return docs

    # ------------------------------------------------------------------
    # Enhanced place records (field names: name_en/name_ar, prices, ...)
    # ------------------------------------------------------------------
    def _enhanced_place_docs(
        self,
        place: Any,
        verification: VerificationMetadata,
        source: SourceMetadata,
    ) -> List[KBDocument]:
        data = place.data
        docs: List[KBDocument] = []
        source_file = "rehlaai_rag_data_enhanced.json"

        def base_meta(section: str) -> Dict[str, Any]:
            meta = self._collection_meta(
                "place", place.place_id, place.name, place.city, place.category,
                _KB_HOLDER.kb, verification, source_file,
            )
            meta.pop("record_id", None)
            meta["place_id"] = place.place_id
            meta["place_name"] = place.name
            if place.name_arabic:
                meta["name_arabic"] = place.name_arabic
            return meta

        pid = place.place_id
        coords = data.get("coordinates") or {}

        # --- overview ----------------------------------------------------
        overview_lines = [
            f"{place.name} ({place.name_arabic})" if place.name_arabic else place.name,
            f"Category: {place.category}" if place.category else "",
            f"City: {place.city}" if place.city else "",
            f"Recommended visit duration: {data.get('recommended_duration_minutes')} minutes"
            if data.get("recommended_duration_minutes") else "",
            f"Description: {data.get('description_en')}" if data.get("description_en") else "",
            f"الوصف: {data.get('description_ar')}" if data.get("description_ar") else "",
            f"Coordinates: lat {coords.get('lat')}, lng {coords.get('lng')}"
            if coords.get("lat") is not None else "Coordinates: not available",
            f"Highlights: {_flatten_list(data.get('highlights_en', []))}" if data.get("highlights_en") else "",
            f"Interests: {_flatten_list(data.get('interests', []))}" if data.get("interests") else "",
            f"Suitable for: {_flatten_list(data.get('suitable_for', []))}" if data.get("suitable_for") else "",
        ]
        for i, chunk in enumerate(self._split_if_needed("\n".join(l for l in overview_lines if l))):
            docs.append(KBDocument(f"{pid}::overview::{i}", chunk, base_meta("overview")))

        # --- practical (opening hours + prices + booking, kept intact) ----
        practical: Dict[str, Any] = {}
        if data.get("opening_hours") is not None:
            practical["opening_hours"] = data["opening_hours"]
        if data.get("prices") is not None:
            practical["prices"] = data["prices"]
        if data.get("booking") is not None:
            practical["booking"] = data["booking"]
        if practical:
            docs.append(
                KBDocument(
                    f"{pid}::practical::0",
                    json.dumps(practical, ensure_ascii=False, indent=2),
                    base_meta("practical"),
                )
            )

        # --- facts --------------------------------------------------------
        facts_lines = []
        if data.get("things_to_do_en"):
            facts_lines.append(f"Things to do: {_flatten_list(data['things_to_do_en'])}")
        if data.get("things_to_do_ar"):
            facts_lines.append(f"أنشطة: {_flatten_list(data['things_to_do_ar'])}")
        if data.get("best_activities"):
            facts_lines.append(f"Best activities: {_flatten_list(data['best_activities'])}")
        if data.get("visit_rules"):
            facts_lines.append(f"Visit rules: {_flatten_list(data['visit_rules'])}")
        if data.get("best_time"):
            facts_lines.append(f"Best time: {json.dumps(data['best_time'], ensure_ascii=False)}")
        if facts_lines:
            for i, chunk in enumerate(self._split_if_needed("\n".join(facts_lines))):
                docs.append(KBDocument(f"{pid}::facts::{i}", chunk, base_meta("facts")))

        # --- access / logistics -------------------------------------------
        access_lines = []
        accessibility = data.get("accessibility")
        if isinstance(accessibility, dict):
            if accessibility.get("details"):
                access_lines.append(f"Accessibility: {accessibility['details']}")
            else:
                access_lines.append(
                    "Accessibility: not available in the knowledge base "
                    f"(status: {accessibility.get('status', 'unverified')})"
                )
        if data.get("indoor_outdoor") is None:
            access_lines.append("Indoor/outdoor classification: not available in the knowledge base")
        else:
            access_lines.append(f"Indoor/outdoor: {data['indoor_outdoor']}")
        if data.get("age_suitability") is None:
            access_lines.append("Age suitability: not available in the knowledge base")
        else:
            access_lines.append(f"Age suitability: {json.dumps(data['age_suitability'], ensure_ascii=False)}")
        t_opts = data.get("transportation_options")
        if not t_opts:
            access_lines.append("Transportation options: not available in the knowledge base")
        else:
            access_lines.append(f"Transportation options: {json.dumps(t_opts, ensure_ascii=False)}")
        if data.get("estimated_travel_time"):
            access_lines.append(f"Estimated travel time: {json.dumps(data['estimated_travel_time'], ensure_ascii=False)}")
        if data.get("estimated_transport_cost"):
            access_lines.append(f"Estimated transport cost: {json.dumps(data['estimated_transport_cost'], ensure_ascii=False)}")
        if data.get("effort_level"):
            access_lines.append(f"Effort level: {data['effort_level']}" + (
                f" ({data['effort_notes']})" if data.get("effort_notes") else ""
            ))
        if data.get("address"):
            access_lines.append(f"Address: {data['address']}" if isinstance(data["address"], str) else
                                f"Address: {json.dumps(data['address'], ensure_ascii=False)}")
        if data.get("google_maps_url"):
            access_lines.append(f"Google Maps: {data['google_maps_url']}")
        if data.get("nearby_places"):
            access_lines.append(f"Nearby places: {_flatten_list(data['nearby_places'])}")
        for i, chunk in enumerate(self._split_if_needed("\n".join(access_lines))):
            docs.append(KBDocument(f"{pid}::accessibility::{i}", chunk, base_meta("accessibility")))

        return docs


class _KBHolder:
    """Tiny context bridge so metadata helpers can see the LoadedKB."""

    kb: Optional[LoadedKB] = None


_KB_HOLDER = _KBHolder()


def _enhanced_record_status(record: Dict[str, Any]) -> VerificationStatus:
    from rag.ingestion.enhanced_loader import map_status

    return map_status(record.get("verification_status") or record.get("data_quality", {}).get("status"))
