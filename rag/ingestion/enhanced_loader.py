"""Dedicated adapter for the enhanced RehlaAI JSON knowledge base.

Schema (v0.6, agentic_readiness schema_version 2.0):
  _meta, cities, places, activities, planning_rules_examples, itineraries,
  transportation_routes (may be empty), transportation_routes_note,
  validation_rules.

Design rules honored here:
  - the JSON file is only ever READ (never written, never rewritten)
  - original record fields (ids, place_id, prices, hours, coordinates,
    data_quality) are preserved verbatim inside StructuredPlace.data
  - per-field verification comes from each record's ``data_quality`` block
    and price/hours ``confidence`` values — nothing is upgraded to
    "verified" without an explicit source marker
  - transportation routes: when the collection is empty, the loader keeps
    the source note so structured retrieval can return an explicit
    "unavailable" result instead of inventing routes
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from rag.ingestion.loader import LoadedKB, KBLoadError
from rag.schemas import SourceMetadata, StructuredPlace, VerificationMetadata, VerificationStatus

logger = logging.getLogger(__name__)

# Enhanced schema markers
_REQUIRED_TOP_KEYS = ("cities", "places", "activities", "planning_rules_examples", "itineraries")

# data_quality / confidence status mapping. Everything not explicitly marked
# as verified/official stays unverified.
_STATUS_MAP = {
    "official": VerificationStatus.VERIFIED,
    "verified": VerificationStatus.VERIFIED,
    "official_page": VerificationStatus.VERIFIED,
    "estimated": VerificationStatus.ESTIMATED,
    "unverified": VerificationStatus.UNVERIFIED,
    "secondary_sources": VerificationStatus.UNVERIFIED,
    "typical_unverified": VerificationStatus.UNVERIFIED,
    "low_conflicting_sources": VerificationStatus.UNVERIFIED,
    "conflicting": VerificationStatus.UNVERIFIED,
    "unconfirmed": VerificationStatus.UNVERIFIED,
    "requires_verification": VerificationStatus.UNVERIFIED,
    "unavailable": VerificationStatus.UNAVAILABLE,
}


def map_status(raw: Any) -> VerificationStatus:
    """Map a raw status/confidence string to a VerificationStatus (never invent)."""
    if raw is None:
        return VerificationStatus.UNVERIFIED
    key = str(raw).strip().lower()
    return _STATUS_MAP.get(key, VerificationStatus.UNVERIFIED)


def _record_verification(record: Dict[str, Any]) -> VerificationMetadata:
    """Record-level verification from the record's own explicit fields."""
    status = map_status(record.get("verification_status"))
    last_verified = record.get("last_verified") or record.get("last_updated")
    notes: List[str] = []
    hours = record.get("opening_hours")
    if isinstance(hours, dict) and hours.get("confidence"):
        notes.append(f"opening_hours confidence: {hours['confidence']}")
    prices = record.get("prices")
    if isinstance(prices, dict) and prices.get("confidence"):
        notes.append(f"prices confidence: {prices['confidence']}")
    return VerificationMetadata(
        status=status,
        last_verified=str(last_verified) if last_verified else None,
        notes=" | ".join(notes) if notes else None,
    )


def _field_status(record: Dict[str, Any], field: str) -> VerificationStatus:
    """Per-field verification from data_quality (exact source semantics)."""
    dq = record.get("data_quality")
    if not isinstance(dq, dict):
        return VerificationStatus.UNVERIFIED
    entry = dq.get(field)
    if isinstance(entry, dict):
        return map_status(entry.get("status"))
    return VerificationStatus.UNVERIFIED


def _field_source_note(record: Dict[str, Any], field: str) -> Optional[str]:
    dq = record.get("data_quality")
    if not isinstance(dq, dict):
        return None
    entry = dq.get(field)
    if isinstance(entry, dict):
        parts = []
        if entry.get("source_url"):
            parts.append(f"source_url: {entry['source_url']}")
        if entry.get("verification_date"):
            parts.append(f"verification_date: {entry['verification_date']}")
        if entry.get("note"):
            parts.append(f"note: {entry['note']}")
        if parts:
            return " | ".join(str(p) for p in parts)
    return None


def _retrieval_aliases(record: Dict[str, Any], exclude: Set[str]) -> List[str]:
    """Aliases for retrieval matching: search_index aliases + keywords.

    Keywords are the source's own search index (``search_index.keywords``) and
    include common user phrasings ("Abu Simbel" for "Abu Simbel Temples").
    Generic words (categories, city names) are excluded so they can't trigger
    false place detections.
    """
    search_index = record.get("search_index") or {}
    aliases = [str(a).strip() for a in search_index.get("aliases", []) if a]
    for kw in search_index.get("keywords", []):
        kw = str(kw).strip()
        if kw and kw.lower() not in exclude and kw not in aliases:
            aliases.append(kw)
    return aliases


def _place_from_record(
    record: Dict[str, Any], source_file: str, kb_meta: Dict[str, Any], exclude: Set[str]
) -> StructuredPlace:
    pid = record.get("place_id") or record.get("id")
    name = record.get("name_en") or record.get("name") or record.get("name_ar")
    if not pid or not name:
        raise KBLoadError(f"{source_file}: place record missing id/name: {record.get('id')}")
    if not record.get("place_id"):
        raise KBLoadError(
            f"{source_file}: place '{pid}' missing required 'place_id' "
            "(validation_rules.required_place_fields)"
        )
    aliases = _retrieval_aliases(record, exclude)
    place = StructuredPlace(
        place_id=str(pid),
        name=str(name),
        name_arabic=record.get("name_ar"),
        alternate_names=aliases,
        city=record.get("city"),
        category=record.get("category"),
        data=record,
        verification=_record_verification(record),
        source=SourceMetadata(
            kb_name=kb_meta.get("kb_name"),
            kb_version=kb_meta.get("kb_version"),
            kb_last_updated=kb_meta.get("kb_last_updated"),
            source_file=source_file,
            record_id=record.get("id"),
            last_updated=record.get("last_updated"),
            disclaimer=kb_meta.get("disclaimer"),
        ),
    )
    return place


def _validate(data: Dict[str, Any], path: Path) -> None:
    for key in _REQUIRED_TOP_KEYS:
        if key not in data or not isinstance(data[key], list):
            raise KBLoadError(f"{path}: missing or invalid '{key}' collection")
    # unique place_id across places
    place_ids = [p.get("place_id") or p.get("id") for p in data["places"] if isinstance(p, dict)]
    if len(place_ids) != len(set(place_ids)):
        raise KBLoadError(f"{path}: duplicate place_id values (validation_rules.unique_keys)")
    for p in data["places"]:
        if not isinstance(p, dict):
            raise KBLoadError(f"{path}: non-object place record")
        for req in ("id", "place_id", "name_en", "name_ar", "city", "category", "prices", "coordinates"):
            if req not in p:
                raise KBLoadError(
                    f"{path}: place '{p.get('id')}' missing required field '{req}'"
                )


def load_enhanced_kb(path: Path) -> LoadedKB:
    """Load the enhanced RehlaAI JSON knowledge base (exact schema adapter)."""
    path = Path(path)
    if not path.is_file():
        raise KBLoadError(f"Enhanced KB file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise KBLoadError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise KBLoadError(f"{path}: top-level JSON must be an object")
    _validate(data, path)

    meta = data.get("_meta") or {}
    # words that must never act as place-name aliases (they would cause
    # false detections): categories and city names (EN + AR)
    exclude: Set[str] = {str(c).lower() for c in (meta.get("categories") or [])}
    for city in data.get("cities") or []:
        for key in ("name_en", "name_ar", "region"):
            if city.get(key):
                exclude.add(str(city[key]).lower())

    kb_meta = {
        "kb_name": meta.get("project") or "RehlaAI",
        "kb_version": meta.get("version"),
        "kb_last_updated": meta.get("generated_on") or meta.get("last_updated"),
        "disclaimer": meta.get("note"),
    }

    kb = LoadedKB(
        kb_name=kb_meta["kb_name"],
        kb_version=kb_meta["kb_version"],
        kb_last_updated=kb_meta["kb_last_updated"],
        disclaimer=kb_meta["disclaimer"],
        raw_index=data,
        cities=list(data.get("cities") or []),
        activities=list(data.get("activities") or []),
        planning_rules=list(data.get("planning_rules_examples") or []),
        itineraries=list(data.get("itineraries") or []),
        transportation_routes=list(data.get("transportation_routes") or []),
        transportation_note=data.get("transportation_routes_note"),
    )

    for record in data["places"]:
        place = _place_from_record(record, path.name, kb_meta, exclude)
        kb.places[place.place_id] = place

    if not kb.places:
        raise KBLoadError(f"No place records found in {path}")
    logger.info(
        "Enhanced KB validated: %s places, %s cities, %s activities, "
        "%s rules, %s itineraries, %s routes",
        len(kb.places), len(kb.cities), len(kb.activities),
        len(kb.planning_rules), len(kb.itineraries), len(kb.transportation_routes),
    )
    return kb
