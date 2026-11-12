"""JSON knowledge base loading and validation.

Supports two knowledge-base formats:

1. **Folder format** (the current repository layout): a global ``index.json``
   plus one folder per landmark containing ``info.json`` (structured facts)
   and optionally ``faq.json`` (Q&A pairs).

2. **Single-file format** (future enhanced RehlaAI file): one JSON document
   with collections such as ``cities`` / ``places`` / ``activities`` /
   ``itineraries`` / ``transportation_routes``. When configured via
   ``KB_SINGLE_FILE`` (or detected automatically), the loader adapts.

The loader **never writes** to the original JSON files.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.schemas import (
    SourceMetadata,
    StructuredPlace,
    VerificationMetadata,
    VerificationStatus,
)

logger = logging.getLogger(__name__)


class KBLoadError(Exception):
    """Raised when the knowledge base cannot be loaded or is invalid."""


@dataclass
class FAQRecord:
    """A single FAQ question/answer pair tied to a place."""

    faq_id: str
    place_id: str
    question: str
    answer: str
    source_file: Optional[str] = None


@dataclass
class LoadedKB:
    """Validated, in-memory view of the knowledge base."""

    kb_name: Optional[str] = None
    kb_version: Optional[str] = None
    kb_last_updated: Optional[str] = None
    disclaimer: Optional[str] = None
    places: Dict[str, StructuredPlace] = field(default_factory=dict)
    faqs: List[FAQRecord] = field(default_factory=list)
    # Enhanced-RehlaAI collections (empty for the legacy folder format)
    cities: List[Dict[str, Any]] = field(default_factory=list)
    activities: List[Dict[str, Any]] = field(default_factory=list)
    planning_rules: List[Dict[str, Any]] = field(default_factory=list)
    itineraries: List[Dict[str, Any]] = field(default_factory=list)
    transportation_routes: List[Dict[str, Any]] = field(default_factory=list)
    transportation_note: Optional[str] = None
    # Raw original files keyed by source path, kept for provenance.
    raw_index: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_places(self) -> int:
        return len(self.places)

    @property
    def num_faqs(self) -> int:
        return len(self.faqs)

    @property
    def is_enhanced(self) -> bool:
        return bool(self.cities or self.activities or self.itineraries)


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
REQUIRED_PLACE_FIELDS = ("id", "name")

# Values that mean "the source explicitly says this is not available".
_EXPLICIT_UNAVAILABLE = {
    "unavailable",
    "not available",
    "n/a",
    "na",
    "null",
    "none",
    "unknown",
    "غير متوفر",
    "غير معروف",
}


def _classify_value(value: Any) -> VerificationStatus:
    """Classify a raw value as unavailable / estimated / unverified."""
    if value is None:
        return VerificationStatus.MISSING
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _EXPLICIT_UNAVAILABLE or lowered.startswith("unknown"):
            return VerificationStatus.UNAVAILABLE
        if "estimate" in lowered or "estimated" in lowered:
            return VerificationStatus.ESTIMATED
    return VerificationStatus.UNVERIFIED


def _build_verification(record: Dict[str, Any]) -> VerificationMetadata:
    """Derive verification metadata from a place record.

    A record counts as ``verified`` only when it carries an explicit
    ``last_verified`` marker (e.g. ``ticket_price.last_verified``).
    """
    last_verified: Optional[str] = None
    notes_parts: List[str] = []

    # Look for explicit verification markers anywhere in the record.
    def _scan(obj: Any, path: str = "") -> None:
        if isinstance(obj, dict):
            for key, val in obj.items():
                key_l = str(key).lower()
                if key_l in ("last_verified", "verified_as_of", "last_updated") and isinstance(
                    val, (str, int, float)
                ):
                    nonlocal last_verified
                    if last_verified is None:
                        last_verified = str(val)
                elif key_l == "notes" and isinstance(val, str):
                    notes_parts.append(val)
                elif key_l in ("estimate", "estimated", "is_estimated") and val:
                    notes_parts.append(f"{path}{key}: estimated value")
                _scan(val, f"{path}{key}.")

    _scan(record)
    notes = " | ".join(dict.fromkeys(notes_parts)) if notes_parts else None
    status = VerificationStatus.VERIFIED if last_verified else VerificationStatus.UNVERIFIED
    return VerificationMetadata(
        status=status, last_verified=last_verified, notes=notes
    )


def _validate_place_record(record: Any, origin: str) -> Dict[str, Any]:
    if not isinstance(record, dict):
        raise KBLoadError(f"{origin}: place record must be a JSON object")
    for req in REQUIRED_PLACE_FIELDS:
        if req not in record or record[req] in (None, ""):
            raise KBLoadError(f"{origin}: place record missing required field '{req}'")
    return record


# ----------------------------------------------------------------------
# Folder-format loader
# ----------------------------------------------------------------------
def _place_from_info(
    info: Dict[str, Any], source_file: str, kb_meta: Dict[str, Any]
) -> StructuredPlace:
    location = info.get("location") or {}
    city = location.get("city") if isinstance(location, dict) else None
    source = SourceMetadata(
        kb_name=kb_meta.get("kb_name"),
        kb_version=kb_meta.get("kb_version"),
        kb_last_updated=kb_meta.get("kb_last_updated"),
        source_file=source_file,
        record_id=info.get("id"),
        last_updated=info.get("last_updated"),
        disclaimer=kb_meta.get("disclaimer"),
    )
    return StructuredPlace(
        place_id=str(info["id"]),
        name=str(info["name"]),
        name_arabic=info.get("name_arabic"),
        alternate_names=[str(n) for n in info.get("alternate_names", []) if n],
        city=city,
        category=info.get("category"),
        data=info,
        verification=_build_verification(info),
        source=source,
    )


def load_kb_folder(kb_path: Path) -> LoadedKB:
    """Load the per-landmark folder knowledge base."""
    kb_path = Path(kb_path)
    index_path = kb_path / "index.json"
    if not index_path.is_file():
        raise KBLoadError(f"Knowledge base index not found: {index_path}")

    with index_path.open("r", encoding="utf-8") as fh:
        try:
            index = json.load(fh)
        except json.JSONDecodeError as exc:
            raise KBLoadError(f"{index_path}: invalid JSON ({exc})") from exc

    kb = LoadedKB(
        kb_name=index.get("knowledge_base_name"),
        kb_version=index.get("version"),
        kb_last_updated=index.get("last_updated"),
        disclaimer=index.get("data_currency_disclaimer"),
        raw_index=index,
    )
    kb_meta = {
        "kb_name": kb.kb_name,
        "kb_version": kb.kb_version,
        "kb_last_updated": kb.kb_last_updated,
        "disclaimer": kb.disclaimer,
    }

    landmarks = index.get("landmarks")
    if not isinstance(landmarks, list) or not landmarks:
        raise KBLoadError(f"{index_path}: 'landmarks' must be a non-empty list")

    for entry in landmarks:
        folder_name = entry.get("folder")
        if not folder_name:
            raise KBLoadError(f"{index_path}: landmark entry missing 'folder': {entry}")
        folder = kb_path / folder_name.rstrip("/")

        info_path = folder / "info.json"
        if not info_path.is_file():
            logger.warning("Missing info.json for landmark '%s', skipping", entry.get("id"))
            continue
        try:
            with info_path.open("r", encoding="utf-8") as fh:
                info = json.load(fh)
        except json.JSONDecodeError as exc:
            raise KBLoadError(f"{info_path}: invalid JSON ({exc})") from exc

        info = _validate_place_record(info, str(info_path))
        place = _place_from_info(info, f"{folder_name.rstrip('/')}/info.json", kb_meta)
        kb.places[place.place_id] = place

        faq_path = folder / "faq.json"
        if faq_path.is_file():
            try:
                with faq_path.open("r", encoding="utf-8") as fh:
                    faq_data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise KBLoadError(f"{faq_path}: invalid JSON ({exc})") from exc
            faqs = faq_data.get("faqs", [])
            landmark_id = faq_data.get("landmark_id", place.place_id)
            for i, pair in enumerate(faqs):
                if not isinstance(pair, dict):
                    continue
                question, answer = pair.get("question"), pair.get("answer")
                if not question or not answer:
                    logger.warning("%s: FAQ #%d missing question/answer, skipped", faq_path, i)
                    continue
                kb.faqs.append(
                    FAQRecord(
                        faq_id=f"{landmark_id}::faq_{i}",
                        place_id=str(landmark_id),
                        question=str(question),
                        answer=str(answer),
                        source_file=f"{folder_name.rstrip('/')}/faq.json",
                    )
                )

    if not kb.places:
        raise KBLoadError(f"No valid landmark records loaded from {kb_path}")
    return kb


# ----------------------------------------------------------------------
# Single-file (enhanced RehlaAI) format loader
# ----------------------------------------------------------------------
def load_kb_single_file(path: Path) -> LoadedKB:
    """Load a single enhanced JSON knowledge base.

    The accepted shape is flexible: any top-level collection mapping or list
    whose records contain ``id``/``place_id`` and ``name`` fields is treated
    as a place collection; records whose ``collection``/``type`` field marks
    them as FAQ entries become FAQ records.
    """
    path = Path(path)
    if not path.is_file():
        raise KBLoadError(f"Knowledge base file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise KBLoadError(f"{path}: invalid JSON ({exc})") from exc
    if not isinstance(data, dict):
        raise KBLoadError(f"{path}: top-level JSON must be an object")

    kb = LoadedKB(
        kb_name=data.get("knowledge_base_name") or data.get("name"),
        kb_version=data.get("version"),
        kb_last_updated=data.get("last_updated"),
        disclaimer=data.get("data_currency_disclaimer") or data.get("disclaimer"),
        raw_index=data,
    )
    kb_meta = {
        "kb_name": kb.kb_name,
        "kb_version": kb.kb_version,
        "kb_last_updated": kb.kb_last_updated,
        "disclaimer": kb.disclaimer,
    }

    place_like_collections = []
    for key, value in data.items():
        if key in ("knowledge_base_name", "version", "last_updated", "description",
                   "data_currency_disclaimer", "disclaimer", "schema_notes"):
            continue
        records = value if isinstance(value, list) else None
        if records is None and isinstance(value, dict):
            # allow {id: record} mappings
            records = list(value.values())
        if records and all(isinstance(r, dict) for r in records):
            place_like_collections.append((key, records))

    for collection_name, records in place_like_collections:
        for record in records:
            if "question" in record and "answer" in record:
                pid = str(record.get("place_id") or record.get("landmark_id") or "")
                kb.faqs.append(
                    FAQRecord(
                        faq_id=str(record.get("id") or f"{collection_name}_{len(kb.faqs)}"),
                        place_id=pid,
                        question=str(record["question"]),
                        answer=str(record["answer"]),
                        source_file=str(path.name),
                    )
                )
                continue
            pid = record.get("place_id") or record.get("id")
            name = record.get("name") or record.get("name_en") or record.get("title")
            if not pid or not name:
                logger.warning("%s: %s record without id/name skipped", path.name, collection_name)
                continue
            record = _validate_place_record({**record, "id": str(pid), "name": name}, str(path))
            if str(pid) in kb.places:
                raise KBLoadError(f"{path}: duplicate place id '{pid}'")
            kb.places[str(pid)] = _place_from_info(
                record, str(path.name), kb_meta
            )

    if not kb.places:
        raise KBLoadError(f"No place records found in {path}")
    return kb


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def load_kb(
    kb_path: Optional[Path] = None,
    single_file: Optional[Path] = None,
) -> LoadedKB:
    """Load and validate the knowledge base.

    Precedence:
      1. single-file enhanced RehlaAI schema (dedicated adapter)
      2. single-file generic format
      3. per-landmark folder format
    """
    if single_file and Path(single_file).is_file():
        from rag.ingestion.enhanced_loader import load_enhanced_kb

        try:
            kb = load_enhanced_kb(Path(single_file))
            logger.info(
                "Loaded enhanced KB '%s': %s places, %s cities, %s activities, "
                "%s rules, %s itineraries, %s routes",
                kb.kb_name, kb.num_places, len(kb.cities), len(kb.activities),
                len(kb.planning_rules), len(kb.itineraries),
                len(kb.transportation_routes),
            )
            return kb
        except KBLoadError:
            # not the enhanced schema -> fall back to the generic single-file loader
            kb = load_kb_single_file(Path(single_file))
            logger.info(
                "Loaded single-file KB: %s places, %s FAQs", kb.num_places, kb.num_faqs
            )
            return kb

    if kb_path is None:
        raise KBLoadError("No knowledge base path provided")

    try:
        kb = load_kb_folder(Path(kb_path))
    except KBLoadError:
        # Folder failed: try treating the path itself as a single file.
        candidate = Path(kb_path)
        if candidate.suffix == ".json" and candidate.is_file():
            kb = load_kb_single_file(candidate)
        else:
            raise
    logger.info(
        "Loaded folder KB '%s': %s places, %s FAQs", kb.kb_name, kb.num_places, kb.num_faqs
    )
    return kb


def validate_kb(kb: LoadedKB) -> List[str]:
    """Run consistency checks; returns a list of human-readable issues.

    This is advisory: the pipeline keeps working, but surfaced issues let the
    caller log data-quality warnings (missing coordinates, unverified prices,
    ...).

    Field names are schema-aware: the enhanced RehlaAI KB uses
    ``coordinates``/``prices``/``accessibility``; the legacy folder format
    uses ``location.coordinates``/``ticket_price``/``accessibility_notes``.
    """
    issues: List[str] = []
    for pid, place in kb.places.items():
        data = place.data
        if kb.is_enhanced:
            coords = data.get("coordinates")
            if not isinstance(coords, dict) or coords.get("lat") is None:
                issues.append(f"{pid}: missing coordinates")
            if data.get("prices") is None:
                issues.append(f"{pid}: no prices field")
            if data.get("opening_hours") is None:
                issues.append(f"{pid}: no opening_hours field")
            if data.get("accessibility") is None:
                issues.append(f"{pid}: no accessibility information")
        else:
            location = data.get("location") or {}
            coords = location.get("coordinates") if isinstance(location, dict) else None
            if not coords:
                issues.append(f"{pid}: missing coordinates")
            if data.get("ticket_price") is None:
                issues.append(f"{pid}: no ticket_price field")
            elif _classify_value(data.get("ticket_price")) == VerificationStatus.UNAVAILABLE:
                issues.append(f"{pid}: ticket_price explicitly unavailable")
            if data.get("opening_hours") is None:
                issues.append(f"{pid}: no opening_hours field")
            if data.get("accessibility_notes") is None:
                issues.append(f"{pid}: no accessibility information")
        if place.verification.status == VerificationStatus.UNVERIFIED:
            issues.append(f"{pid}: no verification marker (treated as unverified)")
    return issues
