"""Structured retrieval: exact lookups over the JSON knowledge base.

This is the component that guarantees exact values: ticket prices, opening
hours, coordinates, accessibility, verification status, etc. Semantic
similarity is never used as a substitute for these lookups.

Design:
  - The structured data layer is the in-memory dict of ``StructuredPlace``
    records built by ``load_kb`` — raw original fields are preserved.
  - Lookups return ``StructuredLookup`` objects that distinguish:
      * a value found (exact source value, with verification metadata)
      * a field that is explicitly null/"unavailable" in the source
      * a field that simply does not exist in the source data
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from rag.ingestion.loader import LoadedKB
from rag.ingestion.enhanced_loader import map_status
from rag.schemas import SourceMetadata, StructuredPlace, VerificationMetadata, VerificationStatus

logger = logging.getLogger(__name__)


class StructuredLookup(BaseModel):
    """Result of a single structured lookup."""

    field: str
    status: VerificationStatus
    value: Any = None
    place_id: Optional[str] = None
    note: Optional[str] = None


class StructuredRetriever:
    """Exact structured lookups over place records and FAQs."""

    def __init__(self, kb: LoadedKB) -> None:
        self.kb = kb
        self._name_index = self._build_name_index()
        self._city_index = self._build_city_index()
        self._activity_name_index = self._build_activity_name_index()

    # ------------------------------------------------------------------
    def _build_name_index(self) -> Dict[str, str]:
        """Map lowercase name variants (English + Arabic) -> place_id."""
        index: Dict[str, str] = {}
        for pid, place in self.kb.places.items():
            names = [place.name, *(place.alternate_names or [])]
            if place.name_arabic:
                names.append(place.name_arabic)
            for name in names:
                if not name:
                    continue
                key = name.strip().lower()
                index.setdefault(key, pid)
                # also index without parenthesized suffixes, e.g. "grand egyptian museum (gem)"
                simplified = re.sub(r"\(.*?\)", "", name).strip().lower()
                if simplified and simplified != key:
                    index.setdefault(simplified, pid)
        return index

    def _build_city_index(self) -> Dict[str, Dict[str, Any]]:
        """Map lowercase city name variants -> city record (enhanced KB)."""
        index: Dict[str, Dict[str, Any]] = {}
        for city in self.kb.cities:
            names = [city.get("name_en"), city.get("name_ar"), city.get("region")]
            for name in names:
                if not name:
                    continue
                simplified = re.sub(r"\(.*?\)", "", str(name)).strip().lower()
                index.setdefault(simplified, city)
        return index

    def _build_activity_name_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for act in self.kb.activities:
            for name in (act.get("name_en"), act.get("name_ar")):
                if not name:
                    continue
                index.setdefault(str(name).strip().lower(), act)
        return index

    # ------------------------------------------------------------------
    def _build_name_index(self) -> Dict[str, str]:
        """Map lowercase name variants (English + Arabic) -> place_id."""
        index: Dict[str, str] = {}
        for pid, place in self.kb.places.items():
            names = [place.name, *(place.alternate_names or [])]
            if place.name_arabic:
                names.append(place.name_arabic)
            for name in names:
                if not name:
                    continue
                key = name.strip().lower()
                index.setdefault(key, pid)
                # also index without parenthesized suffixes, e.g. "grand egyptian museum (gem)"
                simplified = re.sub(r"\(.*?\)", "", name).strip().lower()
                if simplified and simplified != key:
                    index.setdefault(simplified, pid)
        return index

    # ------------------------------------------------------------------
    def find_by_id(self, place_id: str) -> Optional[StructuredPlace]:
        return self.kb.places.get(str(place_id))

    def find_by_name(self, name: str) -> Optional[StructuredPlace]:
        """Exact name match (English, Arabic, or an alternate name)."""
        if not name:
            return None
        key = name.strip().lower()
        pid = self._name_index.get(key)
        if pid:
            return self.kb.places[pid]
        # fuzzy containment: "grand egyptian museum in giza" still matches
        for name_key, pid in self._name_index.items():
            if len(name_key) >= 4 and name_key in key:
                return self.kb.places[pid]
        return None

    def find_places_by_city(self, city: str) -> List[StructuredPlace]:
        """Match a city against the ``city`` and ``governorate`` fields.

        Some records use a locality as city (e.g. Abu Simbel) with the
        governorate as the wider region (Aswan); both are legitimate answers
        to "places in X" and are taken verbatim from the source data.
        """
        if not city:
            return []
        city_l = city.strip().lower()
        out: List[StructuredPlace] = []
        for p in self.kb.places.values():
            location = p.data.get("location") or {}
            governorate = (
                location.get("governorate") if isinstance(location, dict) else None
            )
            if (
                (p.city and city_l in p.city.lower())
                or (governorate and city_l in str(governorate).lower())
            ):
                out.append(p)
        return out

    def find_places_by_category(self, category: str) -> List[StructuredPlace]:
        if not category:
            return []
        cat_l = category.strip().lower()
        return [
            p
            for p in self.kb.places.values()
            if p.category and (cat_l in p.category.lower() or p.category.lower() in cat_l)
        ]

    def find_by_coordinates(self, lat: float, lon: float, tolerance: float = 0.01) -> List[StructuredPlace]:
        out: List[StructuredPlace] = []
        for place in self.kb.places.values():
            coords = place.data.get("coordinates")
            if not isinstance(coords, dict):
                coords = (place.data.get("location") or {}).get("coordinates")
            if not isinstance(coords, dict):
                continue
            lat_p, lon_p = coords.get("lat", coords.get("latitude")), coords.get("lng", coords.get("longitude"))
            if lat_p is None or lon_p is None:
                continue
            if abs(float(lat_p) - lat) <= tolerance and abs(float(lon_p) - lon) <= tolerance:
                out.append(place)
        return out

    # ------------------------------------------------------------------
    # Enhanced-collection lookups (cities, activities, itineraries, rules)
    # ------------------------------------------------------------------
    def find_city(self, name: str) -> Optional[Dict[str, Any]]:
        """Exact city lookup by English/Arabic name or region."""
        if not name:
            return None
        key = re.sub(r"\(.*?\)", "", str(name)).strip().lower()
        return self._city_index.get(key)

    def all_cities(self) -> List[Dict[str, Any]]:
        return list(self.kb.cities)

    def find_activities_by_city(self, city: str) -> List[Dict[str, Any]]:
        if not city:
            return []
        city_l = city.strip().lower()
        return [
            a for a in self.kb.activities
            if any(city_l in str(c).lower() for c in (a.get("cities") or []))
        ]

    def find_activity_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        key = str(name).strip().lower()
        act = self._activity_name_index.get(key)
        if act:
            return act
        for name_key, candidate in self._activity_name_index.items():
            if len(name_key) >= 4 and name_key in key:
                return candidate
        return None

    def find_itineraries_by_city(self, city: str) -> List[Dict[str, Any]]:
        if not city:
            return []
        city_l = city.strip().lower()
        return [
            it for it in self.kb.itineraries
            if it.get("city") and city_l in str(it["city"]).lower()
        ]

    def find_itinerary(self, itinerary_id: str) -> Optional[Dict[str, Any]]:
        for it in self.kb.itineraries:
            if str(it.get("id")) == str(itinerary_id):
                return it
        return None

    def get_planning_rules_for_place(self, place_id: str) -> List[Dict[str, Any]]:
        return [
            rule for rule in self.kb.planning_rules
            if str(place_id) in [str(a) for a in (rule.get("applies_to") or [])]
        ]

    def get_transportation_routes(
        self, from_place_id: Optional[str] = None, to_place_id: Optional[str] = None
    ) -> StructuredLookup:
        """Exact transportation-route lookup.

        When the collection is empty in the source JSON, this returns an
        explicit ``unavailable`` status with the source's own note — routes
        are never invented.
        """
        routes = self.kb.transportation_routes
        if not routes:
            return StructuredLookup(
                field="transportation_routes",
                status=VerificationStatus.UNAVAILABLE,
                value=None,
                place_id=None,
                note=(
                    self.kb.transportation_note
                    or "transportation_routes collection is empty in the source JSON"
                ),
            )
        if from_place_id or to_place_id:
            routes = [
                r for r in routes
                if (not from_place_id or r.get("from_place_id") == str(from_place_id))
                and (not to_place_id or r.get("to_place_id") == str(to_place_id))
            ]
            if not routes:
                return StructuredLookup(
                    field="transportation_routes",
                    status=VerificationStatus.MISSING,
                    value=None,
                    note="no route matching the requested origin/destination",
                )
        return StructuredLookup(
            field="transportation_routes", status=VerificationStatus.UNVERIFIED,
            value=routes,
        )

    # ------------------------------------------------------------------
    # Field-level exact lookups
    # ------------------------------------------------------------------
    def get_field(self, place_id: str, field: str) -> StructuredLookup:
        """Exact value of ``field`` for a place, preserving source semantics."""
        place = self.kb.places.get(str(place_id))
        if place is None:
            return StructuredLookup(
                field=field, status=VerificationStatus.MISSING,
                note=f"Unknown place_id '{place_id}'",
            )
        data = place.data
        aliases = {
            "prices": "prices",
            "price": "prices",
            "ticket_price": "prices",
            "ticket_prices": "prices",
            "hours": "opening_hours",
            "opening": "opening_hours",
            "location.coordinates": "coordinates",
            "accessibility": "accessibility",
            "accessibility_notes": "accessibility",
            "transportation": "transportation_options",
            "transport": "transportation_options",
        }
        field_l = field.lower()
        # candidate paths: alias first, then the raw field name itself
        candidate_paths = [aliases.get(field_l, field)]
        if candidate_paths[0] != field_l:
            candidate_paths.append(field_l)

        value: Any = None
        field_path = None
        for path in candidate_paths:
            parts = path.split(".")
            current: Any = data
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    current = None
                    break
            if current is not None:
                value = current
                field_path = path
                break
            # remember whether the path exists at all (even if value is null)
            if self._path_exists(data, parts):
                field_path = path
                break

        if value is None:
            if field_path is not None:
                # field exists but is null/empty -> explicitly unavailable
                status = VerificationStatus.UNAVAILABLE
                note = f"Field '{field}' is explicitly null/unavailable in the source"
            else:
                status = VerificationStatus.MISSING
                note = f"Field '{field}' does not exist in the source record"
            return StructuredLookup(
                field=field, status=status, value=None,
                place_id=place.place_id, note=note,
            )

        status, note = self._lookup_verification(place, field_path, value)
        return StructuredLookup(
            field=field, status=status, value=value,
            place_id=place.place_id, note=note,
        )

    @staticmethod
    def _path_exists(data: Dict[str, Any], parts: List[str]) -> bool:
        container: Any = data
        for part in parts:
            if isinstance(container, dict) and part in container:
                container = container[part]
            else:
                return False
        return True

    def _lookup_verification(
        self, place: StructuredPlace, field_path: str, value: Any
    ) -> tuple:
        """Verification status for an exact field value.

        Priority: the record's per-field ``data_quality`` entry (exact source
        semantics) -> nested ``last_verified``/``confidence`` markers ->
        record-level verification.
        """
        root_field = field_path.split(".")[0]

        # 1) per-field data_quality (enhanced KB)
        dq = place.data.get("data_quality")
        if isinstance(dq, dict) and isinstance(dq.get(root_field), dict):
            entry = dq[root_field]
            status = map_status(entry.get("status"))
            note_parts = []
            if entry.get("status"):
                note_parts.append(f"data_quality status: {entry['status']}")
            if entry.get("source_url"):
                note_parts.append(f"source_url: {entry['source_url']}")
            if entry.get("verification_date"):
                note_parts.append(f"verification_date: {entry['verification_date']}")
            if entry.get("note"):
                note_parts.append(f"note: {entry['note']}")
            return status, (" | ".join(note_parts) if note_parts else None)

        # 2) nested markers on the value itself
        if isinstance(value, dict):
            if value.get("last_verified"):
                return VerificationStatus.VERIFIED, f"last_verified: {value['last_verified']}"
            if value.get("confidence"):
                status = map_status(value["confidence"])
                note = f"confidence: {value['confidence']}"
                if value.get("checked_on"):
                    note += f" (checked_on: {value['checked_on']})"
                if value.get("verify_at"):
                    note += f" (verify at: {value['verify_at']})"
                return status, note

        # 3) record-level verification
        return place.verification.status, place.verification.notes

    def get_verification(self, place_id: str) -> VerificationMetadata:
        place = self.kb.places.get(str(place_id))
        if place is None:
            return VerificationMetadata(status=VerificationStatus.MISSING)
        return place.verification

    # ------------------------------------------------------------------
    # Composite record retrieval (used by hybrid pipeline + context builder)
    # ------------------------------------------------------------------
    def full_record(self, place_id: str) -> Optional[StructuredPlace]:
        return self.kb.places.get(str(place_id))

    def records_by_ids(self, place_ids: List[str]) -> List[StructuredPlace]:
        seen, out = set(), []
        for pid in place_ids:
            if pid in seen:
                continue
            seen.add(pid)
            place = self.kb.places.get(str(pid))
            if place:
                out.append(place)
        return out

    def faqs_for_place(self, place_id: str) -> List[Dict[str, Any]]:
        return [
            {"faq_id": f.faq_id, "question": f.question, "answer": f.answer}
            for f in self.kb.faqs
            if f.place_id == str(place_id)
        ]

    def as_json(self, place: StructuredPlace) -> str:
        """Compact JSON serialization preserving original field names."""
        return json.dumps(
            {
                "place_id": place.place_id,
                "name": place.name,
                "name_arabic": place.name_arabic,
                "city": place.city,
                "category": place.category,
                "verification_status": place.verification.status.value,
                "last_verified": place.verification.last_verified,
                "source_file": place.source.source_file,
                "record": place.data,
            },
            ensure_ascii=False,
        )

    def summary_of(self, place: StructuredPlace) -> str:
        data = place.data
        coords = data.get("coordinates")
        if not isinstance(coords, dict):
            coords = (data.get("location") or {}).get("coordinates")
        lines = [
            # NOTE: no internal id here — this text goes to the LLM and must
            # not invite id citations in user-facing answers
            f"{place.name}"
            + (f" ({place.name_arabic})" if place.name_arabic else ""),
            f"City: {place.city or 'not available'}",
            f"Category: {place.category or 'not available'}",
            f"Coordinates: "
            + (
                f"lat {coords.get('lat', coords.get('latitude'))}, "
                f"lng {coords.get('lng', coords.get('longitude'))}"
                if isinstance(coords, dict)
                and (coords.get("lat") is not None or coords.get("latitude") is not None)
                else "not available"
            ),
        ]
        # opening hours (enhanced: daily_hours / legacy: season keys)
        hours = data.get("opening_hours")
        if isinstance(hours, dict):
            enhanced_keys = [
                k for k in ("daily_hours", "ramadan_hours", "last_entry", "hours_type")
                if hours.get(k)
            ]
            if enhanced_keys:
                lines.append(
                    "Opening hours: "
                    + "; ".join(f"{k}={hours[k]}" for k in enhanced_keys)
                    + (f" (confidence: {hours['confidence']})" if hours.get("confidence") else "")
                )
            else:
                summer = hours.get("summer_may_to_september")
                winter = hours.get("winter_october_to_april")
                general = hours.get("daily") or hours.get("general")
                if general or summer or winter:
                    lines.append(
                        "Opening hours: "
                        + "; ".join(str(v) for v in (general, summer, winter) if v)
                    )
                else:
                    lines.append("Opening hours: " + json.dumps(hours, ensure_ascii=False))
        # ticket prices (enhanced: tickets[] / legacy: flat numeric fields)
        prices = data.get("prices") if data.get("prices") is not None else data.get("ticket_price")
        if isinstance(prices, dict):
            currency = prices.get("currency", "EGP")
            confidence = prices.get("confidence")
            tickets = prices.get("tickets")
            if isinstance(tickets, list) and tickets:
                parts = []
                for t in tickets:
                    name = t.get("name", "ticket")
                    numeric = {
                        k: v for k, v in t.items()
                        if isinstance(v, (int, float))
                    }
                    if numeric:
                        parts.append(
                            f"{name}: "
                            + "; ".join(f"{k.replace('_', ' ')}={v}" for k, v in numeric.items())
                            + f" {currency}"
                        )
                    else:
                        parts.append(f"{name}: not available")
                label = "Ticket prices"
                if confidence:
                    label += f" (confidence: {confidence})"
                lines.append(label + ": " + " | ".join(parts))
            elif prices.get("note"):
                lines.append(
                    f"Ticket prices (confidence: {confidence}): no numeric ticket "
                    f"prices listed; source note: {prices['note']}"
                )
            else:
                numeric = {
                    k: v
                    for k, v in prices.items()
                    if isinstance(v, (int, float)) and k != "currency"
                }
                if numeric:
                    parts = [
                        f"{k.replace('_', ' ')}: {v} {currency}" for k, v in numeric.items()
                    ]
                    lines.append("Ticket prices (" + ("verified" if place.verification.status == VerificationStatus.VERIFIED else "unverified") + "): " + "; ".join(parts))
                else:
                    lines.append("Ticket prices: not available in the knowledge base")
        elif isinstance(prices, str):
            lines.append(f"Ticket prices: {prices}")
        elif prices is None:
            lines.append("Ticket prices: not present in knowledge base")
        for field in ("accessibility_notes", "nearest_airport"):
            v = data.get(field)
            if v:
                lines.append(f"{field}: {v}")
        # enhanced accessibility object
        accessibility = data.get("accessibility")
        if isinstance(accessibility, dict) and accessibility.get("details"):
            lines.append(f"Accessibility: {accessibility['details']}")
        return "\n".join(lines)
