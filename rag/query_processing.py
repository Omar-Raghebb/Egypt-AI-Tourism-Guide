"""Query processing: normalization, entity extraction, intent detection.

Supports Arabic and English with pattern-driven (not example-hardcoded) intent
detection. Detected intents route queries to the appropriate retrieval
methods:

  - ``structured_exact`` intents (price, hours, coordinates, accessibility,
    transportation, indoor/outdoor, age suitability) trigger structured
    retrieval in addition to hybrid retrieval.
  - ``semantic`` intents (recommendations, descriptions, things-to-do) go
    through hybrid dense+sparse retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.ingestion.loader import LoadedKB
from rag.retrieval.normalize import normalize_text
from rag.retrieval.structured import StructuredRetriever

# Arabic diacritics/tatweel stripped before matching
_STRIP = re.compile(r"[\u064B-\u065F\u0640]")

INTENT_PATTERNS: Dict[str, List[str]] = {
    "price": [
        r"\bticket", r"\bprice", r"\bcost", r"\bfee", r"\badmission",
        r"\bentry\b", r"\bentrance\b", r"how much",
        r"سعر", r"أسعار", r"تذكرة", r"تذاكر", r"تكلفة", r"كم", r"دخول",
    ],
    "hours": [
        r"opening", r"hours", r"close", r"open\b", r"when .*open", r"when .*close",
        r"مواعيد", r"دوام", r"ساعات", r"يفتح", r"يغلق", r"وقت الدخول", r"وقت الفتح",
    ],
    "coordinates": [
        r"coordinates", r"latitude", r"longitude", r"where located", r"exact location",
        r"إحداثيات", r"خط العرض", r"خط الطول", r"الموقع الدقيق",
    ],
    "accessibility": [
        r"accessib", r"wheelchair", r"disab", r"mobility",
        r"ذوي", r"كرسي", r"وصول لذوي", r"ميسر", r"مقاعد",
    ],
    "transportation": [
        r"transport", r"get to", r"getting there", r"how to reach", r"reach\b",
        r"airport", r"bus\b", r"train\b", r"metro\b", r"taxi\b",
        r"مواصلات", r"وسائل", r"تنقل", r"مطار", r"أتوصل", r"ازاي أروح", r"إزاي أروح",
    ],
    "indoor_outdoor": [
        r"indoor", r"outdoor", r"inside or outside",
        r"داخلي", r"خارجي", r"مغلق", r"مفتوح", r"في الهواء",
    ],
    "age_suitability": [
        r"age", r"kids", r"children", r"child", r"family", r"suitable for",
        r"عمر", r"أطفال", r"عائلة", r"مناسب للأطفال",
    ],
    "activities": [
        r"activit", r"things to do", r"what can (i|we) do", r"things to see", r"do in",
        r"أنشطة", r"نشاط", r"أعمل", r"اعمل", r"أشوف", r"أزور", r"أشياء",
    ],
    "itinerary": [
        r"itinerar", r"day plan", r"trip plan", r"schedule",
        r"\b\d+[-\s]day\b", r"\b(?:one|two|three|four|five|six|seven)[-\s]day\b",
        r"\b\d+\s*days?\b", r"\b(?:one|two|three|four|five|six|seven)\s+days?\b",
        r"trip to", r"journey to",
        r"برنامج", r"خط سير", r"خطة", r"جولة يوم", r"خطوط سير",
    ],
    "recommendation": [
        r"best places", r"recommend", r"suggest", r"top places", r"must visit",
        r"visit in", r"places to see",
        r"أفضل", r"أحسن", r"انصح", r"أنصح", r"ترشيح", r"أزور", r"أشوف",
    ],    "definition": [
        r"what is", r"who built", r"who made", r"history of", r"tell me about", r"about\b",
        r"ما هو", r"ما هى", r"من بني", r"من صنع", r"تاريخ", r"عن", r"حدثني", r"اخبرني",
    ],
}

# Intents that must hit the structured data layer (exact values).
STRUCTURED_INTENTS = {
    "price",
    "hours",
    "coordinates",
    "accessibility",
    "transportation",
    "indoor_outdoor",
    "age_suitability",
}

# Intent -> canonical field name in the structured data layer.
# The names are resolved by StructuredRetriever.get_field aliases, which accept
# both the enhanced schema names (prices/coordinates) and the legacy folder
# format names (ticket_price/location.coordinates).
INTENT_FIELD = {
    "price": "prices",
    "hours": "opening_hours",
    "coordinates": "coordinates",
    "accessibility": "accessibility",
    "transportation": "transportation_options",
    "indoor_outdoor": "indoor_outdoor",
    "age_suitability": "age_suitability",
}

_FILTER_KEYS = {
    "city": [r"\bin\s+([A-Z][a-zA-Z]+)\b"],
    "category": [],
}

CITY_ALIASES = {
    "القاهرة": "Cairo",
    "الجيزة": "Giza",
    "الأقصر": "Luxor",
    "اقصر": "Luxor",
    "الاقصر": "Luxor",
    "أسوان": "Aswan",
    "اسوان": "Aswan",
    "الغردقة": "Hurghada",
    "شرم الشيخ": "Sharm El Sheikh",
    "الإسكندرية": "Alexandria",
    "الاسكندرية": "Alexandria",
}


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    raw: str
    normalized: str
    intent: str
    structured_fields: List[str] = field(default_factory=list)
    detected_places: List[str] = field(default_factory=list)
    detected_city: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    warnings: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)  # ALL matched intents


def normalize_query(text: str) -> str:
    """Strip diacritics, collapse whitespace, normalize Arabic letters."""
    text = _STRIP.sub("", text or "")
    text = normalize_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def detect_language(text: str) -> str:
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text or ""))
    latin_chars = len(re.findall(r"[A-Za-z]", text or ""))
    return "ar" if arabic_chars > latin_chars else "en"


class QueryProcessor:
    """Bilingual query analyzer and router."""

    def __init__(self, kb: LoadedKB, structured: Optional[StructuredRetriever] = None):
        self.kb = kb
        self.structured = structured or StructuredRetriever(kb)
        self._name_patterns = self._compile_name_patterns()

    # ------------------------------------------------------------------
    def _compile_name_patterns(self) -> List[Any]:
        """Pre-compiled regexes for place names and aliases (normalized)."""
        patterns: List[Any] = []
        for pid, place in self.kb.places.items():
            names = [place.name, *(place.alternate_names or [])]
            if place.name_arabic:
                names.append(place.name_arabic)
            for name in names:
                if not name:
                    continue
                simplified = re.sub(r"\(.*?\)", "", name).strip()
                simplified = re.sub(r"[-–].*$", "", simplified).strip()
                for candidate in {name, simplified}:
                    if not candidate or len(candidate) < 3:
                        continue
                    # normalize the pattern the same way queries are
                    # normalized, so Arabic hamza/teh-marbuta variants match
                    escaped = re.escape(normalize_query(candidate))
                    patterns.append((pid, re.compile(escaped, re.IGNORECASE), candidate))
                    # "Giza Pyramids and Sphinx" should also match the short
                    # user phrasing "Giza Pyramids"
                    if " and " in candidate.lower():
                        first_part = re.split(r"\s+and\s+", candidate, flags=re.IGNORECASE)[0].strip()
                        if len(first_part) >= 3:
                            patterns.append(
                                (pid, re.compile(re.escape(normalize_query(first_part)), re.IGNORECASE), first_part)
                            )
        return patterns

    # ------------------------------------------------------------------
    def detect_entities(self, query: str) -> Dict[str, Any]:
        """Extract place names, city names, and category hints from a query."""
        normalized = normalize_query(query)
        detected_places: List[str] = []
        for pid, pattern, original in self._name_patterns:
            if pattern.search(normalized):
                if pid not in detected_places:
                    detected_places.append(pid)

        detected_city = self._detect_city(normalized)

        filters: Dict[str, Any] = {}
        if detected_city:
            filters["city"] = detected_city
        if "family" in normalized or "عائلة" in normalized:
            filters["family_friendly"] = True
        if re.search(r"\bindoor\b|مغلق|داخلي", normalized, re.IGNORECASE):
            filters["indoor"] = True
        if re.search(r"\boutdoor\b|خارجي|في الهواء", normalized, re.IGNORECASE):
            filters["indoor"] = False

        return {
            "places": detected_places,
            "city": detected_city,
            "filters": filters,
        }

    def _detect_city(self, normalized: str) -> Optional[str]:
        # Known Egyptian cities: first from the cities collection (enhanced KB),
        # then from place records, then Arabic aliases.
        cities = set()
        for city in getattr(self.kb, "cities", []):
            for key in ("name_en", "name_ar"):
                if city.get(key):
                    cities.add(str(city[key]))
        for place in self.kb.places.values():
            if place.city:
                cities.add(place.city)
        for city in cities:
            if re.search(re.escape(city), normalized, re.IGNORECASE):
                return city
        # Arabic aliases: the query text is normalized (hamza/alef variants
        # collapsed), so compare against a normalized alias as well
        city_by_ar = {}
        for city in getattr(self.kb, "cities", []):
            if city.get("name_en") and city.get("name_ar"):
                city_by_ar[city["name_ar"]] = city["name_en"]
        for ar, en in {**CITY_ALIASES, **city_by_ar}.items():
            if ar in normalized or normalize_text(ar) in normalized:
                return en
        return None

    # ------------------------------------------------------------------
    def detect_intents(self, query: str) -> List[str]:
        """Return matched intents, ordered by specificity."""
        normalized = normalize_query(query)
        matched: List[str] = []
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                # normalize the pattern too so Arabic hamza/alef variants in
                # the patterns match the normalized query text
                if re.search(normalize_query(pattern), normalized, re.IGNORECASE):
                    matched.append(intent)
                    break
        # Order: structured intents first, then activities, itinerary, recommendation, definition
        priority = [
            "price", "hours", "coordinates", "accessibility", "transportation",
            "indoor_outdoor", "age_suitability", "activities", "itinerary",
            "recommendation", "definition",
        ]
        return sorted(matched, key=lambda i: priority.index(i) if i in priority else 99)

    # ------------------------------------------------------------------
    def process(self, query: str) -> ProcessedQuery:
        """Full analysis of a user query."""
        normalized = normalize_query(query)
        language = detect_language(query)
        entities = self.detect_entities(query)
        intents = self.detect_intents(query)

        structured_fields: List[str] = []
        for intent in intents:
            if intent in STRUCTURED_INTENTS:
                field = INTENT_FIELD[intent]
                if field not in structured_fields:
                    structured_fields.append(field)

        # Query without any recognized intent defaults to semantic search.
        intent = intents[0] if intents else "semantic"
        warnings: List[str] = []
        if not entities["places"] and not entities["city"]:
            warnings.append("no_place_detected")

        return ProcessedQuery(
            raw=query,
            normalized=normalized,
            intent=intent,
            structured_fields=structured_fields,
            detected_places=entities["places"],
            detected_city=entities["city"],
            filters=entities["filters"],
            language=language,
            warnings=warnings,
            intents=intents,
        )
