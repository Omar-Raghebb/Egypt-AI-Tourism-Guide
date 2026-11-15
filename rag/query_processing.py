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
        # "رحلة" only when it means a multi-day trip plan, not a single
        # activity ("رحلة عشاء النيلية" = dinner cruise): allow up to two
        # Arabic words between "رحلة" and the plan marker ("رحلة عائلية
        # لمدة 3 أيام"), but require an explicit trip-plan marker after it
        r"رحلة\s+(?:[\u0600-\u06FF]{2,}\s+){0,2}?(?:سياحية|كاملة|لمدة|يومين|يوم|أيام|ايام|\d)",
    ],
    "recommendation": [
        r"best places", r"recommend", r"suggest", r"top places", r"must visit",
        r"visit in", r"places to see", r"attraction",
        r"أفضل", r"أحسن", r"أنصح", r"انصح", r"ترشيح", r"أزور", r"أشوف", r"أهم",
    ],
    "definition": [
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
    detected_cities: List[str] = field(default_factory=list)  # ALL mentioned
    filters: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    warnings: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)  # ALL matched intents
    days: Optional[int] = None  # requested trip length ("plan a 3-day trip")


# Number words -> digits (EN words; Arabic phrases are matched on the raw
# text below). Anything else is matched by digit patterns.
_WORD_DAYS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
}
_AR_WORD_DAYS = {  # matched against the RAW text (un-normalized forms)
    "يوم واحد": 1, "يومين": 2, "اليومين": 2,
    "ثلاثة أيام": 3, "ثلاث ايام": 3,
    "اربعة ايام": 4, "أربعة أيام": 4, "أربع أيام": 4, "اربعة أيام": 4,
    "خمسة أيام": 5, "خمسة ايام": 5, "خمس ايام": 5, "خمس أيام": 5,
    "ستة أيام": 6, "ستة ايام": 6,
    "سبعة أيام": 7, "سبعة ايام": 7,
}

# Western AND Arabic-Indic numerals ("3 أيام", "٣ أيام")
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_DAYS_DIGIT_RE = re.compile(r"\b(\d+)[\s\-]*(?:days?\b|يوم|أيام|ايام)")
_DAYS_WORD_EN_RE = re.compile(
    r"\b(one|two|three|four|five|six|seven)[- ]days?\b", re.IGNORECASE
)


def extract_requested_days(query: str) -> Optional[int]:
    """Requested trip length in days ("plan a 3-day trip", "لمدة يومين").

    Pattern-driven, bilingual; returns None when the query does not state
    a number of days. Numbers followed by budgets/prices/counts are never
    matched (a duration always needs the "days" unit right after the
    number), and the result is capped at 30 days.
    """
    text = (query or "").translate(_ARABIC_INDIC)
    match = _DAYS_DIGIT_RE.search(text)
    if match:
        try:
            value = int(match.group(1))
        except ValueError:
            value = None
        if value and 1 <= value <= 30:
            return value
    match = _DAYS_WORD_EN_RE.search(text)
    if match:
        return _WORD_DAYS.get(match.group(1).lower())
    low = text.lower()
    for phrase, value in _AR_WORD_DAYS.items():
        if phrase in low or phrase in text:
            return value
    return None


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
        matched_patterns: Dict[str, List[str]] = {}
        for pid, pattern, original in self._name_patterns:
            if pattern.search(normalized):
                if pid not in detected_places:
                    detected_places.append(pid)
                    matched_patterns[pid] = []
                if original not in matched_patterns[pid]:
                    matched_patterns[pid].append(original)

        detected_places = self._filter_generic_matches(
            normalized, detected_places, matched_patterns
        )

        detected_cities = self._detect_cities(normalized)
        detected_city = detected_cities[0] if detected_cities else None

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
            "cities": detected_cities,
            "filters": filters,
        }

    # Generic tourism nouns that many records share as keywords — a match on
    # ONLY one of these is not evidence that the user meant that specific
    # place ("opening hours of Karnak Temple" must not pull Philae).
    _GENERIC_PLACE_WORDS = {
        "temple", "temples", "museum", "mosque", "complex", "citadel",
        "bazaar", "market", "street", "church", "synagogue", "palace",
        "village", "island", "bay", "beach", "dam", "fort", "castle",
        "towers", "pyramids", "site", "sites",
        "معبد", "معابد", "متحف", "مسجد", "قصر", "سوق", "شارع",
    }

    @classmethod
    def _filter_generic_matches(
        cls,
        normalized: str,
        detected_places: List[str],
        matched_patterns: Dict[str, List[str]],
    ) -> List[str]:
        """Keep places matched by a distinctive pattern; drop keyword-only hits.

        A pattern is *distinctive* when it contains at least one significant
        token (not a generic tourism noun, length >= 4) that also appears in
        the query. Places matched only through generic nouns are kept only
        when NO distinctive match exists at all (e.g. "which temples are in
        Luxor?" legitimately matches all temples).
        """
        if not detected_places:
            return detected_places

        normalized_lower = (normalized or "").lower()

        def _is_distinctive(pattern_text: str) -> bool:
            tokens = re.split(r"[\s/،,\-–]+", pattern_text or "")
            for token in tokens:
                token = token.strip().lower()
                if len(token) < 4 or token in cls._GENERIC_PLACE_WORDS:
                    continue
                if token in normalized_lower:
                    return True
            return False

        strong = [
            pid for pid in detected_places
            if any(_is_distinctive(text) for text in matched_patterns.get(pid, []))
        ]
        return strong if strong else detected_places

    def _detect_city(self, normalized: str) -> Optional[str]:
        """Single primary city: the FIRST one mentioned in the query."""
        cities = self._detect_cities(normalized)
        return cities[0] if cities else None

    def _detect_cities(self, normalized: str) -> List[str]:
        """ALL cities mentioned in the query, in order of first appearance.

        The previous single-city detection iterated a ``set`` of candidates,
        so when two cities were mentioned ("الجيزة والقاهرة") which one won
        was nondeterministic and the other destination's evidence was never
        retrieved (Bug B). Candidates: cities collection (name_en/name_ar),
        place records, then Arabic aliases; every alias is compared against
        the normalized query (and its own normalized form) so Arabic
        hamza/teh-marbuta variants match. Deduplicated by canonical English
        name; ordering is by earliest match position.
        """
        city_by_ar: Dict[str, str] = {}
        for city in getattr(self.kb, "cities", []):
            if city.get("name_en") and city.get("name_ar"):
                city_by_ar[str(city["name_ar"])] = str(city["name_en"])

        candidates: Dict[str, List[str]] = {}  # canonical EN -> aliases

        def _register(canonical: str, *aliases: str) -> None:
            canonical = (canonical or "").strip()
            if not canonical:
                return
            bucket = candidates.setdefault(canonical, [])
            for alias in aliases:
                alias = (alias or "").strip()
                if alias and alias not in bucket:
                    bucket.append(alias)

        for city in getattr(self.kb, "cities", []):
            _register(str(city.get("name_en") or ""), str(city.get("name_ar") or ""))
        for place in self.kb.places.values():
            if place.city:
                _register(str(place.city).strip(), str(place.city).strip())
        for ar, en in {**CITY_ALIASES, **city_by_ar}.items():
            _register(en, ar)

        hits: List[tuple] = []  # (earliest match position, canonical)
        for canonical, aliases in candidates.items():
            for alias in aliases:
                normalized_alias = normalize_query(alias)
                match = re.search(
                    re.escape(normalized_alias), normalized, re.IGNORECASE
                )
                if match:
                    hits.append((match.start(), canonical))
                    break  # keep the earliest position for this city
        hits.sort(key=lambda hit: (hit[0], hit[1]))

        ordered: List[str] = []
        for _position, canonical in hits:
            if canonical not in ordered:
                ordered.append(canonical)
        return ordered

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
        matched.sort(key=lambda i: priority.index(i) if i in priority else 99)
        # A "what is ... / ما هي ..." prefix inside another intent's question
        # ("what is the ticket price", "ما هي ساعات زيارة ...", "ما هي أفضل
        # الأماكن للزيارة ...") is conversational framing, not a definition
        # request: the other intent defines the scope. Keeping definition
        # would add descriptions/hours to fact or listing answers — and only
        # in Arabic (English "what is" never matches "what are"), which broke
        # Arabic/English response equivalence. Definition survives alone.
        if len(matched) > 1 and "definition" in matched:
            matched = [i for i in matched if i != "definition"]
        return matched

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

        days = extract_requested_days(query)
        # Query without any recognized intent defaults to semantic search.
        intent = intents[0] if intents else "semantic"
        # A requested trip length makes the itinerary the primary intent:
        # "Plan a 3-day family itinerary in Cairo" must answer with the
        # PLAN (not per-place fact fields triggered by "family"), and the
        # itinerary branch of the responder must be the one that renders.
        if days and "itinerary" in intents:
            intents = ["itinerary"] + [i for i in intents if i != "itinerary"]
            intent = "itinerary"
            structured_fields = []
        elif (
            days
            and not set(intents) & STRUCTURED_INTENTS
            and not entities["places"]
        ):
            # Colloquial follow-ups state ONLY the length and no plan
            # keyword ("خليها يومين في الأقصر"): an explicit day count with
            # no structured-fact intent and no specific place is a trip-plan
            # request, so it routes to the itinerary branch like any other
            # plan request. Fact intents ("سعر...", "مواعيد...") and
            # place-specific questions keep their own routing.
            intents = ["itinerary"] + [i for i in intents if i != "itinerary"]
            intent = "itinerary"
            structured_fields = []
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
            detected_cities=entities.get("cities") or [],
            filters=entities["filters"],
            language=language,
            warnings=warnings,
            intents=intents,
            days=extract_requested_days(query),
        )
