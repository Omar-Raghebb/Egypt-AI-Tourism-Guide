"""Fallback generation: deterministic, template-based answers.

When the LLM provider is unavailable, the user still gets a *readable,
structured* answer built from the structured evidence — never a raw dump of
retrieved documents, JSON, document IDs, or internal validation notes.

Principles (user-facing output contract):
  - format only the fields relevant to the query intent
  - natural Arabic and English templates
  - missing facts are stated as unavailable (never invented)
  - unverified/estimated FACT values (prices, hours, coordinates) are NOT
    presented as facts: a requested value that cannot be safely established
    is stated as "currently unavailable" — with NO internal verification
    talk, NO "unverified" labels, and NO fallback notices
  - conflicting values are treated as unavailable (no arbitrary selection)
  - the fallback answer is presented exactly like any other answer
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rag.generation.validator import AnswerValidator, ValidationResult
from rag.retrieval.structured import StructuredRetriever
from rag.schemas import (
    RetrievedContext,
    RetrievalResult,
    StructuredPlace,
    StructuredRecord,
    VerificationStatus,
)

logger = logging.getLogger(__name__)

_NO_INFO_EN = "This information is currently unavailable."
_NO_INFO_AR = "هذه المعلومة غير متاحة حاليًا."

# Detects an answer that honestly reports the requested information is
# unavailable (the prompt mandates a short "currently unavailable" sentence).
# Used to label such requests status="unavailable" in Groq mode too —
# the missing-info contract must be generation-mode independent.
NO_INFO_ASSERTION_RE = re.compile(
    r"knowledge base[^.!?\n]{0,120}?(?:does not contain|doesn't contain|"
    r"does not have|has no|not available|no information|not found|"
    r"couldn't find|cannot find|can't find)"
    r"|(?:not available|not found|couldn't find|cannot find|can't find|"
    r"no information|does not contain|doesn't contain)[^.!?\n]{0,120}?"
    r"knowledge base"
    r"|currently unavailable"
    r"|غير متاح(?:ة|ه)[^.!?\n]{0,40}?حالي"
    r"|قاعدة المعرفة[^.!?؟\n]{0,120}?(?:لا توجد|غير موجودة|غير متوفر|لم أجد|"
    r"لا تحتوي|لا يحتوي)"
    r"|(?:لم أجد|لا توجد|غير موجودة|غير متوفر)[^.!?\n]{0,120}?قاعدة المعرفة",
    re.IGNORECASE,
)

# Unavailability phrases — sentences containing these are "missing-value"
# statements, not data.
_UNAVAIL_PHRASE_RE = re.compile(
    r"currently unavailable|unavailable|not available|not found|no information"
    r"|couldn't find|cannot find|can't find"
    r"|غير متاحة|غير متوفر|لم أجد",
    re.IGNORECASE,
)

# Guard: an answer that actually delivers field data (numbers, prices, hours)
# is a partial/success answer even if it also mentions something being
# "not available" — only data-free answers may be labelled unavailable.
# Bare field labels ("Opening hours are ...") are not data; only actual
# values (currency amounts, times, structured headings) count.
_ANSWER_HAS_DATA_RE = re.compile(
    r"\d+\s*(?:EGP|جنيه|USD)|(?:EGP|جنيه|USD)\s*\d+"
    r"|\d{1,2}[:：]\d{2}"
    r"|\*\*(?:place|المكان)\b|\bAbout\b|نبذة",
    re.IGNORECASE,
)


def _strip_unavailable_sentences(text: str) -> str:
    """Remove every sentence/line that only reports missing information."""
    # split on sentence boundaries AND line breaks (fallback answers are
    # line-oriented: "Ticket prices: currently unavailable." sits on its
    # own line inside an otherwise data-bearing answer)
    sentences = re.split(r"(?<=[.!?؟])\s+|\n+", text)
    kept = [s for s in sentences if not _UNAVAIL_PHRASE_RE.search(s)]
    return " ".join(kept)


def asserts_missing_info(answer: str) -> bool:
    """True when the answer states the info is missing AND carries no data."""
    text = answer or ""
    if not NO_INFO_ASSERTION_RE.search(text):
        return False
    remainder = _strip_unavailable_sentences(text)
    return not _ANSWER_HAS_DATA_RE.search(remainder)

_LABELS = {
    "en": {
        "place": "Place",
        "city": "City",
        "location": "Location",
        "hours": "Visiting hours",
        "prices": "Ticket prices",
        "activities": "Activities",
        "notes": "Important notes",
        "source": "Source",
        "days": "Day",
        "time": "Time",
        "stop": "Destination",
        "tip": "Practical note",
        "suggested_plan": "Suggested plan",
        "overview": "About",
    },
    "ar": {
        "place": "المكان",
        "city": "المدينة",
        "location": "الموقع",
        "hours": "ساعات الزيارة",
        "prices": "أسعار التذاكر",
        "activities": "الأنشطة",
        "notes": "ملاحظات مهمة",
        "source": "المصدر",
        "days": "اليوم",
        "time": "الوقت",
        "stop": "الوجهة",
        "tip": "ملاحظة عملية",
        "suggested_plan": "الخطة المقترحة",
        "overview": "نبذة",
    },
}

_TICKET_LABELS_EN = {
    "foreigner_adult": "Foreign adult",
    "foreigner_student": "Foreign student",
    "egyptian_adult": "Egyptian adult",
    "egyptian_student": "Egyptian student",
}
_TICKET_LABELS_AR = {
    "foreigner_adult": "بالغ أجنبي",
    "foreigner_student": "طالب أجنبي",
    "egyptian_adult": "بالغ مصري",
    "egyptian_student": "طالب مصري",
}


def _lookup_place_name(
    places: Dict[str, StructuredPlace],
    place_id: str,
    lang: str = "en",
    structured: Optional[StructuredRetriever] = None,
) -> Optional[str]:
    """Human place name for an itinerary stop (language-correct, no IDs).

    Returns ``None`` when the stop cannot be resolved to a place — the
    caller omits the stop instead of ever rendering an internal record ID
    in a user-facing answer.
    """
    pid = str(place_id)
    place = places.get(pid)
    if place is None and structured is not None:
        place = structured.kb.places.get(pid)
    if place is None:
        return None
    if lang == "ar" and place.name_arabic:
        return place.name_arabic
    return place.name


# Arabic ordinal day headings ("اليوم الأول" … "اليوم العاشر"); beyond 10
# the numeral itself is used ("اليوم 11") — no fabricated ordinals.
_AR_DAY_ORDINALS = {
    1: "الأول", 2: "الثاني", 3: "الثالث", 4: "الرابع", 5: "الخامس",
    6: "السادس", 7: "السابع", 8: "الثامن", 9: "التاسع", 10: "العاشر",
}


def _day_heading(day_number: int, lang: str, day_title: str = "") -> str:
    """Markdown heading for one itinerary day (language-correct).

    English: "### Day 2 — West Bank" (KB day title). Arabic: "### اليوم
    الثاني" — the KB day titles are English-only, so the title is omitted
    in Arabic answers instead of mixing scripts.
    """
    if lang == "ar":
        label = _AR_DAY_ORDINALS.get(day_number, str(day_number))
        return f"### اليوم {label}"
    title = f" — {day_title}" if day_title else ""
    return f"### Day {day_number}{title}"


class FallbackResponder:
    """Deterministic responder used when Groq generation fails.

    Also exposes ``validate_answer`` for generated-answer guardrails
    (delegates to :class:`rag.generation.validator.AnswerValidator`).
    """

    def __init__(self, structured: Optional[StructuredRetriever] = None) -> None:
        self.structured = structured
        self.validator = AnswerValidator()

    # ------------------------------------------------------------------
    # User-facing fallback answer
    # ------------------------------------------------------------------
    def respond(
        self,
        context: RetrievedContext,
        language: str = "en",
        retrieval: Optional[RetrievalResult] = None,
    ) -> str:
        """Build a clean, natural answer from the structured evidence."""
        lang = language if language in ("en", "ar") else "en"
        L = _LABELS[lang]
        sections: List[str] = []
        places = {
            p.place_id: p for p in (context.structured_records or [])
        }
        intent = retrieval.intent if retrieval is not None else None
        all_intents = list(getattr(retrieval, "intents", []) or []) if retrieval is not None else []
        requested_days = (
            (retrieval.detected_entities or {}).get("days") if retrieval is not None else None
        )

        if intent == "itinerary":
            # itinerary queries: the plan itself (+ city/rules), NOT a dump
            # of every retrieved place's prices and hours
            for extra in context.structured_extras:
                section = self._format_extra(
                    extra, lang, places, requested_days=requested_days
                )
                if section:
                    sections.append(section)
        else:
            # 1) intent-specific formatted facts from structured records.
            # Auto-added definition hours apply only when the answer is about
            # a single place (a "what is X?" question), never to a listing.
            auto_hours = len(context.structured_records) == 1
            for place in context.structured_records:
                section = self._format_place(
                    place, intent, lang, intents=all_intents, auto_hours=auto_hours
                )
                if section:
                    sections.append(section)

            # 2) non-place records (cities, activities, itineraries, rules)
            for extra in context.structured_extras:
                section = self._format_extra(
                    extra, lang, places, requested_days=requested_days
                )
                if section:
                    sections.append(section)

        # 3) no structured evidence at all -> honest "not available"
        if not sections:
            # Nothing found: a short natural statement is the whole answer.
            # No internal notices (nothing was "assembled", nothing to
            # qualify) — Case E: the fallback answer is a normal answer.
            return _NO_INFO_AR if lang == "ar" else _NO_INFO_EN

        # Case E: the fallback answer is presented exactly like any other
        # answer — no "assembled from the knowledge base" notice and no
        # verification qualification. Unverified/estimated/conflicting
        # FACT values are already excluded inline by _render_field.
        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    def _format_place(
        self, place: StructuredPlace, intent: Optional[str], lang: str,
        intents: Optional[List[str]] = None,
        auto_hours: bool = True,
    ) -> Optional[str]:
        L = _LABELS[lang]
        if self.structured is None:
            return None
        name = place.name_arabic if (lang == "ar" and place.name_arabic) else place.name
        lines = [f"**{L['place']}: {name}**"]

        if place.city:
            lines.append(f"{L['city']}: {self._city_display(place.city, lang)}")

        intent_fields = {
            "price": ["prices"],
            "hours": ["opening_hours"],
            "coordinates": ["coordinates"],
            "accessibility": ["accessibility"],
            "transportation": ["transportation_options"],
            "indoor_outdoor": ["indoor_outdoor"],
            "age_suitability": ["age_suitability"],
        }
        # multi-intent questions ("opening hours AND accessibility") render
        # EVERY requested field, not only the first detected intent
        requested = [i for i in (intents or ([intent] if intent else [])) if i]
        overview_intents = (None, "definition", "semantic", "recommendation")
        overview = any(i in overview_intents for i in requested)
        fields: List[str] = []
        for i in requested:
            for f in intent_fields.get(i, []):
                if f not in fields:
                    fields.append(f)
        if not requested:
            fields = ["prices", "opening_hours"]
        for field in fields:
            lookup = self.structured.get_field(place.place_id, field)
            rendered = self._render_field(lookup, field, lang)
            if rendered:
                lines.append(rendered)

        # overview-type queries: describe the place; do not dump prices
        # and hours into a "best places to visit" list
        if overview:
            description = (
                place.data.get("description_ar") if lang == "ar" else place.data.get("description_en")
            )
            if description:
                lines.append(f"{L['overview']}: {description}")
            # a definition question about ONE place may include its hours;
            # a multi-place listing ("main attractions in Luxor") is an
            # overview and must not gain hours automatically — in either
            # language
            if (
                auto_hours
                and any(i in (None, "definition") for i in requested)
                and "opening_hours" not in fields
            ):
                lookup = self.structured.get_field(place.place_id, "opening_hours")
                # auto-added hours must be a confirmed value: an unverified
                # one is omitted (the user did not explicitly request hours,
                # so no "unavailable" line is needed)
                rendered = (
                    self._render_hours(lookup.value, lang)
                    if lookup.status == VerificationStatus.VERIFIED
                    else None
                )
                if rendered:
                    lines.append(rendered)

        return "\n".join(lines)

    def _city_display(self, city: str, lang: str) -> str:
        """City name in the answer language when the KB has it (else raw)."""
        if lang == "ar" and self.structured is not None:
            for record in self.structured.kb.cities:
                if record.get("name_en") == city and record.get("name_ar"):
                    return record["name_ar"]
        return city

    def _render_field(self, lookup: Any, field: str, lang: str) -> Optional[str]:
        """Render one structured field lookup as a labeled, human line.

        When the field was explicitly requested by the query intent and the
        KB has no value, an honest "not available" line is rendered instead
        of silently omitting the requested information.
        """
        L = _LABELS[lang]
        status = lookup.status
        value = lookup.value
        not_available = (
            "غير متاحة حاليًا." if lang == "ar" else "currently unavailable."
        )

        def _label_for(field: str) -> str:
            # the field aliases map onto the human labels
            if field in ("opening_hours", "hours"):
                field = "hours"
            elif field in ("prices", "price", "ticket_price", "ticket_prices"):
                field = "prices"
            return L.get(field, field.replace("_", " ").capitalize())

        def _human_value(v: Any) -> Optional[str]:
            """Dict/list values are never dumped raw."""
            if isinstance(v, dict):
                details = v.get("details") or v.get("note") or v.get("text")
                if isinstance(details, str) and details.strip():
                    return details.strip()
                return None
            if isinstance(v, (list, tuple)):
                parts = [str(x).strip() for x in v if str(x).strip()]
                return ", ".join(parts) if parts else None
            if v is None:
                return None
            return str(v)

        if status in (VerificationStatus.MISSING, VerificationStatus.UNAVAILABLE):
            return f"{_label_for(field)}: {not_available}"

        # Case B: UNVERIFIED or ESTIMATED fact values are never presented as
        # confirmed facts. If the user explicitly requested the value, say it
        # is unavailable — with no internal verification talk. (Applies to
        # fact fields only: descriptive content may still be shown.)
        _FACT_FIELDS = ("prices", "opening_hours", "coordinates")
        if field in _FACT_FIELDS and status in (
            VerificationStatus.UNVERIFIED, VerificationStatus.ESTIMATED
        ):
            return f"{_label_for(field)}: {not_available}"

        if field == "prices":
            return self._render_prices(value, lang)
        if field == "opening_hours":
            return self._render_hours(value, lang)
        if field == "coordinates" and isinstance(value, dict):
            lat, lng = value.get("lat"), value.get("lng")
            if lat is None or lng is None:
                lat, lng = value.get("latitude"), value.get("longitude")
            if lat is None or lng is None:
                return f"{_label_for(field)}: {not_available}"
            return f"{L['location']}: lat {lat}, lng {lng}"

        # generic fallback: human-readable value (descriptive content is
        # shown without verification labels)
        human = _human_value(value)
        if not human:
            return f"{_label_for(field)}: {not_available}"
        return f"{_label_for(field)}: {human}"

    @staticmethod
    def _render_prices(prices: Any, lang: str) -> Optional[str]:
        if not isinstance(prices, dict):
            return None
        L = _LABELS[lang]
        # Case C: conflicting values are never arbitrarily resolved and never
        # shown with a long conflict warning — the field is simply unavailable.
        if prices.get("confidence") in ("conflicting",):
            return f"{L['prices']}: " + (
                "غير متاحة حاليًا." if lang == "ar" else "currently unavailable."
            )
        currency = prices.get("currency", "EGP")
        tickets = prices.get("tickets")
        if not isinstance(tickets, list) or not tickets:
            return None
        labels = _TICKET_LABELS_AR if lang == "ar" else _TICKET_LABELS_EN
        lines = [f"{L['prices']}:"]
        for ticket in tickets:
            name = ticket.get("name")
            parts: List[str] = []
            for key, label in labels.items():
                v = ticket.get(key)
                if isinstance(v, (int, float)):
                    parts.append(f"{label} {v} {currency}" if lang == "ar" else f"{label}: {v} {currency}")
            if not parts:
                continue
            prefix = f"- {name}: " if name else "- "
            lines.append(prefix + ("، ".join(parts) if lang == "ar" else "; ".join(parts)))
        if len(lines) == 1:
            return None
        return "\n".join(lines)

    @staticmethod
    def _render_hours(hours: Any, lang: str) -> Optional[str]:
        if not isinstance(hours, dict):
            return None
        L = _LABELS[lang]
        daily = hours.get("daily_hours")
        if not daily:
            return None
        lines = [f"{L['hours']}: {daily}"]
        if hours.get("ramadan_hours"):
            label = "ساعات رمضان" if lang == "ar" else "Ramadan hours"
            lines.append(f"{label}: {hours['ramadan_hours']}")
        if hours.get("closed_days"):
            closed = hours["closed_days"]
            if closed and str(closed[0]).lower() != "none listed on official page":
                label = "أيام الإغلاق" if lang == "ar" else "Closed days"
                lines.append(f"{label}: {', '.join(str(c) for c in closed)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def _format_extra(
        self, extra: StructuredRecord, lang: str, places: Dict[str, StructuredPlace],
        requested_days: Optional[int] = None,
    ) -> Optional[str]:
        L = _LABELS[lang]
        data = extra.data
        if extra.record_type == "city":
            name = data.get("name_ar") if (lang == "ar" and data.get("name_ar")) else data.get("name_en")
            lines = [f"**{name}**"]
            description = data.get("description_ar") if lang == "ar" else data.get("description_en")
            if description:
                lines.append(description)
            if data.get("suggested_days"):
                label = "أيام مقترحة" if lang == "ar" else "Suggested days"
                lines.append(f"{label}: {data['suggested_days']}")
            return "\n".join(lines)

        if extra.record_type == "activity":
            name = data.get("name_ar") if (lang == "ar" and data.get("name_ar")) else data.get("name_en")
            lines = [f"**{name}**"]
            description = data.get("description_ar") if lang == "ar" else data.get("description_en")
            if description:
                lines.append(description)
            if data.get("typical_duration"):
                label = "المدة المعتادة" if lang == "ar" else "Typical duration"
                lines.append(f"{label}: {data['typical_duration']}")
            if data.get("cities"):
                lines.append(f"{L['city']}: {', '.join(str(c) for c in data['cities'])}")
            return "\n".join(lines)

        if extra.record_type == "itinerary":
            title = data.get("title_ar") if (lang == "ar" and data.get("title_ar")) else data.get("title_en")
            lines = [f"**{L['suggested_plan']}: {title}**"]
            days = data.get("days") or []
            expected_days = data.get("days_count")
            # Render EXACTLY the requested number of days when the user asked
            # for a specific trip length (one coherent plan — never several
            # unrelated itineraries); otherwise render the complete record.
            if requested_days and len(days) > requested_days:
                shown = days[:requested_days]
            else:
                shown = days
            for index, day in enumerate(shown, start=1):
                try:
                    day_number = int(day.get("day"))
                except (TypeError, ValueError):
                    day_number = index  # positional fallback (never None)
                day_lines = [
                    _day_heading(
                        day_number, lang,
                        (day.get("title") or "") if lang == "en" else "",
                    )
                ]
                for stop in day.get("stops") or []:
                    stop_name = _lookup_place_name(
                        places, stop.get("place_id", ""), lang, self.structured
                    )
                    if not stop_name:
                        # unresolvable stop: omit it rather than leak an
                        # internal record ID into the user-facing answer
                        continue
                    day_lines.append(
                        f"- {L['time']} {stop.get('suggested_time', '')} — "
                        f"{L['stop']}: {stop_name}"
                    )
                if day.get("tip"):
                    day_lines.append(f"- {L['tip']}: {day['tip']}")
                lines.append("\n".join(day_lines))
            # requested days > days the evidence can fill: KEEP the requested
            # day headings and state the limitation naturally inside each
            # empty day — never invent attractions just to fill them
            covered = len(shown)
            if requested_days and covered < requested_days:
                for day_number in range(covered + 1, requested_days + 1):
                    empty_note = (
                        "لا تتوفر تفاصيل كافية لبرنامج هذا اليوم، فيمكنك "
                        "تخصيصه للراحة أو لإعادة زيارة معلم أعجبك أو "
                        "لاستكشاف المدينة على مهلك."
                        if lang == "ar"
                        else "No further day details are available, so feel "
                        "free to use it for rest, a relaxed revisit of a "
                        "highlight above, or exploring the city at your "
                        "own pace."
                    )
                    lines.append(
                        "\n".join([
                            _day_heading(day_number, lang),
                            f"- _{empty_note}_",
                        ])
                    )
            elif (
                expected_days
                and len(days) < int(expected_days)
                and not requested_days
            ):
                limitation = (
                    f"هذه الخطة تغطي {len(days)} يومًا."
                    if lang == "ar"
                    else f"This plan covers {len(days)} days."
                )
                lines.append(f"_{limitation}_")
            elif requested_days and len(days) > requested_days:
                # the shown plan is the first part of a longer itinerary —
                # say so briefly instead of claiming an exact-length match
                adaptation = (
                    f"هذه الخطة تغطي أول {requested_days} أيام من برنامج "
                    f"مدته {len(days)} أيام."
                    if lang == "ar"
                    else f"These {requested_days} days are the first part of "
                    f"a {len(days)}-day itinerary."
                )
                lines.append(f"_{adaptation}_")
            # family suitability note ONLY when the record itself supports it
            interests = [str(i).lower() for i in (data.get("interests") or [])]
            title_l = title.lower() if title else ""
            if "famil" in title_l or any("famil" in i for i in interests):
                lines.append("مناسبة للعائلات." if lang == "ar" else "Family-friendly.")
            return "\n\n".join(lines)

        if extra.record_type == "planning_rule":
            # rules are English-only in the KB; in Arabic answers we omit
            # the English text instead of mixing languages (never invent
            # a translation that is not in the knowledge base)
            if lang == "ar" and not data.get("text_ar"):
                return None
            text = data.get("text_en") or ""
            if text:
                return f"**{L['notes']}:** {text}"
            return None

        if extra.record_type == "transportation_route":
            if data.get("status") == "unavailable":
                note = data.get("note") or ""
                base = (
                    "Transportation information is currently unavailable."
                    if lang == "en" else "معلومات المواصلات غير متاحة حاليًا."
                )
                return f"{base}" + (f"\n_{note}_" if note else "")
            return None

        return None

    # ------------------------------------------------------------------
    # Generated-answer guardrails (compatibility API)
    # ------------------------------------------------------------------
    def validate_answer(
        self,
        answer: str,
        context: RetrievedContext,
        retrieval: Optional[RetrievalResult] = None,
    ) -> List[str]:
        """Guardrail checks on a generated answer (list of warning codes)."""
        result: ValidationResult = self.validator.validate(answer, context, retrieval)
        warnings = list(result.warnings)
        if not re.search(r"\[[^\]]+\]", answer or ""):
            warnings.append("missing_source_attribution")
        return warnings
