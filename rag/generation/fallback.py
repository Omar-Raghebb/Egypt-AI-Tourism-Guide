"""Fallback generation: deterministic, template-based answers.

When the LLM provider is unavailable, the user still gets a *readable,
structured* answer built from the structured evidence — never a raw dump of
retrieved documents, JSON, document IDs, or internal validation notes.

Principles:
  - format only the fields relevant to the query intent
  - natural Arabic and English templates
  - missing facts are stated as unavailable (never invented)
  - unverified/estimated facts keep their qualification
  - conflicting values are explained, not silently resolved
  - a short, calm notice states that full AI generation is unavailable
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

# Quiet availability notice (kept stable so tests/integrations can rely on it).
GENERATION_UNAVAILABLE_EN = (
    "(Note: full AI-generated wording is temporarily unavailable, so this "
    "answer was assembled directly from the knowledge base.)"
)
GENERATION_UNAVAILABLE_AR = (
    "(ملاحظة: صياغة الإجابة بالذكاء الاصطناعي غير متوفرة حالياً، لذلك تم تجميع "
    "هذه الإجابة مباشرة من قاعدة المعرفة.)"
)

_NO_INFO_EN = "I couldn't find this information in the knowledge base yet."
_NO_INFO_AR = "لم أجد هذه المعلومة في قاعدة المعرفة حتى الآن."

# Detects a generated answer that honestly reports the requested information
# is absent from the knowledge base (the prompt mandates this statement).
# Used to label such requests status="unavailable" in Groq mode too —
# the missing-info contract must be generation-mode independent.
NO_INFO_ASSERTION_RE = re.compile(
    r"knowledge base[^.!?\n]{0,120}?(?:does not contain|doesn't contain|"
    r"does not have|has no|not available|no information|not found|"
    r"couldn't find|cannot find|can't find)"
    r"|(?:not available|not found|couldn't find|cannot find|can't find|"
    r"no information|does not contain|doesn't contain)[^.!?\n]{0,120}?"
    r"knowledge base"
    r"|قاعدة المعرفة[^.!?؟\n]{0,120}?(?:لا توجد|غير موجودة|غير متوفر|لم أجد|"
    r"لا تحتوي|لا يحتوي)"
    r"|(?:لم أجد|لا توجد|غير موجودة|غير متوفر)[^.!?؟\n]{0,120}?قاعدة المعرفة",
    re.IGNORECASE,
)

# Guard: an answer that actually delivers field data (numbers, prices, hours)
# is a partial/success answer even if it also mentions something being
# "not available" — only data-free answers may be labelled unavailable.
_ANSWER_HAS_DATA_RE = re.compile(
    r"\d+\s*(?:EGP|جنيه)|(?:EGP|جنيه)\s*\d+"
    r"|ticket prices|visiting hours|opening hours|ساعات الزيارة|أسعار التذاكر"
    r"|\*\*(?:place|المكان)\b|\bAbout\b|نبذة",
    re.IGNORECASE,
)


def asserts_missing_info(answer: str) -> bool:
    """True when the answer states the info is missing AND carries no data."""
    return bool(
        NO_INFO_ASSERTION_RE.search(answer or "")
        and not _ANSWER_HAS_DATA_RE.search(answer or "")
    )

_UNVERIFIED_EN = (
    "This information comes from the knowledge base but has not been fully "
    "verified yet, so please double-check before relying on it."
)
_UNVERIFIED_AR = (
    "هذه المعلومة موجودة في قاعدة المعرفة، لكنها لم تُتحقق منها بشكل كافٍ، "
    "فيُفضّل التأكد منها قبل الاعتماد عليها."
)
_CONFLICTING_EN = (
    "The available sources contain conflicting values for this information, "
    "so no single figure is shown as final."
)
_CONFLICTING_AR = (
    "المصادر المتاحة تحتوي على قيم متضاربة لهذه المعلومة، لذلك لم يتم اختيار "
    "قيمة واحدة كنهائية."
)
_UNAVAILABLE_EN = "This information is not available in the knowledge base."
_UNAVAILABLE_AR = "المعلومة دي مش متوفرة حاليًا في قاعدة البيانات."

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
) -> str:
    """Human place name for an itinerary stop (language-correct, no IDs)."""
    pid = str(place_id)
    place = places.get(pid)
    if place is None and structured is not None:
        place = structured.kb.places.get(pid)
    if place is None:
        return pid  # last resort; should not happen with the enhanced KB
    if lang == "ar" and place.name_arabic:
        return place.name_arabic
    return place.name


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

        if intent == "itinerary":
            # itinerary queries: the plan itself (+ city/rules), NOT a dump
            # of every retrieved place's prices and hours
            for extra in context.structured_extras:
                section = self._format_extra(extra, lang, places)
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
                section = self._format_extra(extra, lang, places)
                if section:
                    sections.append(section)

        # 3) no structured evidence at all -> honest "not available"
        if not sections:
            # Nothing found: a short natural statement is the whole answer.
            # No availability notice (nothing was "assembled") and no
            # verification qualification (nothing exists to qualify).
            return _NO_INFO_AR if lang == "ar" else _NO_INFO_EN

        # 4) one calm notice that full generation is offline (not an error dump)
        notice = GENERATION_UNAVAILABLE_AR if lang == "ar" else GENERATION_UNAVAILABLE_EN

        # 5) verification qualification — once, at the end
        qualification = self._verification_qualification(context, lang)
        if qualification:
            sections.append(qualification)

        text = "\n\n".join(sections)
        return f"{text}\n\n{notice}"

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
                rendered = self._render_hours(lookup.value, lang) if (
                    lookup.status not in (VerificationStatus.MISSING, VerificationStatus.UNAVAILABLE)
                ) else None
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
            "غير متوفر في قاعدة المعرفة." if lang == "ar"
            else "not available in the knowledge base."
        )

        def _label_for(field: str) -> str:
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
            line = f"{L['location']}: lat {lat}, lng {lng}"
            if status == VerificationStatus.ESTIMATED:
                line += " (approximate)" if lang == "en" else " (تقريبية)"
            return line

        # generic fallback: human-readable value or honest omission
        human = _human_value(value)
        if not human:
            return f"{_label_for(field)}: {not_available}"
        line = f"{_label_for(field)}: {human}"
        if status == VerificationStatus.UNVERIFIED:
            line += " (unverified)" if lang == "en" else " (غير موثق)"
        return line

    @staticmethod
    def _render_prices(prices: Any, lang: str) -> Optional[str]:
        if not isinstance(prices, dict):
            return None
        currency = prices.get("currency", "EGP")
        tickets = prices.get("tickets")
        if not isinstance(tickets, list) or not tickets:
            return None
        labels = _TICKET_LABELS_AR if lang == "ar" else _TICKET_LABELS_EN
        L = _LABELS[lang]
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
        # conflicting confidence note
        if prices.get("confidence") in ("conflicting",):
            lines.append(
                f"_{_CONFLICTING_AR if lang == 'ar' else _CONFLICTING_EN}_"
            )
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
        self, extra: StructuredRecord, lang: str, places: Dict[str, StructuredPlace]
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
            # ALWAYS render the complete itinerary: every day of the record,
            # each with all of its stops (never truncated mid-itinerary).
            for day in days:
                day_title = day.get("title", "")
                if lang == "ar" and day.get("title_ar"):
                    day_title = day["title_ar"]
                day_lines = [f"{L['days']} {day.get('day')} — {day_title}"]
                for stop in day.get("stops") or []:
                    stop_name = _lookup_place_name(
                        places, stop.get("place_id", ""), lang, self.structured
                    )
                    day_lines.append(
                        f"- {L['time']} {stop.get('suggested_time', '')} — "
                        f"{L['stop']}: {stop_name}"
                    )
                if day.get("tip"):
                    day_lines.append(f"- {L['tip']}: {day['tip']}")
                lines.append("\n".join(day_lines))
            # report an honest limitation if the record itself is incomplete
            if expected_days and len(days) < int(expected_days):
                limitation = (
                    f"خطة السجل تغطي {len(days)} يوم من أصل {expected_days}."
                    if lang == "ar"
                    else f"The itinerary record covers {len(days)} of {expected_days} days."
                )
                lines.append(f"_{limitation}_")
            # family suitability note ONLY when the record itself supports it
            interests = [str(i).lower() for i in (data.get("interests") or [])]
            title_l = title.lower() if title else ""
            if "famil" in title_l or any("famil" in i for i in interests):
                lines.append(
                    "مناسبة للعائلات حسب قاعدة المعرفة."
                    if lang == "ar"
                    else "Family-friendly according to the knowledge base."
                )
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
                base = _UNAVAILABLE_AR if lang == "ar" else _UNAVAILABLE_EN
                return f"{base}" + (f"\n_{note}_" if note else "")
            return None

        return None

    # ------------------------------------------------------------------
    def _verification_qualification(
        self, context: RetrievedContext, lang: str
    ) -> Optional[str]:
        """Single, deduplicated qualification for unverified/conflicting info.

        When part of the evidence is verified and part is not, the wording
        reflects the mixed situation instead of a blanket "unverified".
        """
        statuses = {
            p.verification.status for p in context.structured_records
        }
        statuses |= {e.verification.status for e in context.structured_extras}
        if VerificationStatus.UNAVAILABLE in statuses:
            return None  # already communicated inline
        has_verified = VerificationStatus.VERIFIED in statuses
        has_uncertain = bool(
            statuses & {VerificationStatus.UNVERIFIED, VerificationStatus.ESTIMATED}
        )
        if has_verified and has_uncertain:
            return (
                "جزء من هذه المعلومات موثق جيداً، وأجزاء أخرى لم يتم التحقق "
                "منها بشكل كافٍ، فيُفضّل التأكد قبل الاعتماد عليها."
                if lang == "ar"
                else "Part of this information is well-sourced; other parts "
                "have not been fully verified yet, so please double-check "
                "before relying on it."
            )
        if has_uncertain:
            return _UNVERIFIED_AR if lang == "ar" else _UNVERIFIED_EN
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
