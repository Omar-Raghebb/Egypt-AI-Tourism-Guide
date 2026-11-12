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

        if intent == "itinerary":
            # itinerary queries: the plan itself (+ city/rules), NOT a dump
            # of every retrieved place's prices and hours
            for extra in context.structured_extras:
                section = self._format_extra(extra, lang, places)
                if section:
                    sections.append(section)
        else:
            # 1) intent-specific formatted facts from structured records
            for place in context.structured_records:
                section = self._format_place(place, intent, lang)
                if section:
                    sections.append(section)

            # 2) non-place records (cities, activities, itineraries, rules)
            for extra in context.structured_extras:
                section = self._format_extra(extra, lang, places)
                if section:
                    sections.append(section)

        # 3) no structured evidence at all -> honest "not available"
        if not sections:
            sections.append(_NO_INFO_AR if lang == "ar" else _NO_INFO_EN)

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
        self, place: StructuredPlace, intent: Optional[str], lang: str
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
        fields = intent_fields.get(intent, ["prices", "opening_hours"])
        for field in fields:
            lookup = self.structured.get_field(place.place_id, field)
            rendered = self._render_field(lookup, field, lang)
            if rendered:
                lines.append(rendered)

        # definition-type queries get the description
        if intent in (None, "definition", "semantic", "recommendation"):
            description = (
                place.data.get("description_ar") if lang == "ar" else place.data.get("description_en")
            )
            if description:
                lines.append(f"{L['overview']}: {description}")

        return "\n".join(lines)

    def _city_display(self, city: str, lang: str) -> str:
        """City name in the answer language when the KB has it (else raw)."""
        if lang == "ar" and self.structured is not None:
            for record in self.structured.kb.cities:
                if record.get("name_en") == city and record.get("name_ar"):
                    return record["name_ar"]
        return city

    def _render_field(self, lookup: Any, field: str, lang: str) -> Optional[str]:
        """Render one structured field lookup as a labeled, human line."""
        L = _LABELS[lang]
        status = lookup.status
        value = lookup.value

        if status in (VerificationStatus.MISSING, VerificationStatus.UNAVAILABLE):
            if field in ("prices", "opening_hours"):
                return None  # simply omit — no empty sections
            return None

        if field == "prices":
            return self._render_prices(value, lang)
        if field == "opening_hours":
            return self._render_hours(value, lang)
        if field == "coordinates" and isinstance(value, dict):
            lat, lng = value.get("lat"), value.get("lng")
            if lat is None or lng is None:
                return None
            line = f"{L['location']}: lat {lat}, lng {lng}"
            if status == VerificationStatus.ESTIMATED:
                line += " (approximate)" if lang == "en" else " (تقريبية)"
            return line
        if field == "age_suitability":
            return None  # null -> omit silently; the general qualification covers it
        if field == "transportation_options":
            return None
        if field == "indoor_outdoor":
            return None

        # generic fallback for scalar values
        if value is None:
            return None
        return f"{L.get(field, field)}: {value}"

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
            for day in data.get("days") or []:
                day_lines = [f"{L['days']} {day.get('day')}: {day.get('title', '')}"]
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
            return "\n\n".join(lines)

        if extra.record_type == "planning_rule":
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
        """Single, deduplicated qualification for unverified/conflicting info."""
        statuses = {
            p.verification.status for p in context.structured_records
        }
        statuses |= {e.verification.status for e in context.structured_extras}
        if VerificationStatus.UNAVAILABLE in statuses:
            return None  # already communicated inline
        if VerificationStatus.UNVERIFIED in statuses or VerificationStatus.ESTIMATED in statuses:
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
