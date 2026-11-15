"""Answer validation with graded outcomes.

Outcomes (never silently destructive):
  - ``valid``                every factual claim is supported by the evidence
  - ``partially_supported``  answerable but contains unverified/missing parts
  - ``insufficient_evidence`` the evidence cannot answer the query
  - ``invalid``              empty or contradicts the evidence

Design rule: a useful answer is never *rejected wholesale* merely because
some requested fields are unavailable — unsupported numbers are the only
hard problem (they risk fabrication), and they downgrade the outcome to
``invalid`` only when the answer's key facts cannot be traced. All findings
are returned as warnings so the caller can qualify the answer instead of
throwing it away.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

from rag.schemas import RetrievedContext, RetrievalResult

logger = logging.getLogger(__name__)

ARABIC_TO_WESTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Phrases that indicate the model explicitly qualified information instead of
# asserting it (these never count as fabricated claims).
_HEDGING_PATTERNS = [
    r"not available", r"unavailable", r"not specified", r"no information",
    r"unverified", r"estimated", r"approximately", r"may (differ|change)",
    r"confirm", r"verify",
    r"غير متوف", r"لم يتم التحقق", r"تقديري", r"تقريبا", r"غير محد",
    r"تحقق", r"تأكد",
]

OUTCOME_VALID = "valid"
OUTCOME_PARTIAL = "partially_supported"
OUTCOME_INSUFFICIENT = "insufficient_evidence"
OUTCOME_INVALID = "invalid"

# Arabic ordinal day words used in day headings ("اليوم الأول" ...);
# the English form ("### Day 2") is matched by its own pattern below.
_AR_DAY_ORDINALS = [
    "الأول", "الثاني", "الثالث", "الرابع", "الخامس",
    "السادس", "السابع", "الثامن", "التاسع", "العاشر",
]
_DAY_HEAD_EN_RE = re.compile(r"###\s*Day\s*(\d+)", re.IGNORECASE)


class ValidationResult:
    """Outcome of answer validation."""

    def __init__(self, outcome: str, warnings: List[str]) -> None:
        self.outcome = outcome
        self.warnings = warnings

    @property
    def is_usable(self) -> bool:
        return self.outcome in (OUTCOME_VALID, OUTCOME_PARTIAL)


def extract_numbers(text: str) -> List[str]:
    """Numbers (incl. Arabic-Indic digits) as normalized strings."""
    normalized = (text or "").translate(ARABIC_TO_WESTERN)
    return re.findall(r"\d+(?:[.,]\d+)?", normalized)


def _is_hedged(sentence: str) -> bool:
    return any(
        re.search(p, sentence, re.IGNORECASE) for p in _HEDGING_PATTERNS
    )


def find_day_headings(answer: str) -> set:
    """Day numbers found as distinct day headings (bilingual, structural).

    Recognizes Markdown headings "### Day N" (English) and the Arabic
    ordinal forms "اليوم الأول" .. "اليوم العاشر" (plus "اليوم N" beyond
    ten). This is a structural check — it does NOT count the string
    "days" or numbers in prose.
    """
    found: set = set()
    for match in _DAY_HEAD_EN_RE.finditer(answer or ""):
        try:
            found.add(int(match.group(1)))
        except ValueError:
            continue
    for index, ordinal in enumerate(_AR_DAY_ORDINALS, start=1):
        if re.search(rf"اليوم\s+{re.escape(ordinal)}\b", answer or ""):
            found.add(index)
    for number in range(11, 31):
        if re.search(rf"اليوم\s+{number}\b", answer or ""):
            found.add(number)
    return found


def itinerary_structure_warnings(
    answer: str, requested_days: int
) -> List[str]:
    """Structural validation of a multi-day itinerary answer.

    For an N-day itinerary request (N >= 2) the answer must contain the
    distinct day headings Day 1..N. A flat list (or a plan that puts
    everything under the first day) is missing headings 2..N and is
    reported as a structural failure.
    """
    found = find_day_headings(answer)
    if not found:
        return [f"itinerary_structure:no_day_headings_for_{requested_days}_days"]
    missing = [
        day for day in range(1, requested_days + 1) if day not in found
    ]
    if missing:
        return [
            "itinerary_structure:missing_day_headings_"
            + ",".join(str(d) for d in missing)
        ]
    return []


class AnswerValidator:
    """Validates a generated answer against its retrieved context."""

    def validate(
        self,
        answer: str,
        context: RetrievedContext,
        retrieval: Optional[RetrievalResult] = None,
        language: str = "en",
    ) -> ValidationResult:
        warnings: List[str] = []
        if not answer or not answer.strip():
            return ValidationResult(OUTCOME_INVALID, ["empty_answer"])

        context_text = context.text or ""

        # 1) unsupported numerical claims (fabrication risk).
        # Day-heading indices ("### Day 1", "### اليوم 12") are structural
        # markers required by the day-by-day contract, not factual claims —
        # their numbers are exempt from the unsupported-number check so a
        # padded day (requested > available evidence) is not rejected for
        # using its own heading number.
        heading_safe = re.sub(r"(###\s*Day\s*)\d+", r"\1N", answer)
        heading_safe = re.sub(r"(###\s*اليوم\s*)\d+", r"\1N", heading_safe)
        answer_numbers = extract_numbers(heading_safe)
        context_numbers = set(extract_numbers(context_text))
        unsupported: List[str] = []
        for number in answer_numbers:
            if number in context_numbers:
                continue
            # find the sentence containing the number; hedged mentions of
            # unknown values ("about X", "not available") are not fabrication
            sentence = self._sentence_containing(answer, number)
            if sentence and _is_hedged(sentence):
                continue
            unsupported.append(number)
        if unsupported:
            warnings.append(
                "unsupported_numbers:" + ",".join(unsupported[:5])
            )

        # 2) empty evidence -> the answer cannot be grounded
        if not context_text.strip():
            warnings.append("no_evidence")
            return ValidationResult(OUTCOME_INSUFFICIENT, warnings)

        # 3) internal machinery leaking into the user answer
        leak_warnings = self._structure_leak_warnings(answer, context, retrieval)
        warnings.extend(leak_warnings)

        # 3b) structural validation for multi-day itinerary requests:
        # the answer must contain the requested number of distinct day
        # headings (bilingual). A flattened attraction list is NOT an
        # acceptable itinerary answer — it is repaired by replacing it
        # with the structured fallback plan.
        itinerary_days = None
        if retrieval is not None:
            if getattr(retrieval, "intent", None) == "itinerary":
                itinerary_days = (retrieval.detected_entities or {}).get("days")
        structure_warnings: List[str] = []
        if itinerary_days and int(itinerary_days) >= 2:
            structure_warnings = itinerary_structure_warnings(
                answer, int(itinerary_days)
            )
            warnings.extend(structure_warnings)

        # 4) answer much longer than evidence (possible hallucinated padding)
        if len(answer) > 8 * max(len(context_text), 1):
            warnings.append("answer_much_longer_than_evidence")

        # 5) language consistency (internal note only; never blocks a useful
        #    answer by itself — wrong-language output is a quality defect,
        #    not a fabrication)
        lang_warnings = self._language_warnings(answer, language)
        warnings.extend(lang_warnings)

        # leaks downgrade the outcome (never to invalid) so callers log
        # and qualify instead of discarding the answer; unsupported
        # numbers and a missing day structure ARE hard failures (the
        # answer would misinform or is not the requested itinerary) and
        # replace the answer with the structured fallback
        if unsupported or structure_warnings:
            outcome = OUTCOME_INVALID
        elif leak_warnings:
            outcome = OUTCOME_INSUFFICIENT
        elif lang_warnings:
            outcome = OUTCOME_PARTIAL
        else:
            outcome = OUTCOME_VALID
        return ValidationResult(outcome, warnings)

    # ------------------------------------------------------------------
    _LEAK_PATTERNS = [
        (r"\{['\"]|```json|\"place_id\"\s*:", "raw_json"),
        # record ids may have several underscore-separated parts
        # (giza_pyramids_001, act_nile_cruise) — the id shape requires a
        # snake_case token ending in _NNN, or any context-internal id
        # (checked against the actual retrieved ids below)
        (r"\b[a-z0-9]+(?:_[a-z0-9]+)*_[0-9]{3}\b", "internal_record_id"),
        (r"\bdoc_[a-f0-9]{6,}\b", "internal_doc_id"),
        (r"\b[\w/\\]+\.(?:json|pkl|md|txt)\b", "internal_file_path"),
        (r"retrieval_score|chunk_id|traceback|exception\b", "debug_metadata"),
        (r"provider_error|api[_ ]error|status[_ ]code", "debug_metadata"),
    ]

    def _context_internal_ids(self, context: RetrievedContext,
                              retrieval: Optional[RetrievalResult]) -> List[str]:
        """All internal ids actually present in the retrieved evidence."""
        ids: List[str] = []
        for place in context.structured_records or []:
            ids.append(str(place.place_id))
        for extra in context.structured_extras or []:
            ids.append(str(extra.record_id))
        for doc in context.documents or []:
            if doc.doc_id:
                ids.append(str(doc.doc_id))
                # chunked doc ids are "prefix::section::n"; the model may cite
                # only the record prefix
                prefix = str(doc.doc_id).split("::")[0]
                if prefix and prefix != str(doc.doc_id):
                    ids.append(prefix)
        if retrieval is not None:
            entities = getattr(retrieval, "detected_entities", None) or {}
            ids.extend(str(p) for p in (entities.get("places") or []))
        return [i for i in ids if i and i != "None"]

    def _structure_leak_warnings(
        self, answer: str, context: RetrievedContext,
        retrieval: Optional[RetrievalResult] = None,
    ) -> List[str]:
        """Detect raw JSON, internal IDs, or debug text in the answer."""
        found: List[str] = []
        for pattern, code in self._LEAK_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                if code not in found:
                    found.append(code)
        # precise check: any internal id from THIS retrieval appearing in
        # the user-facing answer (catches ids like city_luxor that have no
        # _NNN suffix)
        for internal_id in self._context_internal_ids(context, retrieval):
            if re.search(rf"\b{re.escape(internal_id)}\b", answer):
                if "internal_record_id" not in found:
                    found.append("internal_record_id")
        return [f"leak:{code}" for code in found]

    _AR_RE = re.compile(r"[\u0600-\u06FF]")
    _LATIN_RE = re.compile(r"[A-Za-z]")

    def _language_warnings(self, answer: str, language: str) -> List[str]:
        """Flag answers whose script contradicts the requested language.

        Numbers and short proper nouns are unavoidable; we require the
        dominant script to match. Only applies when the request is language-
        specific (en/ar).
        """
        if language not in ("en", "ar"):
            return []
        arabic = len(self._AR_RE.findall(answer or ""))
        latin = len(self._LATIN_RE.findall(answer or ""))
        if language == "ar" and latin > arabic and latin > 20:
            return ["language_mismatch:ar_request_latin_answer"]
        if language == "en" and arabic > latin and arabic > 20:
            return ["language_mismatch:en_request_arabic_answer"]
        return []

    @staticmethod
    def _sentence_containing(text: str, token: str) -> Optional[str]:
        for sentence in re.split(r"(?<=[.!؟?\n])\s+", text or ""):
            if token in sentence:
                return sentence
        return None
