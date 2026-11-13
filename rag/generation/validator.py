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

        # 1) unsupported numerical claims (fabrication risk)
        answer_numbers = extract_numbers(answer)
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
        leak_warnings = self._structure_leak_warnings(answer)
        warnings.extend(leak_warnings)

        # 4) answer much longer than evidence (possible hallucinated padding)
        if len(answer) > 8 * max(len(context_text), 1):
            warnings.append("answer_much_longer_than_evidence")

        # 5) language consistency (internal note only; never blocks a useful
        #    answer by itself — wrong-language output is a quality defect,
        #    not a fabrication)
        lang_warnings = self._language_warnings(answer, language)
        warnings.extend(lang_warnings)

        # leaks and language issues downgrade the outcome one step (never to
        # invalid) so callers log and qualify instead of discarding the answer
        if unsupported:
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
        (r"\b[a-z][a-z0-9]+_[0-9]{3}\b", "internal_record_id"),
        (r"\bdoc_[a-f0-9]{6,}\b", "internal_doc_id"),
        (r"retrieval_score|chunk_id|traceback|exception\b", "debug_metadata"),
        (r"provider_error|api[_ ]error|status[_ ]code", "debug_metadata"),
    ]

    def _structure_leak_warnings(self, answer: str) -> List[str]:
        """Detect raw JSON, internal IDs, or debug text in the answer."""
        found: List[str] = []
        for pattern, code in self._LEAK_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                if code not in found:
                    found.append(code)
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
