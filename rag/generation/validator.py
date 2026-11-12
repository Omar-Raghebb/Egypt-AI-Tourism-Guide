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

        # 3) answer much longer than evidence (possible hallucinated padding)
        if len(answer) > 8 * max(len(context_text), 1):
            warnings.append("answer_much_longer_than_evidence")

        outcome = OUTCOME_INVALID if unsupported else OUTCOME_VALID
        return ValidationResult(outcome, warnings)

    @staticmethod
    def _sentence_containing(text: str, token: str) -> Optional[str]:
        for sentence in re.split(r"(?<=[.!؟?\n])\s+", text or ""):
            if token in sentence:
                return sentence
        return None
