"""Arabic/English text normalization for keyword retrieval.

Normalization steps for Arabic (order matters):
  1. strip diacritics/tashkeel and tatweel
  2. normalize alef variants (أ إ آ -> ا)
  3. normalize hamza forms (ؤ -> و, ئ -> ي)
  4. normalize teh marbuta (ة -> ه) and alef maqsura (ى -> ي)
"""

from __future__ import annotations

import re
from typing import List

_AR_DIACRITICS = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"

_ALEF = re.compile("[أإآٱ]")
_WAW_HAMZA = re.compile("ؤ")
_YA_HAMZA = re.compile("ئ")
_TA_MARBUTA = re.compile("ة")
_ALEF_MAQSURA = re.compile("ى")

_TOKEN_RE = re.compile(r"[A-Za-z\u0600-\u06FF0-9]+")


def normalize_arabic(text: str) -> str:
    text = _AR_DIACRITICS.sub("", text or "")
    text = text.replace(_TATWEEL, "")
    text = _ALEF.sub("ا", text)
    text = _WAW_HAMZA.sub("و", text)
    text = _YA_HAMZA.sub("ي", text)
    text = _TA_MARBUTA.sub("ه", text)
    text = _ALEF_MAQSURA.sub("ي", text)
    return text


def normalize_text(text: str) -> str:
    """Normalize mixed Arabic/English text (keeps Latin words as-is)."""
    return normalize_arabic(text or "")


def tokenize(text: str) -> List[str]:
    """Lowercased, normalized token stream for BM25 indexing and queries."""
    text = normalize_text(text)
    return [t.lower() for t in _TOKEN_RE.findall(text)]
