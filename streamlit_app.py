"""RehlaAI — Egyptian tourism assistant UI (Streamlit).

Talks to the RehlaAI RAG FastAPI service (rag/server.py):
    POST /rag/answer   -> user-facing answer contract
    GET  /rag/health   -> component status

The UI renders the natural-language answer (Markdown) and friendly source
references. Verification badges, verification warnings, generation-mode
labels and fallback notices are intentionally hidden from the user: the
fallback answer is presented exactly like any other answer.

Text direction (BiDi): every block (paragraph, list item, heading, table
cell, caption, chat input) resolves its own direction from its first strong
character (dir="auto"), so Arabic blocks render RTL and English blocks LTR
even inside one mixed-language answer.

Run:
    streamlit run streamlit_app.py
Environment:
    RAG_API_URL      API location (default: http://127.0.0.1:8000)
    RAG_API_TIMEOUT  request timeout in seconds (default: 120)
    RAG_UI_DEBUG=1   show technical service status in the sidebar
"""

from __future__ import annotations

import html
import json
import os
import re
import time
from typing import Any, Dict, Iterator, List, Optional

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------
# Configuration (centralized)
# ---------------------------------------------------------------------
API_BASE_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("RAG_API_TIMEOUT", "120"))
SHOW_STATUS = os.environ.get("RAG_UI_DEBUG", "0") == "1"

SUGGESTED_QUESTIONS = [
    ("🎟️", "What is the ticket price of the Giza Pyramids?"),
    ("🕒", "ما مواعيد زيارة معبد الكرنك؟"),
    ("📍", "What are the best places to visit in Luxor?"),
    ("🎯", "أفضل الأنشطة في أسوان"),
    ("🗺️", "Suggest a 3 day itinerary in Cairo"),
    ("🏛️", "Tell me about Abu Simbel"),
]

DESTINATIONS = [
    {"emoji": "🕌", "name": "Cairo", "name_ar": "القاهرة", "query": "Tell me about Cairo as a travel destination"},
    {"emoji": "🔺", "name": "Giza", "name_ar": "الجيزة", "query": "What are the best places to visit in Giza?"},
    {"emoji": "🏛️", "name": "Luxor", "name_ar": "الأقصر", "query": "What are the best places to visit in Luxor?"},
    {"emoji": "⛵", "name": "Aswan", "name_ar": "أسوان", "query": "What should I visit in Aswan?"},
    {"emoji": "🌊", "name": "Alexandria", "name_ar": "الإسكندرية", "query": "Tell me about Alexandria as a travel destination"},
    {"emoji": "🤿", "name": "Sharm El-Sheikh", "name_ar": "شرم الشيخ", "query": "What can I do in Sharm El-Sheikh?"},
]

LANG_OPTIONS = {"Auto · تلقائي": None, "English": "en", "العربية": "ar"}

# ---------------------------------------------------------------------
# Design system — deep navy / muted gold / sandstone, papyrus texture
# ---------------------------------------------------------------------
NAVY = "#0d1b2a"
NAVY_SOFT = "#13263a"
GOLD = "#c9a227"
GOLD_SOFT = "#e3c96b"
SAND = "#f4ecdd"
SAND_DEEP = "#e8dcc3"
INK = "#22303f"

# pyramids + sun silhouette used as the hero decoration (inline, no assets)
HERO_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 400 160'>"
    "<circle cx='310' cy='44' r='26' fill='%23e3c96b' fill-opacity='.35'/>"
    "<path d='M0 160 L120 62 L205 160Z' fill='%23c9a227' fill-opacity='.20'/>"
    "<path d='M135 160 L255 28 L375 160Z' fill='%23c9a227' fill-opacity='.28'/>"
    "<path d='M300 160 L350 104 L400 160Z' fill='%23c9a227' fill-opacity='.18'/>"
    "<rect x='0' y='150' width='400' height='10' fill='%23e3c96b' fill-opacity='.22'/>"
    "</svg>"
)

st.set_page_config(
    page_title="RehlaAI — Egypt Travel Assistant",
    page_icon="𓂀",
    layout="centered",
)

CSS = f"""
<style>
    /* ---------- app shell ---------- */
    .stApp {{
        background:
            repeating-linear-gradient(45deg, rgba(201,162,39,0.03) 0 2px, transparent 2px 14px),
            linear-gradient(180deg, {SAND} 0%, {SAND_DEEP} 100%);
    }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stAppDeployButton"], #MainMenu, [data-testid="stMainMenu"] {{ display: none; }}
    footer {{ visibility: hidden; }}

    /* ---------- hero ---------- */
    .rehla-hero {{
        position: relative; overflow: hidden;
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_SOFT} 60%, #1b3a55 100%);
        border-radius: 18px;
        padding: 28px 30px 24px 30px;
        color: #f6f1e7;
        border: 1px solid rgba(201,162,39,0.35);
        box-shadow: 0 10px 30px rgba(13,27,42,0.35);
        margin-bottom: 14px;
    }}
    .rehla-hero::after {{
        content: ""; position: absolute; right: 0; bottom: 0;
        width: 360px; height: 150px; pointer-events: none;
        background: url("{HERO_SVG}") no-repeat right bottom / contain;
    }}
    .rehla-hero > * {{ position: relative; z-index: 1; }}
    .rehla-brand {{
        font-size: 34px; font-weight: 800; letter-spacing: 0.5px;
        color: {GOLD_SOFT}; margin: 0;
    }}
    .rehla-brand span {{ color: #f6f1e7; }}
    .rehla-tagline {{
        margin: 6px 0 0 0; font-size: 15px; color: #d9d2c2; max-width: 560px;
    }}
    .hero-divider {{
        height: 3px; width: 74px; background: {GOLD};
        border-radius: 3px; margin: 14px 0 12px 0;
    }}
    .hero-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .hero-chips span {{
        font-size: 12.5px; color: #f1e7c8;
        background: rgba(227,201,107,0.12);
        border: 1px solid rgba(227,201,107,0.35);
        border-radius: 999px; padding: 3px 11px;
    }}
    @media (max-width: 640px) {{
        .rehla-hero::after {{ width: 220px; opacity: .55; }}
        .rehla-brand {{ font-size: 28px; }}
    }}

    /* ---------- chat messages ---------- */
    [data-testid="stChatMessage"] {{
        background: #ffffff;
        border: 1px solid {SAND_DEEP};
        border-left: 4px solid {GOLD};
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(34,48,63,0.05);
        padding: 6px 8px;
    }}
    [data-testid="stChatMessage"] .stMarkdown,
    [data-testid="stChatMessage"] .stMarkdown p,
    [data-testid="stChatMessage"] .stMarkdown li,
    [data-testid="stChatMessage"] .stMarkdown td,
    [data-testid="stChatMessage"] .stMarkdown th,
    [data-testid="stChatMessage"] .stMarkdown h1,
    [data-testid="stChatMessage"] .stMarkdown h2,
    [data-testid="stChatMessage"] .stMarkdown h3,
    [data-testid="stChatMessage"] .stMarkdown h4 {{
        color: {INK} !important;
        font-family: "Segoe UI", Tahoma, "Noto Naskh Arabic", "Noto Sans Arabic", sans-serif;
    }}
    [data-testid="stChatMessage"] .stMarkdown h2,
    [data-testid="stChatMessage"] .stMarkdown h3 {{
        color: {NAVY} !important;
        border-bottom: 2px solid rgba(201,162,39,0.45);
        padding-bottom: 4px; margin-top: 0.9rem;
    }}
    [data-testid="stChatMessage"] .stMarkdown strong {{ color: {NAVY} !important; }}
    [data-testid="stChatMessage"] .stMarkdown table {{
        border-collapse: collapse; width: 100%;
    }}
    [data-testid="stChatMessage"] .stMarkdown th {{
        background: rgba(201,162,39,0.16);
    }}
    [data-testid="stChatMessage"] .stMarkdown :is(th, td) {{
        border: 1px solid {SAND_DEEP}; padding: 6px 10px;
    }}

    /* user bubble: navy, so the conversation reads as two sides */
    [data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarUser"],
                                          [data-testid="chatAvatarIcon-user"])) {{
        background: {NAVY_SOFT};
        border-color: rgba(201,162,39,0.40);
        border-left: 1px solid rgba(201,162,39,0.40);
        border-right: 4px solid {GOLD};
    }}
    [data-testid="stChatMessage"]:has(:is([data-testid="stChatMessageAvatarUser"],
                                          [data-testid="chatAvatarIcon-user"]))
        .stMarkdown :is(p, li, strong, span) {{
        color: #f6f1e7 !important;
    }}

    /* ---------- BiDi: each block follows its own first strong character ----------
       (dir="auto" is also set from JS below; this is the CSS default) */
    [data-testid="stChatMessage"] :is(p, li, h1, h2, h3, h4, h5, h6,
                                       blockquote, td, th, summary) {{
        unicode-bidi: plaintext;
        text-align: start;
    }}
    [data-testid="stChatMessage"] :is(ul, ol) {{ padding-inline: 1.6rem; }}
    [data-testid="stChatMessage"] :is(code, pre) {{
        direction: ltr; unicode-bidi: isolate; text-align: left;
    }}
    [data-testid="stCaptionContainer"],
    [data-testid="stChatInput"] textarea {{
        unicode-bidi: plaintext; text-align: start;
    }}

    /* ---------- sources as pills ---------- */
    .src-row {{
        display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
        margin: 2px 4px 16px 4px;
    }}
    .src-label {{ font-size: 13px; color: #7a6f57; font-weight: 600; }}
    .src-pill {{
        background: rgba(201,162,39,0.14);
        border: 1px solid rgba(201,162,39,0.45);
        color: {NAVY}; border-radius: 999px;
        padding: 2px 10px; font-size: 12.5px;
    }}

    /* ---------- section headers ---------- */
    .section-title {{
        font-size: 18px; font-weight: 700; color: {NAVY};
        margin: 22px 0 8px 0;
    }}
    .section-title::after {{
        content: ""; display: block; height: 2px; width: 46px;
        background: {GOLD}; border-radius: 2px; margin-top: 4px;
    }}

    /* ---------- buttons (destinations + suggested questions) ---------- */
    .stApp .stButton > button {{
        background: #ffffff; color: {NAVY};
        border: 1px solid {SAND_DEEP}; border-radius: 12px;
        box-shadow: 0 2px 8px rgba(34,48,63,0.05);
        transition: all .15s ease;
    }}
    .stApp .stButton > button:hover {{
        border-color: {GOLD}; color: {NAVY};
        background: #fffaf0; transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(201,162,39,0.18);
    }}
    .stApp .stButton > button p {{ color: inherit; }}

    /* ---------- chat input ---------- */
    [data-testid="stChatInput"] {{
        border-radius: 16px;
        box-shadow: 0 4px 18px rgba(13,27,42,0.10);
    }}
    [data-testid="stChatInput"]:focus-within {{
        box-shadow: 0 0 0 2px rgba(201,162,39,0.55);
    }}

    /* ---------- sidebar: navy panel ---------- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY} 0%, {NAVY_SOFT} 100%);
        border-right: 1px solid rgba(201,162,39,0.30);
    }}
    [data-testid="stSidebar"] :is(p, span, label, h1, h2, h3, li) {{ color: #e9e2d0; }}
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color: #b9b19c; }}
    [data-testid="stSidebar"] hr {{ border-color: rgba(201,162,39,0.25); }}
    [data-testid="stSidebar"] :is(.stButton, .stDownloadButton) > button {{
        background: rgba(255,255,255,0.05); color: {SAND};
        border: 1px solid rgba(201,162,39,0.35); box-shadow: none;
    }}
    [data-testid="stSidebar"] :is(.stButton, .stDownloadButton) > button:hover {{
        background: rgba(201,162,39,0.18); border-color: {GOLD_SOFT};
        color: #ffffff; transform: none; box-shadow: none;
    }}
    [data-testid="stSidebar"] :is(.stButton, .stDownloadButton) button p {{ color: inherit !important; }}
    [data-testid="stSidebar"] [data-testid="stAlert"] * {{ color: #3b2f00 !important; }}
    .sidebar-welcome {{ color: {GOLD_SOFT}; font-weight: 700; font-size: 20px; }}
    .sidebar-note {{ color: #a9a18c; font-size: 12px; line-height: 1.5; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# Sets dir="auto" on every rendered block (and the chat input) and keeps
# doing it as Streamlit re-renders. dir="auto" makes the browser pick RTL or
# LTR per element from its first strong character, which also puts list
# bullets and text alignment on the correct side.
BIDI_JS = """
<script>
(function () {
  const doc = window.parent.document;
  const SEL = [
    '[data-testid="stChatMessage"] :is(p, li, ul, ol, h1, h2, h3, h4, h5, h6, blockquote, table, td, th)',
    '[data-testid="stCaptionContainer"]',
    '[data-testid="stChatInput"] textarea'
  ].join(',');

  function apply() {
    doc.querySelectorAll(SEL).forEach(function (el) {
      if (el.getAttribute('dir') !== 'auto') el.setAttribute('dir', 'auto');
    });
  }

  try { if (window.parent.__rehlaBidiObs) window.parent.__rehlaBidiObs.disconnect(); } catch (e) {}
  let queued = false;
  const obs = new MutationObserver(function () {
    if (queued) return;
    queued = true;
    window.parent.requestAnimationFrame(function () { queued = false; apply(); });
  });
  obs.observe(doc.body, { childList: true, subtree: true });
  window.parent.__rehlaBidiObs = obs;
  apply();
})();
</script>
"""
components.html(BIDI_JS, height=0)


# ---------------------------------------------------------------------
# API layer (single integration point)
# ---------------------------------------------------------------------
class ApiError(Exception):
    """User-friendly API failure (details stay in logs, not in the UI)."""


def api_answer(query: str, language: Optional[str] = None) -> Dict[str, Any]:
    """POST /rag/answer — returns the user-facing contract dict.

    Raises ApiError with a user-friendly message on any failure; internal
    error details are logged to the server console, never shown raw.
    """
    payload: Dict[str, Any] = {"query": query}
    if language in ("en", "ar"):
        payload["language"] = language
    try:
        resp = requests.post(
            f"{API_BASE_URL}/rag/answer", json=payload, timeout=REQUEST_TIMEOUT
        )
    except requests.exceptions.ConnectionError:
        raise ApiError(
            "I can't reach the RehlaAI service right now. Please make sure the "
            "API server is running (uvicorn rag.server:app) and try again."
        )
    except requests.exceptions.Timeout:
        raise ApiError("The request took too long. Please try again in a moment.")
    except requests.exceptions.RequestException:
        raise ApiError("Something went wrong while contacting the service. Please try again.")

    if resp.status_code == 422:
        raise ApiError("Please enter a valid question (up to 1000 characters).")
    if resp.status_code >= 500:
        raise ApiError("The service had a temporary problem answering. Please try again.")
    if resp.status_code != 200:
        raise ApiError("Unexpected service response. Please try again.")
    try:
        return resp.json()
    except (json.JSONDecodeError, ValueError):
        raise ApiError("The service returned an unreadable response. Please try again.")


@st.cache_data(ttl=60, show_spinner=False)
def api_health() -> Optional[Dict[str, Any]]:
    try:
        resp = requests.get(f"{API_BASE_URL}/rag/health", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


# ---------------------------------------------------------------------
# BiDi helpers
# ---------------------------------------------------------------------
_AR_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_STRONG_RE = re.compile(r"[A-Za-z\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
# trailing punctuation right after Latin text / digits / closing bracket
_TRAIL_PUNCT_RE = re.compile(r"(?<=[A-Za-z0-9)\]])([.!?,;:]+)(?=[*_]*\s*$)")
LRM = "\u200e"


def _contains_arabic(text: str) -> bool:
    return bool(_AR_RE.search(text or ""))


def _is_rtl(text: str) -> bool:
    """True when the first strong (letter) character is Arabic."""
    m = _STRONG_RE.search(text or "")
    return bool(m and _AR_RE.match(m.group()))


def fix_bidi(markdown: str) -> str:
    """Fix the classic 'period jumps to the left' bug.

    In an RTL line that ends with an English phrase
    ("ملاحظة: Travel by train.") the final "." is resolved as RTL and lands
    at the start of the line. Putting an invisible LRM before it binds it to
    the English run. Tables, code blocks and LTR lines are left untouched.
    """
    out = []
    in_code = False
    for line in markdown.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code = not in_code
            out.append(line)
            continue
        if in_code or stripped.startswith("|") or not _is_rtl(line):
            out.append(line)
            continue
        out.append(_TRAIL_PUNCT_RE.sub(lambda m: LRM + m.group(1), line))
    return "\n".join(out)


# ---------------------------------------------------------------------
# Conversation context: follow-up destination resolution (Bug A)
# ---------------------------------------------------------------------
# The RAG API is stateless by design (each /rag/answer request sees only its
# own query — documented in tests/test_user_contract.py, Group F). The chat
# already keeps the conversation in st.session_state.history for rendering,
# so the client — the owner of the conversation — resolves a follow-up's
# destination from that history and, when needed, appends the destination to
# the query TEXT. No new server contract, no new memory system.
_DESTINATION_ALIASES = [
    # (canonical EN, canonical AR, raw-text aliases incl. hamza/ta-marbuta
    #  spelling variants — matched against the RAW message, not normalized)
    ("Cairo", "القاهرة", ["cairo", "القاهرة", "القاهره"]),
    ("Giza", "الجيزة", ["giza", "الجيزة", "الجيزه"]),
    ("Luxor", "الأقصر", ["luxor", "الأقصر", "الاقصر", "اقصر"]),
    ("Aswan", "أسوان", ["aswan", "أسوان", "اسوان", "الأسوان", "الاسوان"]),
    (
        "Alexandria",
        "الإسكندرية",
        ["alexandria", "الإسكندرية", "الاسكندرية", "إسكندرية", "اسكندرية",
         "الاسكندريه", "إسكندريه", "اسكندريه"],
    ),
    ("Hurghada", "الغردقة", ["hurghada", "الغردقة", "الغردقه", "غردقة"]),
    ("Sharm El-Sheikh", "شرم الشيخ", ["sharm", "شرم الشيخ"]),
]

# A follow-up carries the active destination only when it is a plan/itinerary
# request ("عاوز خطة 4 أيام", "Make it a 2-day plan"); other follow-ups
# (prices, hours, a specific place) are complete questions on their own and
# must not inherit the previous destination.
_PLAN_FOLLOWUP_RE = re.compile(
    r"خطة|برنامج|يوم|أيام|ايام|itinerary|trip|plan|days?\b",
    re.IGNORECASE,
)


def find_destinations(text: str) -> List[Dict[str, str]]:
    """Destinations explicitly mentioned in ONE message, in appearance order."""
    low = (text or "").lower()
    hits: List[tuple] = []
    for index, (name_en, name_ar, aliases) in enumerate(_DESTINATION_ALIASES):
        positions = [
            low.find(alias.lower()) for alias in aliases if alias
        ]
        positions = [p for p in positions if p >= 0]
        if positions:
            hits.append((min(positions), index, {"name_en": name_en, "name_ar": name_ar}))
    hits.sort(key=lambda h: (h[0], h[1]))
    return [h[2] for h in hits]


def resolve_followup_destination(
    query: str, history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Resolve the destination of a follow-up message from the conversation.

    Modes: ``explicit`` (the query names its own destination(s) — sent
    unchanged), ``resolved`` (destination taken from the most recent
    destination-bearing USER message — only that one is considered, so older
    unrelated destinations never leak), ``ambiguous`` (the active context
    holds several destinations — ask, don't guess), ``none`` (not a plan
    follow-up, or nothing to resolve — sent unchanged). Assistant messages
    are never scanned: an answer may list many cities, but the user defines
    where they want to go.
    """
    if not _PLAN_FOLLOWUP_RE.search(query or ""):
        return {"mode": "none"}
    destinations = find_destinations(query)
    if len(destinations) == 1:
        return {"mode": "explicit", "destination": destinations[0]}
    if len(destinations) > 1:
        # several destinations requested explicitly — the multi-destination
        # path, not an ambiguity
        return {"mode": "explicit", "destinations": destinations}
    for message in reversed(history or []):
        if message.get("role") != "user":
            continue  # assistant answers mention many cities — never scanned
        found = find_destinations(message.get("content") or "")
        if not found:
            continue  # older message — not the active destination
        if len(found) == 1:
            return {"mode": "resolved", "destination": found[0]}
        return {"mode": "ambiguous", "destinations": found}
    return {"mode": "none"}


def destination_clarification(
    query: str, destinations: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Concise clarification payload when the destination is ambiguous."""
    if _is_rtl(query):
        answer = (
            "لأي مدينة البرنامج؟ ("
            + " أو ".join(d["name_ar"] for d in destinations)
            + ")"
        )
        language = "ar"
    else:
        answer = (
            "Which city is the plan for? ("
            + " or ".join(d["name_en"] for d in destinations)
            + ")"
        )
        language = "en"
    return {"answer": answer, "sources": [], "language": language}


# ---------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------
def _word_stream(text: str) -> Iterator[str]:
    """Typing effect: yield word by word, capped at ~2.5s total."""
    parts = re.split(r"(\s+)", text)
    words = sum(1 for p in parts if p and not p.isspace())
    delay = min(0.02, 2.5 / max(words, 1))
    for p in parts:
        yield p
        if p and not p.isspace():
            time.sleep(delay)


def render_sources(sources: List[str], answer: str, language: Optional[str]) -> None:
    """Sources as small gold pills (place names only, HTML-escaped)."""
    if not sources:
        return
    rtl = _is_rtl(answer) or language == "ar"
    label = "المصادر" if rtl else "Sources"
    pills = "".join(
        f"<span class='src-pill' dir='auto'>📍 {html.escape(str(s))}</span>"
        for s in sources
    )
    st.markdown(
        f"<div class='src-row' dir='auto'><span class='src-label'>{label}</span>{pills}</div>",
        unsafe_allow_html=True,
    )


def render_answer(payload: Dict[str, Any], animate: bool = False) -> None:
    """Render one assistant message: Markdown answer + friendly sources.

    No verification UI, no generation-mode labels, no fallback notices —
    a fallback answer is presented exactly like any other answer.
    Direction is NOT forced per message: each block picks its own.
    """
    answer = payload.get("answer", "")
    with st.chat_message("assistant", avatar="🏺"):
        text = fix_bidi(answer)
        if animate:
            st.write_stream(_word_stream(text))
        else:
            st.markdown(text)
    render_sources(payload.get("sources") or [], answer, payload.get("language"))


def render_history() -> None:
    animate_idx = st.session_state.animate_idx
    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧭"):
                st.markdown(message["content"])
        else:
            render_answer(message["payload"], animate=(i == animate_idx))
    st.session_state.animate_idx = None


def render_welcome() -> None:
    """Friendly first message so the empty state feels like a conversation."""
    with st.chat_message("assistant", avatar="🏺"):
        st.markdown("**أهلاً بك في رحلة! 👋** اسألني عن الأماكن السياحية والأسعار والمواعيد، أو اطلب خطة رحلة جاهزة.")
        st.markdown("**Welcome to RehlaAI!** Ask me about places, ticket prices, opening hours — or request a ready-made itinerary.")


def render_home_cards() -> None:
    """Destination grid + suggested questions (only on the empty state)."""
    st.markdown("<div class='section-title'>Explore destinations · استكشف الوجهات</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    for i, dest in enumerate(DESTINATIONS):
        with cols[i % 3]:
            if st.button(
                f"{dest['emoji']} {dest['name']} · {dest['name_ar']}",
                key=f"home_dest_{dest['name']}",
                use_container_width=True,
            ):
                st.session_state.pending_query = dest["query"]

    st.markdown("<div class='section-title'>Try asking · جرّب تسأل</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, (icon, suggestion) in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(f"{icon} {suggestion}", key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = suggestion


def history_markdown() -> str:
    """Whole conversation as a Markdown file (for the download button)."""
    lines = ["# RehlaAI — my Egypt trip notes", ""]
    for m in st.session_state.history:
        if m["role"] == "user":
            lines += [f"## ❓ {m['content']}", ""]
        else:
            p = m["payload"]
            lines += [p.get("answer", ""), ""]
            if p.get("sources"):
                lines += ["_Sources: " + " · ".join(p["sources"]) + "_", ""]
    return "\n".join(lines)


def ask_and_store(query: str) -> None:
    """Send one request, store both messages. Errors survive the rerun."""
    lang = LANG_OPTIONS.get(st.session_state.get("lang_choice"))
    # Bug A: resolve the follow-up destination BEFORE storing anything. The
    # API is stateless, so an omitted destination is appended to the query
    # text; the ORIGINAL user text is what goes into the history.
    resolution = resolve_followup_destination(query, st.session_state.history)
    if resolution["mode"] == "ambiguous":
        # genuine ambiguity: ask a concise clarification instead of guessing
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({
            "role": "assistant",
            "payload": destination_clarification(query, resolution["destinations"]),
        })
        return
    api_query = query
    if resolution["mode"] == "resolved":
        dest = resolution["destination"]
        if _is_rtl(query):
            api_query = f"{query} في {dest['name_ar']}"
        else:
            api_query = f"{query} in {dest['name_en']}"
    st.session_state.history.append({"role": "user", "content": query})
    spinner_text = "رحلة تجهّز إجابتك…" if _is_rtl(query) else "RehlaAI is planning your answer…"
    with st.spinner(spinner_text):
        try:
            payload = api_answer(api_query, language=lang)
        except ApiError as exc:
            st.session_state.history.pop()  # don't keep dangling user message
            st.session_state.last_error = str(exc)
            return
    st.session_state.history.append({"role": "assistant", "payload": payload})
    st.session_state.animate_idx = len(st.session_state.history) - 1


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending_query", None)
st.session_state.setdefault("last_error", None)
st.session_state.setdefault("animate_idx", None)

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    """
    <div class='rehla-hero'>
        <p class='rehla-brand'>𓂀 Rehla<span>AI</span> · رحلة</p>
        <p class='rehla-tagline'>
            Your AI travel companion for Egypt — grounded answers about places,
            prices, visiting hours, activities and itineraries, in Arabic and English.
        </p>
        <div class='hero-divider'></div>
        <div class='hero-chips'>
            <span>🎟️ Prices · الأسعار</span>
            <span>🕒 Hours · المواعيد</span>
            <span>🗺️ Itineraries · الخطط</span>
            <span>🏛️ Sights · المعالم</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-welcome'>𓂀 RehlaAI</div>", unsafe_allow_html=True)
    st.caption("Egypt, explained — one question at a time.")

    health = api_health()
    if not health:
        st.warning(
            "The assistant is temporarily unavailable. Please try again shortly."
            + (" Start the API with `uvicorn rag.server:app`." if SHOW_STATUS else "")
        )
    elif SHOW_STATUS:
        with st.expander("Service status"):
            llm = health.get("llm_provider") or {}
            st.caption(
                f"Knowledge base: **{health.get('kb_name') or 'RehlaAI'}** · "
                f"{health.get('kb_places', 0)} places · "
                f"{(health.get('dense_index') or {}).get('documents', 0)} indexed documents"
            )
            st.caption(
                "AI generation: "
                + ("🟢 configured" if llm.get("key_configured") else "🟡 offline answers")
            )

    st.divider()
    st.markdown("**Answer language · لغة الإجابة**")
    st.radio(
        "Answer language",
        list(LANG_OPTIONS),
        key="lang_choice",
        label_visibility="collapsed",
    )

    st.divider()
    st.subheader("Explore Egypt")
    for dest in DESTINATIONS:
        if st.button(
            f"{dest['emoji']} {dest['name']} · {dest['name_ar']}",
            key=f"dest_{dest['name']}",
            use_container_width=True,
        ):
            st.session_state.pending_query = dest["query"]

    st.divider()
    has_history = bool(st.session_state.history)
    if st.button("🗑️ New chat · محادثة جديدة", use_container_width=True, disabled=not has_history):
        st.session_state.history = []
        st.session_state.last_error = None
        st.rerun()
    st.download_button(
        "⬇️ Save my trip notes",
        data=history_markdown(),
        file_name="rehlaai-trip-notes.md",
        mime="text/markdown",
        use_container_width=True,
        disabled=not has_history,
    )
    st.markdown(
        "<p class='sidebar-note'>Prices and opening hours can change — please confirm on the "
        "official website before you go.<br>الأسعار والمواعيد قد تتغير، يُرجى التأكد قبل الزيارة.</p>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------
if not st.session_state.history:
    render_welcome()
    render_home_cards()
else:
    render_history()

if st.session_state.last_error:
    st.error(st.session_state.last_error)
    st.session_state.last_error = None

question = st.chat_input("Ask anything about traveling in Egypt… / اسأل عن السفر في مصر…")

# one request per new query — either typed or triggered from a button
query_to_ask = question or st.session_state.pending_query
if query_to_ask:
    st.session_state.pending_query = None
    ask_and_store(query_to_ask)
    st.rerun()