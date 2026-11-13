"""RehlaAI — Egyptian tourism assistant UI (Streamlit).

Talks to the RehlaAI RAG FastAPI service (rag/server.py):
    POST /rag/answer   -> user-facing answer contract
    GET  /rag/health   -> component status

The UI renders ONLY the user-facing contract: the natural-language answer,
verification badge, friendly source references, and user-relevant warnings.
Internal evidence (documents, scores, JSON) is never displayed.

Run:
    streamlit run streamlit_app.py
Configure the API location with the RAG_API_URL environment variable
(default: http://127.0.0.1:8000).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------
# Configuration (centralized)
# ---------------------------------------------------------------------
API_BASE_URL = os.environ.get("RAG_API_URL", "http://127.0.0.1:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.environ.get("RAG_API_TIMEOUT", "120"))

SUGGESTED_QUESTIONS = [
    "What is the ticket price of the Giza Pyramids?",
    "ما مواعيد زيارة معبد الكرنك؟",
    "What are the best places to visit in Luxor?",
    "أفضل الأنشطة في أسوان",
    "Suggest a 3 day itinerary in Cairo",
    "Tell me about Abu Simbel",
]

DESTINATIONS = [
    {"name": "Cairo", "name_ar": "القاهرة", "query": "Tell me about Cairo as a travel destination"},
    {"name": "Giza", "name_ar": "الجيزة", "query": "What are the best places to visit in Giza?"},
    {"name": "Luxor", "name_ar": "الأقصر", "query": "What are the best places to visit in Luxor?"},
    {"name": "Aswan", "name_ar": "أسوان", "query": "What should I visit in Aswan?"},
    {"name": "Alexandria", "name_ar": "الإسكندرية", "query": "Tell me about Alexandria as a travel destination"},
    {"name": "Sharm El-Sheikh", "name_ar": "شرم الشيخ", "query": "What can I do in Sharm El-Sheikh?"},
]

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

st.set_page_config(
    page_title="RehlaAI — Egypt Travel Assistant",
    page_icon="𓂀",
    layout="centered",
)

CSS = f"""
<style>
    /* app background: layered sand tones with a subtle papyrus weave */
    .stApp {{
        background:
            repeating-linear-gradient(45deg, rgba(201,162,39,0.03) 0 2px, transparent 2px 14px),
            linear-gradient(180deg, {SAND} 0%, {SAND_DEEP} 100%);
    }}
    /* hero */
    .rehla-hero {{
        background: linear-gradient(135deg, {NAVY} 0%, {NAVY_SOFT} 60%, #1b3a55 100%);
        border-radius: 18px;
        padding: 28px 30px 24px 30px;
        color: #f6f1e7;
        border: 1px solid rgba(201,162,39,0.35);
        box-shadow: 0 10px 30px rgba(13,27,42,0.35);
        margin-bottom: 8px;
    }}
    .rehla-brand {{
        font-size: 34px; font-weight: 800; letter-spacing: 0.5px;
        color: {GOLD_SOFT}; margin: 0;
    }}
    .rehla-brand span {{ color: #f6f1e7; }}
    .rehla-tagline {{
        margin: 6px 0 0 0; font-size: 15px; color: #d9d2c2;
        max-width: 640px;
    }}
    .hero-divider {{
        height: 3px; width: 74px; background: {GOLD};
        border-radius: 3px; margin: 14px 0 0 0;
    }}
    /* chat bubbles */
    .bubble-user, .bubble-assistant {{
        max-width: 86%;
        padding: 14px 18px;
        border-radius: 16px;
        margin: 8px 0;
        line-height: 1.55;
        font-size: 15px;
        word-wrap: break-word;
    }}
    .bubble-user {{
        margin-left: auto;
        background: {NAVY};
        color: #f6f1e7;
        border-bottom-right-radius: 4px;
    }}
    .bubble-assistant {{
        margin-right: auto;
        background: #ffffff;
        color: {INK};
        border: 1px solid {SAND_DEEP};
        border-left: 4px solid {GOLD};
        border-bottom-left-radius: 4px;
        box-shadow: 0 2px 10px rgba(34,48,63,0.06);
        white-space: pre-wrap;
    }}
    .bubble-meta {{
        font-size: 12px; color: #6b7a89; margin: -2px 2px 10px 2px;
    }}
    /* verification + warning chips */
    .chip {{
        display: inline-block;
        padding: 3px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px 6px 8px 0;
    }}
    .chip-verified {{ background: #e6f2e6; color: #2e6b34; border: 1px solid #bcd9bd; }}
    .chip-unverified {{ background: #fdf3d7; color: #8a6d1a; border: 1px solid #e8d491; }}
    .chip-mixed {{ background: #e8f0fb; color: #2d5c9e; border: 1px solid #b9d0ee; }}
    .chip-info {{ background: #eef1f4; color: #4a5a68; border: 1px solid #ccd6dd; }}
    .note-warn {{
        background: #fdf6e3; border: 1px solid #ecd9a0; color: #6b5313;
        border-radius: 10px; padding: 8px 12px; font-size: 13px; margin: 6px 0;
    }}
    /* section headers */
    .section-title {{
        font-size: 18px; font-weight: 700; color: {NAVY};
        margin: 26px 0 4px 0;
    }}
    .section-title::after {{
        content: ""; display: block; height: 2px; width: 46px;
        background: {GOLD}; border-radius: 2px; margin-top: 4px;
    }}
    /* destination cards */
    .dest-card {{
        background: #ffffff; border: 1px solid {SAND_DEEP};
        border-radius: 12px; padding: 12px 14px; text-align: center;
        box-shadow: 0 2px 8px rgba(34,48,63,0.05);
    }}
    .dest-card .en {{ font-weight: 700; color: {NAVY}; font-size: 14px; }}
    .dest-card .ar {{ color: #8a7f66; font-size: 13px; margin-top: 2px; }}
    /* sidebar */
    .sidebar-welcome {{ color: {GOLD_SOFT}; font-weight: 700; font-size: 18px; }}
    footer {{ visibility: hidden; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# API layer (single integration point)
# ---------------------------------------------------------------------
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
    except json.JSONDecodeError:
        raise ApiError("The service returned an unreadable response. Please try again.")


class ApiError(Exception):
    """User-friendly API failure (details stay in logs, not in the UI)."""


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
# Answer rendering (user-facing contract only)
# ---------------------------------------------------------------------
VERIFICATION_CHIP = {
    "verified": ('✅ Verified', "chip-verified"),
    "mixed": ('🔶 Partly verified', "chip-mixed"),
    "unverified": ('⚠️ Not fully verified', "chip-unverified"),
    "missing": ('ℹ️ Not in knowledge base', "chip-info"),
}


def render_answer(payload: Dict[str, Any]) -> None:
    verification = payload.get("verification") or {}
    status = verification.get("status", "unverified")
    chip_text, chip_class = VERIFICATION_CHIP.get(status, VERIFICATION_CHIP["unverified"])

    st.markdown(f"<div class='bubble-assistant'>{payload['answer']}</div>", unsafe_allow_html=True)

    chips = f"<span class='chip {chip_class}'>{chip_text}</span>"
    mode = payload.get("generation_mode")
    if mode == "fallback":
        chips += "<span class='chip chip-mixed'>📖 From knowledge base</span>"
    st.markdown(chips, unsafe_allow_html=True)

    sources = payload.get("sources") or []
    if sources:
        st.caption("Sources: " + " · ".join(sources))

    for warning in payload.get("warnings") or []:
        st.markdown(f"<div class='note-warn'>ℹ️ {warning}</div>", unsafe_allow_html=True)


def render_history() -> None:
    for message in st.session_state.history:
        if message["role"] == "user":
            st.markdown(
                f"<div class='bubble-user'>{message['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            render_answer(message["payload"])


def ask_and_store(query: str) -> None:
    """Send one request, store both messages. Guarded against rerun dupes."""
    st.session_state.history.append({"role": "user", "content": query})
    with st.spinner("RehlaAI is thinking…"):
        try:
            payload = api_answer(query)
        except ApiError as exc:
            st.session_state.history.pop()  # don't keep dangling user message
            st.error(str(exc))
            return
    st.session_state.history.append({"role": "assistant", "payload": payload})


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------
st.markdown(
    f"""
    <div class='rehla-hero'>
        <p class='rehla-brand'>𓂀 Rehla<span>AI</span> · رحلة</p>
        <p class='rehla-tagline'>
            Your AI travel companion for Egypt — grounded answers about places,
            prices, visiting hours, activities and itineraries, in Arabic and English.
        </p>
        <div class='hero-divider'></div>
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
    if health:
        llm = (health.get("llm_provider") or {})
        st.caption(
            f"Knowledge base: **{health.get('kb_name') or 'RehlaAI'}** · "
            f"{health.get('kb_places', 0)} places · "
            f"{(health.get('dense_index') or {}).get('documents', 0)} indexed documents"
        )
        st.caption(
            "AI generation: "
            + ("🟢 configured" if llm.get("key_configured") else "🟡 fallback mode (no API key)")
        )
    else:
        st.warning("API server not reachable — start it with `uvicorn rag.server:app`.")

    st.divider()
    st.subheader("Explore Egypt")
    for dest in DESTINATIONS:
        if st.button(
            f"{dest['name']} · {dest['name_ar']}",
            key=f"dest_{dest['name']}",
            use_container_width=True,
        ):
            st.session_state.pending_query = dest["query"]

# ---------------------------------------------------------------------
# Chat area
# ---------------------------------------------------------------------
render_history()

question = st.chat_input("Ask anything about traveling in Egypt…")

# suggested questions (each submits a real query)
if not st.session_state.history:
    st.markdown("<div class='section-title'>Try asking</div>", unsafe_allow_html=True)
    cols = st.columns(2)
    for i, suggestion in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 2]:
            if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                st.session_state.pending_query = suggestion

# one request per new query — either typed or triggered from a button
query_to_ask = question or st.session_state.pending_query
if query_to_ask:
    st.session_state.pending_query = None
    ask_and_store(query_to_ask)
    st.rerun()
