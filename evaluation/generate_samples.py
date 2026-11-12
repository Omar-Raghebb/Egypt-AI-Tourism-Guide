"""Generate sample outputs with the production pipeline (fallback mode).

Writes evaluation/sample_outputs.md so the report contains real outputs
captured from this machine, not invented ones.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.pipeline import RAGPipeline

QUERIES = [
    ("en", "What is the ticket price of the Pyramids of Giza?"),
    ("ar", "سعر تذكرة أهرامات الجيزة؟"),
    ("en", "What are the opening hours of the Pyramids of Giza?"),
    ("ar", "ما هي مواعيد فتح معبد الأقصر؟"),
    ("en", "What are the best places to visit in Luxor?"),
    ("en", "What are the exact coordinates of the Giza Pyramids?"),
    ("en", "Is the Giza Plateau accessible for wheelchair users?"),
    ("en", "What is the nearest airport to the Pyramids of Giza?"),
    ("en", "Which activities are available at Karnak Temple?"),
    ("en", "Is the Grand Egyptian Museum suitable for young children?"),
    ("en", "What are the best places to visit in Aswan?"),
]

OUT = Path(__file__).parent / "sample_outputs.md"


def main() -> None:
    pipeline = RAGPipeline()
    pipeline.initialize()
    health = pipeline.health()

    lines = [
        "# Sample RAG outputs (captured locally, fallback mode — no GROQ_API_KEY)",
        "",
        f"- Pipeline health: `{json.dumps(health, ensure_ascii=False)}`",
        "- `generated_by=fallback` means Groq was not called because no API key is",
        "  configured; the text after the notice is the exact retrieved evidence.",
        "",
    ]
    for lang, query in QUERIES:
        response = pipeline.answer(query, language=lang)
        lines += [
            f"## Query ({lang}): {query}",
            "",
            f"- intent: `{response.retrieval.intent}` "
            f"- methods: `{response.retrieval.methods_used}` "
            f"- entities: `{json.dumps(response.retrieval.detected_entities, ensure_ascii=False)}` "
            f"- latency: {response.retrieval.latency_ms:.0f} ms",
            f"- verification_flags: `{json.dumps(response.verification_flags, ensure_ascii=False)}`",
            "",
            "```text",
            response.answer.strip(),
            "```",
            "",
        ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
