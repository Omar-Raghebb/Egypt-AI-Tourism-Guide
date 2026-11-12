"""CLI for building indexes and asking questions from the command line.

Usage:
  python -m rag.cli build-index [--rebuild]
  python -m rag.cli ask "What is the ticket price of the Pyramids?" [--lang en|ar]
  python -m rag.cli ask "سعر تذكرة الأهرامات؟" --lang ar --structured
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def cmd_build_index(args) -> int:
    from rag.config import RAGConfig
    from rag.ingestion.indexer import KBIndexer
    from rag.ingestion.loader import load_kb, validate_kb

    config = RAGConfig()
    kb = load_kb(config.kb_path, config.kb_single_file)
    issues = validate_kb(kb)
    for issue in issues:
        print(f"[data-quality] {issue}")

    indexer = KBIndexer(config)
    stats = indexer.build(kb, rebuild=args.rebuild)
    print(json.dumps(stats.__dict__, indent=2, ensure_ascii=False))
    return 0


def cmd_ask(args) -> int:
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    if args.structured:
        processed = pipeline.query_processor.process(args.query)
        result = pipeline.retrieve(args.query)
        print(f"Intent: {result.intent}")
        print(f"Entities: {result.detected_entities}")
        print(f"Methods: {result.methods_used}")
        for place in result.structured_records:
            print(pipeline.structured.summary_of(place))
        return 0

    response = pipeline.answer(args.query, language=args.lang)
    print(f"[generated_by={response.generated_by}]")
    print(response.answer)
    if response.sources:
        print("\nSources:")
        for s in response.sources:
            print(f"  - {s}")
    return 0


def cmd_repl(_args) -> int:
    """Interactive bilingual shell for quick manual testing."""
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline()
    print("RehlaAI RAG — type a question ('exit' to quit)")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        response = pipeline.answer(query)
        print(f"[{response.generated_by}] {response.answer}")
    return 0


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-index", help="Build the ChromaDB + BM25 indexes")
    p_build.add_argument("--rebuild", action="store_true", help="Wipe and rebuild")
    p_build.set_defaults(func=cmd_build_index)

    p_ask = sub.add_parser("ask", help="Answer a question end-to-end")
    p_ask.add_argument("query")
    p_ask.add_argument("--lang", choices=["en", "ar"], default=None)
    p_ask.add_argument("--structured", action="store_true",
                       help="Show structured retrieval output only (no LLM)")
    p_ask.set_defaults(func=cmd_ask)

    p_repl = sub.add_parser("repl", help="Interactive question loop")
    p_repl.set_defaults(func=cmd_repl)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
