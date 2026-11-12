"""Generation evaluation (requires GROQ_API_KEY in .env).

Metrics:
  - Faithfulness            (LLM-judged: does the answer follow the evidence?)
  - Answer relevancy        (LLM-judged 1-5)
  - Context precision       (are the top retrieved docs actually relevant?)
  - Context recall          (does the evidence contain what the answer needs?)
  - Unsupported claim rate  (numeric claims not traceable to the context)

If no API key is configured, the script exits with a clear message instead of
pretending to run. Judgments are produced by the Groq model itself (LLM-as-a-judge);
this is an *approximation* of RAGAS-style metrics without the extra dependency,
and results should be read with that limitation in mind.

Usage:
    python -m evaluation.run_generation_eval [--cases 10] [--out evaluation/results_generation.json]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional

from rag.config import get_config
from rag.generation.groq_client import GroqClient
from rag.generation.fallback import FallbackResponder
from rag.pipeline import RAGPipeline

DATASET_PATH = Path(__file__).parent / "dataset.json"

JUDGE_PROMPT = """You are an evaluator for a retrieval-augmented tourism Q&A system.

EVIDENCE:
{context}

QUESTION: {question}

ANSWER:
{answer}

Score strictly with JSON only:
{{"faithfulness": <1-5, 5 = every claim in the answer is supported by the evidence, 1 = contradicts or invents facts>,
"relevancy": <1-5, 5 = directly answers the question>,
"context_precision": <1-5, 5 = the evidence provided is on-topic for the question>,
"context_recall": <1-5, 5 = the evidence contains everything needed to answer>,
"unsupported_numeric_claims": [<list any numbers in the ANSWER that do not appear in the EVIDENCE>]}}"""


def judge_case(client: GroqClient, case: Dict, answer: str, context_text: str) -> Optional[Dict]:
    prompt = JUDGE_PROMPT.format(context=context_text, question=case["query"], answer=answer)
    try:
        raw = client.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_output_tokens=300,
        )
    except Exception as exc:
        return {"judge_error": str(exc)}
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"judge_error": "unparseable judge output"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"judge_error": "invalid JSON from judge"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=10, help="Max cases to judge")
    parser.add_argument("--out", default=str(DATASET_PATH.parent / "results_generation.json"))
    args = parser.parse_args(argv)

    config = get_config()
    if not config.has_groq_key:
        print(
            "GROQ_API_KEY is not configured.\n"
            "Generation evaluation cannot run without a real key.\n"
            "Add your key to .env (see .env.example) and re-run this script."
        )
        return 1

    client = GroqClient(config)
    responder = FallbackResponder()
    pipeline = RAGPipeline()

    cases = load_cases(DATASET_PATH)
    results: List[Dict] = []
    for case in cases[: args.cases]:
        response = pipeline.answer(case["query"])
        context_text = response.context.text if response.context else ""
        if response.generated_by != "groq":
            results.append(
                {
                    "id": case["id"],
                    "error": f"generation unavailable (generated_by={response.generated_by})",
                }
            )
            continue
        judge = judge_case(client, case, response.answer, context_text)
        guardrails = responder.validate_answer(response.answer, response.context)
        unsupported = judge.get("unsupported_numeric_claims", []) if isinstance(judge, dict) else []
        results.append(
            {
                "id": case["id"],
                "query": case["query"],
                "language": response.language,
                "faithfulness": judge.get("faithfulness"),
                "relevancy": judge.get("relevancy"),
                "context_precision": judge.get("context_precision"),
                "context_recall": judge.get("context_recall"),
                "unsupported_numeric_claims": unsupported,
                "guardrail_warnings": guardrails,
                "judge_error": judge.get("judge_error") if isinstance(judge, dict) else None,
                "answer_preview": response.answer[:300],
            }
        )
        time.sleep(0.5)  # be gentle with free-tier rate limits

    scored = [r for r in results if r.get("faithfulness") is not None]
    summary = {
        "num_cases": len(cases[: args.cases]),
        "num_scored": len(scored),
        "num_failed": len(results) - len(scored),
        "judge_model": config.groq_model,
        "note": "LLM-as-a-judge approximation of RAGAS metrics; scores are model opinions, not measurements.",
    }
    for metric in ("faithfulness", "relevancy", "context_precision", "context_recall"):
        values = [r[metric] for r in scored if isinstance(r.get(metric), (int, float))]
        summary[f"avg_{metric}"] = round(mean(values), 2) if values else None
    total_unsupported = sum(len(r.get("unsupported_numeric_claims", [])) for r in scored)
    summary["unsupported_claim_rate"] = (
        round(total_unsupported / len(scored), 3) if scored else None
    )

    out = {"summary": summary, "per_case": results}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nPer-case details written to {args.out}")
    return 0


def load_cases(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["cases"]


if __name__ == "__main__":
    raise SystemExit(main())
