"""Run the small RAG evaluation against an isolated database."""

import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "evaluation"
sys.path.insert(0, str(ROOT))


def run_cases(store_directory):
    from app import ingest, retriever
    from app.config import LLM_MODEL, LLM_PROVIDER, TOP_K
    from app.service import ask

    ingest.CHROMA_PERSIST_DIR = store_directory
    retriever.CHROMA_PERSIST_DIR = store_directory
    retriever.reset_vectorstore_cache()
    ingest.run_ingestion(str(EVALUATION / "documents"))
    

    cases = json.loads(
        (EVALUATION / "questions.json").read_text(encoding="utf-8-sig")
    )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": LLM_PROVIDER,
        "model": LLM_MODEL,
        "top_k": TOP_K,
        "results": [],
    }
    output = EVALUATION / "results.json"

    for case in cases:
        started = time.perf_counter()
        row = dict(case)

        try:
            response = ask(case["question"], top_k=TOP_K)
            sources = response["sources"]
            retrieved = sorted({
                Path(source["source"]).name for source in sources
            })
            expected = set(case["expected_sources"])

            row.update({
                "answer": response["answer"],
                "sources": sources,
                "retrieved_sources": retrieved,
                "source_recall": (
                    len(expected.intersection(retrieved)) / len(expected)
                    if expected else None
                ),
                "error": None,
            })
        except Exception as exc:
            row.update({
                "answer": None,
                "sources": [],
                "retrieved_sources": [],
                "source_recall": None,
                "error": type(exc).__name__,
            })

        row["latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 1
        )
        row["manual_verdict"] = None
        report["results"].append(row)

        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        status = row["error"] or "recorded"
        print(f"{case['id']}: {status}", flush=True)

    print(f"Results saved to {output}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_cases(sys.argv[2])
    else:
        # The child process exits before Windows deletes the database.
        with tempfile.TemporaryDirectory() as folder:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()),
                 "--worker", folder],
                cwd=ROOT,
                check=True,
            )