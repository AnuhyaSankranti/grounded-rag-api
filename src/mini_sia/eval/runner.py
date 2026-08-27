import argparse
import asyncio
import json
import statistics
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from mini_sia.config import Settings
from mini_sia.eval.metrics import (
    answer_coverage,
    citation_validity,
    context_precision,
    reciprocal_rank,
    retrieval_recall,
)
from mini_sia.providers import (
    ExtractiveAnswerProvider,
    HashEmbeddingProvider,
    OpenAIAnswerProvider,
    OpenAIEmbeddingProvider,
)
from mini_sia.services import IngestionService, RagService
from mini_sia.store import SQLiteHybridStore


@dataclass(slots=True)
class CaseResult:
    id: str
    retrieval_recall: float
    reciprocal_rank: float
    context_precision: float
    answer_coverage: float
    citation_validity: float
    latency_ms: float
    answer: str

    @property
    def quality_score(self) -> float:
        return statistics.mean(
            [
                self.retrieval_recall,
                self.reciprocal_rank,
                self.context_precision,
                self.answer_coverage,
                self.citation_validity,
            ]
        )


async def run_eval(args: argparse.Namespace) -> dict:
    project_root = Path(__file__).resolve().parents[3]
    dataset_path = Path(args.dataset or project_root / "evals" / "golden.jsonl")
    docs_dir = Path(args.docs or project_root / "data" / "demo")
    cases = [json.loads(line) for line in dataset_path.read_text().splitlines() if line.strip()]

    with tempfile.TemporaryDirectory(prefix="mini-sia-eval-") as directory:
        provider = "local" if args.local else args.provider
        settings = Settings(
            app_env="test",
            database_path=Path(directory) / "eval.db",
            llm_provider="extractive" if provider == "local" else "openai",
            embedding_provider="hash" if provider == "local" else "openai",
        )
        store = SQLiteHybridStore(settings.database_path)
        store.initialize()
        embeddings = (
            HashEmbeddingProvider()
            if provider == "local"
            else OpenAIEmbeddingProvider(settings.embedding_model)
        )
        answer_provider = (
            ExtractiveAnswerProvider()
            if provider == "local"
            else OpenAIAnswerProvider(settings.chat_model, settings.max_answer_tokens)
        )
        ingestion = IngestionService(settings, store, embeddings)
        rag = RagService(settings, store, embeddings, answer_provider)

        for document in sorted(docs_dir.glob("*")):
            if document.suffix.lower() in {".txt", ".md", ".pdf"}:
                await ingestion.ingest(document.name, document.read_bytes())

        results: list[CaseResult] = []
        for case in cases:
            retrieved = await rag.retrieve(case["question"], top_k=case.get("top_k", 4))
            response = await rag.ask(case["question"], top_k=case.get("top_k", 4))
            expected_sources = set(case["expected_sources"])
            results.append(
                CaseResult(
                    id=case["id"],
                    retrieval_recall=retrieval_recall(retrieved, expected_sources),
                    reciprocal_rank=reciprocal_rank(retrieved, expected_sources),
                    context_precision=context_precision(retrieved, expected_sources),
                    answer_coverage=answer_coverage(
                        response.answer, case.get("expected_answer_contains", [])
                    ),
                    citation_validity=citation_validity(response),
                    latency_ms=response.latency_ms,
                    answer=response.answer,
                )
            )

    quality = statistics.mean(result.quality_score for result in results) if results else 0.0
    report = {
        "provider": provider,
        "dataset": str(dataset_path),
        "case_count": len(results),
        "quality_score": round(quality, 4),
        "mean_latency_ms": round(statistics.mean(r.latency_ms for r in results), 2),
        "p95_latency_ms": round(_percentile([r.latency_ms for r in results], 0.95), 2),
        "metrics": {
            "retrieval_recall": _mean(results, "retrieval_recall"),
            "reciprocal_rank": _mean(results, "reciprocal_rank"),
            "context_precision": _mean(results, "context_precision"),
            "answer_coverage": _mean(results, "answer_coverage"),
            "citation_validity": _mean(results, "citation_validity"),
        },
        "cases": [asdict(result) | {"quality_score": result.quality_score} for result in results],
    }
    return report


def _mean(results: list[CaseResult], field: str) -> float:
    return round(statistics.mean(getattr(result, field) for result in results), 4)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Mini SIA retrieval and answer evals")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--local", action="store_true", help="Use deterministic local providers")
    mode.add_argument("--provider", choices=["openai"], default="openai")
    parser.add_argument("--dataset", help="Path to JSONL golden dataset")
    parser.add_argument("--docs", help="Directory containing documents to index")
    parser.add_argument("--output", help="Write the full JSON report to this path")
    parser.add_argument("--fail-under", type=float, default=0.0, help="Minimum quality score")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = asyncio.run(run_eval(args))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    if report["quality_score"] < args.fail_under:
        raise SystemExit(
            f"Eval quality {report['quality_score']:.3f} is below threshold {args.fail_under:.3f}"
        )


if __name__ == "__main__":
    main()

