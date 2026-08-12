import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASELINE_MODEL = "BAAI/bge-m3"

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]
EVALUATE_SCRIPT = SCRIPT_PATH.parent / "evaluate_bge_m3_checkpoint.py"

DEFAULT_QUERIES = CODE_ROOT / "data" / "finetune" / "repo" / "dev_queries.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "finetune_comparison"

COMPARISON_METRICS = (
    "recall@1",
    "recall@5",
    "recall@10",
    "recall@100",
    "precision@1",
    "precision@5",
    "precision@10",
    "f1@1",
    "mrr@5",
    "map",
    "ndcg@5",
    "ndcg@10",
    "ndcg@100",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the baseline BGE-M3 model and a fine-tuned checkpoint on the same split, "
            "then write a metric-by-metric comparison."
        )
    )
    parser.add_argument("--baseline-model", default=DEFAULT_BASELINE_MODEL)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="finetuned")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "dense", "hybrid", "multivector"),
        default="auto",
    )
    parser.add_argument("--query-field", default="expanded")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--candidate-multiplier", type=int, default=2)
    parser.add_argument("--hybrid-max-length", type=int, default=256)
    parser.add_argument("--colbert-score-batch-size", type=int, default=512)
    parser.add_argument("--dense-weight", type=float, default=0.4)
    parser.add_argument("--sparse-weight", type=float, default=0.2)
    parser.add_argument("--colbert-weight", type=float, default=0.4)
    parser.add_argument("--precompute-corpus-hybrid", action="store_true")
    parser.add_argument(
        "--corpus-colbert-dtype",
        choices=("float16", "float32"),
        default="float16",
    )
    parser.add_argument("--log-every-queries", type=int, default=100)
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify_label(label: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip()).strip("_").lower()
    return normalized or "model"


def build_eval_command(
    args: argparse.Namespace,
    model: str,
    results_output: Path,
    metrics_output: Path,
) -> list[str]:
    if not EVALUATE_SCRIPT.is_file():
        raise FileNotFoundError(f"Evaluation script not found: {EVALUATE_SCRIPT}")

    command = [
        sys.executable,
        str(EVALUATE_SCRIPT),
        "--model",
        model,
        "--queries",
        str(args.queries),
        "--corpus",
        str(args.corpus),
        "--output",
        str(results_output),
        "--metrics-output",
        str(metrics_output),
        "--retrieval-mode",
        args.retrieval_mode,
        "--query-field",
        args.query_field,
        "--top-k",
        str(args.top_k),
        "--batch-size",
        str(args.batch_size),
        "--cpu-threads",
        str(args.cpu_threads),
        "--candidate-multiplier",
        str(args.candidate_multiplier),
        "--hybrid-max-length",
        str(args.hybrid_max_length),
        "--colbert-score-batch-size",
        str(args.colbert_score_batch_size),
        "--dense-weight",
        str(args.dense_weight),
        "--sparse-weight",
        str(args.sparse_weight),
        "--colbert-weight",
        str(args.colbert_weight),
        "--corpus-colbert-dtype",
        args.corpus_colbert_dtype,
        "--log-every-queries",
        str(args.log_every_queries),
    ]

    if args.max_length is not None:
        command.extend(["--max-length", str(args.max_length)])
    if args.query_prefix is not None:
        command.extend(["--query-prefix", args.query_prefix])
    if args.max_queries is not None:
        command.extend(["--max-queries", str(args.max_queries)])
    if args.precompute_corpus_hybrid:
        command.append("--precompute-corpus-hybrid")

    return command


def run_eval(label: str, command: list[str]) -> None:
    print(f"Running evaluation for {label}:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def extract_metric_subset(payload: dict[str, Any]) -> dict[str, float]:
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError("Metrics payload is missing the top-level 'metrics' object.")
    return {
        metric_name: float(metrics[metric_name])
        for metric_name in COMPARISON_METRICS
        if metric_name in metrics
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_slug = slugify_label(args.baseline_label)
    candidate_slug = slugify_label(args.candidate_label)

    baseline_results = output_dir / f"{baseline_slug}_results.json"
    baseline_metrics = output_dir / f"{baseline_slug}_metrics.json"
    candidate_results = output_dir / f"{candidate_slug}_results.json"
    candidate_metrics = output_dir / f"{candidate_slug}_metrics.json"
    comparison_output = output_dir / "comparison.json"

    run_eval(
        args.baseline_label,
        build_eval_command(
            args=args,
            model=args.baseline_model,
            results_output=baseline_results,
            metrics_output=baseline_metrics,
        ),
    )
    run_eval(
        args.candidate_label,
        build_eval_command(
            args=args,
            model=args.candidate_model,
            results_output=candidate_results,
            metrics_output=candidate_metrics,
        ),
    )

    baseline_payload = load_json(baseline_metrics)
    candidate_payload = load_json(candidate_metrics)
    baseline_metric_subset = extract_metric_subset(baseline_payload)
    candidate_metric_subset = extract_metric_subset(candidate_payload)

    delta = {
        metric_name: candidate_metric_subset[metric_name] - baseline_metric_subset[metric_name]
        for metric_name in COMPARISON_METRICS
        if metric_name in baseline_metric_subset and metric_name in candidate_metric_subset
    }

    summary = {
        "config": {
            "queries": str(args.queries),
            "corpus": str(args.corpus),
            "query_field": args.query_field,
            "retrieval_mode_requested": args.retrieval_mode,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "max_queries": args.max_queries,
            "candidate_multiplier": args.candidate_multiplier,
            "hybrid_max_length": args.hybrid_max_length,
            "colbert_score_batch_size": args.colbert_score_batch_size,
            "dense_weight": args.dense_weight,
            "sparse_weight": args.sparse_weight,
            "colbert_weight": args.colbert_weight,
            "precompute_corpus_hybrid": args.precompute_corpus_hybrid,
            "corpus_colbert_dtype": args.corpus_colbert_dtype,
        },
        "baseline": {
            "label": args.baseline_label,
            "model": args.baseline_model,
            "results_output": str(baseline_results),
            "metrics_output": str(baseline_metrics),
            "metrics": baseline_payload,
        },
        "candidate": {
            "label": args.candidate_label,
            "model": args.candidate_model,
            "results_output": str(candidate_results),
            "metrics_output": str(candidate_metrics),
            "metrics": candidate_payload,
        },
        "delta_candidate_minus_baseline": delta,
    }
    save_json(comparison_output, summary)

    print(f"Saved comparison to {comparison_output}")
    for metric_name in ("recall@10", "mrr@5", "map", "ndcg@10"):
        if metric_name in delta:
            print(
                f"{metric_name}: "
                f"{baseline_metric_subset[metric_name]:.4f} -> {candidate_metric_subset[metric_name]:.4f} "
                f"(delta {delta[metric_name]:+.4f})"
            )


if __name__ == "__main__":
    main()
