import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_BGE_M3_MAX_LENGTH = 1024
DEFAULT_CPU_THREADS = 6
DEFAULT_RETRIEVAL_MODE = "auto"
DEFAULT_CANDIDATE_MULTIPLIER = 2
DEFAULT_HYBRID_MAX_LENGTH = 256
DEFAULT_DENSE_WEIGHT = 0.4
DEFAULT_SPARSE_WEIGHT = 0.2
DEFAULT_COLBERT_WEIGHT = 0.4
DEFAULT_COLBERT_SCORE_BATCH_SIZE = 512
DEFAULT_CORPUS_COLBERT_DTYPE = "float16"
DEFAULT_LOG_EVERY_QUERIES = 100

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]
RETRIEVAL_SCRIPT = SCRIPT_PATH.parents[1] / "py" / "BiEncoderBGE_m3_hybrid.py"

DEFAULT_QUERIES = CODE_ROOT / "data" / "finetune" / "repo" / "test_queries.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "finetune_results.json"
DEFAULT_METRICS_OUTPUT = REPO_ROOT / "results" / "finetune_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a BGE-M3 checkpoint with the repository retrieval pipeline "
            "and compute the held-out retrieval metrics."
        )
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--retrieval-mode",
        choices=("auto", "dense", "hybrid", "multivector"),
        default=DEFAULT_RETRIEVAL_MODE,
    )
    parser.add_argument("--query-field", default="original")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--query-prefix", default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--candidate-multiplier", type=int, default=DEFAULT_CANDIDATE_MULTIPLIER)
    parser.add_argument("--hybrid-max-length", type=int, default=DEFAULT_HYBRID_MAX_LENGTH)
    parser.add_argument(
        "--colbert-score-batch-size",
        type=int,
        default=DEFAULT_COLBERT_SCORE_BATCH_SIZE,
    )
    parser.add_argument("--dense-weight", type=float, default=DEFAULT_DENSE_WEIGHT)
    parser.add_argument("--sparse-weight", type=float, default=DEFAULT_SPARSE_WEIGHT)
    parser.add_argument("--colbert-weight", type=float, default=DEFAULT_COLBERT_WEIGHT)
    parser.add_argument("--precompute-corpus-hybrid", action="store_true")
    parser.add_argument(
        "--corpus-colbert-dtype",
        choices=("float16", "float32"),
        default=DEFAULT_CORPUS_COLBERT_DTYPE,
    )
    parser.add_argument("--log-every-queries", type=int, default=DEFAULT_LOG_EVERY_QUERIES)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_local_model_config(model_name_or_path: str) -> dict[str, Any] | None:
    model_path = Path(model_name_or_path).expanduser()
    if not model_path.is_dir():
        return None

    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None

    try:
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    return loaded if isinstance(loaded, dict) else None


def is_probably_bge_m3_model(model_name_or_path: str) -> bool:
    normalized = str(model_name_or_path).strip().lower()
    if "bge-m3" in normalized or "bge_m3" in normalized:
        return True

    config = load_local_model_config(model_name_or_path)
    if config is None:
        return False

    config_hints: list[str] = []
    name_or_path = config.get("_name_or_path")
    if isinstance(name_or_path, str):
        config_hints.append(name_or_path.lower())

    model_type = config.get("model_type")
    if isinstance(model_type, str):
        config_hints.append(model_type.lower())

    architectures = config.get("architectures")
    if isinstance(architectures, list):
        config_hints.extend(
            architecture.lower()
            for architecture in architectures
            if isinstance(architecture, str)
        )

    return any("bge-m3" in hint or "bge_m3" in hint for hint in config_hints)


def resolve_retrieval_mode(model_name: str, requested_mode: str) -> str:
    if requested_mode == "multivector":
        return "hybrid"
    if requested_mode != "auto":
        return requested_mode
    if is_probably_bge_m3_model(model_name):
        return "hybrid"
    return "dense"


def pick_query_text(item: dict[str, Any], query_field: str) -> str:
    if query_field != "auto":
        value = item.get(query_field, "")
        return value.strip() if isinstance(value, str) else ""

    for field_name in ("expanded", "original", "text", "query"):
        value = item.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def load_queries(
    path: Path,
    query_field: str,
    max_queries: int | None,
) -> tuple[list[str], list[str]]:
    raw_queries = load_json(path)
    if not isinstance(raw_queries, list):
        raise ValueError("Query JSON must be a list.")

    query_ids: list[str] = []
    gold_pubkeys: list[str] = []

    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        qid = item.get("index", item.get("qid", item.get("id")))
        gold_pubkey = item.get("pubkey")
        if qid is None or gold_pubkey is None:
            continue
        if not pick_query_text(item, query_field):
            continue
        query_ids.append(str(qid))
        gold_pubkeys.append(str(gold_pubkey))
        if max_queries is not None and len(query_ids) >= max_queries:
            break

    if not query_ids:
        raise ValueError("No valid queries were loaded.")

    return query_ids, gold_pubkeys


def run_retrieval(args: argparse.Namespace) -> None:
    if not RETRIEVAL_SCRIPT.is_file():
        raise FileNotFoundError(f"Retrieval script not found: {RETRIEVAL_SCRIPT}")

    command = [
        sys.executable,
        str(RETRIEVAL_SCRIPT),
        "--queries",
        str(args.queries),
        "--corpus",
        str(args.corpus),
        "--output",
        str(args.output),
        "--model",
        args.model,
        "--retrieval-mode",
        args.retrieval_mode,
        "--query-field",
        args.query_field,
        "--top-k",
        str(args.top_k),
        "--batch-size",
        str(args.batch_size),
        "--candidate-multiplier",
        str(args.candidate_multiplier),
        "--multivector-max-length",
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
        "--cpu-threads",
        str(args.cpu_threads),
    ]

    if args.max_length is not None:
        command.extend(["--max-length", str(args.max_length)])
    if args.query_prefix is not None:
        command.extend(["--query-prefix", args.query_prefix])
    if args.max_queries is not None:
        command.extend(["--max-queries", str(args.max_queries)])
    if args.precompute_corpus_hybrid:
        command.append("--precompute-corpus-hybrid")

    print("Running retrieval command:")
    print(" ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


def precision_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    if not ranked or k <= 0:
        return 0.0
    limit = min(k, len(ranked))
    hits = sum(1 for idx in range(limit) if ranked[idx] in gold)
    return hits / float(k)


def dcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    total = 0.0
    for idx, docid in enumerate(ranked[:k], start=1):
        if docid in gold:
            total += 1.0 / math.log2(idx + 1)
    return total


def ndcg_at_k(ranked: list[str], gold: set[str], rel_count: int, k: int) -> float:
    if rel_count <= 0:
        return 0.0
    ideal = sum(1.0 / math.log2(idx + 1) for idx in range(1, min(rel_count, k) + 1))
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked, gold, k) / ideal


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    values_sorted = sorted(values)
    mid = len(values_sorted) // 2
    if len(values_sorted) % 2 == 1:
        return values_sorted[mid]
    return (values_sorted[mid - 1] + values_sorted[mid]) / 2.0


def evaluate_results(
    query_ids: list[str],
    gold_pubkeys: list[str],
    results: dict[str, list[str]],
) -> dict[str, Any]:
    total = len(query_ids)
    if total == 0:
        raise ValueError("No queries to evaluate.")

    hit_at_1 = hit_at_5 = hit_at_10 = hit_at_100 = 0.0
    prec_at_1 = prec_at_5 = prec_at_10 = 0.0
    mrr_total = map_total = 0.0
    ndcg_5 = ndcg_10 = ndcg_100 = 0.0
    per_query_mrr: list[float] = []
    per_query_ndcg10: list[float] = []

    for qid, gold_pubkey in zip(query_ids, gold_pubkeys):
        gold_set = {gold_pubkey}
        ranked = [str(docid) for docid in results.get(qid, [])]
        rank = -1
        for idx, docid in enumerate(ranked):
            if docid in gold_set:
                rank = idx + 1
                break

        if rank == 1:
            hit_at_1 += 1.0
        if 0 < rank <= 5:
            hit_at_5 += 1.0
        if 0 < rank <= 10:
            hit_at_10 += 1.0
        if 0 < rank <= 100:
            hit_at_100 += 1.0

        prec_at_1 += precision_at_k(ranked, gold_set, 1)
        prec_at_5 += precision_at_k(ranked, gold_set, 5)
        prec_at_10 += precision_at_k(ranked, gold_set, 10)

        q_mrr = 1.0 / rank if 0 < rank <= 5 else 0.0
        per_query_mrr.append(q_mrr)
        mrr_total += q_mrr

        avg_precision = 1.0 / rank if rank > 0 else 0.0
        map_total += avg_precision

        q_ndcg_5 = ndcg_at_k(ranked, gold_set, 1, 5)
        q_ndcg_10 = ndcg_at_k(ranked, gold_set, 1, 10)
        q_ndcg_100 = ndcg_at_k(ranked, gold_set, 1, 100)
        ndcg_5 += q_ndcg_5
        ndcg_10 += q_ndcg_10
        ndcg_100 += q_ndcg_100
        per_query_ndcg10.append(q_ndcg_10)

    recall_1 = hit_at_1 / total
    precision_1 = prec_at_1 / total
    f1_at_1 = (
        2.0 * recall_1 * precision_1 / (recall_1 + precision_1)
        if (recall_1 + precision_1) > 0
        else 0.0
    )

    return {
        "queries": total,
        "recall@1": hit_at_1 / total,
        "recall@5": hit_at_5 / total,
        "recall@10": hit_at_10 / total,
        "recall@100": hit_at_100 / total,
        "precision@1": prec_at_1 / total,
        "precision@5": prec_at_5 / total,
        "precision@10": prec_at_10 / total,
        "f1@1": f1_at_1,
        "mrr@5": mrr_total / total,
        "map": map_total / total,
        "ndcg@5": ndcg_5 / total,
        "ndcg@10": ndcg_10 / total,
        "ndcg@100": ndcg_100 / total,
        "per_query_stats": {
            "mrr@5": {
                "min": min(per_query_mrr) if per_query_mrr else 0.0,
                "median": median(per_query_mrr),
                "max": max(per_query_mrr) if per_query_mrr else 0.0,
            },
            "ndcg@10": {
                "min": min(per_query_ndcg10) if per_query_ndcg10 else 0.0,
                "median": median(per_query_ndcg10),
                "max": max(per_query_ndcg10) if per_query_ndcg10 else 0.0,
            },
        },
    }


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be greater than 0.")
    if args.max_length is not None and args.max_length <= 0:
        raise ValueError("--max-length must be greater than 0 when provided.")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be greater than 0 when provided.")
    if args.candidate_multiplier <= 0:
        raise ValueError("--candidate-multiplier must be greater than 0.")
    if args.hybrid_max_length <= 0:
        raise ValueError("--hybrid-max-length must be greater than 0.")
    if args.colbert_score_batch_size <= 0:
        raise ValueError("--colbert-score-batch-size must be greater than 0.")
    if args.log_every_queries <= 0:
        raise ValueError("--log-every-queries must be greater than 0.")

    effective_retrieval_mode = resolve_retrieval_mode(args.model, args.retrieval_mode)
    effective_max_length = (
        args.max_length
        if args.max_length is not None
        else (DEFAULT_BGE_M3_MAX_LENGTH if is_probably_bge_m3_model(args.model) else None)
    )
    query_ids, gold_pubkeys = load_queries(
        path=args.queries,
        query_field=args.query_field,
        max_queries=args.max_queries,
    )

    print(f"Loaded {len(query_ids)} evaluation queries from {args.queries}")
    print(f"Evaluating model: {args.model}")
    print(f"Requested retrieval mode: {args.retrieval_mode}")
    print(f"Effective retrieval mode: {effective_retrieval_mode}")
    print(f"Query field: {args.query_field}")

    run_retrieval(args)

    raw_results = load_json(args.output)
    if not isinstance(raw_results, dict):
        raise ValueError("Retrieval output must be a JSON object mapping qid to ranked docids.")
    results = {
        str(qid): [str(docid) for docid in ranked_docs]
        for qid, ranked_docs in raw_results.items()
        if isinstance(ranked_docs, list)
    }
    metrics = evaluate_results(query_ids, gold_pubkeys, results)

    config = {
        "model": args.model,
        "retrieval_mode_requested": args.retrieval_mode,
        "retrieval_mode_effective": effective_retrieval_mode,
        "query_field": args.query_field,
        "top_k": args.top_k,
        "batch_size": args.batch_size,
        "max_length": effective_max_length,
        "max_queries": args.max_queries,
        "candidate_multiplier": args.candidate_multiplier,
        "hybrid_max_length": args.hybrid_max_length,
        "colbert_score_batch_size": args.colbert_score_batch_size,
        "dense_weight": args.dense_weight,
        "sparse_weight": args.sparse_weight,
        "colbert_weight": args.colbert_weight,
        "precompute_corpus_hybrid": args.precompute_corpus_hybrid,
        "corpus_colbert_dtype": args.corpus_colbert_dtype,
        "log_every_queries": args.log_every_queries,
    }
    save_json(args.metrics_output, {"config": config, "metrics": metrics})

    print(f"Saved retrieval results to {args.output}")
    print(f"Saved metrics to {args.metrics_output}")
    print(f"Recall@1:   {metrics['recall@1']:.4f}")
    print(f"Recall@5:   {metrics['recall@5']:.4f}")
    print(f"Recall@10:  {metrics['recall@10']:.4f}")
    print(f"Recall@100: {metrics['recall@100']:.4f}")
    print(f"MRR@5:      {metrics['mrr@5']:.4f}")
    print(f"MAP:        {metrics['map']:.4f}")
    print(f"nDCG@10:    {metrics['ndcg@10']:.4f}")


if __name__ == "__main__":
    main()
