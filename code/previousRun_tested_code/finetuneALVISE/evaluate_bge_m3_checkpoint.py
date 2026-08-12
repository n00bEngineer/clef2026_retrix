import argparse
import json
import math
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch


DEFAULT_MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 100
DEFAULT_BATCH_SIZE = 128
DEFAULT_BGE_M3_MAX_LENGTH = 1024
DEFAULT_CPU_THREADS = 6

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]

DEFAULT_QUERIES = CODE_ROOT / "data" / "finetune" / "repo" / "test_queries.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "finetune_dense_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a BGE-M3 checkpoint with dense retrieval on a query subset."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional path for the metrics JSON. If omitted, metrics are only printed to stdout.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query-field", default="original")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_BGE_M3_MAX_LENGTH)
    parser.add_argument("--cpu-threads", type=int, default=DEFAULT_CPU_THREADS)
    parser.add_argument("--query-prefix", default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_installed_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def patch_transformers_flash_attn_compat() -> None:
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as transformers_import_utils
    except ImportError:
        return

    if hasattr(transformers_utils, "is_flash_attn_greater_or_equal_2_10"):
        pass
    else:
        legacy_checker = getattr(transformers_utils, "is_flash_attn_greater_or_equal", None)
        if legacy_checker is not None:

            def _is_flash_attn_greater_or_equal_2_10() -> bool:
                try:
                    return bool(legacy_checker("2.1.0"))
                except Exception:
                    return False

            transformers_utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_greater_or_equal_2_10

    if hasattr(transformers_import_utils, "is_torch_fx_available"):
        return

    def _is_torch_fx_available() -> bool:
        try:
            return bool(transformers_import_utils.is_torch_available())
        except Exception:
            return False

    transformers_import_utils.is_torch_fx_available = _is_torch_fx_available


def build_bge_m3_model(model_name: str) -> Any:
    patch_transformers_flash_attn_compat()
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError as error:
        flagembedding_version = get_installed_version("FlagEmbedding")
        transformers_version = get_installed_version("transformers")
        if flagembedding_version is not None:
            version_bits = [f"FlagEmbedding=={flagembedding_version}"]
            if transformers_version is not None:
                version_bits.append(f"transformers=={transformers_version}")
            version_summary = ", ".join(version_bits)
            raise ImportError(
                "FlagEmbedding is installed but could not be imported. "
                f"Detected {version_summary}. "
                "Install compatible versions in the same environment."
            ) from error
        raise ImportError(
            "FlagEmbedding is required to evaluate a BGE-M3 checkpoint. "
            "Install it with: pip install -U FlagEmbedding"
        ) from error

    use_fp16 = torch.cuda.is_available()
    init_attempts = []
    if torch.cuda.is_available():
        init_attempts.append({"devices": "cuda:0", "use_fp16": use_fp16})
        init_attempts.append({"device": "cuda:0", "use_fp16": use_fp16})
    init_attempts.append({"devices": "cpu", "use_fp16": False})
    init_attempts.append({"device": "cpu", "use_fp16": False})

    last_type_error: TypeError | None = None
    for init_kwargs in init_attempts:
        try:
            return BGEM3FlagModel(model_name, **init_kwargs)
        except TypeError as error:
            last_type_error = error

    if last_type_error is not None:
        raise last_type_error

    return BGEM3FlagModel(model_name)


def configure_cpu_threads(cpu_threads: int | None) -> None:
    if cpu_threads is None:
        return
    if cpu_threads <= 0:
        raise ValueError("--cpu-threads must be greater than 0.")
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(cpu_threads)
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))
    except RuntimeError:
        pass


def infer_query_prefix(model_name: str, provided_prefix: str | None) -> str:
    if provided_prefix is not None:
        return provided_prefix
    if model_name.lower().startswith("baai/bge-m3"):
        return ""
    if model_name.lower().startswith("baai/bge"):
        return "Represent this sentence for searching relevant passages: "
    return ""


def pick_query_text(item: dict[str, Any], query_field: str) -> str:
    if query_field != "auto":
        value = item.get(query_field, "")
        return value.strip() if isinstance(value, str) else ""
    for field_name in ("original", "expanded", "text", "query"):
        value = item.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def load_queries(path: Path, query_field: str, query_prefix: str) -> tuple[list[str], list[str], list[str]]:
    raw_queries = load_json(path)
    if not isinstance(raw_queries, list):
        raise ValueError("Query JSON must be a list.")

    query_ids: list[str] = []
    query_texts: list[str] = []
    gold_pubkeys: list[str] = []

    for item in raw_queries:
        if not isinstance(item, dict):
            continue
        qid = item.get("index", item.get("qid", item.get("id")))
        gold_pubkey = item.get("pubkey")
        if qid is None or gold_pubkey is None:
            continue
        text = pick_query_text(item, query_field)
        if not text:
            continue
        query_ids.append(str(qid))
        query_texts.append(f"{query_prefix}{text}" if query_prefix else text)
        gold_pubkeys.append(str(gold_pubkey))

    if not query_ids:
        raise ValueError("No valid queries were loaded.")

    return query_ids, query_texts, gold_pubkeys


def build_document_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title", "")).strip(),
        str(item.get("abstract", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list.")

    pubkeys: list[str] = []
    texts: list[str] = []
    seen_pubkeys: set[str] = set()

    for item in raw_corpus:
        if not isinstance(item, dict):
            continue
        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            continue
        pubkey_str = str(pubkey)
        if pubkey_str in seen_pubkeys:
            continue
        text = build_document_text(item)
        if not text:
            continue
        seen_pubkeys.add(pubkey_str)
        pubkeys.append(pubkey_str)
        texts.append(text)

    if not pubkeys:
        raise ValueError("No valid corpus documents were loaded.")

    return pubkeys, texts


def is_cuda_oom(error: Exception) -> bool:
    if isinstance(error, torch.OutOfMemoryError):
        return True
    return "out of memory" in str(error).lower()


def encode_dense(
    model: Any,
    texts: list[str],
    batch_size: int,
    max_length: int,
    label: str,
) -> np.ndarray:
    current_batch_size = batch_size
    while True:
        try:
            dense_chunks: list[np.ndarray] = []
            total_items = len(texts)
            total_chunks = (total_items + current_batch_size - 1) // current_batch_size
            for chunk_idx, start in enumerate(range(0, total_items, current_batch_size), start=1):
                end = min(total_items, start + current_batch_size)
                print(
                    f"Encoding {label}: chunk {chunk_idx}/{total_chunks} "
                    f"({start + 1}-{end} of {total_items}), batch_size={current_batch_size}..."
                )
                output = model.encode(
                    texts[start:end],
                    batch_size=current_batch_size,
                    max_length=max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                dense = np.asarray(output["dense_vecs"], dtype=np.float32)
                faiss.normalize_L2(dense)
                dense_chunks.append(dense)
            return np.concatenate(dense_chunks, axis=0)
        except Exception as error:
            if not (torch.cuda.is_available() and is_cuda_oom(error) and current_batch_size > 1):
                raise
            next_batch_size = max(1, current_batch_size // 2)
            print(
                f"CUDA OOM while encoding {label} with batch_size={current_batch_size}. "
                f"Retrying with batch_size={next_batch_size}."
            )
            torch.cuda.empty_cache()
            current_batch_size = next_batch_size


def dense_search(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_pubkeys: list[str],
    top_k: int,
) -> dict[str, list[str]]:
    effective_top_k = min(top_k, len(corpus_pubkeys))
    index = faiss.IndexFlatIP(query_embeddings.shape[1])
    index.add(query_embeddings[:0])
    corpus_index = faiss.IndexFlatIP(query_embeddings.shape[1])
    raise RuntimeError("dense_search should not be called directly without a built index")


def build_index(corpus_embeddings: np.ndarray) -> Any:
    index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    index.add(corpus_embeddings)
    return index


def run_dense_search(
    index: Any,
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_pubkeys: list[str],
    top_k: int,
) -> dict[str, list[str]]:
    effective_top_k = min(top_k, len(corpus_pubkeys))
    _, indices = index.search(query_embeddings, effective_top_k)
    results: dict[str, list[str]] = {}
    for row, qid in enumerate(query_ids):
        results[qid] = [
            corpus_pubkeys[col]
            for col in indices[row]
            if 0 <= col < len(corpus_pubkeys)
        ]
    return results


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
        ranked = results.get(qid, [])
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


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than 0.")

    configure_cpu_threads(args.cpu_threads)
    query_prefix = infer_query_prefix(args.model, args.query_prefix)
    query_ids, query_texts, gold_pubkeys = load_queries(args.queries, args.query_field, query_prefix)
    corpus_pubkeys, corpus_texts = load_corpus(args.corpus)

    print(f"Loaded {len(query_ids)} queries from {args.queries}")
    print(f"Loaded {len(corpus_pubkeys)} documents from {args.corpus}")
    print(f"Evaluating model: {args.model}")
    print(f"Query field: {args.query_field}")

    model = build_bge_m3_model(args.model)
    corpus_embeddings = encode_dense(
        model=model,
        texts=corpus_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="corpus",
    )
    query_embeddings = encode_dense(
        model=model,
        texts=query_texts,
        batch_size=args.batch_size,
        max_length=args.max_length,
        label="queries",
    )

    print("Building FAISS index...")
    index = build_index(corpus_embeddings)
    results = run_dense_search(
        index=index,
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_pubkeys=corpus_pubkeys,
        top_k=args.top_k,
    )
    metrics = evaluate_results(query_ids, gold_pubkeys, results)

    config = {
        "model": args.model,
        "mode": "dense",
        "query_field": args.query_field,
        "top_k": args.top_k,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
    }
    save_json(args.output, results)

    print(f"Saved retrieval results to {args.output}")
    if args.metrics_output is not None:
        save_json(args.metrics_output, {"config": config, "metrics": metrics})
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
