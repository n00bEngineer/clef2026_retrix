import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "BAAI/bge-base-en"
DEFAULT_TOP_K = 100
DEFAULT_BATCH_SIZE = 64

SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]

DEFAULT_QUERIES = CODE_ROOT / "data" / "expanded_queries_bge_large.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / "bi_encoder_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode queries and corpus with a bi-encoder and run dense retrieval."
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--query-field", default="auto")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional debug limit. If set, only the first N queries are processed.",
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="Optional prefix added to every query before encoding.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_execution_mode() -> tuple[str, list[str]]:
    """
    Automatically select execution mode based on visible CUDA devices:
    - 0 GPUs -> CPU
    - 1 GPU  -> single-GPU
    - 2+ GPUs -> multi-GPU
    """
    if not torch.cuda.is_available():
        return "cpu", []

    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        return "cuda", []

    devices = [f"cuda:{gpu_idx}" for gpu_idx in range(gpu_count)]
    return "multi-gpu", devices


def infer_query_prefix(model_name: str, provided_prefix: str | None) -> str:
    if provided_prefix is not None:
        return provided_prefix
    if model_name.lower().startswith("baai/bge"):
        return "Represent this sentence for searching relevant passages: "
    return ""


def pick_query_text(item: dict[str, Any], query_field: str) -> str:
    if query_field != "auto":
        value = item.get(query_field, "")
        return value.strip() if isinstance(value, str) else ""

    for field_name in ("embedding", "sparse", "colbert", "expanded", "original", "text", "query"):
        value = item.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def iter_query_records(raw_queries: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(raw_queries, list):
        for position, item in enumerate(raw_queries):
            if isinstance(item, dict):
                qid = str(item.get("index", item.get("qid", item.get("id", position))))
                yield qid, item
    elif isinstance(raw_queries, dict):
        for qid, item in raw_queries.items():
            if isinstance(item, dict):
                yield str(qid), item
            elif isinstance(item, str):
                yield str(qid), {"text": item}


def load_queries(
    path: Path,
    query_field: str,
    max_queries: int | None,
    query_prefix: str,
) -> tuple[list[str], list[str]]:
    raw_queries = load_json(path)

    query_ids: list[str] = []
    query_texts: list[str] = []
    skipped = 0

    for qid, item in iter_query_records(raw_queries):
        text = pick_query_text(item, query_field)
        if not text:
            skipped += 1
            continue

        query_ids.append(qid)
        query_texts.append(f"{query_prefix}{text}" if query_prefix else text)

        if max_queries is not None and len(query_ids) >= max_queries:
            break

    print(f"Loaded {len(query_ids)} queries from {path}")
    if skipped:
        print(f"Skipped {skipped} queries with empty text")

    if not query_ids:
        raise ValueError("No valid queries were loaded.")

    return query_ids, query_texts


def build_document_text(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("title", "")).strip(),
        str(item.get("abstract", "")).strip(),
    ]
    return " ".join(part for part in parts if part)


def load_corpus(path: Path) -> tuple[list[str], list[str]]:
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list of documents.")

    pubkeys: list[str] = []
    documents: list[str] = []
    seen_pubkeys: set[str] = set()
    skipped = 0

    for item in raw_corpus:
        if not isinstance(item, dict):
            skipped += 1
            continue

        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            skipped += 1
            continue

        pubkey_str = str(pubkey)
        if pubkey_str in seen_pubkeys:
            continue

        text = build_document_text(item)
        if not text:
            skipped += 1
            continue

        seen_pubkeys.add(pubkey_str)
        pubkeys.append(pubkey_str)
        documents.append(text)

    print(f"Loaded {len(pubkeys)} documents from {path}")
    if skipped:
        print(f"Skipped {skipped} corpus entries without usable text or id")

    if not pubkeys:
        raise ValueError("No valid corpus documents were loaded.")

    return pubkeys, documents


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
) -> np.ndarray:
    print(f"Encoding {label} ({len(texts)} items)...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def encode_texts_multi_gpu(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int,
    label: str,
    pool: Any,
) -> np.ndarray:
    print(f"Encoding {label} ({len(texts)} items) with multi-GPU...")
    embeddings = model.encode_multi_process(
        texts,
        pool=pool,
        batch_size=batch_size,
    )
    normalized = np.asarray(embeddings, dtype=np.float32)
    faiss.normalize_L2(normalized)
    return normalized


def build_faiss_index(corpus_embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = corpus_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(corpus_embeddings)
    return index


def dense_search(
    query_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_pubkeys: list[str],
    index: faiss.IndexFlatIP,
    top_k: int,
) -> dict[str, list[str]]:
    effective_top_k = min(top_k, len(corpus_pubkeys))
    _, indices = index.search(query_embeddings, effective_top_k)

    results: dict[str, list[str]] = {}
    for row, qid in enumerate(query_ids):
        ranked_pubkeys = [
            corpus_pubkeys[col]
            for col in indices[row]
            if 0 <= col < len(corpus_pubkeys)
        ]
        results[qid] = ranked_pubkeys

    print(f"Completed dense search for {len(query_ids)} queries")
    return results


def save_results(path: Path, results: dict[str, list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} query rankings to {path}")


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.max_queries is not None and args.max_queries <= 0:
        raise ValueError("--max-queries must be greater than 0 when provided.")

    query_prefix = infer_query_prefix(args.model, args.query_prefix)
    execution_mode, target_devices = resolve_execution_mode()
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Detected GPUs: {gpu_count}")
    if execution_mode == "cpu":
        print("Execution mode: CPU (no CUDA GPU detected)")
    elif execution_mode == "cuda":
        print("Execution mode: single-GPU")
        print(f"Primary GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Execution mode: multi-GPU ({', '.join(target_devices)})")

    print(f"Bi-encoder model: {args.model}")
    if query_prefix:
        print(f"Using query prefix: {query_prefix}")

    query_ids, query_texts = load_queries(
        path=args.queries,
        query_field=args.query_field,
        max_queries=args.max_queries,
        query_prefix=query_prefix,
    )
    corpus_pubkeys, corpus_texts = load_corpus(args.corpus)

    if execution_mode == "multi-gpu":
        model = SentenceTransformer(args.model)
        pool = model.start_multi_process_pool(target_devices=target_devices)
        try:
            corpus_embeddings = encode_texts_multi_gpu(
                model=model,
                texts=corpus_texts,
                batch_size=args.batch_size,
                label="corpus",
                pool=pool,
            )
            query_embeddings = encode_texts_multi_gpu(
                model=model,
                texts=query_texts,
                batch_size=args.batch_size,
                label="queries",
                pool=pool,
            )
        finally:
            model.stop_multi_process_pool(pool)
    else:
        model = SentenceTransformer(args.model, device=execution_mode)
        corpus_embeddings = encode_texts(
            model=model,
            texts=corpus_texts,
            batch_size=args.batch_size,
            label="corpus",
        )
        query_embeddings = encode_texts(
            model=model,
            texts=query_texts,
            batch_size=args.batch_size,
            label="queries",
        )

    print("Building FAISS index...")
    index = build_faiss_index(corpus_embeddings)

    results = dense_search(
        query_ids=query_ids,
        query_embeddings=query_embeddings,
        corpus_pubkeys=corpus_pubkeys,
        index=index,
        top_k=args.top_k,
    )
    save_results(args.output, results)


if __name__ == "__main__":
    main()
