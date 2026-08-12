"""
rerank_with_finetuned_crossencoder.py

Reranks the results already reranked by Nemotron using the fine-tuned ModernBERT CrossEncoder.

Input:
    - models/modernbert-large-retrix-nemotron-distilled/final  (fine-tuned model)
    - reranked_results_nemotron_topk400.json  {query_index: [doc_id, ...]}
    - collection_data.json                   [{pubkey, title, abstract, ...}]
    - en_train.json                          [{index, text, pubkey}]  ← for the queries

Output:
    - reranked_results_modernbert_ft_topkN.json   {query_index: [doc_id, ...]}
        (same format of the Nemotron run, drop-in in the pipeline)

Pipeline:
    BM25 → Nemotron (top 400) → fine-tuned ModernBERT (top N) → Evaluator.java

Usage:
    python rerank_with_finetuned_crossencoder.py \
        --model_dir    models/modernbert-large-retrix-nemotron-distilled/final \
        --nemotron_run reranked_results_nemotron_topk400.json \
        --collection   collection_data.json \
        --queries      en_train.json \
        --output       reranked_results_modernbert_ft_top100.json \
        --rerank_depth 100 \
        --batch_size   64
"""

import json
import emoji
import logging
import re
import gc
import argparse
from pathlib import Path

import torch
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_tweet(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Loading file JSON ─────────────────────────────────────────────────────
def load_collection(path: str, max_chars: int = 2000) -> dict[str, str]:
    """
    Loads collection_data.json.
    Returns {str(pubkey): "title. abstract"}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = {}
    for item in data:
        key      = str(item["pubkey"])
        title    = item.get("title", "")    or ""
        abstract = item.get("abstract", "") or ""
        text     = f"{title}. {abstract}".strip(". ")[:max_chars]
        if text:
            corpus[key] = text

    log.info(f"Corpus loaded: {len(corpus):,} documents  ← {path}")
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """
    Loads en_train.json (or any file with {index, text, pubkey}).
    Returns {str(index): query_text_cleaned}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    queries = {}
    for item in data:
        qid          = str(item["index"])
        queries[qid] = clean_tweet(item["text"])

    log.info(f"Queries loaded: {len(queries):,}  ← {path}")
    return queries


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    """
    Loads reranked_results_nemotron_topk400.json.
    Format: {query_index: [doc_id_rank1, doc_id_rank2, ...]}
    Keys and values are normalized to string.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    total_docs = sum(len(v) for v in run.values())
    log.info(f"Nemotron run loaded: {len(run):,} queries, {total_docs:,} total documents  ← {path}")
    return run


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_query(
        model:      CrossEncoder,
        query:      str,
        docs:       list[str],
        doc_ids:    list[str],
        batch_size: int,
        device:     str,
) -> list[tuple[str, float]]:
    """
    Calculates scores for all documents of a query.
    Returns an UNORDERED list of (doc_id, score).
    Handles OOM by reducing the batch_size by half.
    """
    pairs  = [[query, doc] for doc in docs]
    scores = _predict_with_oom_fallback(model, pairs, batch_size)
    return list(zip(doc_ids, scores))


def _predict_with_oom_fallback(
        model:      CrossEncoder,
        pairs:      list[list[str]],
        batch_size: int,
) -> list[float]:
    """predict() with fallback on OOM: halves the batch until it succeeds."""
    current_bs = batch_size
    while current_bs >= 1:
        try:
            scores = model.predict(
                pairs,
                batch_size=current_bs,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # scores can be a 1D array or a scalar
            if isinstance(scores, (float, int)):
                scores = [float(scores)]
            else:
                scores = scores.tolist()
            return scores

        except torch.cuda.OutOfMemoryError:
            log.warning(f"OOM with batch_size={current_bs}. Trying with {current_bs // 2}...")
            torch.cuda.empty_cache()
            gc.collect()
            current_bs //= 2

    # Last resort
    log.error("OOM even with batch_size=1. Returning score 0.0 for everything.")
    return [0.0] * len(pairs)


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check(results: dict, nemotron_run: dict, rerank_depth: int):
    if not results:
        log.error("SANITY CHECK FAILED: output empty!")
        return

    n_out   = len(results)
    n_in    = len(nemotron_run)
    avg_len = sum(len(v) for v in results.values()) / n_out
    coverage = n_out / max(1, n_in)

    log.info("── Sanity check ──────────────────────────────")
    log.info(f"  Queries in input (Nemotron): {n_in:,}")
    log.info(f"  Queries in output:           {n_out:,}  ({coverage:.1%} coverage)")
    log.info(f"  Avg doc per query:           {avg_len:.1f}  (expected: ≤{rerank_depth})")

    # Sample a query for manual inspection
    sample_qid    = next(iter(results))
    sample_top5   = results[sample_qid][:5]
    log.info(f"  Sample qid={sample_qid} top-5:")
    for rank, doc_id in enumerate(sample_top5, 1):
        log.info(f"    rank {rank}: doc_id={doc_id}")
    log.info("──────────────────────────────────────────────")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Reranking with fine-tuned ModernBERT CrossEncoder on Nemotron results"
    )
    parser.add_argument("--model_dir",    required=True,
                        help="Path to the fine-tuned model (e.g. models/.../final)")
    parser.add_argument("--nemotron_run", required=True,
                        help="reranked_results_nemotron_topk400.json")
    parser.add_argument("--collection",  required=True,
                        help="collection_data.json")
    parser.add_argument("--queries",     required=True,
                        help="en_train.json (cantains query's index + text)")
    parser.add_argument("--output",      required=True,
                        help="Path output JSON, es. reranked_results_modernbert_ft_top100.json")
    parser.add_argument("--rerank_depth", type=int, default=100,
                        help="How many docs from the Nemotron run to consider (default: 100)")
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="Batch size per CrossEncoder.predict (default: 64)")
    parser.add_argument("--max_doc_chars", type=int, default=4000,
                        help="Max character per document (default: 2000)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # ── 1. Load model ──────────────────────────────────────────────────
    log.info(f"Loading CrossEncoder from {args.model_dir} ...")
    model = CrossEncoder(
        args.model_dir,
        device=device,
        automodel_args={"torch_dtype": torch.bfloat16},
    )
    log.info(f"  max_length={model.max_length}, num_labels={model.num_labels}")

    # ── 2. Load data ─────────────────────────────────────────────────────
    corpus       = load_collection(args.collection, max_chars=args.max_doc_chars)
    queries      = load_queries(args.queries)
    nemotron_run = load_nemotron_run(args.nemotron_run)

    # ── 3. Reranking ───────────────────────────────────────────────────────
    results      = {}
    n_missing_q  = 0
    n_empty_docs = 0

    for qid, nemotron_docs in tqdm(nemotron_run.items(), desc="Reranking"):
        # Missing queries in the file
        if qid not in queries:
            n_missing_q += 1
            continue

        query = queries[qid]

        # Take the first rerank_depth documents from the Nemotron run
        candidate_ids = nemotron_docs[:args.rerank_depth]

        # Filter documents in the corpus
        valid_ids  = [d for d in candidate_ids if d in corpus]
        valid_docs = [corpus[d] for d in valid_ids]

        if not valid_ids:
            n_empty_docs += 1
            continue

        # Scoring
        scored = score_query(
            model=model,
            query=query,
            docs=valid_docs,
            doc_ids=valid_ids,
            batch_size=args.batch_size,
            device=device,
        )

        # Sort by decreasing score
        ranked    = sorted(scored, key=lambda x: x[1], reverse=True)
        results[qid] = [doc_id for doc_id, _ in ranked]

    log.info(f"Missing queries in the file: {n_missing_q}")
    log.info(f"Queries without document in the corpus: {n_empty_docs}")

    # ── 4. Sanity check ────────────────────────────────────────────────────
    sanity_check(results, nemotron_run, args.rerank_depth)

    # ── 5. Storing ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"✓ Output saved: {out_path}  ({len(results):,} queries)")


if __name__ == "__main__":
    main()