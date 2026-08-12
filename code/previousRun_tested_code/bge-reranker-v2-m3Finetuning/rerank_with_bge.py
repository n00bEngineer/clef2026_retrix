"""
rerank_with_bge.py

Inference with BGE-reranker-v2-m3 fine-tuned using FlagReranker.

IMPORTANT: bge-reranker-v2-m3 does NOT use query prefixes.
Queries are only cleaned (URLs, hashtags, emojis removed).
This is different from BGE bi-encoders which use prefixes like
"Represent this sentence...".

Use FlagReranker (FlagEmbedding) instead of CrossEncoder (sentence-transformers)
to be coherent with the trainig framework.

Usage:
    python rerank_with_bge.py \
        --model_dir    models/bge-reranker-v2-m3-retrix \
        --nemotron_run reranked_results_nemotron_topk400.json \
        --collection   collection_data.json \
        --queries      en_train.json \
        --output       reranked_results_bge_ft_top100.json \
        --rerank_depth 100 \
        --batch_size   256
"""

import json
import logging
import re
import gc
import argparse
import unicodedata
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Preprocessing ──────────────────────────────────────────────────────────────
# IDENTICAL to prepare_data_unified_v3.py — no prefix, only cleanup.

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"&amp;",  "&",  text)
    text = re.sub(r"&lt;",   "<",  text)
    text = re.sub(r"&gt;",   ">",  text)
    text = re.sub(r"&quot;", '"',  text)
    text = re.sub(r"&#39;",  "'",  text)
    text = "".join(
        c for c in text
        if unicodedata.category(c) not in ("So", "Cs", "Sk")
    )
    return re.sub(r"\s+", " ", text).strip()


def doc_text(item: dict, max_chars: int) -> str:
    t = (item.get("title")    or "").strip()
    a = (item.get("abstract") or "").strip()
    return f"{t}. {a}".strip(". ")[:max_chars]


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_collection(path: str, max_chars: int) -> dict[str, str]:
    """
    Autodetect 2026 format (pubkey) vs 2025 format (cord_uid).
    Use 'in' for the check — avoids that pubkey=0 (falsy) goes to cord_uid.
    """
    data   = json.load(open(path, encoding="utf-8"))
    corpus = {}
    for item in data:
        if "pubkey" in item and item["pubkey"] is not None:
            key = str(item["pubkey"])
        elif "cord_uid" in item and item["cord_uid"] is not None:
            key = str(item["cord_uid"])
        else:
            continue
        text = doc_text(item, max_chars)
        if text:
            corpus[key] = text
    log.info(f"Corpus: {len(corpus):,} documents  ← {path}")
    return corpus


def load_queries(path: str) -> dict[str, str]:
    """Loads [{index, text, pubkey}]. Only cleanup, no prefix."""
    data    = json.load(open(path, encoding="utf-8"))
    queries = {}
    for item in data:
        qid          = str(item["index"])
        queries[qid] = clean_text(item["text"])
    log.info(f"Queries: {len(queries):,}  ← {path}")
    return queries


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    raw = json.load(open(path, encoding="utf-8"))
    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    log.info(f"Nemotron run: {len(run):,} queries  ← {path}")
    return run


# ── Scoring ────────────────────────────────────────────────────────────────────

def load_reranker(model_dir: str, use_fp16: bool):
    """
    Loads FlagReranker. Use use_fp16=True for speed on L40S.
    FlagReranker handles internally batching and device.
    """
    try:
        from FlagEmbedding import FlagReranker
        reranker = FlagReranker(model_dir, use_fp16=use_fp16)
        log.info(f"FlagReranker loaded from {model_dir}")
        return reranker, "flag"
    except ImportError:
        log.warning(
            "FlagEmbedding not found. Fallback on sentence-transformers CrossEncoder.\n"
            "To install FlagEmbedding: pip install FlagEmbedding"
        )
        from sentence_transformers.cross_encoder import CrossEncoder
        model = CrossEncoder(
            model_dir,
            automodel_args={"torch_dtype": torch.bfloat16},
        )
        return model, "cross_encoder"


def score_pairs(reranker, mode: str, pairs: list[list[str]], batch_size: int) -> list[float]:
    """Copmutes score for a list of [query, doc] pairs."""
    if mode == "flag":
        try:
            scores = reranker.compute_score(pairs, batch_size=batch_size)
            if isinstance(scores, (float, int)):
                return [float(scores)]
            return [float(s) for s in scores]
        except Exception as e:
            log.warning(f"FlagReranker error: {e}. Trying batch_size=32.")
            gc.collect()
            scores = reranker.compute_score(pairs, batch_size=32)
            if isinstance(scores, (float, int)):
                return [float(scores)]
            return [float(s) for s in scores]
    else:
        # CrossEncoder fallback
        try:
            scores = reranker.predict(
                pairs, batch_size=batch_size,
                show_progress_bar=False, convert_to_numpy=True
            )
            if isinstance(scores, (float, int)):
                return [float(scores)]
            return scores.tolist()
        except torch.cuda.OutOfMemoryError:
            log.warning("OOM. Trying batch_size=16.")
            torch.cuda.empty_cache()
            gc.collect()
            scores = reranker.predict(
                pairs, batch_size=16,
                show_progress_bar=False, convert_to_numpy=True
            )
            return scores.tolist()


# ── Sanity check ───────────────────────────────────────────────────────────────

def sanity_check(results: dict, nemotron_run: dict, rerank_depth: int):
    if not results:
        log.error("SANITY CHECK FAILED: empty output!")
        return
    coverage = len(results) / max(1, len(nemotron_run))
    avg_len  = sum(len(v) for v in results.values()) / len(results)
    log.info(f"Sanity check: {len(results):,} queries ({coverage:.1%} coverage), "
             f"avg {avg_len:.1f} doc/query (expected ≤{rerank_depth})")
    sample_qid = next(iter(results))
    log.info(f"  Sample qid={sample_qid} top-3: {results[sample_qid][:3]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Reranking with BGE-reranker-v2-m3 fine-tuned (FlagReranker)"
    )
    parser.add_argument("--model_dir",     required=True)
    parser.add_argument("--nemotron_run",  required=True)
    parser.add_argument("--collection",    required=True)
    parser.add_argument("--queries",       required=True)
    parser.add_argument("--output",        required=True)
    parser.add_argument("--rerank_depth",  type=int, default=100)
    parser.add_argument("--batch_size",    type=int, default=256,
                        help="FlagReranker is faster than CrossEncoder, "
                             "high batch_size is ok (default 256)")
    parser.add_argument("--max_doc_chars", type=int, default=4000)
    parser.add_argument("--no_fp16",       action="store_true",
                        help="Disables fp16 (usa bf16 o fp32)")
    args = parser.parse_args()

    use_fp16 = not args.no_fp16
    log.info(f"use_fp16={use_fp16}")

    # ── Loads model ─────────────────────────────────────────────────────────
    reranker, mode = load_reranker(args.model_dir, use_fp16=use_fp16)
    log.info(f"Inference mode: {mode}")

    # ── Loads data ────────────────────────────────────────────────────────────
    corpus       = load_collection(args.collection, args.max_doc_chars)
    queries      = load_queries(args.queries)
    nemotron_run = load_nemotron_run(args.nemotron_run)

    # ── Reranking ──────────────────────────────────────────────────────────────
    results      = {}
    n_missing_q  = 0
    n_empty_docs = 0

    for qid, nemotron_docs in tqdm(nemotron_run.items(), desc="Reranking"):
        if qid not in queries:
            n_missing_q += 1
            continue

        query         = queries[qid]
        candidate_ids = nemotron_docs[:args.rerank_depth]
        valid_ids     = [d for d in candidate_ids if d in corpus]
        valid_docs    = [corpus[d] for d in valid_ids]

        if not valid_ids:
            n_empty_docs += 1
            continue

        pairs  = [[query, doc] for doc in valid_docs]
        scores = score_pairs(reranker, mode, pairs, args.batch_size)

        ranked       = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [doc_id for doc_id, _ in ranked]

    if n_missing_q:
        log.warning(f"Missing queries: {n_missing_q}")
    if n_empty_docs:
        log.warning(f"Query with no doc in the corpus: {n_empty_docs}")

    sanity_check(results, nemotron_run, args.rerank_depth)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(f"✓ Output: {out_path}  ({len(results):,} queries)")


if __name__ == "__main__":
    main()