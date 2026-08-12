"""
prepare_data_from_nemotron.py

Prepares the fine-tuning dataset for ModernBERT CrossEncoder starting from:

CORPUS 2026 (current project):
    - en_train.json                          → ground truth: {index, text (query), pubkey (correct doc)}
    - collection_data.json                   → corpus: {pubkey, title, abstract, ...}
    - reranked_results_nemotron_topk400.json → {query_index: [doc_id_rank1, ...]}

CORPUS 2025 (last year project, optional):
    - en_train.tsv                           → ground truth: post_id, tweet_text, cord_uid
    - collection_data2025.json               → corpus: {cord_uid, title, abstract, ...}
    - reranked_results_nemotron_2025.json      (optional, if reranked)

Hard negatives strategy:
    The Nemotron run provides a list sorted by descending score.
    The documents in that list that are NOT the positive one are higher quality
    hard negatives compared to BM25 negatives: Nemotron has already judged them
    as "plausible," but we know they are false positives.

    In practice, we take:
        - Positive:       the document with the correct pubkey/cord_uid (from en_train)
        - Hard negatives: the top-K documents in the Nemotron run, excluding the positive
        (concentrated in the very first ranks: they are the most ambiguous and informative)

Output:
    training_data/
        train_pairs.jsonl
        dev_pairs.jsonl
        stats.json

Output format (for the CrossEncoderTrainer with BinaryCrossEntropyLoss):
  labeled-pair: {query, passage, label}
  with label=1 for positives, label=0 for negatives

Usage:
    python prepare_data_from_nemotron.py \
        --train_json      en_train.json \
        --collection_json collection_data.json \
        --nemotron_run    reranked_results_nemotron_topk400.json \
        --out_dir         training_data \
        [--train_tsv      en_train.tsv] \
        [--collection_json2025 collection_data2025.json] \
        [--nemotron_run2025    reranked_results_nemotron_2025.json] \
        [--num_hard_neg   5] \
        [--neg_offset     0]
"""

import json
import random
import logging
import argparse
import csv
import emoji
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
@dataclass
class Config:
    num_hard_neg:  int   = 5      # how many hard negatives per positive
    # neg_offset: skip the first negatives in the Nemotron run.
    # useful for excluding the most ambiguous almost-positives (rank 1-2)
    # and strarting from the easier ones. Default 0 = start from the top.
    neg_offset:    int   = 0
    dev_ratio:     float = 0.1
    seed:          int   = 42
    max_doc_chars: int   = 6000   # document text truncation
# ────────────────────────────────────────────────────────────────────────────


import re

def clean_tweet(text: str) -> str:
    """Preprocessing aligned with the Java/Lucene pipeline of the project."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def doc_to_text(doc: dict, max_chars: int) -> str:
    """
    Builds the text of the document from title + abstract.
    Handles both 2026 (title/abstract) and 2025 (same structure) formats.
    """
    title    = doc.get("title", "") or ""
    abstract = doc.get("abstract", "") or ""
    text     = f"{title}. {abstract}".strip(". ")
    return text[:max_chars]


# ── Loading corpus ───────────────────────────────────────────────────────
def load_collection_2026(path: str, max_chars: int) -> dict[str, str]:
    """
    Loads collection_data.json (corpus 2026).
    Key: str(pubkey)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = {}
    for item in data:
        key  = str(item["pubkey"])
        text = doc_to_text(item, max_chars)
        if text:
            corpus[key] = text

    log.info(f"Corpus 2026: {len(corpus):,} documents (from {path})")
    return corpus


def load_collection_2025(path: str, max_chars: int) -> dict[str, str]:
    """
    Loads collection_data2025.json (corpus 2025).
    Key: cord_uid (stringa)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    corpus = {}
    for item in data:
        key  = str(item["cord_uid"])
        text = doc_to_text(item, max_chars)
        if text:
            corpus[key] = text

    log.info(f"2025 corpus: {len(corpus):,} documents (from {path})")
    return corpus


# ── Loading ground truth ─────────────────────────────────────────────────
def load_ground_truth_2026(path: str) -> dict[str, str]:
    """
    Loads en_train.json.
    Returns: {str(index): str(pubkey)}
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for item in data:
        qid     = str(item["index"])
        pos_id  = str(item["pubkey"])
        gt[qid] = pos_id

    log.info(f"Ground truth 2026: {len(gt):,} query-document pairs (from {path})")
    return gt


def load_ground_truth_2025(path: str) -> tuple[dict[str, str], dict[str, str]]:
    """
    Loads en_train.tsv.
    Returns:
      queries:    {str(post_id): tweet_text}
      ground_truth: {str(post_id): cord_uid}
    """
    queries = {}
    gt      = {}

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid             = str(row["post_id"])
            queries[qid]    = row["tweet_text"]
            gt[qid]         = str(row["cord_uid"])

    log.info(f"2025 ground truth: {len(gt):,} query-document pairs (from {path})")
    return queries, gt


def load_queries_2026(train_json_path: str) -> dict[str, str]:
    """Extracts the queries from en_train.json. Returns {str(index): text}"""
    with open(train_json_path, encoding="utf-8") as f:
        data = json.load(f)
    queries = {str(item["index"]): clean_tweet(item["text"]) for item in data}
    log.info(f"2026 queries: {len(queries):,}")
    return queries


# ── Loading Nemotron run ─────────────────────────────────────────────────
def load_nemotron_run(path: str) -> dict[str, list[str]]:
    """
    Loads the Nemotron reranked run JSON file.
    Expected format: {str(query_index): [doc_id_1, doc_id_2, ...]}
    (ordered by descending score = rank 1 = most relevant according to Nemotron)

    Handles both int and str keys.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    run = {str(k): [str(d) for d in v] for k, v in raw.items()}
    n_docs = sum(len(v) for v in run.values())
    log.info(f"Nemotron run: {len(run):,} queries, {n_docs:,} doc totali (da {path})")
    return run


# ── Build examples ───────────────────────────────────────────────────────
def build_examples(
        queries:      dict[str, str],    # {qid: query_text}
        corpus:       dict[str, str],    # {doc_id: doc_text}
        ground_truth: dict[str, str],    # {qid: positive_doc_id}
        nemotron_run: dict[str, list[str]],  # {qid: [doc_id ranked by Nemotron]}
        cfg:          Config,
        split_tag:    str = "",
) -> list[dict]:
    """
    Builds pairs (query, passage, label) using:
      - Positive: the correct doc from ground_truth
      - Hard negatives: the top documents in the Nemotron run, excluding the positive

    Negative selection logic:
      Nemotron run[qid] = [d1, d2, d3, ..., d400] (already sorted by desc score)
      We exclude the positive, then take the first num_hard_neg from the list.
      With neg_offset > 0, we skip the very first ones (the most ambiguous).

    Why this is better than BM25 negatives:
      Nemotron has already "filtered" the plausible candidates.
      A document that Nemotron ranked at rank 2-10 but is not the positive
      is a very high-quality hard negative for teaching ModernBERT
      to distinguish the true positive from near-positives.
    """
    examples     = []
    stats        = Counter()

    for qid, query_text in queries.items():
        pos_id = ground_truth.get(qid)
        if pos_id is None:
            stats["no_gt"] += 1
            continue

        if pos_id not in corpus:
            stats["pos_not_in_corpus"] += 1
            continue

        if qid not in nemotron_run:
            stats["no_nemotron_run"] += 1
            # Fallback: no hard negatives, skip
            continue

        nemotron_docs = nemotron_run[qid]  # già ordinati per rank Nemotron

        # Hard negatives: all the docs in the Nemotron run except for the positive one
        hard_negs = [d for d in nemotron_docs if d != pos_id and d in corpus]

        if len(hard_negs) < 2:
            stats["too_few_negs"] += 1
            continue

        # Apply offset and take num_hard_neg
        sliced = hard_negs[cfg.neg_offset : cfg.neg_offset + cfg.num_hard_neg]
        if len(sliced) < cfg.num_hard_neg:
            # If the offset is too high, complete with the next available
            sliced = hard_negs[:cfg.num_hard_neg]

        # Positive row
        examples.append({
            "query":   query_text,
            "passage": corpus[pos_id],
            "label":   1,
            "qid":     qid,
            "doc_id":  pos_id,
        })
        stats["positives"] += 1

        # Negative rows
        for neg_id in sliced:
            examples.append({
                "query":   query_text,
                "passage": corpus[neg_id],
                "label":   0,
                "qid":     qid,
                "doc_id":  neg_id,
            })
            stats["negatives"] += 1

    log.info(
        f"[{split_tag}] Examples: {len(examples):,} "
        f"({stats['positives']:,} pos + {stats['negatives']:,} neg). "
        f"Stored: no_gt={stats['no_gt']}, no_corpus={stats['pos_not_in_corpus']}, "
        f"no_run={stats['no_nemotron_run']}, pochi_neg={stats['too_few_negs']}"
    )
    return examples


# ── Split and storing ──────────────────────────────────────────────────────
def split_by_query(examples: list[dict], dev_ratio: float, seed: int):
    """
    Split train/dev by query (not by row).
    All examples of the same query go into the same split,
    to avoid data leakage.
    """
    # Group by qid
    qid_to_examples = defaultdict(list)
    for ex in examples:
        qid_to_examples[ex["qid"]].append(ex)

    qids = list(qid_to_examples.keys())
    random.Random(seed).shuffle(qids)

    n_dev      = max(10, int(len(qids) * dev_ratio))
    dev_qids   = set(qids[:n_dev])
    train_qids = set(qids[n_dev:])

    train = [ex for qid in train_qids for ex in qid_to_examples[qid]]
    dev   = [ex for qid in dev_qids   for ex in qid_to_examples[qid]]

    return train, dev


def save_jsonl(examples: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(f"Stored: {path} ({len(examples):,} rows)")


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Prepare fine-tuning data using Nemotron run as hard negatives"
    )
    # Corpus 2026 (mandatory)
    parser.add_argument("--train_json",      required=True,
                        help="en_train.json — ground truth 2026")
    parser.add_argument("--collection_json", required=True,
                        help="collection_data.json — corpus 2026")
    parser.add_argument("--nemotron_run",    required=True,
                        help="reranked_results_nemotron_topk400.json")

    # Corpus 2025 (optional)
    parser.add_argument("--train_tsv",          default=None,
                        help="en_train.tsv — ground truth 2025 (optional)")
    parser.add_argument("--collection_json2025", default=None,
                        help="collection_data2025.json — corpus 2025 (optional)")
    parser.add_argument("--nemotron_run2025",    default=None,
                        help="reranked_results_nemotron_2025.json (optional)")

    # Parametri
    parser.add_argument("--out_dir",     default="training_data")
    parser.add_argument("--num_hard_neg", type=int,   default=5)
    parser.add_argument("--neg_offset",   type=int,   default=0,
                        help="Skip the first N negativi in the Nemotron run (default 0)")
    parser.add_argument("--dev_ratio",    type=float, default=0.1)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--max_doc_chars", type=int,  default=5000)
    args = parser.parse_args()

    cfg = Config(
        num_hard_neg=args.num_hard_neg,
        neg_offset=args.neg_offset,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        max_doc_chars=args.max_doc_chars,
    )

    random.seed(cfg.seed)

    all_examples = []

    # ── Corpus 2026 ──────────────────────────────────────────────────────────
    log.info("=== Loading 2026 corpus ===")
    corpus_2026   = load_collection_2026(args.collection_json, cfg.max_doc_chars)
    queries_2026  = load_queries_2026(args.train_json)
    gt_2026       = load_ground_truth_2026(args.train_json)
    nemotron_2026 = load_nemotron_run(args.nemotron_run)

    examples_2026 = build_examples(
        queries_2026, corpus_2026, gt_2026, nemotron_2026, cfg, split_tag="2026"
    )
    all_examples.extend(examples_2026)
    log.info(f"2026 examples: {len(examples_2026):,}")

    # ── Corpus 2025 (opzionale) ───────────────────────────────────────────────
    if args.train_tsv and args.collection_json2025:
        log.info("=== Loading 2025 corpus ===")
        corpus_2025          = load_collection_2025(args.collection_json2025, cfg.max_doc_chars)
        queries_2025, gt_2025 = load_ground_truth_2025(args.train_tsv)
        # Cleanup 2025 queries
        queries_2025 = {qid: clean_tweet(text) for qid, text in queries_2025.items()}

        if args.nemotron_run2025:
            nemotron_2025 = load_nemotron_run(args.nemotron_run2025)
        else:
            # Without Nemotron run 2025, we do not have quality hard negatives.
            # We use the data anyway but with an empty Nemotron run
            # → build_examples will skip these queries (no_nemotron_run).
            # Better to be conservative than to use poor-quality random negatives.
            log.warning(
                "No Nemotron run 2025 provided. "
                "The 2025 data will only be used if you provide --nemotron_run2025. "
                "Tip: rerank the 2025 corpus with Nemotron before using it."
            )
            nemotron_2025 = {}

        examples_2025 = build_examples(
            queries_2025, corpus_2025, gt_2025, nemotron_2025, cfg, split_tag="2025"
        )
        all_examples.extend(examples_2025)
        log.info(f"2025 examples: {len(examples_2025):,}")
    else:
        log.info("2025 corpus not provided, skip.")

    # ── Split and storing ───────────────────────────────────────────────────
    if not all_examples:
        log.error("Nessun example built! Check input files.")
        return

    log.info(f"\nTotal examples: {len(all_examples):,}")
    n_pos = sum(1 for ex in all_examples if ex["label"] == 1)
    n_neg = sum(1 for ex in all_examples if ex["label"] == 0)
    log.info(f"  Positives: {n_pos:,} ({100*n_pos/len(all_examples):.1f}%)")
    log.info(f"  negatives: {n_neg:,} ({100*n_neg/len(all_examples):.1f}%)")
    log.info(f"  Effective neg/pos ration: {n_neg/max(1,n_pos):.1f}x")

    train_examples, dev_examples = split_by_query(all_examples, cfg.dev_ratio, cfg.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_examples, out_dir / "train_pairs.jsonl")
    save_jsonl(dev_examples,   out_dir / "dev_pairs.jsonl")

    # Salva statistiche
    stats = {
        "total": len(all_examples),
        "train": len(train_examples),
        "dev":   len(dev_examples),
        "positives": n_pos,
        "negatives": n_neg,
        "ratio_neg_pos": round(n_neg / max(1, n_pos), 2),
        "corpus_2026_size": len(corpus_2026),
        "num_hard_neg": cfg.num_hard_neg,
        "neg_offset": cfg.neg_offset,
    }
    with open(out_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n✓ Dataset ready in {out_dir}/")
    log.info(f"  train_pairs.jsonl: {len(train_examples):,} rows")
    log.info(f"  dev_pairs.jsonl:   {len(dev_examples):,} rows")
    log.info(
        f"\nNext step:\n"
        f"  python finetune_modernbert_crossencoder.py \\\n"
        f"    --train_jsonl {out_dir}/train_pairs.jsonl \\\n"
        f"    --dev_jsonl   {out_dir}/dev_pairs.jsonl \\\n"
        f"    --output_dir  models/modernbert-large-retrix-nemotron-distilled"
    )


if __name__ == "__main__":
    main()