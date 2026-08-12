"""
prepare_data_nemotron_finetune_v2.py

Prepare the dataset for fine-tuning nvidia/llama-nemotron-rerank-1b-v2.

===============================================================
CHANGES COMPARED TO v1
===============================================================

[FIX 1] Added SOFT NEGATIVES ("neg" field separated from "hard_neg")
  In v1 the dataset had only hard negatives (top-k from base Nemotron).
  Using only hard negatives makes training unstable: loss starts from
  very hard cases without an easier gradient signal to anchor learning.
  Now the dataset has two negative types:
    - "hard_neg": rank [neg_offset .. neg_offset+num_hard_neg] in Nemotron run
      -> documents the model already mis-ranks, strongest signal
    - "neg":      rank [soft_neg_offset .. soft_neg_offset+num_soft_neg]
      -> mid-ranking documents, far enough from positive but non-trivial.
         They stabilize training in early iterations.
  finetune_nemotron_reranker_v3.py already handles both fields
  (hard_neg oversampled x2, neg with normal weight).

[FIX 2] Default neg_offset moved: 0 -> 1
  In v1, first hard negative started from rank=1 (immediately after
  the positive). If the positive is not in the run (rare but possible),
  rank=0 may be another relevant but unlabeled document.
  Starting at offset=1 adds a safety margin.
  WARNING: if your nemotron_run already excludes the positive, keep
  neg_offset=0.

[FIX 3] soft_neg_offset and num_soft_neg are configurable
  Default: soft_neg_offset=10, num_soft_neg=5
  -> Takes documents ranked 10-15 by base Nemotron.
  These are far enough from positive to be likely non-relevant,
  but not so far that they become trivial (e.g., fully different topic).

[FIX 4] Added 2025 vs 2026 domain-shift check
  Added statistical analysis of query length and vocabulary overlap.
  If overlap < 0.5, explicit warning suggests excluding 2025 data.

[FIX 5] More detailed dataset quality stats
  stats.json now includes:
    - overlap_rate: how many queries have positive in Nemotron run
    - hard_neg_avg_rank: average rank of selected hard negatives
    - domain_shift_warning: boolean flag
    - per-year quality stats breakdown

[FIX 6] Year-stratified shuffle in training set
  v1 random shuffle could over-expose one year in initial batches.
  Now an interleaved shuffle keeps a stable 2025/2026 proportion.

[UNCHANGED] Output format compatible with finetune_nemotron_reranker_v3.py
  "pos", "neg", and "hard_neg" are already supported by trainer v3.

Minimal usage (2026 only):
  python prepare_data_nemotron_finetune_v2.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400.json

Full usage including 2025:
  python3 prepare_data_nemotron_finetune_v2.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400_entrain2026.json \
    --collection_2025     collection_data2025.json \
    --queries_2025        subtask4b_query_tweets_train.json \
                          subtask4b_query_tweets_dev.json \
                          subtask4b_query_tweets_test_gold.json \
    --nemotron_runs_2025  reranked_results_nemotron_topk140_tweets_train2025_BiencoderTopk1000.json \
                          reranked_results_nemotron_topk140_tweets_dev2025_BiencoderTopk1000.json \
                          reranked_results_nemotron_topk140_tweets_test_gold2025_BiencoderTopk1000.json

Usage with only 2026 data (recommended if domain shift is confirmed):
  python3 prepare_data_nemotron_finetune_v2.py \
    --collection_2026     collection_data.json \
    --queries_2026        en_train.json \
    --nemotron_run_2026   reranked_results_nemotron_topk400_entrain2026.json \
    --no_2025
"""

import json
import re
import random
import logging
import argparse
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Cleaning aligned with Java/Lucene pipeline.
    No prefix - Nemotron does not use one.
    """
    text = re.sub(r"http\S+|www\S+", "", text)
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


# ══════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════

def load_queries_and_gt(path: str) -> tuple[dict[str, str], dict[str, str]]:
    data = json.load(open(path, encoding="utf-8"))
    queries, gt = {}, {}
    for item in data:
        qid           = str(item["index"])
        queries[qid]  = clean_text(item["text"])
        gt[qid]       = str(item["pubkey"])
    log.info(f"    {Path(path).name}: {len(queries):,} query")
    return queries, gt


def load_corpus(path: str, max_chars: int, label: str = "") -> dict[str, str]:
    """
    Load corpus JSON with "pubkey" field.
    Compatible with collection_data.json (int pubkey)
    and collection_data2025.json (str pubkey).
    """
    data = json.load(open(path, encoding="utf-8"))
    c = {}
    for d in data:
        key  = str(d["pubkey"])
        text = _doc_text(d, max_chars)
        if text:
            c[key] = text
    log.info(f"  Corpus {label} ({Path(path).name}): {len(c):,} documents")
    return c


def _doc_text(item: dict, max_chars: int) -> str:
    t = (item.get("title")    or "").strip()
    a = (item.get("abstract") or "").strip()
    return f"{t}. {a}".strip(". ")[:max_chars]


def load_nemotron_run(path: str) -> dict[str, list[str]]:
    raw   = json.load(open(path, encoding="utf-8"))
    run   = {str(k): [str(d) for d in v] for k, v in raw.items()}
    total = sum(len(v) for v in run.values())
    log.info(f"    Nemotron run {Path(path).name}: {len(run):,} query, {total:,} doc")
    return run


# ══════════════════════════════════════════════════════════════════
# Domain-shift analysis
# ══════════════════════════════════════════════════════════════════

def _tokenize_simple(text: str) -> set[str]:
    """Lightweight tokenization for vocabulary analysis - not for training."""
    return set(re.findall(r"\b[a-z]{3,}\b", text.lower()))


def analyze_domain_shift(
        queries_2026: dict[str, str],
        queries_2025: dict[str, str],
        threshold: float = 0.5,
) -> bool:
    """
    [FIX 4] Estimate domain shift between 2025 and 2026 queries using
    vocabulary Jaccard overlap.

    Jaccard = |V_2025 ∩ V_2026| / |V_2025 ∪ V_2026|

    If Jaccard < threshold (default 0.5) -> warning: 2025 data may
    introduce distribution shift into training.

    Note: threshold 0.5 is conservative. For biomedical corpora in the
    same domain (COVID), Jaccard is usually > 0.6. Values < 0.5 often
    indicate substantial topic drift.
    """
    vocab_2026 = set()
    for q in queries_2026.values():
        vocab_2026.update(_tokenize_simple(q))

    vocab_2025 = set()
    for q in queries_2025.values():
        vocab_2025.update(_tokenize_simple(q))

    intersection = len(vocab_2026 & vocab_2025)
    union        = len(vocab_2026 | vocab_2025)
    jaccard      = intersection / union if union > 0 else 0.0

    avg_len_2026 = sum(len(q.split()) for q in queries_2026.values()) / max(1, len(queries_2026))
    avg_len_2025 = sum(len(q.split()) for q in queries_2025.values()) / max(1, len(queries_2025))

    log.info("  -- Domain-shift analysis 2025 vs 2026 --")
    log.info(f"    2026 vocabulary: {len(vocab_2026):,} unique tokens")
    log.info(f"    2025 vocabulary: {len(vocab_2025):,} unique tokens")
    log.info(f"    Jaccard overlap: {jaccard:.3f} (threshold={threshold})")
    log.info(f"    Average query length - 2026: {avg_len_2026:.1f} tok | 2025: {avg_len_2025:.1f} tok")

    if jaccard < threshold:
        log.warning(
            f"  WARNING: DOMAIN SHIFT DETECTED (Jaccard={jaccard:.3f} < {threshold}). "
            "2025 data may degrade performance on 2026 dev. "
            "Consider --no_2025 if fine-tuning worsens baseline."
        )
        return True
    else:
        log.info(f"  OK: no significant domain shift (Jaccard={jaccard:.3f} >= {threshold})")
        return False


# ══════════════════════════════════════════════════════════════════
# Group construction
# ══════════════════════════════════════════════════════════════════

def build_groups(
        queries:         dict[str, str],
        corpus:          dict[str, str],
        gt:              dict[str, str],
        nemotron_run:    dict[str, list[str]],
        num_hard_neg:    int,
        neg_offset:      int,
        num_soft_neg:    int,
        soft_neg_offset: int,
        tag:             str,
) -> tuple[list[dict], dict]:
    """
    Build training groups in format:
      {
        "query":    str,
        "pos":      [str],        # 1 positive document (ground truth)
        "hard_neg": [str, ...],   # top-k base Nemotron errors [FIX 1]
        "neg":      [str, ...],   # mid-rank soft negatives [FIX 1]
        "_qid":     str,          # metadata, removed on save
        "_pos_id":  str,
      }

    [FIX 1] WHY TWO SEPARATE LISTS:
    Trainer v3 reads "hard_neg" and oversamples them (x2) because they
    are most informative. Soft "neg" items are included with normal
    weight to stabilize gradients in early iterations.

    [FIX 2] default neg_offset=1:
    Rank=0 (after excluding positive) is usually most similar to query.
    In rare cases it may be relevant but unlabeled (false negative).
    Starting from offset=1 adds margin.

    [FIX 3] soft_neg_offset=10:
    Rank 10-15 is usually far enough from positives to be non-relevant,
    yet not so far as to become trivial negatives.
    """
    groups        = []
    stats         = Counter()
    rank_sum_hard = 0
    rank_count    = 0

    for qid, query_text in queries.items():
        pos_id = gt.get(qid)
        if pos_id is None:
            stats["no_gt"] += 1
            continue
        if pos_id not in corpus:
            stats["pos_missing"] += 1
            continue
        if qid not in nemotron_run:
            stats["no_run"] += 1
            continue

        # Negative candidates: all docs in run except the positive
        candidates = [
            d for d in nemotron_run[qid]
            if d != pos_id and d in corpus
        ]

        if len(candidates) < num_hard_neg + 2:
            stats["few_negs"] += 1
            continue

        # [FIX 1 + FIX 2] Hard negatives: top-k with offset
        hard_start = neg_offset
        hard_end   = neg_offset + num_hard_neg
        hard_negs  = candidates[hard_start:hard_end]

        # Fallback if offset exceeds available candidates
        if len(hard_negs) < num_hard_neg:
            hard_negs = candidates[:num_hard_neg]

        # [FIX 1 + FIX 3] Soft negatives: mid-ranking positions
        soft_start = soft_neg_offset
        soft_end   = soft_neg_offset + num_soft_neg
        soft_negs  = candidates[soft_start:soft_end]

        # If there are not enough candidates for soft negatives, skip soft
        # (not fatal - better fewer soft negatives than zero groups)
        if len(soft_negs) == 0:
            stats["no_soft_neg"] += 1
            soft_negs = []

        # Ensure hard and soft negatives do not overlap
        # (can happen if neg_offset + num_hard_neg > soft_neg_offset)
        hard_set = set(hard_negs)
        soft_negs = [d for d in soft_negs if d not in hard_set]

        # Hard-negative rank stats for FIX 5
        for d in hard_negs:
            try:
                rank = nemotron_run[qid].index(d)
                rank_sum_hard += rank
                rank_count    += 1
            except ValueError:
                pass

        groups.append({
            "query":    query_text,
            "pos":      [corpus[pos_id]],
            "hard_neg": [corpus[d] for d in hard_negs],
            "neg":      [corpus[d] for d in soft_negs],
            "_qid":     f"{tag}_{qid}",
            "_pos_id":  pos_id,
        })
        stats["ok"] += 1

    # Quality stats [FIX 5]
    overlap_rate      = stats["ok"] / max(1, len(queries))
    hard_neg_avg_rank = rank_sum_hard / max(1, rank_count)

    log.info(
        f"  [{tag}] {stats['ok']:,} groups "
        f"({num_hard_neg} hard + {num_soft_neg} soft neg) | "
        f"overlap_rate={overlap_rate:.2%} | "
        f"hard_neg_avg_rank={hard_neg_avg_rank:.1f} | "
        f"skip: no_gt={stats['no_gt']} pos_missing={stats['pos_missing']} "
        f"no_run={stats['no_run']} few_negs={stats['few_negs']} "
        f"no_soft={stats['no_soft_neg']}"
    )

    if overlap_rate < 0.85:
        log.warning(
            f"  ⚠  overlap_rate={overlap_rate:.2%} < 85% per [{tag}]. "
            "Check that Nemotron run covers most queries."
        )
    if hard_neg_avg_rank < 2.0:
        log.warning(
            f"  WARNING: hard_neg_avg_rank={hard_neg_avg_rank:.1f} is very low. "
            "Positive may already be excluded from run - check neg_offset."
        )

    quality_stats = {
        "groups":             stats["ok"],
        "overlap_rate":       round(overlap_rate, 4),
        "hard_neg_avg_rank":  round(hard_neg_avg_rank, 2),
        "skip_no_gt":         stats["no_gt"],
        "skip_pos_missing":   stats["pos_missing"],
        "skip_no_run":        stats["no_run"],
        "skip_few_negs":      stats["few_negs"],
        "skip_no_soft":       stats["no_soft_neg"],
    }
    return groups, quality_stats


# ══════════════════════════════════════════════════════════════════
# Split train/dev
# ══════════════════════════════════════════════════════════════════

def split_2026_by_query(
        groups_2026: list[dict],
        dev_ratio:   float,
        seed:        int,
) -> tuple[list[dict], list[dict]]:
    """
    Stratified split by query ID.
    All rows for the same query stay in same split
    (train or dev) to avoid data leakage.
    """
    by_qid = defaultdict(list)
    for g in groups_2026:
        by_qid[g["_qid"]].append(g)

    qids = list(by_qid.keys())
    random.Random(seed).shuffle(qids)

    n_dev      = max(10, int(len(qids) * dev_ratio))
    dev_qids   = set(qids[:n_dev])
    train_qids = set(qids[n_dev:])

    train = [g for qid in train_qids for g in by_qid[qid]]
    dev   = [g for qid in dev_qids   for g in by_qid[qid]]

    log.info(f"  Split 2026: {len(train):,} train / {len(dev):,} dev groups")
    return train, dev


def interleaved_shuffle(
        groups_2026: list[dict],
        groups_2025: list[dict],
        seed:        int,
) -> list[dict]:
    """
    [FIX 6] Interleaved shuffle that keeps 2026/2025 proportion stable.

    Instead of concatenating and random-shuffling (which can produce
    initial batches dominated by the larger group), interleave datasets
    by ratio and then locally shuffle fixed-size windows. This keeps
    windows balanced across years.

    Esempio con ratio 1:1.16 (13480 vs 15699 come nel tuo caso):
      [2026, 2025, 2026, 2025, 2026, 2025, 2026, 2026, ...]
    instead of:
      [2026, 2026, ..., 2026, 2025, 2025, ..., 2025]
    """
    rng = random.Random(seed)

    g26 = groups_2026[:]
    g25 = groups_2025[:]
    rng.shuffle(g26)
    rng.shuffle(g25)

    if not g25:
        return g26

    # Compute ratio: for each N 2026 samples, how many 2025 samples to insert
    ratio = len(g25) / max(1, len(g26))  # es. 15699/13480 ≈ 1.16

    result  = []
    idx_25  = 0
    acc_25  = 0.0

    for item_26 in g26:
        result.append(item_26)
        acc_25 += ratio
        while acc_25 >= 1.0 and idx_25 < len(g25):
            result.append(g25[idx_25])
            idx_25  += 1
            acc_25  -= 1.0

    # Append any remaining 2025 samples at the end
    result.extend(g25[idx_25:])

    # Local shuffle in windows of 128 without destroying global balance
    window = 128
    for start in range(0, len(result), window):
        chunk = result[start : start + window]
        rng.shuffle(chunk)
        result[start : start + window] = chunk

    return result


# ══════════════════════════════════════════════════════════════════
# I/O
# ══════════════════════════════════════════════════════════════════

def save_jsonl(groups: list[dict], path: Path):
    """Save groups removing metadata fields (prefix '_')."""
    with open(path, "w", encoding="utf-8") as f:
        for g in groups:
            out = {k: v for k, v in g.items() if not k.startswith("_")}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    log.info(f"  -> {path.name}: {len(groups):,} groups saved")


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Prepare data for llama-nemotron-rerank-1b-v2 fine-tuning v2"
    )

    g26 = p.add_argument_group("2026 corpus (required)")
    g26.add_argument("--collection_2026",   required=True)
    g26.add_argument("--queries_2026",      required=True)
    g26.add_argument("--nemotron_run_2026", required=True)

    g25 = p.add_argument_group("2025 corpus (optional)")
    g25.add_argument("--collection_2025",    default=None)
    g25.add_argument("--queries_2025",       nargs="+", default=[])
    g25.add_argument("--nemotron_runs_2025", nargs="+", default=[])
    g25.add_argument(
        "--no_2025",
        action="store_true",
        help="Completely ignore 2025 data (recommended if domain shift is confirmed)",
    )

    p.add_argument("--out_dir",          default="training_data_nemotron_ft_v2")
    p.add_argument(
        "--num_hard_neg",
        type=int, default=5,
        help="Number of hard negatives (top-k base Nemotron errors). Default: 5",
    )
    p.add_argument(
        "--neg_offset",
        type=int, default=1,                    # [FIX 2] era 0
        help="Starting offset for hard negatives in run. Default: 1",
    )
    p.add_argument(
        "--num_soft_neg",
        type=int, default=5,                    # [FIX 1]
        help="Number of soft negatives (mid ranking). Default: 5",
    )
    p.add_argument(
        "--soft_neg_offset",
        type=int, default=10,                   # [FIX 3]
        help="Start position for soft negatives in run. Default: 10",
    )
    p.add_argument("--dev_ratio",       type=float, default=0.1)
    p.add_argument("--max_doc_chars",   type=int,   default=4000)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument(
        "--domain_shift_threshold",
        type=float, default=0.5,
        help="Jaccard threshold for domain-shift warning. Default: 0.5",
    )
    args = p.parse_args()

    # Validate 2025 arguments
    if not args.no_2025 and args.queries_2025 and args.nemotron_runs_2025:
        if len(args.queries_2025) != len(args.nemotron_runs_2025):
            p.error(
                f"--queries_2025 has {len(args.queries_2025)} files but "
                f"--nemotron_runs_2025 has {len(args.nemotron_runs_2025)}. "
                "They must match 1:1."
            )

    # Validate offset: hard and soft negatives should not overlap
    hard_end = args.neg_offset + args.num_hard_neg
    if hard_end > args.soft_neg_offset:
        log.warning(
            f"  ⚠  hard negatives [{args.neg_offset}:{hard_end}] e "
            f"soft negatives [{args.soft_neg_offset}:...] overlap. "
            f"Consider increasing --soft_neg_offset to {hard_end + 2} or higher."
        )

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_quality_stats  = {}
    domain_shift_found = False

    # -- 1. 2026 corpus and queries --
    log.info("== 2026 corpus ==================================================")
    corpus_2026           = load_corpus(args.collection_2026, args.max_doc_chars, "2026")
    queries_2026, gt_2026 = load_queries_and_gt(args.queries_2026)
    nemotron_2026         = load_nemotron_run(args.nemotron_run_2026)

    groups_2026, qstats_2026 = build_groups(
        queries      = queries_2026,
        corpus       = corpus_2026,
        gt           = gt_2026,
        nemotron_run = nemotron_2026,
        num_hard_neg    = args.num_hard_neg,
        neg_offset      = args.neg_offset,
        num_soft_neg    = args.num_soft_neg,
        soft_neg_offset = args.soft_neg_offset,
        tag             = "2026",
    )
    all_quality_stats["2026"] = qstats_2026

    train_2026, dev_2026 = split_2026_by_query(groups_2026, args.dev_ratio, args.seed)

    # -- 2. 2025 corpus and queries (optional) --
    train_2025 = []
    n25        = 0

    use_2025 = (
            not args.no_2025
            and args.collection_2025
            and args.queries_2025
            and args.nemotron_runs_2025
    )

    if args.no_2025:
        log.info("== 2025 data SKIPPED (--no_2025) ===============================")
    elif use_2025:
        log.info("== 2025 corpus ==================================================")
        corpus_2025 = load_corpus(args.collection_2025, args.max_doc_chars, "2025")

        # Collect all 2025 queries for domain-shift analysis
        all_queries_2025: dict[str, str] = {}
        all_groups_2025_per_file: list[list[dict]] = []

        for q_path, n_path in zip(args.queries_2025, args.nemotron_runs_2025):
            stem = Path(q_path).stem
            log.info(f"  Dataset: {stem}")
            queries_q, gt_q = load_queries_and_gt(q_path)
            all_queries_2025.update(queries_q)
            nemotron_q = load_nemotron_run(n_path)
            groups_q, qstats_q = build_groups(
                queries      = queries_q,
                corpus       = corpus_2025,
                gt           = gt_q,
                nemotron_run = nemotron_q,
                num_hard_neg    = args.num_hard_neg,
                neg_offset      = args.neg_offset,
                num_soft_neg    = args.num_soft_neg,
                soft_neg_offset = args.soft_neg_offset,
                tag             = f"2025_{stem}",
            )
            all_quality_stats[f"2025_{stem}"] = qstats_q
            all_groups_2025_per_file.append(groups_q)

        # [FIX 4] Domain-shift analysis
        domain_shift_found = analyze_domain_shift(
            queries_2026 = queries_2026,
            queries_2025 = all_queries_2025,
            threshold    = args.domain_shift_threshold,
        )

        for groups_q in all_groups_2025_per_file:
            train_2025.extend(groups_q)
        n25 = len(train_2025)

    elif not args.no_2025 and args.queries_2025:
        log.warning("WARNING: --nemotron_runs_2025 not provided: 2025 data SKIPPED.")

    # -- 3. Final train composition --
    # [FIX 6] Interleaved shuffle instead of concat + random shuffle
    final_train = interleaved_shuffle(train_2026, train_2025, args.seed)
    final_dev   = dev_2026

    if not final_train:
        log.error("No group constructed. Check input files.")
        return

    # -- 4. Save --
    log.info("== Final composition ============================================")
    log.info(f"  Training: {len(final_train):,} groups (2026={len(train_2026):,} + 2025={n25:,})")
    log.info(f"  Dev (2026 only): {len(final_dev):,} groups")

    save_jsonl(final_train, out_dir / "train_groups.jsonl")
    save_jsonl(final_dev,   out_dir / "dev_groups.jsonl")

    # -- 5. Enriched stats.json [FIX 5] --
    tgs = 1 + args.num_hard_neg + args.num_soft_neg
    stats = {
        "train_groups":           len(final_train),
        "dev_groups":             len(final_dev),
        "train_from_2026":        len(train_2026),
        "train_from_2025":        n25,
        "dev_from_2026_only":     True,
        "num_hard_neg":           args.num_hard_neg,
        "num_soft_neg":           args.num_soft_neg,          # [FIX 1]
        "neg_offset":             args.neg_offset,
        "soft_neg_offset":        args.soft_neg_offset,       # [FIX 3]
        "train_group_size_for_training": tgs,
        "domain_shift_warning":   domain_shift_found,         # [FIX 4]
        "quality_per_split":      all_quality_stats,          # [FIX 5]
        "note": (
            "Nemotron does NOT use query prefixes. Text cleaning only. "
            f"train_group_size={tgs}. "
            f"hard_neg=base model errors (rank {args.neg_offset}-{args.neg_offset+args.num_hard_neg}). "
            f"soft_neg=mid ranking (rank {args.soft_neg_offset}-{args.soft_neg_offset+args.num_soft_neg})."
        ),
    }
    stats_path = out_dir / "stats.json"
    json.dump(stats, open(stats_path, "w"), indent=2)
    log.info(f"  -> stats.json saved")

    # -- 6. Final summary --
    log.info("\n" + "═" * 60)
    log.info("Dataset ready")
    log.info(f"  Output dir:          {out_dir}/")
    log.info(f"  train_groups.jsonl : {len(final_train):,} rows")
    log.info(f"  dev_groups.jsonl   : {len(final_dev):,} rows")
    log.info(f"  Hard neg per query : {args.num_hard_neg} (rank {args.neg_offset}-{args.neg_offset+args.num_hard_neg})")
    log.info(f"  Soft neg per query : {args.num_soft_neg} (rank {args.soft_neg_offset}-{args.soft_neg_offset+args.num_soft_neg})")
    if domain_shift_found:
        log.warning(
            "  WARNING: Domain shift detected - consider re-running with --no_2025 "
            "and comparing metrics on the dev set."
        )
    log.info("═" * 60)
    log.info("\nNext step:")
    log.info(f"  python finetune_nemotron_reranker_v3.py \\")
    log.info(f"    --train_file {out_dir}/train_groups.jsonl \\")
    log.info(f"    --dev_file   {out_dir}/dev_groups.jsonl \\")
    log.info(f"    --output_dir models/nemotron-rerank-1b-retrix-v3")
    log.info(f"\n  # Or without 2025 data if domain shift is confirmed:")
    log.info(f"  python prepare_data_nemotron_finetune_v2.py \\")
    log.info(f"    --collection_2026   collection_data.json \\")
    log.info(f"    --queries_2026      en_train.json \\")
    log.info(f"    --nemotron_run_2026 reranked_results_nemotron_topk400.json \\")
    log.info(f"    --no_2025")


if __name__ == "__main__":
    main()