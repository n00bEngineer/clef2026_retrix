"""
Reranker_gte_modernbert_v2.py — GTE-Reranker-ModernBERT re-ranking
====================================================================

Supports both the base model and the fine-tuned version with --model_name.

Alibaba-NLP/gte-reranker-modernbert-base is an encoder-only cross-encoder
based on ModernBERT (AutoModelForSequenceClassification).

Mechanism:
    1. Tokenize the pair as a pair: tokenizer([[query, doc], ...])
    2. Forward pass → model(inputs).logits  shape [B, 1]
    3. sigmoid(logits.view(-1)) → score in (0, 1)
        - high score (~1) = relevant document
        - low score (~0) = irrelevant document
    4. Ranking: sorted(..., reverse=True)

COMPATIBILITY WITH FINE-TUNED MODEL:
    The model fine-tuned by finetune_modernbert_reranker.py is saved
    with AutoModelForSequenceClassification.save_pretrained() — identical to the
    base model. Loading is identical, sigmoid is already applied here,
    there is NO double-sigmoid because we do not use the CrossEncoder wrapper.

Reads:
  - JSON queries    (field "original" or "text", field "index")
  - JSON collection (corpus, title+abstract, field "pubkey")
  - JSON run        (output reranker/BM25: { qid → [pubkey, ...] })

Writes:
  - JSON output  (same format: { qid → [pubkey, ...] })

Usage:
  # base mdoel (default)
  python Reranker_gte_modernbert_v2.py \
    --queries  data/expanded_queries_bge_large.json \
    --papers   data/collection_data.json \
    --bm25     data/bi_encoder_results_513.json \
    --output   results/reranked_results_gte_modernbert_base.json

  # Fine-tuned model
  python Reranker_gte_modernbert_v2.py \
    --model_name models/gte-modernbert-base-retrix/final \
    --queries  data/expanded_queries_bge_large.json \
    --papers   data/collection_data.json \
    --bm25     data/bi_encoder_results_513.json \
    --output   results/reranked_results_gte_modernbert_ft.json
"""

import argparse
import gc
import json
import os

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="Alibaba-NLP/gte-reranker-modernbert-base",
    help=(
        "Model to use for reranking. It can be:\n"
        "  - 'Alibaba-NLP/gte-reranker-modernbert-base' (default, base model)\n"
        "  - Local path to fine-tuned model, e.g.:\n"
        "    'models/gte-modernbert-base-retrix/final'"
    ),
)
parser.add_argument(
    "--queries",
    default="../../../../../../data/expanded_queries_bge_large.json",
)
parser.add_argument(
    "--papers",
    default="../../../../../../data/collection_data.json",
)
parser.add_argument(
    "--bm25",
    default="../../../../../../data/bi_encoder_results_513.json",
)
parser.add_argument(
    "--output",
    default="../../../../../../../results/reranked_results_gte_modernbert_v2.json",
)
parser.add_argument(
    "--top_k",
    type=int,
    default=100,
    help="Candidates to pass to the reranker for each query (default: 100).",
)
parser.add_argument(
    "--batch",
    type=int,
    default=64,
    help="Pairs per GPU forward pass (default: 64).",
)
parser.add_argument(
    "--query_chunk",
    type=int,
    default=8,
    help="Queries per elaboration chunk (default: 8).",
)
parser.add_argument(
    "--max_length",
    type=int,
    default=1024,
    help="Max token length per query+doc pair (default: 1024).",
)


# ──────────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────────

def sanity_check_scores(score_fn) -> bool:
    """
    Verify that the model produces discriminative scores.
    After sigmoid: relevant → close to 1, irrelevant → close to 0.
    """
    test_pairs = [
        (
            "COVID-19 mask effectiveness public transport",
            "A study on the effectiveness of face masks in reducing COVID-19 "
            "transmission in public transport settings, showing significant reduction.",
        ),
        (
            "COVID-19 mask effectiveness public transport",
            "The history of ancient Rome and its political structure "
            "in the first century BC.",
        ),
    ]
    try:
        scores = score_fn(test_pairs)
        s0, s1 = float(scores[0]), float(scores[1])
        diff = abs(s0 - s1)
        print(
            f"\n  [Sanity check] Relevant score={s0:.4f} | "
            f"Irrelevant score={s1:.4f} | Δ={diff:.4f}"
        )
        if diff < 0.05:
            print("  WARNING: almost identical scores — check the model/tokenizer!")
            return False
        print("  [Sanity check] OK.")
        return True
    except Exception as e:
        print(f"  WARNING: sanity check failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────────
# CPU helpers
# ──────────────────────────────────────────────────────────────────

def build_pairs_for_queries(
    query_items: list,
    paper_texts: dict,
    query_texts: dict,
    top_k: int,
) -> tuple[list, list, int]:
    """
    Builds (query_text, doc_text) pairs for a query chunk.

    Returns:
      all_pairs : plain list of (str, str)
      meta      : [(qid, valid_keys, n_pairs, fallback_candidates), ...]
      missing   : number of documents not found in the corpus
    """
    all_pairs = []
    meta      = []
    missing   = 0

    for qid, candidate_pubkeys in query_items:
        query_text = query_texts.get(str(qid), "")
        candidates = candidate_pubkeys[:top_k]

        if not query_text:
            print(
                f"  WARNING: query text not found for qid={qid}, "
                "using original sorting",
                flush=True,
            )
            meta.append((qid, [], 0, candidates))
            continue

        pairs      = []
        valid_keys = []
        for pk in candidates:
            doc_text = paper_texts.get(str(pk))
            if doc_text:
                pairs.append((query_text, doc_text))
                valid_keys.append(pk)
            else:
                missing += 1

        meta.append((qid, valid_keys, len(pairs), candidates))
        all_pairs.extend(pairs)

    return all_pairs, meta, missing


def scores_to_results(meta: list, scores_flat: list) -> dict:
    """
    Rebuilds qid→[pubkey] from the plain score array.
    Score in (0,1) after sigmoid: reverse=True puts the most relevant on the top.
    """
    results = {}
    offset  = 0

    for qid, valid_keys, n_pairs, fallback_candidates in meta:
        if n_pairs == 0:
            results[qid] = fallback_candidates
            continue
        scores = scores_flat[offset : offset + n_pairs]
        offset += n_pairs
        ranked = sorted(
            zip(valid_keys, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        results[qid] = [pk for pk, _ in ranked]

    return results


# ──────────────────────────────────────────────────────────────────
# Loading model
# ──────────────────────────────────────────────────────────────────

def load_reranker(model_name: str, device: str, max_length: int):
    """
    Loads the GTE-ModernBERT reranker (base or fine-tuned).

    Uses AutoModelForSequenceClassification directly — NOT the CrossEncoder
    wrapper from sentence-transformers. This ensures:
        1. No double-sigmoid: the fine-tuned model is saved with
            Sigmoid as the default_activation_function in sentence_bert_config.json,
            but it is NOT applied when loading with AutoModel. The sigmoid
            is applied explicitly here in score_fn.
        2. Compatibility with both models (base and fine-tuned) without
            changing code.
        3. OOM handling with automatic batch size halving.

    The tokenizer receives a pair-list [[query, doc], ...] — identical for
    both the base model and the fine-tuned one.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  Loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"  Loading model: {model_name} → {device}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    model.eval()

    is_finetuned = model_name != "Alibaba-NLP/gte-reranker-modernbert-base"
    print(
        f"  Model type: {'fine-tuned' if is_finetuned else 'base'}",
        flush=True,
    )

    def score_fn(pairs: list[tuple[str, str]]) -> list[float]:
        """
        Scores a list of (query, doc).
        Returns float scores in (0, 1) after sigmoid.

        The GTE tokenizer accepts a pair-list [[q, d], ...]:
        it internally handles the query/doc separation with [SEP].
        """
        pair_list = [[q, d] for q, d in pairs]
        inputs = tokenizer(
            pair_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits        # shape: [B, 1]
            scores = torch.sigmoid(logits.view(-1)).cpu().tolist()

        return scores if isinstance(scores, list) else [scores]

    return score_fn


# ──────────────────────────────────────────────────────────────────
# Inference with OOM handling
# ──────────────────────────────────────────────────────────────────

def predict_scores(score_fn, pairs: list, batch_size: int) -> list:
    """
    Iterates pairs in batches, calls score_fn, and aggregates results.
    Handles OOM by halving the batch size and restoring it for the next chunk.
    Fallback score = 0.5 (neutral) if OOM occurs even with batch=1.
    """
    all_scores    = []
    current_batch = batch_size
    i             = 0

    while i < len(pairs):
        batch = pairs[i : i + current_batch]
        try:
            scores = score_fn(batch)
            all_scores.extend(scores)
            i            += current_batch
            current_batch = batch_size  # restores after reducing batch

        except (torch.OutOfMemoryError, RuntimeError) as e:
            is_oom = (
                isinstance(e, torch.OutOfMemoryError)
                or "out of memory" in str(e).lower()
            )
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            gc.collect()
            if current_batch <= 1:
                print(
                    f"  WARNING: OOM even with batch=1 on {len(batch)} pairs. "
                    "Fallback score=0.5 (original sorting maintained).",
                    flush=True,
                )
                all_scores.extend([0.5] * len(batch))
                i            += current_batch
                current_batch = batch_size
            else:
                current_batch = max(1, current_batch // 2)
                print(
                    f"  OOM → reducing batch to {current_batch} for this chunk",
                    flush=True,
                )

    return all_scores


# ──────────────────────────────────────────────────────────────────
# Worker per-GPU (path multi-GPU)
# ──────────────────────────────────────────────────────────────────

def rerank_worker(
    gpu_id:      int,
    model_name:  str,
    query_items: list,
    paper_texts: dict,
    query_texts: dict,
    top_k:       int,
    batch_size:  int,
    query_chunk: int,
    max_length:  int,
    result_queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda:0"
    print(
        f"[GPU {gpu_id}] {torch.cuda.get_device_name(0)} — loading model...",
        flush=True,
    )

    score_fn = load_reranker(model_name, device, max_length)

    if gpu_id == 0:
        sanity_check_scores(score_fn)

    local_results     = {}
    local_missing     = 0
    local_pairs_total = 0

    chunks = [
        query_items[i : i + query_chunk]
        for i in range(0, len(query_items), query_chunk)
    ]

    for chunk in tqdm(
        chunks,
        desc=f"Re-ranking [GPU {gpu_id}]",
        position=gpu_id,
        leave=True,
    ):
        all_pairs, meta, missing = build_pairs_for_queries(
            chunk, paper_texts, query_texts, top_k
        )
        local_missing += missing

        if not all_pairs:
            for qid, _, _, fallback in meta:
                local_results[qid] = fallback
            continue

        local_pairs_total += len(all_pairs)
        scores_flat = predict_scores(score_fn, all_pairs, batch_size)
        local_results.update(scores_to_results(meta, scores_flat))

    result_queue.put((local_results, local_missing, local_pairs_total))


# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parser.parse_args()

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        DEVICE = "cpu"
        print("Device: cpu (no CUDA GPU found)")
    else:
        DEVICE = "cuda:0"
        print(f"Device: cuda — {NUM_GPUS} available GPU:")
        for i in range(NUM_GPUS):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    print(f"\nModel: {args.model_name}")
    print("Loading data...")

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)

    with open(args.papers, "r", encoding="utf-8") as f:
        papers_raw = json.load(f)

    with open(args.bm25, "r", encoding="utf-8") as f:
        bm25_results: dict = json.load(f)

    paper_texts: dict = {
        str(p["pubkey"]): (
            p.get("title", "") + " " + p.get("abstract", "")
        ).strip()
        for p in papers_raw
    }

    # Supports both "original" (BGE expanded queries) and "text" (raw queries)
    query_texts: dict = {
        str(q["index"]): q.get("original", q.get("text", ""))
        for q in queries_raw
    }

    print(
        f"\nCorpus: {len(paper_texts)} documents | "
        f"Queries: {len(query_texts)} | "
        f"Run input: {len(bm25_results)} queries"
    )

    # Sanity check dati
    sample_qid        = next(iter(bm25_results))
    sample_candidates = bm25_results[sample_qid][:5]
    hits = sum(1 for pk in sample_candidates if str(pk) in paper_texts)
    print(
        f"Sanity check corpus — qid='{sample_qid}': "
        f"{hits}/{len(sample_candidates)} candidates found"
    )
    if hits == 0:
        print(
            "CRITICAL ERROR: 0 candidates found in the corpus. "
            f"pubkey type in the run={type(sample_candidates[0])}, "
            "corpus key type=str — check the file --papers."
        )

    sample_qt = query_texts.get(str(sample_qid), "")
    print(f"Sanity check queries — qid='{sample_qid}': '{sample_qt[:80]}...'")
    if not sample_qt:
        print(
            "WARNING: query text empty. "
            "Check field 'original'/'text' in the file --queries."
        )

    all_query_items = list(bm25_results.items())
    print(
        f"\nRe-ranking {len(bm25_results)} queries "
        f"(top_k={args.top_k}, batch={args.batch}, "
        f"query_chunk={args.query_chunk}, max_length={args.max_length})..."
    )

    # ── Path single-GPU / CPU ──────────────────────────────────────
    if NUM_GPUS <= 1:
        score_fn = load_reranker(args.model_name, DEVICE, args.max_length)

        ok = sanity_check_scores(score_fn)
        if not ok:
            print("\nWARNING: sanity check failed. Continuing anyway.\n")

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        chunks = [
            all_query_items[i : i + args.query_chunk]
            for i in range(0, len(all_query_items), args.query_chunk)
        ]

        for chunk in tqdm(chunks, desc="Re-ranking"):
            all_pairs, meta, missing = build_pairs_for_queries(
                chunk, paper_texts, query_texts, args.top_k
            )
            missing_docs += missing

            if not all_pairs:
                for qid, _, _, fallback in meta:
                    reranked_results[qid] = fallback
                continue

            total_pairs += len(all_pairs)
            scores_flat  = predict_scores(score_fn, all_pairs, args.batch)
            reranked_results.update(scores_to_results(meta, scores_flat))

    # ── Path multi-GPU ─────────────────────────────────────────────
    else:
        partitions = [[] for _ in range(NUM_GPUS)]
        for i, item in enumerate(all_query_items):
            partitions[i % NUM_GPUS].append(item)

        for i, part in enumerate(partitions):
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {len(part)} queries")

        mp.set_start_method("spawn", force=True)
        result_queue: mp.Queue = mp.Queue()

        processes = []
        for gpu_id in range(NUM_GPUS):
            proc = mp.Process(
                target=rerank_worker,
                args=(
                    gpu_id,
                    args.model_name,
                    partitions[gpu_id],
                    paper_texts,
                    query_texts,
                    args.top_k,
                    args.batch,
                    args.query_chunk,
                    args.max_length,
                    result_queue,
                ),
            )
            proc.start()
            processes.append(proc)

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        for _ in range(NUM_GPUS):
            local_results, local_missing, local_pairs = result_queue.get()
            reranked_results.update(local_results)
            missing_docs += local_missing
            total_pairs  += local_pairs

        for proc in processes:
            proc.join()

        # Preserves the original sorting of the queries
        reranked_results = {
            qid: reranked_results[qid]
            for qid in bm25_results
            if qid in reranked_results
        }

    # ── Output ─────────────────────────────────────────────────────
    print(f"\nTotal pairs scored : {total_pairs:,}")
    print(f"Queries re-ranked  : {len(reranked_results):,}")
    if missing_docs:
        print(f"Pubkey not found : {missing_docs} (original sorting maintained)")
    coverage = len(reranked_results) / max(1, len(bm25_results)) * 100
    print(f"Coverage           : {coverage:.1f}% ({len(reranked_results)}/{len(bm25_results)})")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, indent=2, ensure_ascii=False)

    print(f"\nStored → {args.output}")
    print("Done.")
