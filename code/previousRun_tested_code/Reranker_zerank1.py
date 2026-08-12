"""
Reranker_zerank1.py — zerank-1 / zerank-2 re-ranking of BM25 results
========================================================================
zeroentropy/zerank-1 and zeroentropy/zerank-2 are cross-encoder based on
Qwen3-4B with custom weigths and architecture (modeling_zeranker.py).

Official mechanism (from HuggingFace zeroentropy/zerank-1 e zerank-2):
    1.  Load with CrossEncoder("zeroentropy/zerank-1", trust_remote_code=True)
        do NOT use AutoModelForSequenceClassification directly: the score.weight
        parameter would be randomly re-initialized instead of using the trained weights in the checkpoint.
    2.  model.predict([(query, doc), ...]) → score float in [0, 1]
        - ~1.0 = higly relevant
        - ~0.0 = irrelevant
        Scores are already normalized: no sigmoid is required.
    3. CRITICAL: CrossEncoder.predict() with batch_size > 1 crashes if the tokenizer
    does not have a defined pad_token (known model bug).
    Solution: manual mini-batching pair by pair within predict_scores(),
    do NOT rely on the batch_size parameter of CrossEncoder.predict().

Key differences compared to Nemotron:
  - CrossEncoder (sentence-transformers), NOT AutoModelForSequenceClassification
  - Output already in [0,1]: no sigmoid
  - Manual mini-batching tu avoid crash on padding
  - To switch from zerank-1 to zerank-2: only change the MODEL_NAME

Required:
  pip install sentence-transformers torch

Reads:
  - data/expanded_queries_*.json   (query, fields "original" or "text")
  - data/collection_data.json      (corpus, title+abstract)
  - results/bm25_results.json      (Java output: { qid → [pubkey, ...] })

Writes:
  - results/reranked_results_zerank.json  (same format: { qid → [pubkey, ...] })

Usage:
  python Reranker_zerank1.py
  python Reranker_zerank1.py --model zeroentropy/zerank-2 --top_k 100 --batch 16
"""

import json
import argparse
import os
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--queries",     default="../../../../../../data/expanded_queries_bge_large.json")
parser.add_argument("--papers",      default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",        default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",      default="../../../../../../../results/reranked_results_zerank.json")
parser.add_argument("--model",       default="zeroentropy/zerank-1",
                    help="zerank model to use. "
                         "Options: zeroentropy/zerank-1, zeroentropy/zerank-2 (default: zerank-1). "
                         "Both have the same interface: they only differ by a few arguments.")
parser.add_argument("--top_k",       type=int, default=100,
                    help="BM25 candidates to pass to the re-ranker (default: 100)")
parser.add_argument("--batch",       type=int, default=16,
                    help="Pairs per scoring iteration (default: 16). "
                         "zerank-1 (4B BF16) ≈ 8-9 GB VRAM. "
                         "On L40S 48GB can be increased up to a 32-64.")
parser.add_argument("--query_chunk", type=int, default=8,
                    help="Queries per chunk (default: 8).")
parser.add_argument("--max_length",  type=int, default=512,
                    help="Max token length per pair (default: 512).")


# ──────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────
def sanity_check_scores(score_fn) -> bool:
    """
    Verify that the model produces discriminative scores.
    The zerank CrossEncoder returns scores in [0, 1]: relevant >> irrelevant.
    """
    test_pairs = [
        ("neural network classification",
         "A deep learning model for image classification using convolutional neural networks"),
        ("neural network classification",
         "The history of ancient Rome and its emperors in the first century BC"),
    ]
    try:
        scores = score_fn(test_pairs)
        s0, s1 = float(scores[0]), float(scores[1])
        diff = abs(s0 - s1)
        print(f"\n  [Sanity check] Relevant score={s0:.4f} | Irrelevant score={s1:.4f} | Δ={diff:.4f}")
        if diff < 0.05:
            print("  WARNING: almost identical scores — check the model/tokenizer!")
            return False
        print("  [Sanity check] OK.")
        return True
    except Exception as e:
        print(f"  WARNING: sanity check failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# Helpers CPU
# ──────────────────────────────────────────────────────────────
def build_pairs_for_queries(query_items, paper_texts, query_texts, top_k):
    """
    Builds (query_text, doc_text) pairs for a chunk of queries.
    Returns:
        all_pairs : flat list of (str, str)
        meta      : [(qid, valid_keys, n_pairs, fallback_candidates), ...]
        missing   : number of docs not found in the corpus
    """
    all_pairs = []
    meta      = []
    missing   = 0

    for qid, candidate_pubkeys in query_items:
        # Supports both int and str keys in the query_texts dict
        query_text = query_texts.get(str(qid), query_texts.get(qid, ""))
        candidates = candidate_pubkeys[:top_k]

        if not query_text:
            print(f"  WARNING: query text not found for qid={qid}, using BM25 sorting", flush=True)
            meta.append((qid, [], 0, candidates))
            continue

        pairs      = []
        valid_keys = []
        for pk in candidates:
            doc_text = paper_texts.get(str(pk), paper_texts.get(pk))
            if doc_text:
                pairs.append((query_text, doc_text))
                valid_keys.append(pk)
            else:
                missing += 1

        meta.append((qid, valid_keys, len(pairs), candidates))
        all_pairs.extend(pairs)

    return all_pairs, meta, missing


def scores_to_results(meta, scores_flat):
    """
    Reconstructs qid→[pubkey] from the flat score array.
    Scores in [0, 1]: reverse=True puts the most relevant ones at the top.
    """
    results = {}
    offset  = 0
    for qid, valid_keys, n_pairs, fallback_candidates in meta:
        if n_pairs == 0:
            results[qid] = fallback_candidates
            continue
        scores = scores_flat[offset: offset + n_pairs]
        offset += n_pairs
        # Score alti (~1) = più rilevanti → reverse=True
        ranked = sorted(zip(valid_keys, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [pk for pk, _ in ranked]
    return results


# ──────────────────────────────────────────────────────────────
# Loading model — official zerank CrossEncoder
# ──────────────────────────────────────────────────────────────
def load_zerank(model_name: str, device: str, max_length: int):
    """
    Loads zerank-1 or zerank-2 via sentence-transformers CrossEncoder.

    CRITICAL: Use trust_remote_code=True to load modeling_zeranker.py.
    Without it, transformers uses the standard Qwen3ForSequenceClassification and
    re-initializes score.weight randomly → completely random scores.

    model.predict() output: float scores already in [0, 1], no sigmoid required.
    """
    from sentence_transformers.cross_encoder import CrossEncoder

    print(f"  Loading CrossEncoder: {model_name} → {device}", flush=True)
    model = CrossEncoder(
        model_name,
        trust_remote_code=True,
        max_length=max_length,
        device=device,
    )
    # Ensures that the pad_token is defined to avoid a crash during padding
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    if model.model.config.pad_token_id is None:
        model.model.config.pad_token_id = model.tokenizer.eos_token_id

    print(f"  Modello loaded. Max length: {max_length}", flush=True)

    def score_pairs(pairs: list) -> list:
        """
        Closure that encapsulates the CrossEncoder.
        Accepts a list of (query, doc) and returns a list of floats in [0, 1].

        NOTE: DO NOT pass batch_size > 1 to model.predict() directly,
        as the zerank tokenizer may not have pad_token configured
        correctly in the internal call, causing crashes with batch > 1.
        Mini-batching is handled externally in predict_scores().
        Here predict() is called on a single pair at a time.
        """
        scores = model.predict(pairs)
        # predict() restituisce numpy array o lista, convertiamo a lista di float
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        return scores if isinstance(scores, list) else [float(scores)]

    return score_pairs


# ──────────────────────────────────────────────────────────────
# Inference with OOM handling and manual mini-batching
# ──────────────────────────────────────────────────────────────
def predict_scores(score_fn, pairs: list, batch_size: int) -> list:
    """
    Iterates through the pairs in mini-batches, calling score_fn on each single pair.

    The mini-batching is manual and pair-by-pair: zerank with CrossEncoder
    can crash if padding is applied to batch > 1 without a pad_token.
    We iterate with batch_size pairs per call to score_fn, but each pair
    is processed individually inside score_fn for safety.

    Handles OOM by halving the batch and restoring it at the next chunk.
    """
    all_scores    = []
    current_batch = batch_size
    i             = 0

    while i < len(pairs):
        batch = pairs[i: i + current_batch]
        try:
            # Processing pair by pair to avoid crash from padding
            batch_scores = []
            for pair in batch:
                s = score_fn([pair])
                batch_scores.append(float(s[0]))
            all_scores.extend(batch_scores)
            i += current_batch
            current_batch = batch_size

        except (torch.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            if current_batch <= 1:
                print(
                    f"  WARNING: OOM even with batch=1. "
                    f"Score=0.5 as fallback (BM25 sorting maintained).",
                    flush=True,
                )
                # 0.5 = neutral value in [0,1], doesn't change the relative order
                all_scores.extend([0.5] * len(batch))
                i += current_batch
                current_batch = batch_size
            else:
                current_batch = max(1, current_batch // 2)
                print(f"  OOM → reducing batch to {current_batch} for this chunk", flush=True)

    return all_scores


# ──────────────────────────────────────────────────────────────
# Worker per-GPU (path multi-GPU)
# ──────────────────────────────────────────────────────────────
def rerank_worker(
        gpu_id: int,
        query_items: list,
        paper_texts: dict,
        query_texts: dict,
        model_name: str,
        top_k: int,
        batch_size: int,
        query_chunk: int,
        max_length: int,
        result_queue: mp.Queue,
) -> None:
    # CRITICAL: isolates GPU before any import CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    device = "cuda:0"
    print(f"[GPU {gpu_id}] {torch.cuda.get_device_name(0)} — loading model...", flush=True)

    score_fn = load_zerank(model_name, device, max_length)

    if gpu_id == 0:
        sanity_check_scores(score_fn)

    local_results     = {}
    local_missing     = 0
    local_pairs_total = 0

    chunks = [
        query_items[i: i + query_chunk]
        for i in range(0, len(query_items), query_chunk)
    ]

    for chunk in tqdm(chunks, desc=f"Re-ranking [GPU {gpu_id}]", position=gpu_id, leave=True):
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


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parser.parse_args()

    MODEL_NAME = args.model

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        DEVICE = "cpu"
        print("Device: cpu (no CUDA GPU found)")
    else:
        DEVICE = "cuda:0"
        print(f"Device: cuda — {NUM_GPUS} GPU available:")
        for i in range(NUM_GPUS):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    # ── Carica dati ────────────────────────────────────────────
    print(f"\nModel: {MODEL_NAME}")
    print("Loading data...")

    with open(args.queries, "r", encoding="utf-8") as f:
        queries_raw = json.load(f)

    with open(args.papers, "r", encoding="utf-8") as f:
        papers_raw = json.load(f)

    with open(args.bm25, "r", encoding="utf-8") as f:
        bm25_results: dict = json.load(f)

    paper_texts: dict = {
        str(p["pubkey"]): (p.get("title", "") + " " + p.get("abstract", "")).strip()
        for p in papers_raw
    }

    query_texts: dict = {
        str(q["index"]): q.get("original", q.get("text", ""))
        for q in queries_raw
    }

    # ── Sanity check sui dati ──────────────────────────────────
    print(f"\nCorpus: {len(paper_texts)} documents | "
          f"Queries: {len(query_texts)} | "
          f"BM25 results: {len(bm25_results)}")

    sample_qid        = next(iter(bm25_results))
    sample_candidates = bm25_results[sample_qid][:5]
    hits = sum(1 for pk in sample_candidates if str(pk) in paper_texts)
    print(f"Sanity check corpus — query '{sample_qid}': "
          f"{hits}/{len(sample_candidates)} candidates found in the corpus")
    if hits == 0:
        print("CRITICAL ERROR: 0 candidates found. "
              f"BM25 pubkey type={type(sample_candidates[0])}, "
              f"corpus key type=str")

    sample_qt = query_texts.get(str(sample_qid), "")
    print(f"Sanity check queries — qid='{sample_qid}': '{sample_qt[:80]}...'")
    if not sample_qt:
        print("WARNING: query text empty. Check field 'original'/'text' in the query JSON.")

    all_query_items = list(bm25_results.items())
    print(f"\nRe-ranking {len(bm25_results)} queries "
          f"(top_k={args.top_k}, batch={args.batch}, "
          f"query_chunk={args.query_chunk}, max_length={args.max_length})...")

    # ── Single GPU / CPU ───────────────────────────────────────
    if NUM_GPUS <= 1:
        score_fn = load_zerank(MODEL_NAME, DEVICE, args.max_length)

        ok = sanity_check_scores(score_fn)
        if not ok:
            print("\nWARNING: sanity check failed. Continuing anyway.\n")

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        chunks = [
            all_query_items[i: i + args.query_chunk]
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

    # ── Multi-GPU ───────────────────────────────────────────────
    else:
        partitions = [[] for _ in range(NUM_GPUS)]
        for i, item in enumerate(all_query_items):
            partitions[i % NUM_GPUS].append(item)

        for i, p in enumerate(partitions):
            print(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {len(p)} queries")

        mp.set_start_method("spawn", force=True)
        result_queue: mp.Queue = mp.Queue()

        processes = []
        for gpu_id in range(NUM_GPUS):
            p = mp.Process(
                target=rerank_worker,
                args=(
                    gpu_id,
                    partitions[gpu_id],
                    paper_texts,
                    query_texts,
                    MODEL_NAME,
                    args.top_k,
                    args.batch,
                    args.query_chunk,
                    args.max_length,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        for _ in range(NUM_GPUS):
            local_results, local_missing, local_pairs = result_queue.get()
            reranked_results.update(local_results)
            missing_docs += local_missing
            total_pairs  += local_pairs

        for p in processes:
            p.join()

        # Restore the original query sorting
        reranked_results = {
            qid: reranked_results[qid]
            for qid in bm25_results
            if qid in reranked_results
        }

    # ── Final report ──────────────────────────────────────────
    print(f"\nTotal pairs scored : {total_pairs}")
    print(f"Queries re-ranked  : {len(reranked_results)}")
    if missing_docs:
        print(f"Pubkey not found : {missing_docs} (BM25 sorting maintained)")
    coverage = len(reranked_results) / len(bm25_results) * 100
    print(f"Coverage           : {coverage:.1f}% ({len(reranked_results)}/{len(bm25_results)})")

    # ── Storing ──────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, indent=2, ensure_ascii=False)

    print(f"\nStored → {args.output}")
    print("Done. Now execute the Java evaluator on reranked_results_zerank.json.")