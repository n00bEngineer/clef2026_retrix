"""
Reranker_Nemotron_1b.py — Llama-Nemotron-Rerank-1B re-ranking of BM25 results
======================================================================================
nvidia/llama-nemotron-rerank-1b-v2 is a cross-encoder (AutoModelForSequenceClassification)
fine-tuned with bidirectional attention on Llama-3.2-1B.

Official mechanism (from HuggingFace nvidia/llama-nemotron-rerank-1b-v2):
    1. Format the pair with the template: "question:{q} \n \n passage:{p}"
    2. Tokenize as a single sequence (NOT as a pair)
    3. Forward pass → model(**batch).logits shape [B, 1]
    4. logits.view(-1) → scalar score per pair (raw logit, not bounded)
    5. Optional: sigmoid to convert into probability [0,1]

Key differences compared to Qwen3-Reranker:
    - AutoModelForSequenceClassification, NOT AutoModelForCausalLM
    - No fixed prefix/suffix to pre-tokenize
    - No yes/no logic on vocabulary: the score is already in the output logit
    - torch_dtype=torch.bfloat16 (not float16: the model is native BF16)
    - padding_side="left" maintained (decoder model with bidirectional attention)
    - Requires transformers >= 4.44

Reads:
  - data/expanded_queries_*.json   (query, field "original")
  - data/collection_data.json      (corpus, title+abstract)
  - results/bm25_results.json      (output Java: { qid → [pubkey, ...] })

Writed:
  - results/reranked_results_nemotron.json  (same format: { qid → [pubkey, ...] })

Usage:
  python LASTReranker_Nemotron_1b.py
  python LASTReranker_Nemotron_1b.py --top_k 100 --batch 32 --query_chunk 8
"""

import argparse
import json
import math
import os
import traceback

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
parser.add_argument("--output",      default="../../../../../../../results/reranked_results_nemotron.json")
parser.add_argument("--top_k",       type=int, default=100,
                    help="BM25 candidates to pass to the re-ranker (default: 100)")
parser.add_argument("--batch",       type=int, default=32,
                    help="Pairs per GPU forward pass (default: 32). "
                         "Nemotron-1B BF16 ≈ 2.5 GB VRAM. "
                         "On RTX 3090 24GB with max_length=512 can be increased up to 64.")
parser.add_argument("--query_chunk", type=int, default=8,
                    help="Queries per chunk (default: 8).")
parser.add_argument("--max_length",  type=int, default=512,
                    help="Max tokens per pair (default: 512). "
                         "The model supports up to 8192 but 512 is sufficient for titolo+abstract.")

MODEL_NAME = "nvidia/llama-nemotron-rerank-1b-v2"


# ──────────────────────────────────────────────────────────────
# Input formatting — official Nemotron template
# ──────────────────────────────────────────────────────────────
def format_pair(query: str, doc: str) -> str:
    """
    Official nvidia/llama-nemotron-rerank-1b-v2 template.
    The input is a single sequence (NOT pairs from tokenizer), as from docs.
    """
    return f"question:{query} \n \n passage:{doc}"


# ──────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────
def sanity_check_scores(score_fn) -> bool:
    """
    Verify that the model produces discriminative scores.
    Nemotron returns raw logits: relevant >> irrelevant.
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
        print(f"\n  [Sanity check] Relevant score={s0:.4f} | Irrilevant score={s1:.4f} | Δ={diff:.4f}")
        # Nemotron produces raw logits (range ~[-30, +30]): diff > 1.0 is already a good sign
        if diff < 1.0:
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
        query_text = query_texts.get(str(qid), "")
        candidates = candidate_pubkeys[:top_k]

        if not query_text:
            print(f"  WARNING: query text not found for qid={qid}, using BM25 sorting", flush=True)
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


def scores_to_results(meta, scores_flat):
    """
    Rebuilds qid→[pubkey] from the flat score array.
    """
    results = {}
    offset  = 0
    for qid, valid_keys, n_pairs, fallback_candidates in meta:
        if n_pairs == 0:
            results[qid] = fallback_candidates
            continue
        scores = scores_flat[offset: offset + n_pairs]
        offset += n_pairs
        ranked = sorted(zip(valid_keys, scores), key=lambda x: x[1], reverse=True)
        results[qid] = [pk for pk, _ in ranked]
    return results


def build_query_tasks(
        query_items: list,
        max_queries_per_task: int,
        num_gpus: int,
) -> tuple[list[list], int]:
    """
    Splits the work into finer tasks than the single-GPU path.
    In multi-GPU we want more tasks than available GPUs, so workers can
    pick up new work as soon as they finish and avoid unbalanced fixed queues.
    """
    if max_queries_per_task <= 0:
        raise ValueError("--query_chunk must be greater than 0.")
    if not query_items:
        return [], max_queries_per_task

    target_tasks = max(num_gpus * 4, 1)
    effective_queries_per_task = max(
        1,
        min(max_queries_per_task, math.ceil(len(query_items) / target_tasks)),
    )
    tasks = [
        query_items[i: i + effective_queries_per_task]
        for i in range(0, len(query_items), effective_queries_per_task)
    ]
    return tasks, effective_queries_per_task


# ──────────────────────────────────────────────────────────────
# Loading model — AutoModelForSequenceClassification
# ──────────────────────────────────────────────────────────────
def load_nemotron_reranker(device: str, max_length: int):
    """
    Loads nvidia/llama-nemotron-rerank-1b-v2 via AutoModelForSequenceClassification.

    Architecture: cross-encoder (Llama-3.2-1B fine-tuned) with bidirectional
    attention and binary classification head. Output: raw scalar logit per
    pair, NOT a yes/no vector on vocabulary.

    DO NOT use AutoModelForCausalLM (produces incorrect output on this model).
    Requires transformers >= 4.44.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"  Loading tokenizer: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="left",   # left padding — coherente with decoder architecture
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {MODEL_NAME} → {device}", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,   # BF16: dtype native of the model (not float16)
        device_map=device,
    )
    # Syncronyze pad_token_id even in model config
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.eos_token_id
    model.eval()

    def score_pairs(pairs: list) -> list:
        """
        Closure that encapsulates model and tokenizer.
        Accepts list of (query, doc) and returns list of floats (raw logit).

        Official Nemotron logic:
          texts = [format_pair(q, d) for q, d in pairs]
          tokenizer(texts, ...) → batch_dict (single sequence, NOT pair)
          model(**batch_dict).logits → shape [B, 1]
          logits.view(-1) → [B] raw score
        """
        texts = [format_pair(q, d) for q, d in pairs]

        batch_dict = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

        with torch.inference_mode():
            logits = model(**batch_dict).logits  # [B, 1]
            scores = torch.sigmoid(logits.view(-1)).cpu().tolist()

        return scores if isinstance(scores, list) else [scores]

    return score_pairs


# ──────────────────────────────────────────────────────────────
# Inference with OOM handling
# ──────────────────────────────────────────────────────────────
def predict_scores(score_fn, pairs: list, batch_size: int) -> list:
    """
    Iterates through pairs in batches, calls score_fn and aggregates the results.
    Handles OOM by halving the batch and restoring it at the next chunk.
    """
    all_scores    = []
    current_batch = batch_size
    i             = 0

    while i < len(pairs):
        batch = pairs[i: i + current_batch]
        try:
            scores = score_fn(batch)
            all_scores.extend(scores)
            i += current_batch
            current_batch = batch_size

        except (torch.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            if current_batch <= 1:
                print(
                    f"  WARNING: OOM even with batch=1 on {len(batch)} pairs. "
                    f"Score=0 as fallback (BM25 sorting maintained).",
                    flush=True,
                )
                all_scores.extend([0.0] * len(batch))
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
        task_queue: mp.Queue,
        paper_texts: dict,
        query_texts: dict,
        top_k: int,
        batch_size: int,
        max_length: int,
        result_queue: mp.Queue,
) -> None:
    # CRITICAL: isolate GPU before any CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    device = "cuda:0"
    print(f"[GPU {gpu_id}] {torch.cuda.get_device_name(0)} — loading model...", flush=True)

    try:
        score_fn = load_nemotron_reranker(device, max_length)

        if gpu_id == 0:
            sanity_check_scores(score_fn)

        local_results         = {}
        local_missing         = 0
        local_pairs_total     = 0
        local_tasks_processed = 0

        while True:
            chunk = task_queue.get()
            if chunk is None:
                break

            all_pairs, meta, missing = build_pairs_for_queries(
                chunk, paper_texts, query_texts, top_k
            )
            local_missing += missing

            if not all_pairs:
                for qid, _, _, fallback in meta:
                    local_results[qid] = fallback
                local_tasks_processed += 1
                continue

            local_pairs_total += len(all_pairs)
            scores_flat = predict_scores(score_fn, all_pairs, batch_size)
            local_results.update(scores_to_results(meta, scores_flat))
            local_tasks_processed += 1

        result_queue.put({
            "gpu_id": gpu_id,
            "results": local_results,
            "missing": local_missing,
            "pairs": local_pairs_total,
            "tasks": local_tasks_processed,
            "error": None,
        })
    except Exception:
        result_queue.put({
            "gpu_id": gpu_id,
            "results": {},
            "missing": 0,
            "pairs": 0,
            "tasks": 0,
            "error": traceback.format_exc(),
        })


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    args = parser.parse_args()

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        DEVICE = "cpu"
        print("Device: cpu (no CUDA GPU found)")
    else:
        DEVICE = "cuda:0"
        print(f"Device: cuda — {NUM_GPUS} GPU available:")
        for i in range(NUM_GPUS):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    # ── Load data ────────────────────────────────────────────
    print("\nLoading data...")

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

    # ── Sanity check on data ──────────────────────────────────
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
        print("WARNING: query text empty. Check field 'original'/'text' nel JSON query.")

    all_query_items = list(bm25_results.items())
    print(f"\nRe-ranking {len(bm25_results)} queries "
          f"(top_k={args.top_k}, batch={args.batch}, "
          f"query_chunk={args.query_chunk}, max_length={args.max_length})...")

    # ── Single GPU / CPU ───────────────────────────────────────
    if NUM_GPUS <= 1:
        score_fn = load_nemotron_reranker(DEVICE, args.max_length)

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
        tasks, effective_query_chunk = build_query_tasks(
            all_query_items,
            args.query_chunk,
            NUM_GPUS,
        )
        print(
            "  Multi-GPU scheduler: "
            f"{len(tasks)} dynamic tasks, up  to {effective_query_chunk} query/task"
        )

        mp.set_start_method("spawn", force=True)
        task_queue: mp.Queue = mp.Queue()
        result_queue: mp.Queue = mp.Queue()

        for task in tasks:
            task_queue.put(task)
        for _ in range(NUM_GPUS):
            task_queue.put(None)

        processes = []
        for gpu_id in range(NUM_GPUS):
            p = mp.Process(
                target=rerank_worker,
                args=(
                    gpu_id,
                    task_queue,
                    paper_texts,
                    query_texts,
                    args.top_k,
                    args.batch,
                    args.max_length,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        reranked_results: dict = {}
        missing_docs  = 0
        total_pairs   = 0

        worker_task_counts = {}
        worker_errors = []

        for _ in range(NUM_GPUS):
            payload = result_queue.get()
            if payload["error"] is not None:
                worker_errors.append((payload["gpu_id"], payload["error"]))
                continue

            reranked_results.update(payload["results"])
            missing_docs += payload["missing"]
            total_pairs  += payload["pairs"]
            worker_task_counts[payload["gpu_id"]] = payload["tasks"]

        for p in processes:
            p.join()

        if worker_errors:
            error_lines = []
            for gpu_id, error in worker_errors:
                error_lines.append(f"[GPU {gpu_id}] {error}")
            raise RuntimeError(
                "One or more multi-GPU worker failed:\n" + "\n".join(error_lines)
            )

        for gpu_id in range(NUM_GPUS):
            tasks_done = worker_task_counts.get(gpu_id, 0)
            print(f"  GPU {gpu_id} processed {tasks_done} dynamic tasks")

        # Restore original sorting
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

    # ── Store ──────────────────────────────────────────────────
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, indent=2, ensure_ascii=False)

    print(f"\nSalvato → {args.output}")
    print("Done. Now run the Java evaluator on reranked_results_nemotron.json.")
