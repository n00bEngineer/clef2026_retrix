"""
Reranker_qwen3.py — Qwen3-Reranker-0.6B re-ranking of BM25 results
===================================================================
Qwen3-Reranker-0.6B is a GENERATIVE model (AutoModelForCausalLM), not a
classifier. Do NOT use sentence-transformers CrossEncoder.

Official mechanism (from HuggingFace Qwen/Qwen3-Reranker-0.6B):
  1. Build prompt with fixed system prefix/suffix + <Instruct>/<Query>/<Document>
  2. Forward pass -> logits[:, -1, :]  (last generated token)
  3. Extract logits for token "yes" and token "no"
  4. log_softmax over those two logits -> exp(logit "yes") = score in [0, 1]

Requires: transformers >= 4.51.0

Reads:
  - data/expanded_queries_*.json   (queries, "original" field)
  - data/collection_data.json      (corpus, title+abstract)
  - results/bm25_results.json      (Java output: { qid -> [pubkey, ...] })

Writes:
  - results/reranked_results.json  (same format: { qid -> [pubkey, ...] })

Usage:
  python Reranker_qwen3.py
  python Reranker_qwen3.py --top_k 100 --batch 8 --query_chunk 8
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
MODEL_NAME = "Qwen/Qwen3-Reranker-4B"

parser = argparse.ArgumentParser()
parser.add_argument("--queries",     default="../../../../../../data/expanded_queries_bge_large.json")
parser.add_argument("--papers",      default="../../../../../../data/collection_data.json")
parser.add_argument("--bm25",        default="../../../../../../../results/bm25_results.json")
parser.add_argument("--output",      default = f"../../../../../../../results/reranked_results_{MODEL_NAME.split("/")[-1]}.json")
parser.add_argument("--top_k",       type=int, default=100,
                    help="BM25 candidates to pass to the re-ranker (default: 100)")
parser.add_argument("--batch",       type=int, default=32,
                    help="Pairs per GPU forward pass (default: 8). "
                         "Qwen3-0.6B fp16 ≈ 1.2 GB VRAM. "
                         "On RTX 3090 24GB with max_length=512 you can increase up to 32.")
parser.add_argument("--query_chunk", type=int, default=8,
                    help="Queries per chunk (default: 8).")
parser.add_argument("--max_length",  type=int, default=512,
                    help="Maximum token length per pair (default: 512). "
                         "The model supports 32k, but 512 is enough for title+abstract.")


# Task-specific instruction for scientific papers
# Using a relevant instruction usually improves metrics by 1-5% (official docs)
TASK_INSTRUCTION = (
    "Given a short user query (like a tweet) and a formal academic paper, judge how relevant the paper is to the query."
    "Return a higher score for relevant papers, and a lower score for irrelevant papers."
)

# ──────────────────────────────────────────────────────────────
# Prompt formatting - official Qwen3-Reranker API
# ──────────────────────────────────────────────────────────────
# Prefix and suffix are fixed and defined by official docs.
# They are tokenized once when loading the model.
PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    "the Instruct provided. Return an higher score for relevant papers and a lower score for irrelevant papers."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_pair(query: str, doc: str, instruction: str = TASK_INSTRUCTION) -> str:
    """
    Format a (query, doc) pair in the Qwen3-Reranker expected format.
    Template: <Instruct>: ...\n<Query>: ...\n<Document>: ...
    """
    return (
        f"<Instruct>: {instruction}\n"
        f"<Query>: {query}\n"
        f"<Document>: {doc}"
    )


# ──────────────────────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────────────────────
def sanity_check_scores(score_fn) -> bool:
    """
    Verify that the model produces discriminative scores.
    A relevant document should have a significantly higher score
    than an irrelevant one.
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
        print(f"\n  [Sanity check] Relevant score={s0:.4f} | Irrelevant score={s1:.4f} | Delta={diff:.4f}")
        if diff < 0.05:
            print("  WARNING: scores are almost identical - check model/tokenizer!")
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
    Build (query_text, doc_text) pairs for a query chunk.
    Returns:
      all_pairs : flat list of (str, str)
      meta      : [(qid, valid_keys, n_pairs, fallback_candidates), ...]
      missing   : number of docs not found in corpus
    """
    all_pairs = []
    meta      = []
    missing   = 0

    for qid, candidate_pubkeys in query_items:
        query_text = query_texts.get(str(qid), "")
        candidates = candidate_pubkeys[:top_k]

        if not query_text:
            print(f"  WARNING: query text not found for qid={qid}, using BM25 order", flush=True)
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
    Rebuild qid->[pubkey] from the flat score array.
    """
    results = {}
    offset  = 0
    for qid, valid_keys, n_pairs, fallback_candidates in meta:
        if n_pairs == 0:
            results[qid] = fallback_candidates
            continue
        scores = scores_flat[offset: offset + n_pairs]
        offset += n_pairs
        ranked = sorted(zip(valid_keys, scores), key=lambda x: -x[1])
        results[qid] = [pk for pk, _ in ranked]
    return results


# ──────────────────────────────────────────────────────────────
# Model loading - official AutoModelForCausalLM
# ──────────────────────────────────────────────────────────────
def load_qwen3_reranker(device: str, max_length: int):
    """
    Load Qwen3-Reranker-0.6B with AutoModelForCausalLM.

    This model is GENERATIVE: it produces a score by extracting the last-token logit
    for tokens "yes" and "no", NOT through a classification head.
    di classificazione. Usare CrossEncoder o AutoModelForSequenceClassification
    would produce wrong scores (or compatibility errors).

    Richiede transformers >= 4.51.0.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"  Loading tokenizer: {MODEL_NAME}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        padding_side="left",   # CRITICAL: left padding for causal models
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading model: {MODEL_NAME} → {device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()

    # Token IDs for "yes" and "no" - fixed in the Qwen3 vocabulary
    token_true_id  = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    print(f"  Token IDs — yes: {token_true_id}, no: {token_false_id}", flush=True)
    if token_true_id == tokenizer.unk_token_id or token_false_id == tokenizer.unk_token_id:
        raise ValueError(
            "Token 'yes' or 'no' not found in vocabulary. "
            "Check that the tokenizer is correct for Qwen3-Reranker-0.6B."
        )

    # Pre-tokenize prefix and suffix (constant for all pairs)
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)
    overhead = len(prefix_tokens) + len(suffix_tokens)
    # max_length for content (query+doc), excluding prefix/suffix
    content_max_length = max_length - overhead

    print(f"  Prefix tokens: {len(prefix_tokens)}, Suffix tokens: {len(suffix_tokens)}, "
          f"Content max length: {content_max_length}", flush=True)

    def score_pairs(pairs: list) -> list:
        """
        Closure wrapping model, tokenizer, and token IDs.
        Accepts a list of (query, doc) and returns list of floats in [0,1].

        Official Qwen3-Reranker logic:
          logits[:, -1, :]              -> last-token logits (next token position)
          log_softmax([false, true])    -> normalized log-probability over yes/no
          exp(log_p_yes)                → score in [0, 1]
        """
        formatted = [format_pair(q, d) for q, d in pairs]

        # Tokenize content (query+doc) with truncation
        inputs = tokenizer(
            formatted,
            padding=False,
            truncation=True,
            max_length=content_max_length,
            return_attention_mask=False,
            add_special_tokens=False,
        )

        # Add prefix and suffix to each sequence
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = prefix_tokens + ids + suffix_tokens

        # Batch padding (left-padding, already configured in tokenizer)
        inputs = tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=max_length,
        )
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)

        with torch.no_grad():
            logits = model(**inputs).logits  # [B, seq_len, vocab_size]

        # Last-token logit: what the model would "produce" after suffix
        last_logits = logits[:, -1, :]  # [B, vocab_size]

        true_vec  = last_logits[:, token_true_id]   # [B]
        false_vec = last_logits[:, token_false_id]  # [B]

        # log_softmax on [false, true] -> exp of "true" logit = P(yes)
        stacked = torch.stack([false_vec, true_vec], dim=1)  # [B, 2]
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        scores = log_probs[:, 1].exp().tolist()  # P(yes) for each pair

        return scores if isinstance(scores, list) else [scores]

    return score_pairs


# ──────────────────────────────────────────────────────────────
# Inference with OOM handling
# ──────────────────────────────────────────────────────────────
def predict_scores(score_fn, pairs: list, batch_size: int) -> list:
    """
    Iterate pairs by batch, call score_fn, and aggregate results.

    current_batch is local: an OOM on this chunk does not lower the
    batch size for following chunks.
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
            current_batch = batch_size  # restore after success

        except (torch.OutOfMemoryError, RuntimeError) as e:
            is_oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
            if not is_oom:
                raise
            torch.cuda.empty_cache()
            if current_batch <= 1:
                print(
                    f"  WARNING: OOM even with batch=1 on {len(batch)} pairs. "
                    f"Score=0 as fallback (BM25 order preserved).",
                    flush=True,
                )
                all_scores.extend([0.0] * len(batch))
                i += current_batch
                current_batch = batch_size
            else:
                current_batch = max(1, current_batch // 2)
                print(f"  OOM -> reducing batch to {current_batch} for this chunk", flush=True)

    return all_scores


# ──────────────────────────────────────────────────────────────
# Per-GPU worker (multi-GPU path)
# ──────────────────────────────────────────────────────────────
def rerank_worker(
        gpu_id: int,
        query_items: list,
        paper_texts: dict,
        query_texts: dict,
        top_k: int,
        batch_size: int,
        query_chunk: int,
        max_length: int,
        result_queue: mp.Queue,
) -> None:
    # CRITICAL: isolate GPU before any CUDA import
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch

    device = "cuda:0"
    print(f"[GPU {gpu_id}] {torch.cuda.get_device_name(0)} - loading model...", flush=True)

    score_fn = load_qwen3_reranker(device, max_length)

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

    NUM_GPUS = torch.cuda.device_count()
    if NUM_GPUS == 0:
        DEVICE = "cpu"
        print("Device: cpu (no CUDA GPU found)")
    else:
        DEVICE = "cuda:0"
        print(f"Device: cuda - {NUM_GPUS} GPUs available:")
        for i in range(NUM_GPUS):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")

    # -- Load data --
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

    # -- Data sanity checks --
    print(f"\nCorpus: {len(paper_texts)} documents | "
          f"Queries: {len(query_texts)} | "
          f"BM25 results: {len(bm25_results)}")

    sample_qid        = next(iter(bm25_results))
    sample_candidates = bm25_results[sample_qid][:5]
    hits = sum(1 for pk in sample_candidates if str(pk) in paper_texts)
    print(f"Sanity check corpus — query '{sample_qid}': "
          f"{hits}/{len(sample_candidates)} candidates found in corpus")
    if hits == 0:
        print("CRITICAL ERROR: 0 candidates found. "
              f"bm25 pubkey type={type(sample_candidates[0])}, "
              f"corpus key type=str")

    sample_qt = query_texts.get(str(sample_qid), "")
    print(f"Sanity check queries — qid='{sample_qid}': '{sample_qt[:80]}...'")
    if not sample_qt:
        print("WARNING: empty query text. Check 'original'/'text' field in query JSON.")

    all_query_items = list(bm25_results.items())
    print(f"\nRe-ranking {len(bm25_results)} queries "
          f"(top_k={args.top_k}, batch={args.batch}, "
          f"query_chunk={args.query_chunk}, max_length={args.max_length})...")

    # -- Single GPU / CPU --
    if NUM_GPUS <= 1:
        score_fn = load_qwen3_reranker(DEVICE, args.max_length)

        ok = sanity_check_scores(score_fn)
        if not ok:
            print("\nWARNING: sanity check failed. Proceeding anyway.\n")

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

    # -- Multi-GPU --
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

        # Restore original order
        reranked_results = {
            qid: reranked_results[qid]
            for qid in bm25_results
            if qid in reranked_results
        }

    # -- Final report --
    print(f"\nTotal pairs scored : {total_pairs}")
    print(f"Queries re-ranked  : {len(reranked_results)}")
    if missing_docs:
        print(f"Pubkeys not found : {missing_docs} (BM25 order preserved)")
    coverage = len(reranked_results) / len(bm25_results) * 100
    print(f"Coverage           : {coverage:.1f}% ({len(reranked_results)}/{len(bm25_results)})")

    # -- Save --
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reranked_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved -> {args.output}")
    print("Done. Now run the Java evaluator pointing to reranked_results.json.")
