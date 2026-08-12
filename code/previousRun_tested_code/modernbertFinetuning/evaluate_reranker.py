"""
evaluate_reranker.py

Evaluates the fine-tuned reranker on a dev/test set.
Computes: MRR@5, MRR@10, NDCG@10, MAP, Precision@1

Can be used to:
    1. Evaluate checkpoints during training
    2. Compare the fine-tuned model vs the baseline (e.g., GTE-ModernBERT or Nemotron)

Usage:
    python evaluate_reranker.py \
        --checkpoint ./checkpoints/modernbert-reranker/best_checkpoint \
        --dev_file   training_data/dev_pairs.jsonl

    # Or on a complete BM25 run:
    python evaluate_reranker.py \
        --checkpoint ./checkpoints/modernbert-reranker/best_checkpoint \
        --corpus     corpus.tsv \
        --queries    queries.tsv \
        --qrels      qrels.txt \
        --bm25_run   bm25_run.txt \
    --rerank_depth 100
"""

import json
import math
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_peft_model(checkpoint_path: str, device):
    """
    Load a fine-tuned model with PEFT/LoRA.
    Handles both PEFT checkpoint and merged models.
    """
    checkpoint_path = Path(checkpoint_path)

    # Check if it is a PEFT checkpoint (has adapter_config.json)
    if (checkpoint_path / "adapter_config.json").exists():
        log.info(f"Loadign PEFT model from {checkpoint_path}")
        peft_config  = PeftConfig.from_pretrained(str(checkpoint_path))
        base_model   = AutoModelForSequenceClassification.from_pretrained(
            peft_config.base_model_name_or_path,
            num_labels=1,
            torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))
    else:
        log.info(f"Loading merged model from {checkpoint_path}")
        model     = AutoModelForSequenceClassification.from_pretrained(
            str(checkpoint_path), num_labels=1, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_path))

    model = model.to(device)
    model.eval()
    return model, tokenizer


class PairDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str, str, int]]):
        """pairs: [(qid, doc_id, text, relevance)]"""
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def score_pairs(
        model,
        tokenizer,
        query: str,
        docs: list[str],
        max_length: int = 512,
        batch_size: int = 32,
        device=None,
) -> list[float]:
    """Computes the score for a query and a document list."""
    scores = []

    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i : i + batch_size]
        enc = tokenizer(
            [query] * len(batch_docs),
            batch_docs,
            max_length=max_length,
            truncation="only_second",
            padding=True,
            return_tensors="pt",
            )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**enc)

        batch_scores = out.logits.squeeze(-1).float().cpu().tolist()
        if isinstance(batch_scores, float):
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

    return scores


def compute_metrics(
        qid_to_results: dict[str, list[tuple[float, int]]],
        cutoffs: list[int] = [5, 10],
) -> dict:
    """
    qid_to_results: {qid: [(score, relevance), ...]} already sorted by decreasing score
    """
    mrr_scores   = {k: [] for k in cutoffs}
    ndcg_scores  = {k: [] for k in cutoffs}
    ap_scores    = []
    p1_scores    = []

    for qid, results in qid_to_results.items():
        # MRR@k
        for k in cutoffs:
            mrr = 0.0
            for rank, (score, rel) in enumerate(results[:k], start=1):
                if rel >= 1:
                    mrr = 1.0 / rank
                    break
            mrr_scores[k].append(mrr)

        # NDCG@k
        for k in cutoffs:
            dcg  = sum(
                (2 ** rel - 1) / math.log2(rank + 1)
                for rank, (score, rel) in enumerate(results[:k], start=1)
            )
            ideal = sorted([rel for _, rel in results], reverse=True)[:k]
            idcg = sum(
                (2 ** rel - 1) / math.log2(rank + 1)
                for rank, rel in enumerate(ideal, start=1)
            )
            ndcg_scores[k].append(dcg / idcg if idcg > 0 else 0.0)

        # AP
        n_rel = 0
        ap    = 0.0
        for rank, (score, rel) in enumerate(results, start=1):
            if rel >= 1:
                n_rel += 1
                ap    += n_rel / rank
        total_rel = sum(1 for _, rel in results if rel >= 1)
        ap_scores.append(ap / total_rel if total_rel > 0 else 0.0)

        # P@1
        p1_scores.append(1.0 if results and results[0][1] >= 1 else 0.0)

    metrics = {}
    for k in cutoffs:
        metrics[f"MRR@{k}"]   = sum(mrr_scores[k])  / len(mrr_scores[k])
        metrics[f"NDCG@{k}"]  = sum(ndcg_scores[k]) / len(ndcg_scores[k])
    metrics["MAP"]  = sum(ap_scores)  / len(ap_scores)
    metrics["P@1"]  = sum(p1_scores)  / len(p1_scores)
    metrics["n_queries"] = len(qid_to_results)

    return metrics


def evaluate_on_run(
        model, tokenizer, device,
        corpus_path, queries_path, qrels_path, bm25_run_path,
        rerank_depth: int = 100, max_length: int = 512, batch_size: int = 32,
):
    """VEvaluate on a complete BM25 run"""
    from prepare_training_data import load_corpus, load_queries, load_qrels, load_run, clean_tweet

    corpus  = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    qrels   = load_qrels(qrels_path)
    bm25    = load_run(bm25_run_path, top_k=rerank_depth)

    qid_to_results = {}

    for qid, query_text in tqdm(queries.items(), desc="Evaluation"):
        if qid not in bm25:
            continue

        retrieved = bm25[qid][:rerank_depth]
        docs      = [clean_tweet(corpus.get(d, "")) for d in retrieved]
        query     = clean_tweet(query_text)

        scores = score_pairs(model, tokenizer, query, docs,
                             max_length=max_length, batch_size=batch_size, device=device)

        doc_rels = qrels.get(qid, {})
        ranked   = sorted(
            [(s, doc_rels.get(did, 0)) for s, did in zip(scores, retrieved)],
            key=lambda x: x[0], reverse=True
        )
        qid_to_results[qid] = ranked

    return compute_metrics(qid_to_results)


def evaluate_on_pairs(model, tokenizer, device, dev_file: str,
                      max_length: int = 512, batch_size: int = 32):
    """Evaluate on dev_pairs.jsonl."""
    qid_to_results = defaultdict(list)

    with open(dev_file, encoding="utf-8") as f:
        examples = [json.loads(l) for l in f]

    for i, ex in enumerate(tqdm(examples, desc="Valutazione")):
        qid   = ex.get("qid", str(i))
        query = ex["query"]
        docs  = [ex["positive"]] + ex["negatives"]
        rels  = [1] + [0] * len(ex["negatives"])

        scores = score_pairs(model, tokenizer, query, docs,
                             max_length=max_length, batch_size=batch_size, device=device)

        ranked = sorted(zip(scores, rels), key=lambda x: x[0], reverse=True)
        qid_to_results[qid].extend(ranked)

    return compute_metrics(dict(qid_to_results))


def main():
    parser = argparse.ArgumentParser(description="evaluate fine-tuned reranker")
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--dev_file",    default=None, help="dev_pairs.jsonl")
    parser.add_argument("--corpus",      default=None)
    parser.add_argument("--queries",     default=None)
    parser.add_argument("--qrels",       default=None)
    parser.add_argument("--bm25_run",    default=None)
    parser.add_argument("--rerank_depth", type=int, default=100)
    parser.add_argument("--max_length",  type=int, default=512)
    parser.add_argument("--batch_size",  type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_peft_model(args.checkpoint, device)

    if args.dev_file:
        log.info("Evaluation on dev_pairs.jsonl")
        metrics = evaluate_on_pairs(model, tokenizer, device,
                                    args.dev_file, args.max_length, args.batch_size)
    elif args.bm25_run:
        log.info("Evaluation on BM25 run")
        metrics = evaluate_on_run(
            model, tokenizer, device,
            args.corpus, args.queries, args.qrels, args.bm25_run,
            args.rerank_depth, args.max_length, args.batch_size,
        )
    else:
        parser.error("Specify --dev_file or --bm25_run")

    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:15s}: {v:.4f}")
        else:
            print(f"  {k:15s}: {v}")
    print("="*50)


if __name__ == "__main__":
    main()