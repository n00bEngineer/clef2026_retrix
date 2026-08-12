"""
Fine-tuning of answerdotai/ModernBERT-large as reranker
for dataset with tweet-style queries and scientific corpus.

Basato su: https://huggingface.co/blog/train-reranker
Framework: sentence-transformers >= 4.0

Expected input file structure:
  - train.jsonl / dev.jsonl / test.jsonl  (output of prepare_reranker_data.py)
    Every row: {"query": str, "positive": str, "negatives": [str, ...], "qid": int, "pubkey_gold": int}

  alternatively:
  - topics.json   : [{"index": int, "text": str, "pubkey": int}, ...]
  - corpus.json   : [{"index": int, "text": str, "pubkey": int}, ...]
  - qrels.json    : {str(qid): {str(pubkey): int, ...}, ...}   (rilevanze)

Usage:
  python finetune_modernbert_reranker.py \
      --train_jsonl   data/train.jsonl \
      --dev_jsonl     data/dev.jsonl \
      --output_dir    models/reranker-modernbert-large \
      [--topics       data/topics.json] \
      [--corpus       data/corpus.json] \
      [--qrels        data/qrels.json] \
      [--num_negatives 5] \
      [--epochs 3] \
      [--batch_size 16] \
      [--lr 2e-5] \
      [--max_length 512] \
      [--use_faiss]
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for loading data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def jsonl_to_labeled_pairs(records: list[dict], num_negatives: int = 5) -> dict:
    """
    Converts record {query, positive, negatives, ...} in pairs (query, passage, label)
    compatible with BinaryCrossEntropyLoss.
    label=1 → positive, label=0 → negative
    """
    queries, passages, labels = [], [], []
    for rec in records:
        q = rec["query"]
        pos = rec["positive"]
        negs = rec.get("negatives", [])

        # positives
        queries.append(q)
        passages.append(pos)
        labels.append(1.0)

        # negatives (up to num_negatives)
        for neg in negs[:num_negatives]:
            queries.append(q)
            passages.append(neg)
            labels.append(0.0)

    return {"query": queries, "passage": passages, "label": labels}


def build_reranking_samples(records: list[dict], min_negatives: int = 4) -> list[dict]:
    """
    Builds samples for CrossEncoderRerankingEvaluator:
    [{'query': str, 'positive': [str, ...], 'negative': [str, ...]}]

    CrossEncoderRerankingEvaluator requires at least 1 negative per query
    (otherwise ndcg_score of sklearn crashes with "only 1 document").
    If negatives are missing or are too few, integrates them with positives from other queries
    (which are "random" negatives for the current query).
    """
    # COllect all positives as pool from which we take random negatives
    all_positives = [rec["positive"] for rec in records]

    samples = []
    for i, rec in enumerate(records):
        negs = list(rec.get("negatives", []))

        # If there aren't enough negatives, take positives from other queries
        if len(negs) < min_negatives:
            pool = [p for j, p in enumerate(all_positives) if j != i and p not in negs]
            random.shuffle(pool)
            negs += pool[: min_negatives - len(negs)]

        samples.append({
            "query": rec["query"],
            "positive": [rec["positive"]],
            "negative": negs,
        })
    return samples


# ---------------------------------------------------------------------------
# Mining hard negatives from topics/corpus/qrels (alternative path)
# ---------------------------------------------------------------------------

def mine_negatives_from_raw(
    topics_path: str,
    corpus_path: str,
    qrels_path: str,
    num_negatives: int,
    use_faiss: bool,
) -> tuple[Dataset, list[dict]]:
    """
    If pre-processed JSONL are not available, mines hard negatives
    directly from topics + corpus + qrels.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives

    logger.info("Caricamento topics, corpus, qrels...")
    with open(topics_path) as f:
        topics = json.load(f)
    with open(corpus_path) as f:
        corpus_list = json.load(f)
    with open(qrels_path) as f:
        qrels = json.load(f)  # {str(qid): {str(pubkey): score}}

    def doc_to_text(item):
        """Corpus: title + abstract. Robust to missing fields."""
        title    = item.get("title", "").strip()
        abstract = item.get("abstract", "").strip()
        if title and abstract:
            return f"{title}. {abstract}"
        return title or abstract or item.get("text", "")

    pubkey_to_text = {str(item["pubkey"]): doc_to_text(item) for item in corpus_list}

    queries_list, answers_list = [], []
    for topic in topics:
        qid = str(topic["index"])
        pubkey = str(topic["pubkey"])
        q_text = topic["text"]
        if pubkey in pubkey_to_text:
            queries_list.append(q_text)
            answers_list.append(pubkey_to_text[pubkey])
        elif qid in qrels:
            # Take the first relevant from qrels
            for pk, score in qrels[qid].items():
                if score > 0 and pk in pubkey_to_text:
                    queries_list.append(q_text)
                    answers_list.append(pubkey_to_text[pk])
                    break

    raw_dataset = Dataset.from_dict({
        "question": queries_list,
        "answer": answers_list,
    })
    logger.info(f"query-answer pairs: {len(raw_dataset)}")

    logger.info("Mining hard negatives with static-retrieval-mrl-en-v1...")
    embedding_model = SentenceTransformer(
        "sentence-transformers/static-retrieval-mrl-en-v1", device="cpu"
    )

    hard_dataset = mine_hard_negatives(
        raw_dataset,
        embedding_model,
        num_negatives=num_negatives,
        range_min=5,
        range_max=50,
        max_score=0.85,
        margin=0.05,
        sampling_strategy="top",
        batch_size=512,
        output_format="labeled-pair",
        use_faiss=use_faiss,
    )

    # Rename columns
    hard_dataset = hard_dataset.rename_column("question", "query")
    hard_dataset = hard_dataset.rename_column("answer", "passage")

    # Split 90/10 for train/dev
    split = hard_dataset.train_test_split(test_size=0.1, seed=42)

    # Samples for the reranking evaluator (from the raw dev set)
    # Add random negatives (positives from other queries) to avoid a crash with ndcg_score
    eval_answers = answers_list[:500]
    eval_samples = []
    for i, (q, a) in enumerate(zip(queries_list[:500], eval_answers)):
        pool = [x for j, x in enumerate(eval_answers) if j != i]
        random.shuffle(pool)
        eval_samples.append({
            "query": q,
            "positive": [a],
            "negative": pool[:4],
        })

    return split, eval_samples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ModernBERT-large as reranker")

    # Input dati
    parser.add_argument("--train_jsonl", type=str, default=None,
                        help="Path to train.jsonl (output of prepare_reranker_data.py)")
    parser.add_argument("--dev_jsonl", type=str, default=None,
                        help="Path to dev.jsonl")
    parser.add_argument("--topics", type=str, default=None,
                        help="Path to topics.json (alternative to JSONL)")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Path to corpus.json")
    parser.add_argument("--qrels", type=str, default=None,
                        help="Path to qrels.json")

    # Modello
    parser.add_argument("--model", type=str,
                        default="answerdotai/ModernBERT-large",
                        help="Base HuggingFace model")
    parser.add_argument("--output_dir", type=str,
                        default="models/reranker-modernbert-large",
                        help="Output directory")

    # Iperparametri
    parser.add_argument("--num_negatives", type=int, default=5,
                        help="NNumber of negatives per positive sample")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size per device (riduce if OOM)")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max token length per query+passage pair")
    parser.add_argument("--use_faiss", action="store_true",
                        help="Use FAISS for mining (requires faiss-gpu or faiss-cpu)")
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use bfloat16 (recommended on L40S)")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Use fp16 (alternative to bf16)")
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="Gradient checkpointing to save VRAM")
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=600)
    parser.add_argument("--run_name", type=str, default="reranker-modernbert-large")
    parser.add_argument("--loss", type=str,
                        choices=["bce", "mnrl", "lambda"],
                        default="bce",
                        help="Loss: bce=BinaryCrossEntropy, mnrl=CachedMNRL, lambda=LambdaLoss")

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Import sentence-transformers (>= 4.0 required)
    # ------------------------------------------------------------------
    try:
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
        from sentence_transformers.cross_encoder.losses import (
            BinaryCrossEntropyLoss,
            CachedMultipleNegativesRankingLoss,
            LambdaLoss,
        )
        from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
    except ImportError as e:
        logger.error(
            "sentence-transformers >= 4.0 required.\n"
            "Install with: pip install -U sentence-transformers\n"
            f"Errore: {e}"
        )
        raise

    # ------------------------------------------------------------------
    # Caricamento / preparazione dati
    # ------------------------------------------------------------------
    eval_samples = None

    if args.train_jsonl and args.dev_jsonl:
        logger.info("Loading data from pre-processed JSONL...")
        train_records = load_jsonl(args.train_jsonl)
        dev_records = load_jsonl(args.dev_jsonl)

        train_data = jsonl_to_labeled_pairs(train_records, args.num_negatives)
        dev_data = jsonl_to_labeled_pairs(dev_records, args.num_negatives)

        train_dataset = Dataset.from_dict(train_data)
        dev_dataset = Dataset.from_dict(dev_data)

        eval_samples = build_reranking_samples(dev_records)

        logger.info(f"Train samples: {len(train_dataset)} | Dev samples: {len(dev_dataset)}")

    elif args.topics and args.corpus:
        logger.info("No JSONL found → mining hard negatives from topics/corpus...")
        split, eval_samples = mine_negatives_from_raw(
            topics_path=args.topics,
            corpus_path=args.corpus,
            qrels_path=args.qrels,
            num_negatives=args.num_negatives,
            use_faiss=args.use_faiss,
        )
        train_dataset = split["train"]
        dev_dataset = split["test"]
    else:
        raise ValueError(
            "Must provide --train_jsonl + --dev_jsonl  or  --topics + --corpus (+ optionally --qrels)"
        )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logger.info(f"Loading base model: {args.model}")
    model = CrossEncoder(
        args.model,
        num_labels=1,          # reranker: single scalar score
        max_length=args.max_length,
        default_activation_function=torch.nn.Sigmoid(),  # → score in [0,1]
    )

    if args.grad_checkpoint:
        model.model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    if args.loss == "bce":
        # Required columns: query, passage, label (0/1)
        # Excellent for dataset with labeled pairs
        loss = BinaryCrossEntropyLoss(model)
        logger.info("Loss: BinaryCrossEntropyLoss")

    elif args.loss == "mnrl":
        # In-batch negatives: every positive uses the other samples in the batch as negatives
        # Doesn't require label column → remove it if present
        if "label" in train_dataset.column_names:
            train_dataset = train_dataset.filter(lambda x: x["label"] == 1.0)
            train_dataset = train_dataset.remove_columns(["label"])
        if "label" in dev_dataset.column_names:
            dev_dataset = dev_dataset.filter(lambda x: x["label"] == 1.0)
            dev_dataset = dev_dataset.remove_columns(["label"])
        loss = CachedMultipleNegativesRankingLoss(model)
        logger.info("Loss: CachedMultipleNegativesRankingLoss")

    elif args.loss == "lambda":
        # LambdaLoss: directly optimized ranking metrics (NDCG)
        # Requires columns: query, passage, label
        loss = LambdaLoss(model)
        logger.info("Loss: LambdaLoss (optimizes NDCG directly)")

    # ------------------------------------------------------------------
    # Evaluator
    # ------------------------------------------------------------------
    evaluator = None
    if eval_samples:
        evaluator = CrossEncoderRerankingEvaluator(
            samples=eval_samples,
            name="dev-reranking",
            batch_size=args.batch_size,
            show_progress_bar=True,
        )
        logger.info(f"Evaluator: CrossEncoderRerankingEvaluator on {len(eval_samples)} queries")

    # ------------------------------------------------------------------
    # Training Arguments
    # ------------------------------------------------------------------
    training_args = CrossEncoderTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=args.fp16,
        bf16=args.bf16,
        # Storing and eval strategy
        eval_strategy="steps" if evaluator else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=1,
        load_best_model_at_end=True if evaluator else False,
        metric_for_best_model="dev-reranking_map" if evaluator else None,
        greater_is_better=True,
        logging_steps=50,
        run_name=args.run_name,
        # Memory optimization
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        # Gradient accumulation (increases effective batch size if there's no more VRAM)
        gradient_accumulation_steps=2,  # effective_batch = batch_size * 2
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset if evaluator is None else None,
        loss=loss,
        evaluator=evaluator,
    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("AVVIO FINE-TUNING")
    logger.info(f"  Base model    : {args.model}")
    logger.info(f"  Loss          : {args.loss}")
    logger.info(f"  Epoche        : {args.epochs}")
    logger.info(f"  Batch size    : {args.batch_size} (grad_accum=2 → eff {args.batch_size * 2})")
    logger.info(f"  LR            : {args.lr}")
    logger.info(f"  Max length    : {args.max_length}")
    logger.info(f"  Output        : {args.output_dir}")
    logger.info("=" * 60)

    trainer.train()

    # ------------------------------------------------------------------
    # Salvataggio finale
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    logger.info(f"Modello stored in: {final_path}")

    # ------------------------------------------------------------------
    # Final evaluation on dev set
    # ------------------------------------------------------------------
    if evaluator:
        logger.info("Final evaluation on dev set...")
        results = evaluator(model)
        logger.info("Final results:")
        for k, v in results.items():
            logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
