"""
finetune_modernbert_crossencoder.py

Fine-tuning of ModernBERT-large CrossEncoder, updated version
compatible with the dataset produced by prepare_data_from_nemotron.py.

Differences compared to finetune_modernbert_crossencoder.py:
    - Reads train_pairs.jsonl / dev_pairs.jsonl directly (already prepared)
    - Does not handle hard negatives construction (already done by prepare_data_from_nemotron.py)
    - Dynamically calculates pos_weight from the actual dataset ratio
    - Evaluator built from the dev set (for CrossEncoderRerankingEvaluator)

Usage:
  python finetune_modernbert_crossencoder.py \
    --train_jsonl training_data/train_pairs.jsonl \
    --dev_jsonl   training_data/dev_pairs.jsonl \
    --output_dir  models/modernbert-large-retrix-nemotron-distilled \
    [--batch_size 16] [--epochs 3] [--lr 2e-5]
"""

import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderModelCardData,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.evaluation import CrossEncoderRerankingEvaluator
from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {path}: {len(rows):,} rows")
    return rows


def rows_to_hf_dataset(rows: list[dict]) -> Dataset:
    """Convers a list of {query, passage, label} in HuggingFace Dataset."""
    return Dataset.from_dict({
        "query":   [r["query"]   for r in rows],
        "passage": [r["passage"] for r in rows],
        "label":   [float(r["label"]) for r in rows],
    })


def build_reranking_eval_samples(
        dev_rows: list[dict],
        max_queries: int = 500,
        seed: int = 42,
) -> list[dict]:
    """
    Builds samples for CrossEncoderRerankingEvaluator from the dev set.
    Groups the rows by qid and builds:
      {"query": ..., "positive": [...], "documents": [...]}
    """
    qid_groups = defaultdict(lambda: {"positive": [], "documents": []})

    for row in dev_rows:
        qid = row.get("qid", row["query"])   # fallback: uses query text as key
        qid_groups[qid]["query"] = row["query"]
        qid_groups[qid]["documents"].append(row["passage"])
        if row["label"] == 1:
            qid_groups[qid]["positive"].append(row["passage"])

    samples = [
        {
            "query":     v["query"],
            "positive":  v["positive"],
            "documents": v["documents"],
        }
        for v in qid_groups.values()
        if v["positive"] and len(v["documents"]) >= 2
    ]

    if len(samples) > max_queries:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_queries]

    log.info(f"Eval samples: {len(samples):,} queries")
    return samples


def compute_pos_weight(rows: list[dict]) -> float:
    """
    Calculates pos_weight = n_neg / n_pos from the real dataset.
    This is the correct value for BinaryCrossEntropyLoss,
    better than a simple fixed num_hard_neg.
    """
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    ratio = n_neg / max(1, n_pos)
    log.info(f"pos_weight calculated: {ratio:.2f} ({n_neg:,} neg / {n_pos:,} pos)")
    return ratio


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fine-tuning ModernBERT-large CrossEncoder (v2 with pre-built jsonl)"
    )
    parser.add_argument("--train_jsonl",  required=True)
    parser.add_argument("--dev_jsonl",    required=True)
    parser.add_argument("--output_dir",   default="models/modernbert-large-retrix-nemotron-distilled")
    parser.add_argument("--model_name",   default="answerdotai/ModernBERT-large")
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--epochs",       type=int,   default=1)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--eval_steps",   type=int,   default=4000)
    parser.add_argument("--save_steps",   type=int,   default=4000)
    parser.add_argument("--seed",         type=int,   default=12)
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. Loading data ─────────────────────────────────────────────────────
    log.info("Loading dataset...")
    train_rows = load_jsonl(args.train_jsonl)
    dev_rows   = load_jsonl(args.dev_jsonl)

    train_dataset = rows_to_hf_dataset(train_rows)
    dev_dataset   = rows_to_hf_dataset(dev_rows)

    log.info(f"Train: {len(train_dataset):,} | Dev: {len(dev_dataset):,}")

    # ── 2. Model ─────────────────────────────────────────────────────────
    log.info(f"Loading model: {args.model_name}")
    model = CrossEncoder(
        args.model_name,
        max_length=args.max_length,
        model_card_data=CrossEncoderModelCardData(
            language="en",
            license="apache-2.0",
            model_name="ModernBERT-large fine-tuned reranker (Nemotron distillation)",
        ),
    )

    # ── 3. Loss with dynamic pos_weight ────────────────────────────────────
    pos_weight = compute_pos_weight(train_rows)
    loss = BinaryCrossEntropyLoss(
        model=model,
        pos_weight=torch.tensor(pos_weight),
    )

    # ── 4. Evaluator ───────────────────────────────────────────────────────
    log.info("Building evaluator reranking...")
    eval_samples = build_reranking_eval_samples(dev_rows, max_queries=500, seed=args.seed)
    evaluator = CrossEncoderRerankingEvaluator(
        samples=eval_samples,
        batch_size=args.batch_size,
        name="retrix-dev",
        always_rerank_positives=False,
    )

    # Baseline pre-training
    log.info("Baseline (non fine-tuned model):")
    evaluator(model)

    # ── 5. Training args ───────────────────────────────────────────────────
    short_name = args.model_name.split("/")[-1]
    run_name   = f"reranker-{short_name}-retrix-nemotron"
    output_dir = Path(args.output_dir)

    training_args = CrossEncoderTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        bf16=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="eval_retrix-dev_ndcg@10",
        greater_is_better=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=0,
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        seed=args.seed,
        run_name=run_name,
        gradient_checkpointing=True,
        weight_decay=0.01,
    )


    # ── 6. Trainer ─────────────────────────────────────────────────────────
    trainer = CrossEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        loss=loss,
        evaluator=evaluator,
    )

    log.info("Launching training...")
    trainer.train()

    # ── 7. Final evaluation ───────────────────────────────────────────────
    log.info("Final evaluation:")
    final_metrics = evaluator(model)
    log.info(f"Final metrics: {final_metrics}")

    # ── 8. Storing ─────────────────────────────────────────────────────
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    log.info(f"✓ Model stored in {final_dir}")
    log.info(
        f"\nTo use it in the pipeline pipeline:\n"
        f"  python rerank_with_crossencoder.py \\\n"
        f"    --model_dir {final_dir} \\\n"
        f"    --bm25_run  <run.txt> --corpus <corpus.tsv> --queries <queries.tsv>"
    )


if __name__ == "__main__":
    main()