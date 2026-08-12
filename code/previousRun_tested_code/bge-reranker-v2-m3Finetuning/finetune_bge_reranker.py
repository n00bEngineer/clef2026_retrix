"""
finetune_bge_reranker.py

Fine-tuning of BAAI/bge-reranker-v2-m3 with official FlagEmbedding.

Correct module (confirmed by official doc):
    FlagEmbedding.finetune.reranker.encoder_only.base

Key parameters (from official doc for encoder_only):
    --learning_rate       6e-5   (NOT 2e-5 which is for decoders)
    --train_group_size    N+1    (1 pos + N neg, must match the data)
    --query_max_len       512
    --passage_max_len     512
    --pad_to_multiple_of  8
    --dataloader_drop_last True

Installation:
    pip install FlagEmbedding

Usage:
    python finetune_bge_reranker.py \
        --train_file       training_data/train_groups.jsonl \
        --dev_file         training_data/dev_groups.jsonl \
        --output_dir       models/bge-reranker-v2-m3-retrix \
        --train_group_size 6   # = 1 + num_hard_neg used in prepare_data
"""

import logging
import argparse
import importlib.metadata
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def check_flag_embedding():
    try:
        import FlagEmbedding  # noqa: F401
        version = importlib.metadata.version("FlagEmbedding")
        log.info(f"FlagEmbedding found: {version}")
        return True
    except ImportError:
        log.error(
            "FlagEmbedding not found. Install with:\n"
            "  pip install FlagEmbedding"
        )
        return False


def count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning BGE-reranker-v2-m3 with FlagEmbedding encoder_only"
    )
    parser.add_argument("--train_file",       required=True)
    parser.add_argument("--dev_file",         required=True,
                        help="Only used for logging. FlagEmbedding doesn't eval during training.")
    parser.add_argument("--output_dir",       default="models/bge-reranker-v2-m3-retrix")
    parser.add_argument("--model_name",       default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--train_group_size", type=int, default=6,
                        help="1 + num_hard_neg used in prepare_data (default 6 = 1+5)")
    parser.add_argument("--epochs",           type=int,   default=2)
    parser.add_argument("--batch_size",       type=int,   default=4,
                        help="Per GPU. With L40S 48GB you can try 4-8.")
    parser.add_argument("--grad_accum",       type=int,   default=4,
                        help="Gradient accumulation. Effective batch = batch_size * grad_accum.")
    parser.add_argument("--lr",               type=float, default=6e-5,
                        help="Official learning rate for encoder_only (default 6e-5)")
    parser.add_argument("--query_max_len",    type=int,   default=512)
    parser.add_argument("--passage_max_len",  type=int,   default=512)
    parser.add_argument("--warmup_ratio",     type=float, default=0.1)
    parser.add_argument("--save_steps",       type=int,   default=500)
    parser.add_argument("--logging_steps",    type=int,   default=100)
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--num_gpus",         type=int,   default=1,
                        help="Nomber of GPUs per torchrun (default 1)")
    args = parser.parse_args()

    if not check_flag_embedding():
        sys.exit(1)

    n_train = count_lines(args.train_file)
    n_dev   = count_lines(args.dev_file)
    log.info(f"Train: {n_train:,} groups | Dev: {n_dev:,} groups (only for reference)")
    log.info(f"train_group_size: {args.train_group_size}  "
             f"(1 pos + {args.train_group_size - 1} neg per query)")
    log.info(f"Effective batch size: {args.batch_size} x {args.grad_accum} = "
             f"{args.batch_size * args.grad_accum}")
    log.info(f"Number of GPUs: {args.num_gpus}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Launched with torchrun because FlagEmbedding uses dist.get_rank()
    # which requires an initialized process group
    cmd = [
        "torchrun",
        "--nproc_per_node", str(args.num_gpus),
        "-m",
        "FlagEmbedding.finetune.reranker.encoder_only.base",
        "--model_name_or_path",          args.model_name,
        "--train_data",                  args.train_file,
        "--output_dir",                  str(output_dir),
        "--train_group_size",            str(args.train_group_size),
        "--query_max_len",               str(args.query_max_len),
        "--passage_max_len",             str(args.passage_max_len),
        "--pad_to_multiple_of",          "8",
        "--knowledge_distillation",      "False",
        "--num_train_epochs",            str(args.epochs),
        "--per_device_train_batch_size", str(args.batch_size),
        "--gradient_accumulation_steps", str(args.grad_accum),
        "--learning_rate",               str(args.lr),
        "--warmup_ratio",                str(args.warmup_ratio),
        "--weight_decay",                "0.01",
        "--bf16",
        "--gradient_checkpointing",
        "--dataloader_drop_last",        "True",
        "--dataloader_num_workers",      "4",
        "--overwrite_output_dir",
        "--save_total_limit",            "1",
        "--save_only_model",             "True",
        "--save_steps",                  str(args.save_steps),
        "--logging_steps",               str(args.logging_steps),
        "--seed",                        str(args.seed),
    ]

    log.info("Launching training with torchrun + FlagEmbedding encoder_only...")
    log.info("Command:\n  " + " \\\n  ".join(cmd))

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        log.error(
            f"\nTraining failed (exit code {result.returncode}).\n\n"
            "Try executing the command directly:\n\n"
            f"  torchrun --nproc_per_node {args.num_gpus} \\\n"
            f"    -m FlagEmbedding.finetune.reranker.encoder_only.base \\\n"
            f"    --model_name_or_path {args.model_name} \\\n"
            f"    --train_data {args.train_file} \\\n"
            f"    --output_dir {output_dir} \\\n"
            f"    --train_group_size {args.train_group_size} \\\n"
            f"    --query_max_len {args.query_max_len} \\\n"
            f"    --passage_max_len {args.passage_max_len} \\\n"
            f"    --pad_to_multiple_of 8 \\\n"
            f"    --knowledge_distillation False \\\n"
            f"    --num_train_epochs {args.epochs} \\\n"
            f"    --per_device_train_batch_size {args.batch_size} \\\n"
            f"    --gradient_accumulation_steps {args.grad_accum} \\\n"
            f"    --learning_rate {args.lr} \\\n"
            f"    --warmup_ratio {args.warmup_ratio} \\\n"
            f"    --weight_decay 0.01 \\\n"
            f"    --bf16 \\\n"
            f"    --gradient_checkpointing \\\n"
            f"    --dataloader_drop_last True \\\n"
            f"    --overwrite_output_dir \\\n"
            f"    --save_total_limit 1 \\\n"
            f"    --save_only_model True \\\n"
            f"    --save_steps {args.save_steps} \\\n"
            f"    --logging_steps {args.logging_steps} \\\n"
            f"    --seed {args.seed}"
        )
        sys.exit(1)

    log.info(f"\n✓ Training complete. Modello saved in {output_dir}")
    log.info(f"\nIn order to use it:")
    log.info(f"  python rerank_with_bge.py \\")
    log.info(f"    --model_dir    {output_dir} \\")
    log.info(f"    --nemotron_run reranked_results_nemotron_topk400.json \\")
    log.info(f"    --collection   collection_data.json \\")
    log.info(f"    --queries      en_train.json \\")
    log.info(f"    --output       reranked_results_bge_ft_top100.json")


if __name__ == "__main__":
    main()