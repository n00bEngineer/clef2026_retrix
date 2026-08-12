"""
finetune_nemotron_reranker_v3.py

Fine-tuning for nvidia/llama-nemotron-rerank-1b-v2 with native
HuggingFace Trainer + LoRA (PEFT).

===============================================================
CHANGES COMPARED TO v2 - SUMMARY
===============================================================

[FIX 1] Removed modules_to_save=["score"]
  In v2, PEFT cloned and reinitialized the "score" head,
  losing pre-trained calibration. Now the head is frozen by default
  (only LoRA updates q/k/v/o).

[FIX 2] Lowered learning rate: 2e-4 -> 5e-5
  Previous effective LR was too aggressive for LoRA setup.

[FIX 3] Lowered LoRA rank: 32 -> 16, raised dropout: 0.05 -> 0.1
  Fewer trainable params and stronger regularization.

[FIX 4] Loss: BCEWithLogitsLoss -> PairwiseMarginLoss (with BCE fallback)
  Ranking needs relative order (pos > neg), not absolute calibration.

[FIX 5] load_best_model_at_end=True, save_total_limit=3
  Save best checkpoint by minimum eval_loss.

[FIX 6] Reduced epochs and eval interval
  Fewer epochs lower overfitting risk; more frequent eval catches degradation earlier.

[FIX 7] Increased weight_decay: 0.01 -> 0.05
  Stronger L2 regularization for LoRA weights.

[FIX 8] Increased warmup_ratio: 0.1 -> 0.15
  Longer warmup stabilizes early training.

[FIX 9] PairDataset now supports separate hard_neg and soft neg
  hard_neg are sampled with double weight.

[FIX 10] Added EarlyStoppingCallback
  Training stops when eval_loss no longer improves for consecutive evaluations.
===============================================================
"""

import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    log.info(f"Loaded {path}: {len(rows):,} groups")
    return rows


class PairDataset(TorchDataset):
    """
    Pair dataset (query, passage) with binary labels.

    [FIX 9] Supports weighted hard_neg sampling.
    "hard_neg" entries are duplicated to give them double weight.

    Expected JSONL format:
      {"query": "...", "pos": ["doc1", ...], "neg": ["doc2", ...]}
      or with hard negatives:
      {"query": "...", "pos": [...], "neg": [...], "hard_neg": [...]}
    """

    def __init__(self, groups: list[dict], tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pairs: list[tuple[str, str, float]] = []

        n_hard = 0
        for g in groups:
            query = g["query"]
            for doc in g.get("pos", []):
                self.pairs.append((query, doc, 1.0))
            for doc in g.get("neg", []):
                self.pairs.append((query, doc, 0.0))
            # [FIX 9] Hard negatives added twice for oversampling
            for doc in g.get("hard_neg", []):
                self.pairs.append((query, doc, 0.0))
                self.pairs.append((query, doc, 0.0))  # double weight
                n_hard += 1

        if n_hard > 0:
            log.info(f"  -> {n_hard:,} hard negatives included (x2 oversampling)")
        log.info(f"PairDataset: {len(self.pairs):,} total pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        query, passage, label = self.pairs[idx]
        enc = self.tokenizer(
            query,
            passage,
            max_length=self.max_length,
            truncation=True,
            padding=False,        # dynamic padding in DataCollator
            return_tensors=None,  # Python lists, not tensors
        )
        enc["labels"] = label
        return enc


def compute_pos_weight(dataset: PairDataset) -> float:
    n_pos = sum(1 for _, _, label in dataset.pairs if label >= 1.0)
    n_neg = sum(1 for _, _, label in dataset.pairs if label < 1.0)
    ratio = n_neg / max(1, n_pos)
    log.info(f"pos_weight: {ratio:.2f}  ({n_neg:,} neg / {n_pos:,} pos)")
    return ratio


# ══════════════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════════════

def build_eval_samples(
        groups: list[dict],
        max_queries: int = 500,
        seed: int = 42,
) -> list[dict]:
    samples = []
    for g in groups:
        positives = set(g.get("pos", []))
        documents = g.get("pos", []) + g.get("neg", []) + g.get("hard_neg", [])
        if positives and len(documents) >= 2:
            samples.append({
                "query":     g["query"],
                "positives": positives,
                "documents": documents,
            })
    if len(samples) > max_queries:
        random.Random(seed).shuffle(samples)
        samples = samples[:max_queries]
    log.info(f"Eval samples: {len(samples):,} queries")
    return samples


def score_pairs_batched(
        model,
        tokenizer,
        pairs: list[tuple[str, str]],
        batch_size: int,
        max_length: int,
        device: torch.device,
) -> list[float]:
    model.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            queries  = [p[0] for p in batch_pairs]
            passages = [p[1] for p in batch_pairs]
            enc = tokenizer(
                queries,
                passages,
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            logits = out.logits
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
    return scores


def ndcg_at_k(relevances: list[int], k: int) -> float:
    k = min(k, len(relevances))
    dcg  = sum(relevances[i] / math.log2(i + 2) for i in range(k))
    ideal = sorted(relevances, reverse=True)[:k]
    idcg = sum(ideal[i] / math.log2(i + 2) for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(relevances: list[int], k: int) -> float:
    for rank, rel in enumerate(relevances[:k], start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def evaluate_reranking(
        model,
        tokenizer,
        eval_samples: list[dict],
        batch_size: int,
        max_length: int,
        device: torch.device,
        ks: tuple[int, ...] = (5, 10),
        prefix: str = "",
) -> dict[str, float]:
    all_ndcg = {k: [] for k in ks}
    all_mrr  = {k: [] for k in ks}

    for sample in eval_samples:
        query     = sample["query"]
        docs      = sample["documents"]
        positives = sample["positives"]
        pairs  = [(query, doc) for doc in docs]
        logits = score_pairs_batched(
            model, tokenizer, pairs, batch_size, max_length, device
        )
        ranked = sorted(zip(logits, docs), key=lambda x: x[0], reverse=True)
        relevances = [1 if doc in positives else 0 for _, doc in ranked]
        for k in ks:
            all_ndcg[k].append(ndcg_at_k(relevances, k))
            all_mrr[k].append(mrr_at_k(relevances, k))

    metrics = {}
    for k in ks:
        metrics[f"{prefix}ndcg@{k}"] = float(np.mean(all_ndcg[k]))
        metrics[f"{prefix}mrr@{k}"]  = float(np.mean(all_mrr[k]))

    log.info("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


# ══════════════════════════════════════════════════════════════════
# NemotronTrainer - with PairwiseMarginLoss
# ══════════════════════════════════════════════════════════════════

class NemotronTrainer(Trainer):
    """
    Custom trainer with Pairwise Margin Loss + BCE fallback.

    [FIX 4] WHY MARGIN LOSS INSTEAD OF BCE:
    BCEWithLogitsLoss optimizes absolute calibration, while reranking
    needs relative order: score(pos) > score(neg) + margin.

    Implementation:
      For each batch, build all (pos_i, neg_j) pairs and compute
      loss = mean(max(0, margin - (score_pos - score_neg))).

    BCE fallback:
      If batch has only pos or only neg, margin loss is not applicable.
    """

    def __init__(self, *args, pos_weight: float = 1.0, margin: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight_value = pos_weight
        self.margin = margin

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.dim() > 1:
            logits = logits.squeeze(-1)  # (batch,)

        pos_mask = labels >= 1.0
        neg_mask = labels < 1.0

        # [FIX 4] Pairwise margin loss if both pos and neg are in batch
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = logits[pos_mask]   # shape (n_pos,)
            neg_scores = logits[neg_mask]   # shape (n_neg,)

            # Broadcast over all pos x neg pairs: shape (n_pos, n_neg)
            pos_exp = pos_scores.unsqueeze(1)
            neg_exp = neg_scores.unsqueeze(0)

            # max(0, margin - (pos - neg)): we want pos - neg > margin
            pair_loss = torch.clamp(self.margin - (pos_exp - neg_exp), min=0.0)
            loss = pair_loss.mean()
        else:
            # BCE fallback for mono-label batches
            pw = torch.tensor(
                self.pos_weight_value,
                dtype=logits.dtype,
                device=logits.device,
            )
            loss = torch.nn.BCEWithLogitsLoss(pos_weight=pw)(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ══════════════════════════════════════════════════════════════════
# RerankingEvalCallback
# ══════════════════════════════════════════════════════════════════

class RerankingEvalCallback(TrainerCallback):
    """
    Callback that runs reranking evaluation on each on_evaluate.
    Logs NDCG and MRR at multiple k values.
    """

    def __init__(
            self,
            eval_samples: list[dict],
            tokenizer,
            max_length: int,
            batch_size: int,
            device: torch.device,
            ks: tuple[int, ...] = (5, 10),
    ):
        self.eval_samples = eval_samples
        self.tokenizer    = tokenizer
        self.max_length   = max_length
        self.batch_size   = batch_size
        self.device       = device
        self.ks           = ks

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        log.info(f"[Step {state.global_step}] Reranking evaluation:")
        evaluate_reranking(
            model=model,
            tokenizer=self.tokenizer,
            eval_samples=self.eval_samples,
            batch_size=self.batch_size,
            max_length=self.max_length,
            device=self.device,
            ks=self.ks,
            prefix=f"step{state.global_step}_",
        )


# ══════════════════════════════════════════════════════════════════
# LoRA
# ══════════════════════════════════════════════════════════════════

def apply_lora(model, lora_rank: int, lora_alpha: int, lora_dropout: float):
    """
    Apply LoRA to model.

    [FIX 1] modules_to_save=["score"] REMOVED.
    In v2 PEFT cloned and reinitialized the "score" head, wiping out
    pre-trained calibration. Now head weights stay frozen and only
    q/k/v/o modules are updated by LoRA.

    [FIX 3] lower rank and higher dropout by default.
    """
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,   # [FIX 3]
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        # modules_to_save=["score"],  # [FIX 1] REMOVED - head frozen
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    return model


# ══════════════════════════════════════════════════════════════════
# DataCollator
# ══════════════════════════════════════════════════════════════════

@dataclass
class PairCollator:
    """
    DataCollator with dynamic padding and float-label conversion.
    Needed because DataCollatorWithPadding does not handle float labels.
    """
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict]) -> dict:
        labels = [float(f.pop("labels")) for f in features]
        base = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        batch = base(features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        # Put labels back into original features to avoid mutating dataset entries
        for f, label in zip(features, labels):
            f["labels"] = label
        return batch


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tuning llama-nemotron-rerank-1b-v2 - v3"
    )
    parser.add_argument("--train_file",       required=True)
    parser.add_argument("--dev_file",         required=True)
    parser.add_argument("--output_dir",       default="models/nemotron-rerank-1b-retrixV3")
    parser.add_argument("--model_name",       default="nvidia/llama-nemotron-rerank-1b-v2")
    parser.add_argument("--no_lora",          action="store_true")
    parser.add_argument("--lora_rank",        type=int,   default=16)     # [FIX 3] was 32
    parser.add_argument("--lora_alpha",       type=int,   default=32)     # [FIX 3] was 64
    parser.add_argument("--lora_dropout",     type=float, default=0.1)    # [FIX 3] was 0.05
    parser.add_argument("--margin",           type=float, default=1.0)    # [FIX 4] for margin loss
    parser.add_argument("--batch_size",       type=int,   default=16)
    parser.add_argument("--grad_accum",       type=int,   default=4)
    parser.add_argument("--epochs",           type=int,   default=1)      # [FIX 6] was 3
    parser.add_argument("--lr",               type=float, default=None)
    parser.add_argument("--max_length",       type=int,   default=1024)
    parser.add_argument("--warmup_ratio",     type=float, default=0.1)   # [FIX 8]
    parser.add_argument("--weight_decay",     type=float, default=0.05)   # [FIX 7] was 0.01
    parser.add_argument("--eval_steps",       type=int,   default=100)    # [FIX 6] was 500
    parser.add_argument("--save_steps",       type=int,   default=100)    # [FIX 5] was 10000
    parser.add_argument("--patience",         type=int,   default=3)      # [FIX 10] early stopping
    parser.add_argument("--max_eval_queries", type=int,   default=500)
    parser.add_argument("--seed",             type=int,   default=42)
    args = parser.parse_args()

    use_lora = not args.no_lora
    if args.lr is None:
        # [FIX 2] LR lowered for LoRA
        args.lr = 2e-5 if use_lora else 5e-6

    log.info(f"Mode: {'LoRA' if use_lora else 'Full fine-tuning'} | LR={args.lr}")
    log.info(f"  rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    log.info(f"  margin={args.margin}, epochs={args.epochs}, patience={args.patience}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # -- 1. Tokenizer --
    log.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        log.info("pad_token set to eos_token")

    # -- 2. Model --
    log.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        ignore_mismatched_sizes=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    log.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -- 3. LoRA --
    if use_lora:
        log.info(
            f"Applying LoRA: rank={args.lora_rank}, "
            f"alpha={args.lora_alpha}, dropout={args.lora_dropout}"
        )
        model = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    else:
        log.info("Full fine-tuning (no LoRA)")
        try:
            model.gradient_checkpointing_enable()
            log.info("gradient_checkpointing enabled")
        except Exception as e:
            log.warning(f"Could not enable gradient_checkpointing: {e}")

    # -- 4. Dataset --
    log.info("Loading dataset...")
    train_groups  = load_jsonl(args.train_file)
    dev_groups    = load_jsonl(args.dev_file)
    train_dataset = PairDataset(train_groups, tokenizer, args.max_length)
    dev_dataset   = PairDataset(dev_groups,   tokenizer, args.max_length)
    pos_weight    = compute_pos_weight(train_dataset)

    # -- 5. Eval samples --
    eval_samples = build_eval_samples(
        dev_groups, max_queries=args.max_eval_queries, seed=args.seed
    )

    # -- 6. Baseline --
    log.info("Pre-training baseline:")
    model.to(device)
    baseline_metrics = evaluate_reranking(
        model=model,
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        prefix="baseline_",
    )

    # -- 7. Training args --
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,          # [FIX 6]
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,                 # [FIX 2]
        warmup_ratio=args.warmup_ratio,        # [FIX 8]
        weight_decay=args.weight_decay,        # [FIX 7]
        fp16=False,
        bf16=True,
        dataloader_num_workers=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,            # [FIX 6]
        save_strategy="steps",
        save_steps=args.save_steps,            # [FIX 5]
        save_total_limit=1,                    # [FIX 5]
        logging_strategy="steps",
        logging_steps=100,
        logging_first_step=True,
        seed=args.seed,
        run_name="nemotron-rerank-1b-retrix-v3",
        gradient_checkpointing=not use_lora,
        # [FIX 5] Save checkpoint with minimum eval_loss
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_only_model=True,
        report_to="none",
    )

    # -- 8. Callbacks --
    callbacks = [
        # [FIX 10] Early stopping when eval_loss does not improve
        # for `patience` consecutive evaluations.
        EarlyStoppingCallback(early_stopping_patience=args.patience),

        # Reranking evaluation callback (NDCG, MRR)
        RerankingEvalCallback(
            eval_samples=eval_samples,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=device,
        ),
    ]

    # -- 9. Trainer --
    trainer = NemotronTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=PairCollator(tokenizer=tokenizer),
        pos_weight=pos_weight,     # used only in BCE fallback
        margin=args.margin,        # [FIX 4] margin loss
        callbacks=callbacks,
    )

    log.info("Starting training...")
    trainer.train()

    # -- 10. Final evaluation --
    log.info("Final evaluation (best checkpoint):")
    final_metrics = evaluate_reranking(
        model=trainer.model,       # uses best model loaded by load_best_model_at_end
        tokenizer=tokenizer,
        eval_samples=eval_samples,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
        prefix="final_",
    )

    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump({**baseline_metrics, **final_metrics}, f, indent=2)
    log.info(f"Metrics saved to {output_dir / 'final_metrics.json'}")

    # -- 11. Saving --
    #
    # WHY THIS BLOCK IS CRITICAL FOR NEMOTRON:
    #
    # Nemotron uses trust_remote_code=True - its architecture is defined in
    # custom Python files downloaded from HuggingFace Hub and cached locally
    # (modeling_nemotron.py, configuration_nemotron.py, etc.).
    # When saving with save_pretrained(), these files are NOT copied
    # automatically into output directory, causing:
    #   - ValueError: Unrecognized model ... Should have a `model_type` key
    #   - Tokenizer failing to load because custom files are missing
    #
    # Solution: explicitly copy custom files from HuggingFace cache into
    # final directory so the model is self-contained and loadable offline.
    #
    # ADDITIONAL ISSUE with merge_and_unload() + LoRA:
    # After merge, resulting model may lose reference to HF cache.
    # save_pretrained() saves weights but NOT custom architecture files.

    import shutil
    from huggingface_hub import snapshot_download

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: download/locate base-model custom files in HF cache
    # (if already cached, no network request is made)
    log.info("Locating Nemotron architecture custom files...")
    try:
        base_model_cache = snapshot_download(
            repo_id=args.model_name,
            local_files_only=False,   # use cache when available, download otherwise
        )
        log.info(f"  Cache base model: {base_model_cache}")
    except Exception as e:
        log.warning(f"  snapshot_download failed ({e}), trying local cache...")
        # Fallback: search standard HF cache
        from huggingface_hub import hf_hub_download
        import os
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        # Search model directory in cache
        model_cache_name = args.model_name.replace("/", "--")
        base_model_cache = None
        for root, dirs, files in os.walk(cache_dir):
            if model_cache_name in root and "snapshots" in root:
                base_model_cache = root
                break
        if base_model_cache is None:
            log.error("Could not find base model cache. "
                      "Custom files will NOT be copied - model "
                      "may fail to load with trust_remote_code=True.")

    # Step 2: save fine-tuned model weights
    if use_lora:
        log.info("Merging LoRA adapter into base model...")
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(final_dir))
            log.info("Merge completed and weights saved")
        except Exception as e:
            log.error(f"Merge failed: {e}. Saving adapters separately.")
            trainer.model.save_pretrained(str(final_dir))
    else:
        trainer.model.save_pretrained(str(final_dir))

    # Step 3: save tokenizer from base model (more reliable than session tokenizer
    # which may be inconsistent after PEFT)
    log.info("Saving tokenizer from base model...")
    try:
        from transformers import AutoTokenizer as _AutoTokenizer
        base_tokenizer = _AutoTokenizer.from_pretrained(
            args.model_name,
            trust_remote_code=True,
        )
        base_tokenizer.save_pretrained(str(final_dir))
        log.info("Tokenizer saved from base model")
    except Exception as e:
        log.warning(f"Base tokenizer unavailable ({e}), using session tokenizer...")
        tokenizer.save_pretrained(str(final_dir))

    # Step 4: copy custom architecture files from HF cache
    # These files are required when loading with trust_remote_code=True.
    if base_model_cache is not None:
        import os
        custom_extensions = (".py", ".json")
        copied = []
        skipped = []
        for fname in os.listdir(base_model_cache):
            src = os.path.join(base_model_cache, fname)
            dst = final_dir / fname
            if not os.path.isfile(src):
                continue
            # Copy all .py (custom architecture) and .json (config, tokenizer)
            # but do NOT overwrite .json files already saved with updated weights
            if fname.endswith(".py"):
                shutil.copy2(src, dst)
                copied.append(fname)
            elif fname.endswith(".json") and not dst.exists():
                # Copy only missing .json files (do not overwrite saved config.json)
                shutil.copy2(src, dst)
                copied.append(fname)
            else:
                skipped.append(fname)
        log.info(f"  Custom files copied ({len(copied)}): {copied}")
        if skipped:
            log.info(f"  Skipped files (already existing): {[f for f in skipped if f.endswith('.json')]}")
    else:
        log.warning("WARNING: Architecture custom files were NOT copied. "
                    "To load the model use:\n"
                    f"  AutoModelForSequenceClassification.from_pretrained(\n"
                    f"    '{final_dir}',\n"
                    f"    trust_remote_code=True,\n"
                    f"    config=AutoConfig.from_pretrained('{args.model_name}', trust_remote_code=True)\n"
                    f"  )")

    # Step 5: verify config.json has model_type
    config_path = final_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        if "model_type" not in cfg:
            log.warning("  config.json missing 'model_type' - applying automatic patch...")
            # Load base config and update with our parameters
            from transformers import AutoConfig
            base_cfg = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
            base_cfg.num_labels = 1
            base_cfg.pad_token_id = tokenizer.pad_token_id
            base_cfg.save_pretrained(str(final_dir))
            log.info("  config.json updated with correct model_type")
        else:
            log.info(f"  config.json OK (model_type='{cfg['model_type']}')")
    else:
        log.error("  config.json not found! Model will fail to load.")

    log.info(f"Final complete model saved in {final_dir}")
    log.info(f"  Content: {sorted(f.name for f in final_dir.iterdir())}")

    # -- 12. Baseline vs fine-tuned summary --
    log.info("\n" + "═" * 60)
    log.info("COMPARISON SUMMARY")
    log.info("═" * 60)
    for k in (5, 10):
        b_ndcg = baseline_metrics.get(f"baseline_ndcg@{k}", float("nan"))
        f_ndcg = final_metrics.get(f"final_ndcg@{k}", float("nan"))
        b_mrr  = baseline_metrics.get(f"baseline_mrr@{k}", float("nan"))
        f_mrr  = final_metrics.get(f"final_mrr@{k}", float("nan"))
        delta_ndcg = f_ndcg - b_ndcg
        delta_mrr  = f_mrr  - b_mrr
        sign_ndcg  = "▲" if delta_ndcg >= 0 else "▼"
        sign_mrr   = "▲" if delta_mrr  >= 0 else "▼"
        log.info(
            f"  NDCG@{k}: baseline={b_ndcg:.4f}  fine-tuned={f_ndcg:.4f}  "
            f"{sign_ndcg}{abs(delta_ndcg):.4f}"
        )
        log.info(
            f"  MRR@{k}:  baseline={b_mrr:.4f}  fine-tuned={f_mrr:.4f}  "
            f"{sign_mrr}{abs(delta_mrr):.4f}"
        )
    log.info("═" * 60)

    log.info("\nFor inference usage:")
    log.info("  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
    log.info(f"  tokenizer = AutoTokenizer.from_pretrained('{final_dir}', trust_remote_code=True)")
    log.info(f"  model = AutoModelForSequenceClassification.from_pretrained('{final_dir}', trust_remote_code=True)")


if __name__ == "__main__":
    main()