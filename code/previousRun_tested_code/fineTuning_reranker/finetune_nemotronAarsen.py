import argparse
import json
import logging
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from huggingface_hub import login, create_repo, HfApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/llama-nemotron-rerank-1b-v2"
HUB_MODEL_ID = "Gigi332/nemotronFT_Aarsen_EnTrain2026"


# ---------------------------------------------------------------------------
# Official NVIDIA prompt format
# ---------------------------------------------------------------------------

def make_prompt(query: str, passage: str) -> str:
    return f"question:{query} \n \n passage:{passage}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def doc_to_text(item: dict) -> str:
    """Corpus: title + abstract. Robust to missing fields."""
    title    = item.get("title", "").strip()
    abstract = item.get("abstract", "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract or item.get("text", "")


def get_positive(rec: dict) -> str:
    """Supports both 'positive' (mining) and 'pos' (dataset JSONL)."""
    return rec.get("positive") or rec.get("pos", "")


def get_negatives(rec: dict) -> list[str]:
    """
    Supports:
      - 'negatives': list[str]   (mining)
      - 'pos'/'neg'/'hard_neg'   (JSONL dataset with single field)
    """
    if "negatives" in rec:
        return rec["negatives"]
    negs = []
    for field in ("neg", "hard_neg"):
        val = rec.get(field)
        if isinstance(val, list):
            negs.extend(val)
        elif isinstance(val, str) and val:
            negs.append(val)
    return negs


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class RerankDataset(Dataset):
    """
    Every sample is a (query, passage) pair with label float 0.0 or 1.0.
    Applies the NVIDIA prompt format before tokenization.
    Supports both mining (positive/negatives) and JSONL (pos/neg/hard_neg) dataset formats.
    """

    def __init__(self, records: list[dict], num_negatives: int = 5):
        self.pairs = []   # (prompt_text, label)
        skipped = 0
        for rec in records:
            q   = rec.get("query", "")
            pos = get_positive(rec)
            negs = get_negatives(rec)

            if not q or not pos:
                skipped += 1
                continue

            self.pairs.append((make_prompt(q, pos), 1.0))
            for neg in negs[:num_negatives]:
                if neg:
                    self.pairs.append((make_prompt(q, neg), 0.0))

        if skipped:
            logger.warning(f"Dataset: skipped {skipped} record without query or positive.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


def collate_fn(batch, tokenizer, max_length):
    texts  = [x[0] for x in batch]
    labels = torch.tensor([x[1] for x in batch], dtype=torch.float32)
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc, labels


# ---------------------------------------------------------------------------
# Mining hard negatives from topics/corpus/qrels
# ---------------------------------------------------------------------------

def mine_negatives_from_raw(
    topics_path: str,
    corpus_path: str,
    qrels_path: str,
    num_negatives: int,
    use_faiss: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (train_records, dev_records) in the same JSONL format.
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives
    from datasets import Dataset as HFDataset

    logger.info("Loading topics, corpus, qrels...")
    topics      = load_json(topics_path)
    corpus_list = load_json(corpus_path)
    qrels       = load_json(qrels_path) if qrels_path else {}

    pubkey_to_text = {str(item["pubkey"]): doc_to_text(item) for item in corpus_list}

    queries_list, answers_list, pubkeys_list, qids_list = [], [], [], []
    for topic in topics:
        qid    = str(topic["index"])
        pubkey = str(topic["pubkey"])
        q_text = topic.get("text", "")

        if pubkey in pubkey_to_text:
            queries_list.append(q_text)
            answers_list.append(pubkey_to_text[pubkey])
            pubkeys_list.append(pubkey)
            qids_list.append(qid)
        elif qid in qrels:
            for pk, score in qrels[qid].items():
                if int(score) > 0 and pk in pubkey_to_text:
                    queries_list.append(q_text)
                    answers_list.append(pubkey_to_text[pk])
                    pubkeys_list.append(pk)
                    qids_list.append(qid)
                    break

    logger.info(f"query-answer pairs: {len(queries_list)}")

    raw_dataset = HFDataset.from_dict({
        "question": queries_list,
        "answer":   answers_list,
    })

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
        output_format="triplet",   # → question, answer, negative
        use_faiss=use_faiss,
    )

    # Converts in internal JSONL format
    records = []
    for i, row in enumerate(hard_dataset):
        records.append({
            "query":      row["question"],
            "positive":   row["answer"],
            "negatives":  [row["negative"]],
            "qid":        int(qids_list[i % len(qids_list)]),
            "pubkey_gold": int(pubkeys_list[i % len(pubkeys_list)]),
        })

    # Groups negatives
    from collections import defaultdict
    grouped = defaultdict(lambda: {"query": "", "positive": "", "negatives": [], "qid": 0, "pubkey_gold": 0})
    for rec in records:
        key = rec["query"]
        grouped[key]["query"]       = rec["query"]
        grouped[key]["positive"]    = rec["positive"]
        grouped[key]["negatives"].extend(rec["negatives"])
        grouped[key]["qid"]         = rec["qid"]
        grouped[key]["pubkey_gold"] = rec["pubkey_gold"]

    all_records = list(grouped.values())
    random.shuffle(all_records)

    split_idx   = int(len(all_records) * 0.9)
    train_records = all_records[:split_idx]
    dev_records   = all_records[split_idx:]

    logger.info(f"Train records: {len(train_records)} | Dev records: {len(dev_records)}")
    return train_records, dev_records


# ---------------------------------------------------------------------------
# Valutazione su dev set (MRR@5 locale)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mrr(model, tokenizer, dev_records: list[dict], max_length: int,
                 batch_size: int, device: str, k: int = 5) -> float:
    """
    Computes MRR@k on the dev set grouped by query.
    """
    model.eval()
    mrr_scores = []

    for rec in dev_records:
        q    = rec.get("query", "")
        pos  = get_positive(rec)
        negs = get_negatives(rec)

        if not q or not pos:
            continue

        candidates = [pos] + [n for n in negs if n]
        prompts    = [make_prompt(q, c) for c in candidates]
        labels_gt  = [1.0] + [0.0] * (len(candidates) - 1)

        scores = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            enc = tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())

        ranked = sorted(zip(scores, labels_gt), key=lambda x: x[0], reverse=True)
        mrr = 0.0
        for rank, (_, label) in enumerate(ranked[:k], start=1):
            if label == 1.0:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

    model.train()
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0


# ---------------------------------------------------------------------------
# Push su Hugging Face Hub
# ---------------------------------------------------------------------------

def _format_mrr(metrics: dict) -> str:
    """
    Returns a string formatted for MRR to use in the commit message.
    Handles both float values and 'N/A' without crashing.
    """
    if not metrics:
        return ""
    val = metrics.get("mrr", metrics.get("best_mrr"))
    if val is None:
        return ""
    if isinstance(val, float):
        return f" (MRR: {val:.4f})"
    # Stringa (es. 'N/A') — non formattare come float
    return f" (MRR: {val})"


def push_to_hub(model, tokenizer, checkpoint_name=None, metrics=None, is_final=False):
    """
    Push of the model on Hugging Face Hub.
    Use the environment variable HF_TOKEN for authentication.
    Repository: Gigi332/nemotronFT_Aarsen_EnTrain2026
    """
    import tempfile

    # Determines the path in the repository
    repo_path = "" if (is_final or checkpoint_name is None) else checkpoint_name

    # Temporarely saves the model
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / repo_path if repo_path else Path(tmp_dir)

        # Saves model and tokenizer
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        # Saves metrics
        if metrics:
            with open(save_path / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

        # creates the README only for the final model
        if is_final:
            best_mrr_val = metrics.get("best_mrr", metrics.get("mrr")) if metrics else None
            if isinstance(best_mrr_val, float):
                mrr_str = f"{best_mrr_val:.4f}"
            else:
                mrr_str = str(best_mrr_val) if best_mrr_val is not None else "N/A"

            readme_content = f"""---
tags:
- reranker
- peft
- lora
---

# Nemotron Reranker Model

## Metrics
- Best MRR@5: {mrr_str}

## Usage
```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

base_model = AutoModelForSequenceClassification.from_pretrained(
    "nvidia/llama-nemotron-rerank-1b-v2", trust_remote_code=True
)
model = PeftModel.from_pretrained(base_model, "Gigi332/nemotronFT_Aarsen_EnTrain2026")
tokenizer = AutoTokenizer.from_pretrained("Gigi332/nemotronFT_Aarsen_EnTrain2026")
tokenizer.padding_side = "left"
```"""
            with open(Path(tmp_dir) / "README.md", "w") as f:
                f.write(readme_content)

        # Upload on Hub
        api = HfApi()
        commit_msg = f"Upload {checkpoint_name or 'final model'}" + _format_mrr(metrics)
        api.upload_folder(
            folder_path=str(save_path),
            repo_id=HUB_MODEL_ID,
            repo_type="model",
            commit_message=commit_msg,
        )

        logger.info(f"✅ Modello pushed on Hub: {HUB_MODEL_ID}/{repo_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Nemotron-Rerank-1B")

    # Input
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--dev_jsonl",   type=str, default=None)
    parser.add_argument("--topics",      type=str, default=None)
    parser.add_argument("--corpus",      type=str, default=None)
    parser.add_argument("--qrels",       type=str, default=None)

    # Modello
    parser.add_argument("--model",      type=str, default=MODEL_ID)
    parser.add_argument("--output_dir", type=str, default="models/reranker-nemotron-1b")

    # Iperparametri
    parser.add_argument("--num_negatives",  type=int,   default=5)
    parser.add_argument("--epochs",         type=int,   default=3)
    parser.add_argument("--batch_size",     type=int,   default=8)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--warmup_ratio",   type=float, default=0.1)
    parser.add_argument("--max_length",     type=int,   default=512)
    parser.add_argument("--grad_accum",     type=int,   default=4,
                        help="Gradient accumulation steps (effective_batch = batch_size * grad_accum)")
    parser.add_argument("--use_faiss",      action="store_true")
    parser.add_argument("--eval_steps",     type=int,   default=200)
    parser.add_argument("--save_steps",     type=int,   default=600)

    # LoRA
    parser.add_argument("--lora_r",         type=int,   default=16)
    parser.add_argument("--lora_alpha",     type=int,   default=32)
    parser.add_argument("--lora_dropout",   type=float, default=0.05)

    # Hugging Face Hub
    parser.add_argument("--push_to_hub",    action="store_true", default=True,
                        help="Push checkpoints to Hugging Face Hub")
    parser.add_argument("--hub_strategy",   type=str, default="best",
                        choices=["best", "final", "all", "every_save"],
                        help="Strategy for pushing checkpoints to Hub")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Setup Hugging Face Hub
    # ------------------------------------------------------------------
    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            logger.info("✅ Autenticated on Hugging Face Hub via HF_TOKEN")
        else:
            logger.warning("⚠️ HF_TOKEN not found in the evironment variables. Trying interactive login...")
            login()

        try:
            create_repo(repo_id=HUB_MODEL_ID, exist_ok=True)
            logger.info(f"✅ Repository Hub ready: {HUB_MODEL_ID}")
        except Exception as e:
            logger.warning(f"Error in repo verification/creation: {e}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.train_jsonl and args.dev_jsonl:
        logger.info("Loading data from pre-processed JSONL...")
        train_records = load_jsonl(args.train_jsonl)
        dev_records   = load_jsonl(args.dev_jsonl)
    elif args.topics and args.corpus:
        logger.info("Mining hard negatives from topics/corpus...")
        train_records, dev_records = mine_negatives_from_raw(
            topics_path=args.topics,
            corpus_path=args.corpus,
            qrels_path=args.qrels,
            num_negatives=args.num_negatives,
            use_faiss=args.use_faiss,
        )
    else:
        raise ValueError(
            "Provide --train_jsonl + --dev_jsonl  or  --topics + --corpus"
        )

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    from transformers import AutoConfig
    logger.info("loading config with trust_remote_code=True...")
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        num_labels=1,
    )
    config.pad_token_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Base model
    # ------------------------------------------------------------------
    logger.info(f"Loading base model: {args.model}")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )

    # ------------------------------------------------------------------
    # LoRA
    # NOTE: modules_to_save=["score"] REMOVED: it would cause the re-initialization
    # of Nemotron's pre-trained ranking head, destroying its calibration.
    # The ranking head is left frozen; only the LoRA layers are updated.
    # ------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        # modules_to_save=["score"] deliberately removed
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    model.to(device)

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    train_dataset = RerankDataset(train_records, args.num_negatives)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length),
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Train pairs: {len(train_dataset)} | Dev records: {len(dev_records)}")

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ------------------------------------------------------------------
    # Log parametri
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("LAUNCHING NEMOTRON FINE-TUNING")
    logger.info(f"  Model        : {args.model}")
    logger.info(f"  Epoche       : {args.epochs}")
    logger.info(f"  Batch size   : {args.batch_size} (grad_accum={args.grad_accum} → eff {args.batch_size * args.grad_accum})")
    logger.info(f"  LR           : {args.lr}")
    logger.info(f"  LoRA r/alpha : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"  Max length   : {args.max_length}")
    logger.info(f"  Output       : {args.output_dir}")
    if args.push_to_hub:
        logger.info(f"  Hub push     : enabled → {HUB_MODEL_ID}")
        logger.info(f"  Hub strategy : {args.hub_strategy}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_mrr    = 0.0
    global_step = 0
    pushed_checkpoints = set()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (enc, labels) in enumerate(pbar, start=1):
            enc    = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            if device == "cuda":
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(**enc).logits.squeeze(-1)
                    loss   = F.binary_cross_entropy_with_logits(logits, labels)
                    loss   = loss / args.grad_accum
            else:
                logits = model(**enc).logits.squeeze(-1)
                loss   = F.binary_cross_entropy_with_logits(logits, labels)
                loss   = loss / args.grad_accum

            loss.backward()
            epoch_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                pbar.set_postfix(loss=f"{epoch_loss / step:.4f}", step=global_step)

                # Eval periodica
                if global_step % args.eval_steps == 0:
                    mrr = evaluate_mrr(
                        model, tokenizer, dev_records,
                        args.max_length, args.batch_size * 2, device
                    )
                    logger.info(f"Step {global_step} | MRR@5 dev: {mrr:.4f}")

                    if mrr > best_mrr:
                        best_mrr  = mrr
                        best_path = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        logger.info(f"  → Local new best stored (MRR@5={mrr:.4f})")

                        if args.push_to_hub and args.hub_strategy in ["best", "every_save"]:
                            push_to_hub(
                                model=model,
                                tokenizer=tokenizer,
                                checkpoint_name="best",
                                metrics={"mrr": mrr, "step": global_step},
                                is_final=False,
                            )

                # Periodic checkpoint
                if global_step % args.save_steps == 0:
                    ckpt_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt_path)
                    tokenizer.save_pretrained(ckpt_path)
                    logger.info(f"Local checkpoint stored: {ckpt_path}")

                    if args.push_to_hub and args.hub_strategy in ["all", "every_save"]:
                        if ckpt_path not in pushed_checkpoints:
                            push_to_hub(
                                model=model,
                                tokenizer=tokenizer,
                                checkpoint_name=f"checkpoint-{global_step}",
                                metrics={"loss": epoch_loss / step, "step": global_step},
                                is_final=False,
                            )
                            pushed_checkpoints.add(ckpt_path)

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} complete | avg loss: {avg_loss:.4f}")

        # Eval after epoch
        mrr = evaluate_mrr(
            model, tokenizer, dev_records,
            args.max_length, args.batch_size * 2, device
        )
        logger.info(f"Epoch {epoch} | MRR@5 dev: {mrr:.4f}")

        if mrr > best_mrr:
            best_mrr  = mrr
            best_path = os.path.join(args.output_dir, "best")
            model.save_pretrained(best_path)
            tokenizer.save_pretrained(best_path)
            logger.info(f"  → New best stored (MRR@5={mrr:.4f})")

            if args.push_to_hub and args.hub_strategy in ["best", "every_save"]:
                push_to_hub(
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_name="best",
                    metrics={"mrr": mrr, "epoch": epoch},
                    is_final=False,
                )

    # ------------------------------------------------------------------
    # Final store
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Final model stored locally in: {final_path}")

    if args.push_to_hub:
        logger.info("Final model push on Hugging Face Hub...")
        push_to_hub(
            model=model,
            tokenizer=tokenizer,
            checkpoint_name="final",
            metrics={"best_mrr": best_mrr},
            is_final=True,
        )

        if best_mrr > 0 and args.hub_strategy != "best":
            logger.info("Loading best model as main model...")
            from peft import PeftModel
            best_model = PeftModel.from_pretrained(base_model, best_path)
            push_to_hub(
                model=best_model,
                tokenizer=tokenizer,
                checkpoint_name=None,
                metrics={"best_mrr": best_mrr},
                is_final=True,
            )

    logger.info(f"Best MRR@5 dev reached: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
