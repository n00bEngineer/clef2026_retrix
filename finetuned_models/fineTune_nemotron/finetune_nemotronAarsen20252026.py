"""
Fine-tuning di nvidia/llama-nemotron-rerank-1b-v2 come reranker
per dataset con query tweet-style e corpus scientifico (title+abstract).

Supporta dati multi-anno (2025 + 2026):
  - 2025: topics da subtask4b_query_tweets_{dev,test_gold,train}.json
          corpus: collection_data2025.json
          pubkey: stringa alfanumerica (es. "umvrwgaw")
          uso:    SOLO training (mai nel dev set)
  - 2026: topics da topics.json
          corpus: collection_data.json (o corpus.json)
          pubkey: intero
          uso:    90% training + 10% dev (dev set ufficiale per MRR@5)

Il dev set è sempre costruito SOLO da dati 2026 per essere consistente
con il contest CT26 Task 1.

Architettura: AutoModelForSequenceClassification (SEQ_CLS) su base LLaMA
Loss:         BinaryCrossEntropyWithLogits su score scalare sigmoid
Prompt:       "question:{q} \\n \\n passage:{p}"  (formato ufficiale NVIDIA)
LoRA:         TaskType.SEQ_CLS, r=16, targeting q/k/v/o/gate/up/down proj

Uso (con entrambi gli anni):
  python3 finetune_nemotron_rerankerAarsen20252026.py \\
      --topics_2026       en_train.json \\
      --corpus_2026       collection_data.json \\
      --qrels_2026        TRAINqrels.json \\
      --topics_2025       subtask4b_query_tweets_train.json \\
                          subtask4b_query_tweets_dev.json \\
                          subtask4b_query_tweets_test_gold.json \\
      --corpus_2025       collection_data2025.json \\
      --output_dir        models/reranker-nemotron-1bAarsen20252026 \\
      --epochs 3 --batch_size 8 --lr 2e-5

Uso (solo 2026):
  python finetune_nemotron_reranker.py \\
      --topics_2026  data/topics.json \\
      --corpus_2026  data/corpus.json \\
      --output_dir   models/reranker-nemotron-1b
"""

import argparse
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_ID = "nvidia/llama-nemotron-rerank-1b-v2"


# ---------------------------------------------------------------------------
# Prompt format ufficiale NVIDIA
# ---------------------------------------------------------------------------

def make_prompt(query: str, passage: str) -> str:
    return f"question:{query} \n \n passage:{passage}"


# ---------------------------------------------------------------------------
# Helpers I/O
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
    """Corpus: title + abstract. Robusto a campi mancanti (funziona per 2025 e 2026)."""
    title    = item.get("title", "").strip()
    abstract = item.get("abstract", "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract or item.get("text", "")


# ---------------------------------------------------------------------------
# Dataset PyTorch
# ---------------------------------------------------------------------------

class RerankDataset(Dataset):
    """
    Ogni campione è una coppia (query, passage) con label 0.0 o 1.0.
    Applica il prompt format NVIDIA prima della tokenizzazione.
    """

    def __init__(self, records: list[dict], num_negatives: int = 5):
        self.pairs = []
        for rec in records:
            q    = rec["query"]
            pos  = rec["positive"]
            negs = rec.get("negatives", [])
            self.pairs.append((make_prompt(q, pos), 1.0))
            for neg in negs[:num_negatives]:
                self.pairs.append((make_prompt(q, neg), 0.0))

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
# Mining hard negatives — funzione generica per un singolo anno
# ---------------------------------------------------------------------------

def mine_for_year(
    topics: list[dict],
    pubkey_to_text: dict,
    qrels: dict,
    num_negatives: int,
    use_faiss: bool,
    year_label: str,
) -> list[dict]:
    """
    Esegue il mining hard negatives per un singolo anno/corpus.
    Restituisce una lista di records {query, positive, negatives, qid, pubkey_gold}.
    I qid vengono prefissati con year_label per evitare collisioni tra anni
    (es. qid=16 del 2025 e qid=16 del 2026 sono query diverse).
    """
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import mine_hard_negatives
    from datasets import Dataset as HFDataset

    queries_list, answers_list, pubkeys_list, qids_list = [], [], [], []

    for topic in topics:
        qid    = str(topic["index"])
        pubkey = str(topic["pubkey"])
        q_text = topic.get("text", "")
        if not q_text:
            continue

        gold_pk = None
        if pubkey in pubkey_to_text:
            gold_pk = pubkey
        elif qid in qrels:
            for pk, score in qrels[qid].items():
                if int(score) > 0 and pk in pubkey_to_text:
                    gold_pk = pk
                    break

        if gold_pk:
            queries_list.append(q_text)
            answers_list.append(pubkey_to_text[gold_pk])
            pubkeys_list.append(gold_pk)
            qids_list.append(f"{year_label}_{qid}")

    logger.info(f"  [{year_label}] Coppie (Q, PD) valide: {len(queries_list)}")

    if not queries_list:
        logger.warning(f"  [{year_label}] Nessuna coppia valida — skip")
        return []

    raw_dataset = HFDataset.from_dict({
        "question": queries_list,
        "answer":   answers_list,
    })

    logger.info(f"  [{year_label}] Mining hard negatives con static-retrieval-mrl-en-v1...")
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
        absolute_margin=0.05,
        sampling_strategy="top",
        batch_size=512,
        output_format="triplet",
        use_faiss=use_faiss,
    )
    del embedding_model
    torch.cuda.empty_cache()

    # Converti triplet → record raggruppati per query
    grouped = defaultdict(lambda: {
        "query": "", "positive": "", "negatives": [],
        "qid": "", "pubkey_gold": "",
    })

    for i, row in enumerate(hard_dataset):
        key = row["question"]
        idx = i % len(qids_list)
        grouped[key]["query"]       = row["question"]
        grouped[key]["positive"]    = row["answer"]
        grouped[key]["negatives"].append(row["negative"])
        grouped[key]["qid"]         = qids_list[idx]
        grouped[key]["pubkey_gold"] = pubkeys_list[idx]

    records = list(grouped.values())
    logger.info(f"  [{year_label}] Records con hard negatives: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Pipeline mining multi-anno
# ---------------------------------------------------------------------------

def build_train_dev_records(args) -> tuple[list[dict], list[dict]]:
    """
    Costruisce train_records e dev_records combinando 2025 e 2026.

    Regola:
      - 2026 → split 90/10: 90% train + 10% dev
      - 2025 → 100% train (mai nel dev)

    Il dev è sempre solo 2026 per consistenza con il contest.
    """
    all_train_records = []
    dev_records       = []

    # ------------------------------------------------------------------
    # Anno 2026
    # ------------------------------------------------------------------
    if args.topics_2026 and args.corpus_2026:
        logger.info("\n--- Mining 2026 ---")
        corpus_2026 = load_json(args.corpus_2026)
        pubkey_to_text_2026 = {str(item["pubkey"]): doc_to_text(item) for item in corpus_2026}

        topics_2026 = load_json(args.topics_2026)
        qrels_2026  = load_json(args.qrels_2026) if args.qrels_2026 else {}

        records_2026 = mine_for_year(
            topics=topics_2026,
            pubkey_to_text=pubkey_to_text_2026,
            qrels=qrels_2026,
            num_negatives=args.num_negatives,
            use_faiss=args.use_faiss,
            year_label="2026",
        )

        # Split 90/10 — dev sempre da 2026
        random.shuffle(records_2026)
        split = int(len(records_2026) * 0.9)
        train_2026 = records_2026[:split]
        dev_records = records_2026[split:]

        all_train_records.extend(train_2026)
        logger.info(f"2026 → train: {len(train_2026)} | dev: {len(dev_records)}")
    else:
        logger.warning("Nessun dato 2026 fornito — dev set sarà vuoto!")

    # ------------------------------------------------------------------
    # Anno 2025 — tutti i topics vanno in training
    # ------------------------------------------------------------------
    if args.topics_2025 and args.corpus_2025:
        logger.info("\n--- Mining 2025 ---")
        corpus_2025 = load_json(args.corpus_2025)
        pubkey_to_text_2025 = {str(item["pubkey"]): doc_to_text(item) for item in corpus_2025}

        # Carica e concatena tutti i file topics 2025
        topics_2025 = []
        for topics_path in args.topics_2025:
            loaded = load_json(topics_path)
            topics_2025.extend(loaded)
            logger.info(f"  Caricato {topics_path}: {len(loaded)} topics")

        # Deduplica per (index, pubkey) nel caso ci siano overlap tra i file
        seen_keys = set()
        topics_2025_dedup = []
        for t in topics_2025:
            key = (t["index"], str(t["pubkey"]))
            if key not in seen_keys:
                seen_keys.add(key)
                topics_2025_dedup.append(t)
        logger.info(f"  Topics 2025 totali (dopo dedup): {len(topics_2025_dedup)}")

        records_2025 = mine_for_year(
            topics=topics_2025_dedup,
            pubkey_to_text=pubkey_to_text_2025,
            qrels={},          # i topics 2025 hanno già il pubkey diretto
            num_negatives=args.num_negatives,
            use_faiss=args.use_faiss,
            year_label="2025",
        )

        # Tutto in training
        all_train_records.extend(records_2025)
        logger.info(f"2025 → train: {len(records_2025)} | dev: 0 (by design)")

    # Shuffle finale del training set misto
    random.shuffle(all_train_records)

    logger.info(f"\nTotale train: {len(all_train_records)} | dev: {len(dev_records)}")
    return all_train_records, dev_records


# ---------------------------------------------------------------------------
# Valutazione MRR@k sul dev set
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mrr(
    model, tokenizer, dev_records: list[dict],
    max_length: int, batch_size: int, device: str, k: int = 5,
) -> float:
    model.eval()
    mrr_scores = []

    for rec in dev_records:
        q    = rec["query"]
        pos  = rec["positive"]
        negs = rec.get("negatives", [])
        if not negs:
            continue

        candidates = [pos] + negs
        prompts    = [make_prompt(q, c) for c in candidates]
        labels_gt  = [1.0] + [0.0] * len(negs)
        scores     = []

        for i in range(0, len(prompts), batch_size):
            enc = tokenizer(
                prompts[i:i + batch_size],
                padding=True, truncation=True,
                max_length=max_length, return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.squeeze(-1)
            scores.extend(torch.sigmoid(logits).cpu().tolist())

        ranked = sorted(zip(scores, labels_gt), key=lambda x: x[0], reverse=True)
        for rank, (_, label) in enumerate(ranked[:k], start=1):
            if label == 1.0:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    model.train()
    return sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Nemotron-Rerank-1B con dati 2025+2026"
    )

    # --- Dati 2026 ---
    parser.add_argument("--topics_2026", type=str, default=None,
                        help="topics.json del 2026 (corpus collection_data.json)")
    parser.add_argument("--corpus_2026", type=str, default=None,
                        help="collection_data.json (corpus 2026, pubkey=int)")
    parser.add_argument("--qrels_2026",  type=str, default=None,
                        help="qrels.json del 2026 (opzionale se i topics hanno già pubkey)")

    # --- Dati 2025 ---
    parser.add_argument("--topics_2025", type=str, nargs="+", default=None,
                        help="Uno o più file topics 2025 (subtask4b_query_tweets_*.json)")
    parser.add_argument("--corpus_2025", type=str, default=None,
                        help="collection_data2025.json (pubkey=stringa alfanumerica)")

    # --- Percorso alternativo: JSONL pre-processati ---
    parser.add_argument("--train_jsonl", type=str, default=None)
    parser.add_argument("--dev_jsonl",   type=str, default=None)

    # --- Modello ---
    parser.add_argument("--model",      type=str, default=MODEL_ID)
    parser.add_argument("--output_dir", type=str, default="models/reranker-nemotron-1b")

    # --- Mining ---
    parser.add_argument("--num_negatives", type=int,  default=5)
    parser.add_argument("--use_faiss",     action="store_true")

    # --- Training ---
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length",   type=int,   default=512)
    parser.add_argument("--eval_steps",   type=int,   default=200)
    parser.add_argument("--save_steps",   type=int,   default=800)

    # --- LoRA ---
    parser.add_argument("--lora_r",       type=int,   default=16)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args   = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Dati
    # ------------------------------------------------------------------
    if args.train_jsonl and args.dev_jsonl:
        logger.info("Percorso JSONL pre-processati...")
        train_records = load_jsonl(args.train_jsonl)
        dev_records   = load_jsonl(args.dev_jsonl)

    elif args.topics_2026 or args.topics_2025:
        logger.info("Mining automatico hard negatives (multi-anno)...")
        train_records, dev_records = build_train_dev_records(args)

    else:
        raise ValueError(
            "Fornisci almeno uno tra:\n"
            "  --topics_2026 + --corpus_2026\n"
            "  --topics_2025 + --corpus_2025\n"
            "  --train_jsonl + --dev_jsonl"
        )

    if not train_records:
        raise RuntimeError("Training set vuoto — controlla i percorsi dei file.")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    logger.info(f"Caricamento tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Config + Modello (workaround RoPE incompatibility)
    # ------------------------------------------------------------------
    logger.info("Caricamento config con trust_remote_code=True...")
    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        num_labels=1,
    )
    config.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Caricamento modello base: {args.model}")
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
        modules_to_save=["score"],
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
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Train pairs: {len(train_dataset)} | Dev records: {len(dev_records)}")

    # ------------------------------------------------------------------
    # Optimizer e scheduler
    # ------------------------------------------------------------------
    optimizer    = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    total_steps  = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("AVVIO FINE-TUNING NEMOTRON (multi-anno 2025+2026)")
    logger.info(f"  Modello      : {args.model}")
    logger.info(f"  Epoche       : {args.epochs}")
    logger.info(f"  Batch size   : {args.batch_size} × grad_accum={args.grad_accum} → eff {args.batch_size * args.grad_accum}")
    logger.info(f"  LR           : {args.lr}")
    logger.info(f"  LoRA r/alpha : {args.lora_r}/{args.lora_alpha}")
    logger.info(f"  Max length   : {args.max_length}")
    logger.info(f"  Output       : {args.output_dir}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    best_mrr    = 0.0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for step, (enc, labels) in enumerate(pbar, start=1):
            enc    = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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

                # Eval periodica (solo se dev set disponibile)
                if dev_records and global_step % args.eval_steps == 0:
                    mrr = evaluate_mrr(
                        model, tokenizer, dev_records,
                        args.max_length, args.batch_size * 2, device,
                    )
                    logger.info(f"Step {global_step} | MRR@5 dev (2026): {mrr:.4f}")
                    if mrr > best_mrr:
                        best_mrr = mrr
                        best_path = os.path.join(args.output_dir, "best")
                        model.save_pretrained(best_path)
                        tokenizer.save_pretrained(best_path)
                        logger.info(f"  → Best MRR@5={mrr:.4f} salvato in {best_path}")

                if global_step % args.save_steps == 0:
                    ckpt = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    model.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

        if dev_records:
            mrr = evaluate_mrr(
                model, tokenizer, dev_records,
                args.max_length, args.batch_size * 2, device,
            )
            logger.info(f"Epoch {epoch} | MRR@5 dev (2026): {mrr:.4f}")
            if mrr > best_mrr:
                best_mrr = mrr
                model.save_pretrained(os.path.join(args.output_dir, "best"))
                tokenizer.save_pretrained(os.path.join(args.output_dir, "best"))
                logger.info(f"  → Best MRR@5={mrr:.4f}")

    # ------------------------------------------------------------------
    # Salvataggio finale
    # ------------------------------------------------------------------
    final_path = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Modello finale: {final_path}")
    logger.info(f"Best MRR@5 dev raggiunto: {best_mrr:.4f}")


if __name__ == "__main__":
    main()
