"""
Valutazione del reranker Nemotron fine-tunato con LoRA sul dataset Retrix.

Supporta due modalità di caricamento:
  A) --model_dir con adapter LoRA (adapter_config.json + adapter_model.safetensors)
     → carica il base model da HF, fonde i pesi LoRA con merge_and_unload()
  B) --model_dir con pesi completi (config.json + model.safetensors)
     → caricamento diretto senza PEFT (es. modello base non fine-tunato)

Lo script rileva automaticamente quale modalità usare in base alla presenza
di adapter_config.json nella directory.

Produce {qid → [pubkey, ...]} compatibile con Evaluator.java.
Stampa MRR@5, MAP, nDCG@10 locali se --qrels è fornito (richiede ranx).

Uso (modello fine-tunato con LoRA):
  python evaluate_nemotron_reranker.py \\
      --model_dir    models/reranker-nemotron-1b/best \\
      --base_model   nvidia/llama-nemotron-rerank-1b-v2 \\
      --topics       data/topics.json \\
      --corpus       data/corpus.json \\
      --bm25_results data/bm25_results.json \\
      --output       data/reranked_results_nemotron.json \\
      --top_k        100 \\
      --rerank_top   20 \\
      --qrels        data/qrels.json

Uso (modello base senza fine-tuning, per confronto):
  python evaluate_nemotron_reranker.py \\
      --model_dir    nvidia/llama-nemotron-rerank-1b-v2 \\
      --topics       data/topics.json \\
      --corpus       data/corpus.json \\
      --bm25_results data/bm25_results.json \\
      --output       data/reranked_results_nemotron_base.json \\
      --top_k        100 --rerank_top 20 --qrels data/qrels.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

BASE_MODEL_ID = "nvidia/llama-nemotron-rerank-1b-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def doc_to_text(item: dict) -> str:
    """Corpus: title + abstract. Robusto a campi mancanti."""
    title    = item.get("title", "").strip()
    abstract = item.get("abstract", "").strip()
    if title and abstract:
        return f"{title}. {abstract}"
    return title or abstract or item.get("text", "")


def make_prompt(query: str, passage: str) -> str:
    """Formato prompt ufficiale NVIDIA Nemotron."""
    return f"question:{query} \n \n passage:{passage}"


def is_lora_adapter(model_dir: str) -> bool:
    """Controlla se model_dir contiene un adapter LoRA o pesi completi."""
    return os.path.isfile(os.path.join(model_dir, "adapter_config.json"))


# ---------------------------------------------------------------------------
# Caricamento modello — rileva automaticamente LoRA vs pesi completi
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str, base_model_id: str):
    """
    Carica tokenizer e modello in modo appropriato:
      - Se model_dir ha adapter_config.json → carica base model + fonde LoRA
      - Altrimenti → carica direttamente come modello completo

    Restituisce (model, tokenizer) pronti per l'inferenza.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

    # Determina se è un adapter LoRA o un modello completo
    lora_mode = is_lora_adapter(model_dir)

    if lora_mode:
        logger.info(f"Rilevato adapter LoRA in: {model_dir}")
        logger.info(f"Base model: {base_model_id}")
    else:
        logger.info(f"Pesi completi rilevati, caricamento diretto da: {model_dir}")
        base_model_id = model_dir  # usa model_dir come sorgente diretta

    # ------------------------------------------------------------------
    # Tokenizer — sempre dal base model (il tokenizer non cambia con LoRA)
    # ------------------------------------------------------------------
    logger.info("Caricamento tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Config — workaround incompatibilità RoPE con transformers recenti
    # ------------------------------------------------------------------
    logger.info("Caricamento config...")
    config = AutoConfig.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        num_labels=1,
    )
    config.pad_token_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Modello base
    # ------------------------------------------------------------------
    logger.info("Caricamento modello base...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )

    # ------------------------------------------------------------------
    # Fusione LoRA (solo se in modalità adapter)
    # ------------------------------------------------------------------
    if lora_mode:
        from peft import PeftModel

        logger.info("Caricamento e fusione adapter LoRA...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Esecuzione merge_and_unload()...")
        model = peft_model.merge_and_unload()
        logger.info("Fusione completata — modello pronto per inferenza")
    else:
        model = base_model

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inferenza
# ---------------------------------------------------------------------------

@torch.no_grad()
def batch_score(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_length: int,
    device: str,
) -> list[float]:
    """Calcola score Nemotron per una lista di prompt (query+passage)."""
    scores = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i: i + batch_size]
        try:
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits.squeeze(-1)
            # Nemotron: sigmoid sul logit scalare → score in [0,1]
            batch_scores = torch.sigmoid(logits).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM nel batch {i//batch_size} → fallback a score 0.5")
                torch.cuda.empty_cache()
                scores.extend([0.5] * len(batch))
            else:
                raise
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Valuta Nemotron fine-tunato (LoRA o pesi completi)"
    )
    parser.add_argument("--model_dir",    required=True,
                        help="Directory con adapter LoRA o pesi completi")
    parser.add_argument("--base_model",   type=str, default=BASE_MODEL_ID,
                        help="HF model ID del base model (usato solo se model_dir è un adapter LoRA)")
    parser.add_argument("--topics",       required=True)
    parser.add_argument("--corpus",       required=True)
    parser.add_argument("--bm25_results", required=True,
                        help="JSON {qid: [pubkey, ...]} dal primo stage BM25")
    parser.add_argument("--output",       required=True,
                        help="JSON output per Evaluator.java")
    parser.add_argument("--top_k",        type=int, default=100,
                        help="Quanti candidati BM25 il reranker vede per query")
    parser.add_argument("--rerank_top",   type=int, default=20,
                        help="Quanti pubkey scrivere nel JSON finale per query")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--qrels",        type=str, default=None,
                        help="qrels.json per metriche locali (opzionale, richiede ranx)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Caricamento modello
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model)

    # ------------------------------------------------------------------
    # Caricamento dati
    # ------------------------------------------------------------------
    logger.info("Caricamento topics e corpus...")
    topics       = load_json(args.topics)
    corpus_list  = load_json(args.corpus)
    bm25_results = load_json(args.bm25_results)

    pubkey_to_text = {str(item["pubkey"]): doc_to_text(item) for item in corpus_list}
    qid_to_text    = {str(t["index"]): t.get("text", "") for t in topics}

    logger.info(f"Corpus: {len(pubkey_to_text)} documenti | Topics: {len(qid_to_text)} query")

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    logger.info(f"Reranking — top_k={args.top_k}, rerank_top={args.rerank_top}...")
    final_results = {}

    for qid, candidate_pubkeys in tqdm(bm25_results.items(), desc="Queries"):
        q_text = qid_to_text.get(str(qid), "")
        if not q_text:
            # Query non trovata nei topics → mantieni ordine BM25
            final_results[str(qid)] = [str(pk) for pk in candidate_pubkeys[:args.rerank_top]]
            continue

        candidates      = [str(pk) for pk in candidate_pubkeys[:args.top_k]]
        candidate_texts = [pubkey_to_text.get(pk, "") for pk in candidates]

        # Rimuovi candidati senza testo nel corpus
        valid = [(pk, txt) for pk, txt in zip(candidates, candidate_texts) if txt]
        if not valid:
            final_results[str(qid)] = candidates[:args.rerank_top]
            continue

        valid_pks, valid_txts = zip(*valid)
        prompts = [make_prompt(q_text, txt) for txt in valid_txts]

        scores = batch_score(
            model, tokenizer, list(prompts),
            args.batch_size, args.max_length, device,
        )

        ranked = sorted(zip(valid_pks, scores), key=lambda x: x[1], reverse=True)
        final_results[str(qid)] = [pk for pk, _ in ranked[:args.rerank_top]]

    # ------------------------------------------------------------------
    # Salvataggio output per Evaluator.java
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Risultati salvati in: {out_path}")

    # ------------------------------------------------------------------
    # Metriche locali (opzionale — richiede ranx)
    # ------------------------------------------------------------------
    if args.qrels:
        try:
            from ranx import Qrels, Run, evaluate

            qrels_data = load_json(args.qrels)
            qrels_dict = {
                str(qid): {str(pk): int(s) for pk, s in docs.items()}
                for qid, docs in qrels_data.items()
            }
            run_dict = {
                qid: {pk: (args.rerank_top - i) for i, pk in enumerate(pks)}
                for qid, pks in final_results.items()
            }

            metrics = evaluate(Qrels(qrels_dict), Run(run_dict), ["mrr@5", "map", "ndcg@10"])
            logger.info("=" * 50)
            logger.info("METRICHE LOCALI (ranx)")
            for m, v in metrics.items():
                logger.info(f"  {m:15s}: {v:.4f}")
            logger.info("=" * 50)

        except ImportError:
            logger.warning("ranx non installato → pip install ranx")


if __name__ == "__main__":
    main()
