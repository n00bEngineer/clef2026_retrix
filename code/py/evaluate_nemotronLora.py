"""
Evaluate a Nemotron reranker checkpoint on Retrix candidate rankings.

Two checkpoint layouts are supported:
  A) `--model_dir` points to a LoRA adapter
     (`adapter_config.json` + `adapter_model.safetensors`). The script loads
     the base model and merges the adapter with `merge_and_unload()`.
  B) `--model_dir` points to full model weights
     (`config.json` + `model.safetensors`). The script loads the model directly
     without PEFT.

The output is a `{qid: [pubkey, ...]}` JSON object compatible with the Java
evaluator. If qrels are provided and `ranx` is installed, local MRR@5, MAP, and
nDCG@10 metrics are logged for quick validation.

LoRA checkpoint usage:
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

Base model comparison usage:
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
    """Load a UTF-8 JSON document.

    Args:
        path: Path to the JSON file.

    Returns:
        The decoded JSON object.

    Raises:
        OSError: If the file cannot be read.
        json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def doc_to_text(item: dict) -> str:
    """Build the passage text used by the cross-encoder.

    Args:
        item: Corpus record that may contain `title`, `abstract` and `authors`.

    Returns:
        A non-normalized passage string assembled from available fields.
    """
    title    = item.get("title", "").strip()
    abstract = item.get("abstract", "").strip()
    authors  = item.get("authors", "").strip()
    
    body = f"{title}. {abstract}" if title and abstract else title or abstract
    if authors:
        body = f"{body} Authors: {authors}"
    return body or item.get("text", "")



def make_prompt(query: str, passage: str) -> str:
    """Format a query-passage pair using NVIDIA's Nemotron template."""
    return f"question:{query} \n \n passage:{passage}"


def is_lora_adapter(model_dir: str) -> bool:
    """Return whether `model_dir` contains PEFT LoRA adapter metadata."""
    return os.path.isfile(os.path.join(model_dir, "adapter_config.json"))


# ---------------------------------------------------------------------------
# Model loading: auto-detect LoRA adapters versus full checkpoints.
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_dir: str, base_model_id: str):
    """Load a Nemotron reranker model and matching tokenizer.

    Args:
        model_dir: LoRA adapter directory, full checkpoint directory, or model ID.
        base_model_id: Hugging Face model ID used when `model_dir` is a LoRA
            adapter.

    Returns:
        A `(model, tokenizer)` tuple ready for inference.

    Raises:
        ImportError: If required transformer or PEFT packages are unavailable.
        OSError: If the checkpoint or tokenizer files cannot be loaded.
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

    # The presence of adapter_config.json is the PEFT convention for LoRA output.
    lora_mode = is_lora_adapter(model_dir)

    if lora_mode:
        logger.info(f"LoRA adapter detected in: {model_dir}")
        logger.info(f"Base model: {base_model_id}")
    else:
        logger.info(f"Full weights detected, loading directly from: {model_dir}")
        base_model_id = model_dir  # Use model_dir as the direct checkpoint source.

    # ------------------------------------------------------------------
    # Tokenizer: LoRA adapters do not change the base tokenizer vocabulary.
    # ------------------------------------------------------------------
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Config: set pad_token_id explicitly to avoid classification-head padding
    # ambiguities across transformers versions.
    # ------------------------------------------------------------------
    logger.info("Loading config...")
    config = AutoConfig.from_pretrained(
        base_model_id,
        trust_remote_code=True,
        num_labels=1,
    )
    config.pad_token_id = tokenizer.pad_token_id

    # ------------------------------------------------------------------
    # Base model.
    # ------------------------------------------------------------------
    logger.info("Loading base model...")
    base_model = AutoModelForSequenceClassification.from_pretrained(
        base_model_id,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        ignore_mismatched_sizes=True,
    )

    # ------------------------------------------------------------------
    # Merge LoRA weights only when the checkpoint is an adapter.
    # ------------------------------------------------------------------
    if lora_mode:
        from peft import PeftModel

        logger.info("Loading and merging LoRA adapter...")
        peft_model = PeftModel.from_pretrained(
            base_model,
            model_dir,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Executing merge_and_unload()...")
        model = peft_model.merge_and_unload()
        logger.info("Merge completed — model ready for inference")
    else:
        model = base_model

    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
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
    """Score query-passage prompts with the Nemotron classification head.

    Args:
        model: Sequence-classification model returning one logit per prompt.
        tokenizer: Matching tokenizer.
        prompts: Formatted `question: ... passage: ...` strings.
        batch_size: Number of prompts processed per forward pass.
        max_length: Maximum token length after truncation.
        device: Device where tokenized tensors are moved before inference.

    Returns:
        Relevance scores in the `[0, 1]` interval. If a CUDA OOM occurs for a
        batch, neutral `0.5` fallback scores are emitted for that batch so the
        caller can preserve candidate ordering.

    Raises:
        RuntimeError: For non-OOM inference failures.
    """
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
            # Nemotron returns a single relevance logit per prompt.
            batch_scores = torch.sigmoid(logits).cpu().tolist()
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            scores.extend(batch_scores)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM in batch {i//batch_size} → fallback to score 0.5")
                torch.cuda.empty_cache()
                scores.extend([0.5] * len(batch))
            else:
                raise
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run checkpoint loading, reranking, optional evaluation, and JSON export.

    Side effects:
        Reads topics, corpus, BM25 candidates, and optional qrels from disk;
        loads a large model; writes reranked results; and logs progress and
        optional local metrics.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Nemotron (LoRA or full weights)"
    )
    parser.add_argument("--model_dir",    required=True,
                        help="Directory with LoRA adapter or full weights")
    parser.add_argument("--base_model",   type=str, default=BASE_MODEL_ID,
                        help="HF model ID of base model (only used if model_dir is a new LoRA adapter)")
    parser.add_argument("--topics",       required=True)
    parser.add_argument("--corpus",       required=True)
    parser.add_argument("--bm25_results", required=True,
                        help="JSON {qid: [pubkey, ...]} from the first BM25 stage")
    parser.add_argument("--output",       required=True,
                        help="JSON output for Evaluator.java")
    parser.add_argument("--top_k",        type=int, default=100,
                        help="How many BM25 candidates the reranker sees per query")
    parser.add_argument("--rerank_top",   type=int, default=20,
                        help="How many pubkeys to write in the final JSON per query")
    parser.add_argument("--batch_size",   type=int, default=32)
    parser.add_argument("--max_length",   type=int, default=512)
    parser.add_argument("--qrels",        type=str, default=None,
                        help="qrels.json for local metrics (optional, requires ranx)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------------
    # Model loading.
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args.model_dir, args.base_model)

    # ------------------------------------------------------------------
    # Data loading.
    # ------------------------------------------------------------------
    logger.info("Loading topics and corpus...")
    topics       = load_json(args.topics)
    corpus_list  = load_json(args.corpus)
    bm25_results = load_json(args.bm25_results)

    pubkey_to_text = {str(item["pubkey"]): doc_to_text(item) for item in corpus_list}
    qid_to_text    = {str(t["index"]): t.get("text", "") for t in topics}

    logger.info(f"Corpus: {len(pubkey_to_text)} documents | Topics: {len(qid_to_text)} queries")

    # ------------------------------------------------------------------
    # Reranking.
    # ------------------------------------------------------------------
    logger.info(f"Reranking — top_k={args.top_k}, rerank_top={args.rerank_top}...")
    final_results = {}

    for qid, candidate_pubkeys in tqdm(bm25_results.items(), desc="Queries"):
        q_text = qid_to_text.get(str(qid), "")
        if not q_text:
            # Missing topics cannot be scored; preserve first-stage ordering.
            final_results[str(qid)] = [str(pk) for pk in candidate_pubkeys[:args.rerank_top]]
            continue

        candidates      = [str(pk) for pk in candidate_pubkeys[:args.top_k]]
        candidate_texts = [pubkey_to_text.get(pk, "") for pk in candidates]

        # Candidates without corpus text cannot be scored by the cross-encoder.
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
    # Output for Evaluator.java.
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Results saved in: {out_path}")

    # ------------------------------------------------------------------
    # Optional local metrics; ranx is intentionally not a hard dependency.
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
            logger.info("LOCAL METRICS (ranx)")
            for m, v in metrics.items():
                logger.info(f"  {m:15s}: {v:.4f}")
            logger.info("=" * 50)

        except ImportError:
            logger.warning("ranx not installed → pip install ranx")


if __name__ == "__main__":
    main()
