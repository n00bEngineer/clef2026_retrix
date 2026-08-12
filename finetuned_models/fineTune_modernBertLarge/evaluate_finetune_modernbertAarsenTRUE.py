"""
Valutazione del reranker ModernBERT fine-tunato sul dataset Retrix.

Produce il file JSON {qid → [pubkey, ...]} compatibile con Evaluator.java
e stampa MRR@5, MAP, nDCG@10 locali usando ranx.

Uso:
  python evaluate_modernbert_reranker.py \
      --model_dir     models/reranker-modernbert-large/final \
      --topics        data/topics.json \
      --corpus        data/corpus.json \
      --bm25_results  data/bm25_results.json \
      --output        data/reranked_results.json \
      [--top_k        100] \
      [--rerank_top   20] \
      [--batch_size   64] \
      [--qrels        data/qrels.json]

Formato bm25_results.json:
  { "qid": ["pubkey1", "pubkey2", ...], ... }   (pubkey come stringa o int)
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def batch_score(model, pairs: list[tuple[str, str]], batch_size: int) -> list[float]:
    """Calcola score per una lista di coppie (query, passage) in batch."""
    scores = []
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i: i + batch_size]
        try:
            batch_scores = model.predict(batch, show_progress_bar=False)
            # model.predict restituisce array numpy
            scores.extend(batch_scores.tolist())
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM nel batch → fallback a score 0.5")
                torch.cuda.empty_cache()
                scores.extend([0.5] * len(batch))
            else:
                raise
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--topics", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--bm25_results", required=True,
                        help="JSON {qid: [pubkey, ...]} dal primo stage BM25")
    parser.add_argument("--output", required=True,
                        help="Percorso output JSON per Evaluator.java")
    parser.add_argument("--top_k", type=int, default=100,
                        help="Quanti candidati BM25 considerare")
    parser.add_argument("--rerank_top", type=int, default=20,
                        help="Quanti restituire dopo il reranking")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--qrels", type=str, default=None,
                        help="Path qrels per calcolo metriche locali (opzionale)")
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Caricamento modello
    # ------------------------------------------------------------------
    from sentence_transformers import CrossEncoder

    logger.info(f"Caricamento modello da: {args.model_dir}")
    model = CrossEncoder(
        args.model_dir,
        max_length=args.max_length,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model.model.eval()

    # ------------------------------------------------------------------
    # Caricamento dati
    # ------------------------------------------------------------------
    logger.info("Caricamento topics e corpus...")
    topics = load_json(args.topics)
    corpus_list = load_json(args.corpus)
    bm25_results = load_json(args.bm25_results)

    # Costruisci indice pubkey → testo (corpus: title + abstract)
    def doc_to_text(item):
        title    = item.get("title", "").strip()
        abstract = item.get("abstract", "").strip()
        if title and abstract:
            return f"{title}. {abstract}"
        return title or abstract or item.get("text", "")

    pubkey_to_text = {str(item["pubkey"]): doc_to_text(item) for item in corpus_list}
    # Indice qid → topic text (query: campo "text")
    qid_to_text = {str(t["index"]): t.get("text", "") for t in topics}

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------
    logger.info("Reranking in corso...")
    final_results = {}

    for qid, candidate_pubkeys in tqdm(bm25_results.items(), desc="Queries"):
        q_text = qid_to_text.get(str(qid), "")
        if not q_text:
            # Query non trovata → mantieni ordine BM25
            final_results[str(qid)] = [str(pk) for pk in candidate_pubkeys[:args.rerank_top]]
            continue

        # Prendi top_k candidati
        candidates = [str(pk) for pk in candidate_pubkeys[:args.top_k]]
        candidate_texts = [pubkey_to_text.get(pk, "") for pk in candidates]

        # Rimuovi candidati senza testo
        valid = [(pk, txt) for pk, txt in zip(candidates, candidate_texts) if txt]
        if not valid:
            final_results[str(qid)] = candidates[:args.rerank_top]
            continue

        valid_pks, valid_txts = zip(*valid)
        pairs = [(q_text, txt) for txt in valid_txts]

        scores = batch_score(model, list(pairs), args.batch_size)

        # Ordina per score decrescente
        ranked = sorted(zip(valid_pks, scores), key=lambda x: x[1], reverse=True)
        ranked_pubkeys = [pk for pk, _ in ranked[:args.rerank_top]]
        final_results[str(qid)] = ranked_pubkeys

    # ------------------------------------------------------------------
    # Salvataggio output per Evaluator.java
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Risultati salvati in: {out_path}")

    # ------------------------------------------------------------------
    # Metriche locali (opzionale, richiede ranx)
    # ------------------------------------------------------------------
    if args.qrels:
        try:
            from ranx import Qrels, Run, evaluate

            qrels_data = load_json(args.qrels)
            # ranx si aspetta {qid: {pubkey: score}}
            qrels_dict = {
                str(qid): {str(pk): int(s) for pk, s in docs.items()}
                for qid, docs in qrels_data.items()
            }
            run_dict = {
                qid: {pk: (args.rerank_top - i) for i, pk in enumerate(pks)}
                for qid, pks in final_results.items()
            }

            qrels_obj = Qrels(qrels_dict)
            run_obj = Run(run_dict)

            metrics = evaluate(
                qrels_obj, run_obj,
                ["mrr@5", "map", "ndcg@10"]
            )
            logger.info("=" * 50)
            logger.info("METRICHE LOCALI (ranx)")
            for m, v in metrics.items():
                logger.info(f"  {m:15s}: {v:.4f}")
            logger.info("=" * 50)

        except ImportError:
            logger.warning("ranx non installato → metriche locali skippate. pip install ranx")


if __name__ == "__main__":
    main()
