import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]

DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_BASE_INTERNAL = CODE_ROOT / "data" / "finetune" / "flagembedding" / "train_base.internal.jsonl"
DEFAULT_OUTPUT_INTERNAL = CODE_ROOT / "data" / "finetune" / "flagembedding" / "train_hardneg.internal.jsonl"
DEFAULT_OUTPUT = CODE_ROOT / "data" / "finetune" / "flagembedding" / "train_hardneg.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stronger FlagEmbedding training file by replacing easy negatives "
            "with hard negatives from one or more retrieval run files."
        )
    )
    parser.add_argument("--base-internal", type=Path, default=DEFAULT_BASE_INTERNAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--run", dest="runs", action="append", required=True)
    parser.add_argument("--top-k-per-run", type=int, default=100)
    parser.add_argument("--max-negatives-per-run", type=int, default=8)
    parser.add_argument("--negatives-per-example", type=int, default=15)
    parser.add_argument(
        "--reuse-base-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse base-file negatives after run-based negatives if needed.",
    )
    parser.add_argument(
        "--random-fill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill missing negatives with random corpus documents if needed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-internal", type=Path, default=DEFAULT_OUTPUT_INTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_document_text(item: dict[str, Any]) -> str:
    title = normalize_text(str(item.get("title", "")).strip())
    abstract = normalize_text(str(item.get("abstract", "")).strip())
    return " ".join(part for part in (title, abstract) if part)


def load_corpus_texts(path: Path) -> dict[str, str]:
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list.")

    texts: dict[str, str] = {}
    for item in raw_corpus:
        if not isinstance(item, dict):
            continue
        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            continue
        pubkey_str = str(pubkey)
        if pubkey_str in texts:
            continue
        text = build_document_text(item)
        if text:
            texts[pubkey_str] = text

    if not texts:
        raise ValueError("No valid documents were loaded from the corpus.")

    return texts


def load_run(path: Path) -> dict[str, list[str]]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Run file must be a JSON object: {path}")

    run: dict[str, list[str]] = {}
    for qid, ranked_docs in raw.items():
        if isinstance(ranked_docs, list):
            run[str(qid)] = [str(docid) for docid in ranked_docs]
    return run


def sample_random_negatives(
    rng: random.Random,
    all_docids: list[str],
    excluded: set[str],
    count: int,
) -> list[str]:
    candidates = [docid for docid in all_docids if docid not in excluded]
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, count)


def main() -> None:
    args = parse_args()
    if args.top_k_per_run <= 0:
        raise ValueError("--top-k-per-run must be greater than 0.")
    if args.max_negatives_per_run <= 0:
        raise ValueError("--max-negatives-per-run must be greater than 0.")
    if args.negatives_per_example <= 0:
        raise ValueError("--negatives-per-example must be greater than 0.")

    base_rows = load_jsonl(args.base_internal)
    if not base_rows:
        raise ValueError("Base internal training file is empty.")

    corpus_texts = load_corpus_texts(args.corpus)
    all_docids = list(corpus_texts.keys())
    runs = [load_run(Path(run_path)) for run_path in args.runs]
    rng = random.Random(args.seed)

    internal_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    source_counter: Counter = Counter()

    for row in base_rows:
        qid = str(row["qid"])
        gold_pubkey = str(row["gold_pubkey"])
        pos_pubkeys = [str(pubkey) for pubkey in row.get("pos_pubkeys", [gold_pubkey])]
        base_neg_pubkeys = [str(pubkey) for pubkey in row.get("neg_pubkeys", [])]

        chosen_negatives: list[str] = []
        seen_docids = set(pos_pubkeys)
        seen_docids.add(gold_pubkey)

        for run_idx, run in enumerate(runs):
            accepted_this_run = 0
            for docid in run.get(qid, [])[: args.top_k_per_run]:
                if docid in seen_docids or docid not in corpus_texts:
                    continue
                chosen_negatives.append(docid)
                seen_docids.add(docid)
                accepted_this_run += 1
                source_counter[f"run_{run_idx + 1}"] += 1
                if accepted_this_run >= args.max_negatives_per_run:
                    break
                if len(chosen_negatives) >= args.negatives_per_example:
                    break
            if len(chosen_negatives) >= args.negatives_per_example:
                break

        if args.reuse_base_negatives and len(chosen_negatives) < args.negatives_per_example:
            for docid in base_neg_pubkeys:
                if docid in seen_docids or docid not in corpus_texts:
                    continue
                chosen_negatives.append(docid)
                seen_docids.add(docid)
                source_counter["base_random"] += 1
                if len(chosen_negatives) >= args.negatives_per_example:
                    break

        if args.random_fill and len(chosen_negatives) < args.negatives_per_example:
            random_negatives = sample_random_negatives(
                rng=rng,
                all_docids=all_docids,
                excluded=seen_docids,
                count=args.negatives_per_example - len(chosen_negatives),
            )
            chosen_negatives.extend(random_negatives)
            source_counter["random_fill"] += len(random_negatives)

        chosen_negatives = chosen_negatives[: args.negatives_per_example]
        negative_texts = [corpus_texts[docid] for docid in chosen_negatives]

        internal_row = {
            "qid": qid,
            "variant": row.get("variant", "unknown"),
            "gold_pubkey": gold_pubkey,
            "query": row["query"],
            "pos": row["pos"],
            "neg": negative_texts,
            "pos_pubkeys": pos_pubkeys,
            "neg_pubkeys": chosen_negatives,
        }
        export_row = {
            "query": row["query"],
            "pos": row["pos"],
            "neg": negative_texts,
        }

        internal_rows.append(internal_row)
        export_rows.append(export_row)

    write_jsonl(args.output_internal, internal_rows)
    write_jsonl(args.output, export_rows)

    print(f"Wrote hard-negative internal data to {args.output_internal}")
    print(f"Wrote FlagEmbedding training data to {args.output}")
    print(
        "Negative sources: "
        + ", ".join(f"{name}={count}" for name, count in sorted(source_counter.items()))
    )


if __name__ == "__main__":
    main()
