import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from back_translation import build_back_translation_map


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]
REPO_ROOT = SCRIPT_PATH.parents[7]

DEFAULT_QUERIES = CODE_ROOT / "data" / "expanded_queries_bge_large.json"
DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_OUTPUT_DIR = CODE_ROOT / "data" / "finetune"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare FlagEmbedding fine-tuning data and held-out evaluation splits "
            "from the repository corpus and expanded-query dataset."
        )
    )
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-query-variants",
        nargs="+",
        default=("original", "expanded"),
        help="Query fields to turn into separate training examples.",
    )
    parser.add_argument(
        "--random-negatives-per-example",
        type=int,
        default=15,
        help="Fallback random negatives used to build a valid base training file.",
    )
    parser.add_argument(
        "--augment-back-translation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add back-translated query variants to the training split only.",
    )
    parser.add_argument(
        "--back-translation-source-lang",
        default="en",
        help="Source language for the training queries used during back translation.",
    )
    parser.add_argument(
        "--back-translation-pivot-langs",
        nargs="+",
        default=("es",),
        help="Intermediate languages used for back translation.",
    )
    parser.add_argument(
        "--back-translation-model-template",
        default="Helsinki-NLP/opus-mt-{src}-{tgt}",
        help=(
            "Hugging Face model template used for translation. "
            "It must support {src} and {tgt} placeholders."
        ),
    )
    parser.add_argument(
        "--back-translation-batch-size",
        type=int,
        default=8,
        help="Batch size used during back-translation generation.",
    )
    parser.add_argument(
        "--back-translation-max-input-length",
        type=int,
        default=192,
        help="Tokenizer max_length used by the translation models.",
    )
    parser.add_argument(
        "--back-translation-num-beams",
        type=int,
        default=4,
        help="Beam size used when generating back-translated variants.",
    )
    parser.add_argument(
        "--back-translation-device",
        default="auto",
        help="Torch device for translation models, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--back-translation-cache",
        type=Path,
        default=None,
        help="Optional JSON cache for translation results. Defaults under the output directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def build_document_text(item: dict[str, Any]) -> str:
    title = normalize_text(str(item.get("title", "")).strip())
    abstract = normalize_text(str(item.get("abstract", "")).strip())
    return " ".join(part for part in (title, abstract) if part)


def load_corpus(path: Path) -> tuple[list[dict[str, str]], dict[str, str]]:
    raw_corpus = load_json(path)
    if not isinstance(raw_corpus, list):
        raise ValueError("Corpus JSON must be a list.")

    documents: list[dict[str, str]] = []
    doc_text_by_pubkey: dict[str, str] = {}
    for item in raw_corpus:
        if not isinstance(item, dict):
            continue
        pubkey = item.get("pubkey", item.get("id"))
        if pubkey is None:
            continue
        pubkey_str = str(pubkey)
        if pubkey_str in doc_text_by_pubkey:
            continue
        text = build_document_text(item)
        if not text:
            continue
        document = {
            "id": pubkey_str,
            "pubkey": pubkey_str,
            "title": normalize_text(str(item.get("title", "")).strip()),
            "abstract": normalize_text(str(item.get("abstract", "")).strip()),
            "text": text,
        }
        documents.append(document)
        doc_text_by_pubkey[pubkey_str] = text

    if not documents:
        raise ValueError("No valid documents were loaded from the corpus.")

    return documents, doc_text_by_pubkey


def load_queries(path: Path, doc_text_by_pubkey: dict[str, str]) -> list[dict[str, Any]]:
    raw_queries = load_json(path)
    if not isinstance(raw_queries, list):
        raise ValueError("Query JSON must be a list.")

    queries: list[dict[str, Any]] = []
    skipped = 0
    for item in raw_queries:
        if not isinstance(item, dict):
            skipped += 1
            continue

        qid = item.get("index", item.get("qid", item.get("id")))
        gold_pubkey = item.get("pubkey")
        original = normalize_text(str(item.get("original", "")).strip())
        expanded = normalize_text(str(item.get("expanded", "")).strip())
        if qid is None or gold_pubkey is None or not original:
            skipped += 1
            continue

        gold_pubkey_str = str(gold_pubkey)
        if gold_pubkey_str not in doc_text_by_pubkey:
            skipped += 1
            continue

        query = {
            "index": str(qid),
            "pubkey": gold_pubkey_str,
            "original": original,
            "expanded": expanded or original,
            "keywords": normalize_text(str(item.get("keywords", "")).strip()),
            "text": normalize_text(str(item.get("text", "")).strip()),
            "query": normalize_text(str(item.get("query", "")).strip()),
            "raw": item,
        }
        queries.append(query)

    if not queries:
        raise ValueError("No valid queries were loaded.")

    if skipped:
        print(f"Skipped {skipped} queries without required fields or matching gold documents.")

    return queries


def validate_ratios(train_ratio: float, dev_ratio: float) -> None:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not (0.0 < dev_ratio < 1.0):
        raise ValueError("--dev-ratio must be between 0 and 1.")
    if train_ratio + dev_ratio >= 1.0:
        raise ValueError("--train-ratio + --dev-ratio must be less than 1.")


def split_queries(
    queries: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    items = list(queries)
    rng = random.Random(seed)
    rng.shuffle(items)

    total = len(items)
    train_end = int(total * train_ratio)
    dev_end = train_end + int(total * dev_ratio)

    splits = {
        "train": items[:train_end],
        "dev": items[train_end:dev_end],
        "test": items[dev_end:],
    }

    if not splits["train"] or not splits["dev"] or not splits["test"]:
        raise ValueError("One split is empty. Adjust train/dev ratios.")

    return splits


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def repo_query_record(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "index": int(query["index"]) if query["index"].isdigit() else query["index"],
        "original": query["original"],
        "expanded": query["expanded"],
        "keywords": query["keywords"],
        "pubkey": int(query["pubkey"]) if query["pubkey"].isdigit() else query["pubkey"],
    }


def pick_query_variant(query: dict[str, Any], variant: str) -> str:
    value = normalize_text(str(query.get(variant, "")).strip())
    if value:
        return value
    if variant == "expanded":
        return query["original"]
    return ""


def build_eval_queries(split_queries: list[dict[str, Any]], query_field: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for query in split_queries:
        text = pick_query_variant(query, query_field)
        if not text:
            continue
        rows.append({"id": query["index"], "text": text})
    return rows


def build_qrels(split_queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "qid": query["index"],
            "docid": query["pubkey"],
            "relevance": 1,
        }
        for query in split_queries
    ]


def sample_random_negatives(
    rng: random.Random,
    all_docids: list[str],
    excluded: set[str],
    count: int,
) -> list[str]:
    candidates = [docid for docid in all_docids if docid not in excluded]
    if not candidates or count <= 0:
        return []
    if count >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, count)


def append_train_row(
    *,
    internal_rows: list[dict[str, Any]],
    export_rows: list[dict[str, Any]],
    qid: str,
    variant: str,
    gold_pubkey: str,
    query_text: str,
    positive_text: str,
    negative_pubkeys: list[str],
    negative_texts: list[str],
) -> None:
    internal_rows.append(
        {
            "qid": qid,
            "variant": variant,
            "gold_pubkey": gold_pubkey,
            "query": query_text,
            "pos": [positive_text],
            "neg": negative_texts,
            "pos_pubkeys": [gold_pubkey],
            "neg_pubkeys": negative_pubkeys,
        }
    )
    export_rows.append(
        {
            "query": query_text,
            "pos": [positive_text],
            "neg": negative_texts,
        }
    )


def build_train_rows(
    train_queries: list[dict[str, Any]],
    doc_text_by_pubkey: dict[str, str],
    all_docids: list[str],
    train_query_variants: list[str],
    random_negatives_per_example: int,
    seed: int,
    back_translation_map: dict[str, list[dict[str, str]]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    rng = random.Random(seed)
    internal_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    variant_counter: Counter = Counter()

    for query in train_queries:
        seen_variant_texts: set[str] = set()
        gold_pubkey = query["pubkey"]
        positive_text = doc_text_by_pubkey[gold_pubkey]

        for variant in train_query_variants:
            query_text = pick_query_variant(query, variant)
            if not query_text or query_text in seen_variant_texts:
                continue
            seen_variant_texts.add(query_text)
            variant_counter[variant] += 1

            negative_pubkeys = sample_random_negatives(
                rng=rng,
                all_docids=all_docids,
                excluded={gold_pubkey},
                count=random_negatives_per_example,
            )
            negative_texts = [doc_text_by_pubkey[docid] for docid in negative_pubkeys]
            append_train_row(
                internal_rows=internal_rows,
                export_rows=export_rows,
                qid=query["index"],
                variant=variant,
                gold_pubkey=gold_pubkey,
                query_text=query_text,
                positive_text=positive_text,
                negative_pubkeys=negative_pubkeys,
                negative_texts=negative_texts,
            )

            for augmented_variant in (back_translation_map or {}).get(query_text, []):
                augmented_text = normalize_text(augmented_variant.get("text", ""))
                if not augmented_text or augmented_text in seen_variant_texts:
                    continue

                seen_variant_texts.add(augmented_text)
                augmented_variant_name = f"{variant}_bt_{augmented_variant['pivot_lang']}"
                variant_counter[augmented_variant_name] += 1
                append_train_row(
                    internal_rows=internal_rows,
                    export_rows=export_rows,
                    qid=query["index"],
                    variant=augmented_variant_name,
                    gold_pubkey=gold_pubkey,
                    query_text=augmented_text,
                    positive_text=positive_text,
                    negative_pubkeys=negative_pubkeys,
                    negative_texts=negative_texts,
                )

    return internal_rows, export_rows, variant_counter


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.dev_ratio)
    if args.random_negatives_per_example <= 0:
        raise ValueError("--random-negatives-per-example must be greater than 0.")
    if args.back_translation_batch_size <= 0:
        raise ValueError("--back-translation-batch-size must be greater than 0.")
    if args.back_translation_max_input_length <= 0:
        raise ValueError("--back-translation-max-input-length must be greater than 0.")
    if args.back_translation_num_beams <= 0:
        raise ValueError("--back-translation-num-beams must be greater than 0.")

    query_variants = [variant.strip() for variant in args.train_query_variants if variant.strip()]
    if not query_variants:
        raise ValueError("At least one --train-query-variants value is required.")

    documents, doc_text_by_pubkey = load_corpus(args.corpus)
    queries = load_queries(args.queries, doc_text_by_pubkey)
    splits = split_queries(
        queries=queries,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    back_translation_cache = args.back_translation_cache
    back_translation_map: dict[str, list[dict[str, str]]] | None = None
    normalized_pivot_langs = [
        pivot_lang.strip() for pivot_lang in args.back_translation_pivot_langs if pivot_lang.strip()
    ]
    if args.augment_back_translation:
        if not normalized_pivot_langs:
            raise ValueError(
                "At least one --back-translation-pivot-langs value is required when "
                "--augment-back-translation is enabled."
            )
        if back_translation_cache is None:
            back_translation_cache = (
                args.output_dir / "augmentation" / "back_translation_cache.json"
            )

        unique_train_texts = []
        for query in splits["train"]:
            for variant in query_variants:
                text = pick_query_variant(query, variant)
                if text:
                    unique_train_texts.append(text)

        back_translation_map = build_back_translation_map(
            unique_train_texts,
            pivot_langs=normalized_pivot_langs,
            source_lang=args.back_translation_source_lang.strip(),
            model_template=args.back_translation_model_template,
            batch_size=args.back_translation_batch_size,
            device_name=args.back_translation_device,
            max_input_length=args.back_translation_max_input_length,
            num_beams=args.back_translation_num_beams,
            cache_path=back_translation_cache,
        )

    all_docids = [document["id"] for document in documents]
    internal_rows, export_rows, variant_counter = build_train_rows(
        train_queries=splits["train"],
        doc_text_by_pubkey=doc_text_by_pubkey,
        all_docids=all_docids,
        train_query_variants=query_variants,
        random_negatives_per_example=args.random_negatives_per_example,
        seed=args.seed,
        back_translation_map=back_translation_map,
    )

    repo_dir = args.output_dir / "repo"
    flag_dir = args.output_dir / "flagembedding"
    eval_dir = flag_dir / "eval"

    write_json(repo_dir / "train_queries.json", [repo_query_record(query) for query in splits["train"]])
    write_json(repo_dir / "dev_queries.json", [repo_query_record(query) for query in splits["dev"]])
    write_json(repo_dir / "test_queries.json", [repo_query_record(query) for query in splits["test"]])

    write_jsonl(flag_dir / "corpus.jsonl", [{"id": doc["id"], "text": doc["text"]} for doc in documents])
    write_jsonl(
        flag_dir / "candidate_pool.jsonl",
        [{"id": doc["id"], "text": doc["text"]} for doc in documents],
    )
    write_jsonl(flag_dir / "train_base.internal.jsonl", internal_rows)
    write_jsonl(flag_dir / "train_base.jsonl", export_rows)

    for split_name in ("train", "dev", "test"):
        split_items = splits[split_name]
        write_jsonl(eval_dir / f"{split_name}_queries_original.jsonl", build_eval_queries(split_items, "original"))
        write_jsonl(eval_dir / f"{split_name}_queries_expanded.jsonl", build_eval_queries(split_items, "expanded"))
        write_jsonl(eval_dir / f"{split_name}_qrels.jsonl", build_qrels(split_items))

    stats = {
        "corpus_documents": len(documents),
        "total_queries": len(queries),
        "splits": {
            split_name: len(split_items)
            for split_name, split_items in splits.items()
        },
        "train_query_variants": query_variants,
        "train_examples": len(export_rows),
        "train_examples_by_variant": dict(variant_counter),
        "random_negatives_per_example": args.random_negatives_per_example,
        "seed": args.seed,
        "augmentation": {
            "back_translation": {
                "enabled": args.augment_back_translation,
                "source_lang": args.back_translation_source_lang.strip(),
                "pivot_langs": normalized_pivot_langs,
                "model_template": args.back_translation_model_template,
                "batch_size": args.back_translation_batch_size,
                "max_input_length": args.back_translation_max_input_length,
                "num_beams": args.back_translation_num_beams,
                "device": args.back_translation_device,
                "cache_path": str(back_translation_cache) if back_translation_cache else None,
                "unique_source_texts": len(back_translation_map or {}),
                "generated_augmented_texts": sum(
                    len(variants) for variants in (back_translation_map or {}).values()
                ),
            }
        },
        "paths": {
            "repo_train_queries": str(repo_dir / "train_queries.json"),
            "repo_dev_queries": str(repo_dir / "dev_queries.json"),
            "repo_test_queries": str(repo_dir / "test_queries.json"),
            "train_base": str(flag_dir / "train_base.jsonl"),
            "train_base_internal": str(flag_dir / "train_base.internal.jsonl"),
            "corpus_jsonl": str(flag_dir / "corpus.jsonl"),
            "candidate_pool_jsonl": str(flag_dir / "candidate_pool.jsonl"),
        },
    }
    write_json(args.output_dir / "stats.json", stats)

    print(f"Prepared fine-tuning data in {args.output_dir}")
    print(f"Corpus documents: {len(documents)}")
    print(
        "Split sizes: "
        + ", ".join(f"{name}={len(items)}" for name, items in splits.items())
    )
    print(
        "Training examples: "
        f"{len(export_rows)} from variants {', '.join(query_variants)}"
    )
    if args.augment_back_translation:
        print(
            "Back-translation augmented texts: "
            f"{sum(len(variants) for variants in (back_translation_map or {}).values())}"
        )
        print(
            "Training example variants (including augmentation): "
            + ", ".join(sorted(variant_counter.keys()))
        )


if __name__ == "__main__":
    main()
