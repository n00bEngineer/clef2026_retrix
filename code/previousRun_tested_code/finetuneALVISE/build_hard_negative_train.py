import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
CODE_ROOT = SCRIPT_PATH.parents[6]

DEFAULT_CORPUS = CODE_ROOT / "data" / "collection_data.json"
DEFAULT_BASE_INTERNAL = CODE_ROOT / "data" / "finetune" / "flagembedding" / "train_base.internal.jsonl"
DEFAULT_OUTPUT_INTERNAL = CODE_ROOT / "data" / "finetune" / "flagembedding" / "train_hardneg.internal.jsonl"
DEFAULT_OUTPUT = CODE_ROOT / "data" / "finetune" / "flagembedding" / "train_hardneg.jsonl"

MAX_REPORTED_VALIDATION_ISSUES = 20
QID_KEYS = ("qid", "query_id", "queryId")
DOCID_KEYS = ("docid", "doc_id", "document_id", "documentId", "pubkey", "docno", "id")
RANK_KEYS = ("rank", "position")
SCORE_KEYS = ("score", "similarity", "value")


@dataclass(slots=True)
class RunEntry:
    docid: str
    rank: int
    score: float | None = None


@dataclass(slots=True)
class RunDiagnostics:
    format_name: str
    source_items_read: int = 0
    parsed_entries: int = 0
    normalized_entries: int = 0
    unique_qids_recognized: int = 0
    skipped_reasons: Counter[str] = field(default_factory=Counter)
    training_qid_overlap: int = 0
    training_qid_overlap_ratio: float = 0.0
    training_qids_with_usable_candidates: int = 0
    training_qids_with_usable_ratio: float = 0.0
    usable_entries_on_training_qids: int = 0
    corpus_missing_entries_on_training_qids: int = 0
    positive_filtered_entries_on_training_qids: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "format_name": self.format_name,
            "source_items_read": self.source_items_read,
            "parsed_entries": self.parsed_entries,
            "normalized_entries": self.normalized_entries,
            "unique_qids_recognized": self.unique_qids_recognized,
            "skipped_reasons": dict(self.skipped_reasons),
            "training_qid_overlap": self.training_qid_overlap,
            "training_qid_overlap_ratio": self.training_qid_overlap_ratio,
            "training_qids_with_usable_candidates": self.training_qids_with_usable_candidates,
            "training_qids_with_usable_ratio": self.training_qids_with_usable_ratio,
            "usable_entries_on_training_qids": self.usable_entries_on_training_qids,
            "corpus_missing_entries_on_training_qids": self.corpus_missing_entries_on_training_qids,
            "positive_filtered_entries_on_training_qids": self.positive_filtered_entries_on_training_qids,
        }


@dataclass(slots=True)
class LoadedRun:
    name: str
    path: Path
    rows_by_qid: dict[str, list[RunEntry]]
    diagnostics: RunDiagnostics


@dataclass(slots=True)
class PreparedBaseRow:
    qid: str
    variant: str
    gold_pubkey: str
    query: str
    positive_texts: list[str]
    pos_pubkeys: list[str]
    base_neg_pubkeys: list[str]


@dataclass(slots=True)
class TrainingQidMetadata:
    qid: str
    gold_pubkey: str
    row_count: int = 0
    pos_pubkeys: set[str] = field(default_factory=set)
    positive_texts: set[str] = field(default_factory=set)


@dataclass(slots=True)
class CandidateEvidence:
    docid: str
    best_rank: int = 10**9
    rank_sum: int = 0
    rrf_score: float = 0.0
    run_ranks: dict[str, int] = field(default_factory=dict)

    def add_observation(self, *, run_name: str, rank: int, rank_constant: int) -> None:
        if run_name in self.run_ranks:
            return
        self.run_ranks[run_name] = rank
        self.best_rank = min(self.best_rank, rank)
        self.rank_sum += rank
        self.rrf_score += 1.0 / (rank_constant + rank)

    @property
    def support_count(self) -> int:
        return len(self.run_ranks)

    @property
    def mean_rank(self) -> float:
        if self.support_count == 0:
            return float("inf")
        return self.rank_sum / self.support_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a stronger FlagEmbedding training file by replacing easy negatives "
            "with fused hard negatives from one or more retrieval runs."
        )
    )
    parser.add_argument("--base-internal", type=Path, default=DEFAULT_BASE_INTERNAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        required=True,
        help=(
            "Retrieval run file. Supported formats: JSON {qid: [docid, ...]}, "
            "JSON {qid: {docid: score}}, JSONL/list of run records, or standard TREC text runs."
        ),
    )
    parser.add_argument("--top-k-per-run", type=int, default=100)
    parser.add_argument(
        "--max-negatives-per-run",
        type=int,
        default=8,
        help="How many valid candidates to collect from each run before fusion.",
    )
    parser.add_argument("--negatives-per-example", type=int, default=15)
    parser.add_argument(
        "--rank-constant",
        type=int,
        default=60,
        help="RRF constant used when fusing candidates coming from multiple runs.",
    )
    parser.add_argument(
        "--max-doc-frequency",
        type=int,
        default=0,
        help=(
            "Optional global cap on how many examples can reuse the same negative document. "
            "0 disables the cap."
        ),
    )
    parser.add_argument(
        "--reuse-base-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse base-file negatives after fused run-based negatives if needed.",
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
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional JSON file with selection statistics and coverage diagnostics.",
    )
    parser.add_argument(
        "--min-run-qid-overlap",
        type=float,
        default=0.10,
        help="Fail if any run overlaps with fewer than this fraction of training qids.",
    )
    parser.add_argument(
        "--min-run-usable-qid-overlap",
        type=float,
        default=0.10,
        help="Fail if any run provides usable corpus-backed candidates for too few training qids.",
    )
    parser.add_argument(
        "--min-rows-with-run-negatives-ratio",
        type=float,
        default=0.25,
        help="Fail if too few output rows contain at least one true run-based negative.",
    )
    parser.add_argument(
        "--max-fallback-only-row-ratio",
        type=float,
        default=0.75,
        help="Fail if fallback-only rows dominate the output dataset.",
    )
    parser.add_argument(
        "--max-excess-fallback-row-ratio",
        type=float,
        default=0.50,
        help=(
            "Fail if too many rows need more fallback negatives than the configuration "
            "makes structurally unavoidable."
        ),
    )
    parser.add_argument(
        "--min-avg-run-candidates-per-row",
        type=float,
        default=1.0,
        help="Fail if the average real candidate pool is too small.",
    )
    parser.add_argument(
        "--min-average-negative-tokens",
        type=float,
        default=5.0,
        help="Fail if the final negatives look implausibly short on average.",
    )
    parser.add_argument(
        "--require-multi-run-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When multiple runs are provided, require at least some overlap candidates across runs.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc.msg}") from exc
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def normalize_identifier(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list | tuple | set):
        items = value
    else:
        items = [value]
    result: list[str] = []
    for item in items:
        item_str = str(item).strip()
        if item_str:
            result.append(item_str)
    return result


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
        pubkey_str = normalize_identifier(pubkey)
        if pubkey_str is None or pubkey_str in texts:
            continue
        text = build_document_text(item)
        if text:
            texts[pubkey_str] = text

    if not texts:
        raise ValueError("No valid documents were loaded from the corpus.")

    return texts


def maybe_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_first_present(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def normalize_ranked_entries(entries: list[RunEntry]) -> list[RunEntry]:
    unique_by_docid: dict[str, RunEntry] = {}
    for fallback_rank, entry in enumerate(entries, start=1):
        rank = entry.rank if entry.rank > 0 else fallback_rank
        if entry.docid in unique_by_docid:
            if rank < unique_by_docid[entry.docid].rank:
                unique_by_docid[entry.docid] = RunEntry(entry.docid, rank, entry.score)
            continue
        unique_by_docid[entry.docid] = RunEntry(entry.docid, rank, entry.score)

    normalized = list(unique_by_docid.values())
    normalized.sort(
        key=lambda entry: (
            entry.rank,
            -(entry.score if entry.score is not None else float("-inf")),
            entry.docid,
        )
    )
    return [
        RunEntry(entry.docid, index, entry.score)
        for index, entry in enumerate(normalized, start=1)
    ]


def normalize_grouped_run_entries(
    grouped_entries: dict[str, list[RunEntry]],
    diagnostics: RunDiagnostics,
) -> dict[str, list[RunEntry]]:
    rows_by_qid: dict[str, list[RunEntry]] = {}
    for qid, entries in grouped_entries.items():
        normalized_entries = normalize_ranked_entries(entries)
        if not normalized_entries:
            diagnostics.skipped_reasons["empty_ranking"] += 1
            continue
        rows_by_qid[qid] = normalized_entries
    return rows_by_qid


def parse_run_list_value(items: list[Any], diagnostics: RunDiagnostics) -> list[RunEntry]:
    entries: list[RunEntry] = []
    for fallback_rank, item in enumerate(items, start=1):
        if isinstance(item, dict):
            docid = normalize_identifier(extract_first_present(item, DOCID_KEYS))
            if docid is None:
                diagnostics.skipped_reasons["missing_docid"] += 1
                continue
            rank = maybe_int(extract_first_present(item, RANK_KEYS)) or fallback_rank
            score = maybe_float(extract_first_present(item, SCORE_KEYS))
        else:
            docid = normalize_identifier(item)
            if docid is None:
                diagnostics.skipped_reasons["missing_docid"] += 1
                continue
            rank = fallback_rank
            score = None

        diagnostics.parsed_entries += 1
        entries.append(RunEntry(docid, rank, score))

    return normalize_ranked_entries(entries)


def parse_run_mapping_value(items: dict[Any, Any], diagnostics: RunDiagnostics) -> list[RunEntry]:
    entries: list[RunEntry] = []
    has_explicit_rank = False
    has_score_only_entries = False

    for fallback_rank, (raw_docid, raw_value) in enumerate(items.items(), start=1):
        docid = normalize_identifier(raw_docid)
        if docid is None:
            diagnostics.skipped_reasons["missing_docid"] += 1
            continue

        rank: int | None = None
        if isinstance(raw_value, dict):
            score = maybe_float(extract_first_present(raw_value, SCORE_KEYS))
            rank = maybe_int(extract_first_present(raw_value, RANK_KEYS))
        else:
            score = maybe_float(raw_value)

        if rank is not None:
            has_explicit_rank = True
        elif score is not None:
            has_score_only_entries = True

        diagnostics.parsed_entries += 1
        entries.append(RunEntry(docid, rank or fallback_rank, score))

    if has_explicit_rank:
        return normalize_ranked_entries(entries)

    if has_score_only_entries:
        entries.sort(
            key=lambda entry: (
                -(entry.score if entry.score is not None else float("-inf")),
                entry.docid,
            )
        )
        return [
            RunEntry(entry.docid, index, entry.score)
            for index, entry in enumerate(entries, start=1)
        ]

    return normalize_ranked_entries(entries)


def parse_run_records(
    records: list[Any],
    diagnostics: RunDiagnostics,
    *,
    count_source_items: bool = True,
) -> dict[str, list[RunEntry]]:
    grouped_entries: dict[str, list[RunEntry]] = defaultdict(list)

    for fallback_index, record in enumerate(records, start=1):
        if count_source_items:
            diagnostics.source_items_read += 1
        if not isinstance(record, dict):
            diagnostics.skipped_reasons["non_object_record"] += 1
            continue

        qid = normalize_identifier(extract_first_present(record, QID_KEYS))
        if qid is None:
            diagnostics.skipped_reasons["missing_qid"] += 1
            continue

        docid = normalize_identifier(extract_first_present(record, DOCID_KEYS))
        if docid is None:
            diagnostics.skipped_reasons["missing_docid"] += 1
            continue

        rank = maybe_int(extract_first_present(record, RANK_KEYS)) or fallback_index
        score = maybe_float(extract_first_present(record, SCORE_KEYS))
        diagnostics.parsed_entries += 1
        grouped_entries[qid].append(RunEntry(docid, rank, score))

    return normalize_grouped_run_entries(grouped_entries, diagnostics)


def load_jsonl_records(path: Path, diagnostics: RunDiagnostics) -> list[Any]:
    records: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            diagnostics.source_items_read += 1
            line = raw_line.strip()
            if not line:
                diagnostics.skipped_reasons["blank_line"] += 1
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of run {path}: {exc.msg}") from exc
    return records


def load_text_run(path: Path, diagnostics: RunDiagnostics) -> dict[str, list[RunEntry]]:
    grouped_entries: dict[str, list[RunEntry]] = defaultdict(list)
    record_index = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            diagnostics.source_items_read += 1
            line = raw_line.strip()
            if not line:
                diagnostics.skipped_reasons["blank_line"] += 1
                continue
            if line.startswith("#"):
                diagnostics.skipped_reasons["comment_line"] += 1
                continue

            record_index += 1
            parts = line.split()
            qid: str | None = None
            docid: str | None = None
            rank: int | None = None
            score: float | None = None

            if len(parts) >= 6 and parts[1].upper() == "Q0":
                qid = parts[0]
                docid = parts[2]
                rank = maybe_int(parts[3])
                score = maybe_float(parts[4])
            elif len(parts) >= 4:
                qid = parts[0]
                docid = parts[1]
                rank = maybe_int(parts[2])
                score = maybe_float(parts[3])
            elif len(parts) >= 2:
                qid = parts[0]
                docid = parts[1]
            else:
                diagnostics.skipped_reasons["unrecognized_text_line"] += 1
                continue

            normalized_qid = normalize_identifier(qid)
            if normalized_qid is None:
                diagnostics.skipped_reasons["missing_qid"] += 1
                continue

            normalized_docid = normalize_identifier(docid)
            if normalized_docid is None:
                diagnostics.skipped_reasons["missing_docid"] += 1
                continue

            diagnostics.parsed_entries += 1
            grouped_entries[normalized_qid].append(
                RunEntry(normalized_docid, rank or record_index, score)
            )

    return normalize_grouped_run_entries(grouped_entries, diagnostics)


def finalize_loaded_run(
    run_name: str,
    path: Path,
    rows_by_qid: dict[str, list[RunEntry]],
    diagnostics: RunDiagnostics,
) -> LoadedRun:
    diagnostics.unique_qids_recognized = len(rows_by_qid)
    diagnostics.normalized_entries = sum(len(entries) for entries in rows_by_qid.values())
    return LoadedRun(run_name, path, rows_by_qid, diagnostics)


def load_run(path: Path, run_name: str) -> LoadedRun:
    suffix = path.suffix.lower()

    if suffix == ".json":
        raw = load_json(path)
        if isinstance(raw, dict):
            diagnostics = RunDiagnostics(format_name="json_mapping")
            rows_by_qid: dict[str, list[RunEntry]] = {}

            for raw_qid, ranked_docs in raw.items():
                diagnostics.source_items_read += 1
                qid = normalize_identifier(raw_qid)
                if qid is None:
                    diagnostics.skipped_reasons["missing_qid"] += 1
                    continue

                if isinstance(ranked_docs, list):
                    parsed_entries = parse_run_list_value(ranked_docs, diagnostics)
                elif isinstance(ranked_docs, dict):
                    parsed_entries = parse_run_mapping_value(ranked_docs, diagnostics)
                else:
                    diagnostics.skipped_reasons["unsupported_qid_payload"] += 1
                    continue

                if not parsed_entries:
                    diagnostics.skipped_reasons["empty_ranking"] += 1
                    continue
                rows_by_qid[qid] = parsed_entries

            return finalize_loaded_run(run_name, path, rows_by_qid, diagnostics)

        if isinstance(raw, list):
            diagnostics = RunDiagnostics(format_name="json_records")
            rows_by_qid = parse_run_records(raw, diagnostics)
            return finalize_loaded_run(run_name, path, rows_by_qid, diagnostics)

        raise ValueError(f"Unsupported JSON run format: {path}")

    if suffix == ".jsonl":
        diagnostics = RunDiagnostics(format_name="jsonl_records")
        records = load_jsonl_records(path, diagnostics)
        rows_by_qid = parse_run_records(records, diagnostics, count_source_items=False)
        return finalize_loaded_run(run_name, path, rows_by_qid, diagnostics)

    diagnostics = RunDiagnostics(format_name="text_run")
    rows_by_qid = load_text_run(path, diagnostics)
    return finalize_loaded_run(run_name, path, rows_by_qid, diagnostics)


def prepare_base_rows(
    base_rows: list[dict[str, Any]],
    corpus_texts: dict[str, str],
) -> tuple[list[PreparedBaseRow], dict[str, TrainingQidMetadata]]:
    prepared_rows: list[PreparedBaseRow] = []
    qid_metadata: dict[str, TrainingQidMetadata] = {}

    for row_index, row in enumerate(base_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Base row #{row_index} is not a JSON object.")
        if "qid" not in row or "gold_pubkey" not in row or "query" not in row:
            raise ValueError(f"Base row #{row_index} is missing one of: qid, gold_pubkey, query.")

        qid = normalize_identifier(row["qid"])
        if qid is None:
            raise ValueError(f"Base row #{row_index} has an empty qid.")

        gold_pubkey = normalize_identifier(row["gold_pubkey"])
        if gold_pubkey is None:
            raise ValueError(f"Base row #{row_index} has an empty gold_pubkey.")

        query_text = str(row["query"])
        if not normalize_text(query_text):
            raise ValueError(f"Base row #{row_index} has an empty query.")

        positive_texts = [
            normalized
            for normalized in (
                normalize_text(text) for text in coerce_str_list(row.get("pos"))
            )
            if normalized
        ]
        if not positive_texts and gold_pubkey in corpus_texts:
            positive_texts = [corpus_texts[gold_pubkey]]
        positive_texts = dedupe_preserve_order(positive_texts)
        if not positive_texts:
            raise ValueError(
                f"Base row #{row_index} has no usable positive text and gold_pubkey {gold_pubkey} is not in the corpus."
            )

        pos_pubkeys = coerce_str_list(row.get("pos_pubkeys"))
        pos_pubkeys.append(gold_pubkey)
        pos_pubkeys = dedupe_preserve_order(pos_pubkeys)
        base_neg_pubkeys = dedupe_preserve_order(coerce_str_list(row.get("neg_pubkeys")))

        prepared_rows.append(
            PreparedBaseRow(
                qid=qid,
                variant=str(row.get("variant", "unknown")),
                gold_pubkey=gold_pubkey,
                query=query_text,
                positive_texts=positive_texts,
                pos_pubkeys=pos_pubkeys,
                base_neg_pubkeys=base_neg_pubkeys,
            )
        )

        metadata = qid_metadata.get(qid)
        if metadata is None:
            metadata = TrainingQidMetadata(qid=qid, gold_pubkey=gold_pubkey)
            qid_metadata[qid] = metadata
        elif metadata.gold_pubkey != gold_pubkey:
            raise ValueError(
                f"Training qid {qid} maps to multiple gold_pubkeys ({metadata.gold_pubkey} vs {gold_pubkey})."
            )

        metadata.row_count += 1
        metadata.pos_pubkeys.update(pos_pubkeys)
        metadata.positive_texts.update(positive_texts)

    if not prepared_rows:
        raise ValueError("Base internal training file is empty after validation.")

    return prepared_rows, qid_metadata


def inspect_run_alignment(
    *,
    run: LoadedRun,
    training_qids: dict[str, TrainingQidMetadata],
    corpus_texts: dict[str, str],
    top_k_per_run: int,
) -> None:
    diagnostics = run.diagnostics
    total_training_qids = len(training_qids)
    overlapping_qids = sorted(set(run.rows_by_qid).intersection(training_qids))
    diagnostics.training_qid_overlap = len(overlapping_qids)
    diagnostics.training_qid_overlap_ratio = safe_ratio(
        diagnostics.training_qid_overlap,
        total_training_qids,
    )

    qids_with_usable_candidates = 0
    usable_entries = 0
    missing_corpus_entries = 0
    positive_filtered_entries = 0

    for qid in overlapping_qids:
        metadata = training_qids[qid]
        has_usable_candidate = False
        excluded_docids = metadata.pos_pubkeys

        for entry in run.rows_by_qid[qid][:top_k_per_run]:
            if entry.docid in excluded_docids:
                positive_filtered_entries += 1
                continue
            if entry.docid not in corpus_texts:
                missing_corpus_entries += 1
                continue
            usable_entries += 1
            has_usable_candidate = True

        if has_usable_candidate:
            qids_with_usable_candidates += 1

    diagnostics.training_qids_with_usable_candidates = qids_with_usable_candidates
    diagnostics.training_qids_with_usable_ratio = safe_ratio(
        qids_with_usable_candidates,
        total_training_qids,
    )
    diagnostics.usable_entries_on_training_qids = usable_entries
    diagnostics.corpus_missing_entries_on_training_qids = missing_corpus_entries
    diagnostics.positive_filtered_entries_on_training_qids = positive_filtered_entries


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(counter.items()))


def print_run_diagnostics(runs: list[LoadedRun], training_qid_count: int) -> None:
    print(f"Loaded {len(runs)} run(s) for {training_qid_count} unique training qid(s).")
    for run in runs:
        diagnostics = run.diagnostics
        print(
            f"[run:{run.name}] format={diagnostics.format_name}, "
            f"source_items={diagnostics.source_items_read}, "
            f"parsed_entries={diagnostics.parsed_entries}, "
            f"normalized_entries={diagnostics.normalized_entries}, "
            f"recognized_qids={diagnostics.unique_qids_recognized}"
        )
        print(f"  skipped: {format_counter(diagnostics.skipped_reasons)}")
        print(
            "  training_overlap: "
            f"{diagnostics.training_qid_overlap}/{training_qid_count} "
            f"({diagnostics.training_qid_overlap_ratio:.1%}), "
            "usable_training_qids: "
            f"{diagnostics.training_qids_with_usable_candidates}/{training_qid_count} "
            f"({diagnostics.training_qids_with_usable_ratio:.1%}), "
            f"usable_entries={diagnostics.usable_entries_on_training_qids}, "
            f"missing_corpus_entries={diagnostics.corpus_missing_entries_on_training_qids}, "
            f"positive_filtered_entries={diagnostics.positive_filtered_entries_on_training_qids}"
        )


def validate_ratio_arg(name: str, value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")


def validate_run_diagnostics(
    *,
    runs: list[LoadedRun],
    training_qid_count: int,
    min_run_qid_overlap: float,
    min_run_usable_qid_overlap: float,
) -> None:
    issues: list[str] = []

    for run in runs:
        diagnostics = run.diagnostics
        if diagnostics.unique_qids_recognized == 0:
            issues.append(
                f"{run.name}: no valid qids were recognized after parsing {run.path}."
            )
            continue

        if diagnostics.training_qid_overlap == 0:
            issues.append(
                f"{run.name}: 0/{training_qid_count} training qids overlap with the run."
            )
            continue

        if diagnostics.training_qid_overlap_ratio < min_run_qid_overlap:
            issues.append(
                f"{run.name}: training qid overlap is only "
                f"{diagnostics.training_qid_overlap}/{training_qid_count} "
                f"({diagnostics.training_qid_overlap_ratio:.1%}), below the required "
                f"{min_run_qid_overlap:.1%}."
            )

        if diagnostics.training_qids_with_usable_candidates == 0:
            issues.append(
                f"{run.name}: matching qids exist, but none yield usable corpus-backed candidates."
            )
            continue

        if diagnostics.training_qids_with_usable_ratio < min_run_usable_qid_overlap:
            issues.append(
                f"{run.name}: usable qid coverage is only "
                f"{diagnostics.training_qids_with_usable_candidates}/{training_qid_count} "
                f"({diagnostics.training_qids_with_usable_ratio:.1%}), below the required "
                f"{min_run_usable_qid_overlap:.1%}."
            )

    if issues:
        raise ValueError("Run validation failed:\n- " + "\n- ".join(issues))


def build_candidate_pool(
    *,
    qid: str,
    runs: list[LoadedRun],
    corpus_texts: dict[str, str],
    excluded_docids: set[str],
    top_k_per_run: int,
    max_negatives_per_run: int,
    rank_constant: int,
) -> dict[str, CandidateEvidence]:
    candidate_pool: dict[str, CandidateEvidence] = {}

    for run in runs:
        accepted_for_run = 0
        for entry in run.rows_by_qid.get(qid, [])[:top_k_per_run]:
            if accepted_for_run >= max_negatives_per_run:
                break
            if entry.docid in excluded_docids or entry.docid not in corpus_texts:
                continue

            evidence = candidate_pool.setdefault(entry.docid, CandidateEvidence(docid=entry.docid))
            evidence.add_observation(
                run_name=run.name,
                rank=entry.rank,
                rank_constant=rank_constant,
            )
            accepted_for_run += 1

    return candidate_pool


def rank_candidates(candidates: dict[str, CandidateEvidence]) -> list[CandidateEvidence]:
    return sorted(
        candidates.values(),
        key=lambda evidence: (
            -evidence.support_count,
            -evidence.rrf_score,
            evidence.best_rank,
            evidence.mean_rank,
            evidence.docid,
        ),
    )


def can_use_doc(
    *,
    docid: str,
    corpus_texts: dict[str, str],
    used_docids: set[str],
    blocked_texts: set[str],
    doc_usage_counter: Counter[str],
    max_doc_frequency: int,
) -> bool:
    if docid in used_docids or docid not in corpus_texts:
        return False
    if max_doc_frequency > 0 and doc_usage_counter[docid] >= max_doc_frequency:
        return False
    text_key = normalize_text(corpus_texts[docid])
    if not text_key or text_key in blocked_texts:
        return False
    return True


def register_doc(
    *,
    docid: str,
    corpus_texts: dict[str, str],
    used_docids: set[str],
    blocked_texts: set[str],
) -> None:
    used_docids.add(docid)
    blocked_texts.add(normalize_text(corpus_texts[docid]))


def sample_random_negatives(
    *,
    rng: random.Random,
    all_docids: list[str],
    corpus_texts: dict[str, str],
    used_docids: set[str],
    blocked_texts: set[str],
    doc_usage_counter: Counter[str],
    max_doc_frequency: int,
    count: int,
) -> list[str]:
    candidates = [
        docid
        for docid in all_docids
        if can_use_doc(
            docid=docid,
            corpus_texts=corpus_texts,
            used_docids=used_docids,
            blocked_texts=blocked_texts,
            doc_usage_counter=doc_usage_counter,
            max_doc_frequency=max_doc_frequency,
        )
    ]
    if count <= 0 or not candidates:
        return []
    if count >= len(candidates):
        rng.shuffle(candidates)
        return candidates
    return rng.sample(candidates, count)


def build_run_name(index: int, path: Path) -> str:
    stem = path.stem or f"run_{index}"
    safe_stem = "".join(character if character.isalnum() else "_" for character in stem).strip("_")
    if not safe_stem:
        safe_stem = "run"
    return f"run_{index}_{safe_stem}"


def compute_fallback_expectations(
    *,
    run_count: int,
    top_k_per_run: int,
    max_negatives_per_run: int,
    negatives_per_example: int,
) -> dict[str, int]:
    config_run_capacity_per_run = min(top_k_per_run, max_negatives_per_run)
    config_run_negative_capacity_per_row = min(
        negatives_per_example,
        run_count * config_run_capacity_per_run,
    )
    config_min_fallback_negatives_per_row = max(
        0,
        negatives_per_example - config_run_negative_capacity_per_row,
    )
    return {
        "config_run_capacity_per_run": config_run_capacity_per_run,
        "config_run_negative_capacity_per_row": config_run_negative_capacity_per_row,
        "config_min_fallback_negatives_per_row": config_min_fallback_negatives_per_row,
    }


def compute_build_quality_metrics(
    *,
    internal_rows: list[dict[str, Any]],
    stats: Counter[str],
    candidate_total: int,
    overlap_candidate_total: int,
    selected_run_negative_total: int,
    fallback_negative_total: int,
    excess_fallback_negative_total: int,
    fallback_expectations: dict[str, int],
) -> dict[str, float]:
    row_count = len(internal_rows)
    return {
        "rows": float(row_count),
        "config_run_capacity_per_run": float(
            fallback_expectations["config_run_capacity_per_run"]
        ),
        "config_run_negative_capacity_per_row": float(
            fallback_expectations["config_run_negative_capacity_per_row"]
        ),
        "config_min_fallback_negatives_per_row": float(
            fallback_expectations["config_min_fallback_negatives_per_row"]
        ),
        "avg_unique_run_candidates_per_row": safe_ratio(candidate_total, row_count),
        "avg_overlap_candidates_per_row": safe_ratio(overlap_candidate_total, row_count),
        "avg_selected_negatives_per_row": safe_ratio(
            sum(len(row["neg_pubkeys"]) for row in internal_rows),
            row_count,
        ),
        "avg_selected_run_negatives_per_row": safe_ratio(
            selected_run_negative_total,
            row_count,
        ),
        "avg_fallback_negatives_per_row": safe_ratio(fallback_negative_total, row_count),
        "avg_excess_fallback_negatives_per_row": safe_ratio(
            excess_fallback_negative_total,
            row_count,
        ),
        "rows_with_run_negatives_ratio": safe_ratio(stats["rows_with_run_negatives"], row_count),
        "rows_with_overlap_candidates_ratio": safe_ratio(stats["rows_with_overlap_candidates"], row_count),
        "rows_with_overlap_negatives_ratio": safe_ratio(stats["rows_with_overlap_negatives"], row_count),
        "rows_with_any_fallback_ratio": safe_ratio(stats["rows_with_any_fallback"], row_count),
        "rows_with_excess_fallback_ratio": safe_ratio(
            stats["rows_with_excess_fallback"],
            row_count,
        ),
        "rows_with_only_fallback_ratio": safe_ratio(stats["rows_with_only_fallback"], row_count),
    }


def print_build_quality_snapshot(
    *,
    stats: Counter[str],
    build_quality: dict[str, float],
) -> None:
    row_count = int(build_quality["rows"])
    config_min_fallback = int(build_quality["config_min_fallback_negatives_per_row"])
    print(
        "Build quality snapshot: "
        f"rows_with_run_negatives={stats['rows_with_run_negatives']}/{row_count} "
        f"({build_quality['rows_with_run_negatives_ratio']:.1%}), "
        f"rows_with_overlap_candidates={stats['rows_with_overlap_candidates']}/{row_count} "
        f"({build_quality['rows_with_overlap_candidates_ratio']:.1%}), "
        f"rows_with_any_fallback={stats['rows_with_any_fallback']}/{row_count} "
        f"({build_quality['rows_with_any_fallback_ratio']:.1%}), "
        f"rows_with_excess_fallback={stats['rows_with_excess_fallback']}/{row_count} "
        f"({build_quality['rows_with_excess_fallback_ratio']:.1%}), "
        f"rows_with_only_fallback={stats['rows_with_only_fallback']}/{row_count} "
        f"({build_quality['rows_with_only_fallback_ratio']:.1%}), "
        f"avg_unique_run_candidates_per_row={build_quality['avg_unique_run_candidates_per_row']:.2f}, "
        f"config_min_fallback_per_row={config_min_fallback}"
    )


def validate_build_quality(
    *,
    runs: list[LoadedRun],
    stats: Counter[str],
    build_quality: dict[str, float],
    negatives_per_example: int,
    min_rows_with_run_negatives_ratio: float,
    max_fallback_only_row_ratio: float,
    max_excess_fallback_row_ratio: float,
    min_avg_run_candidates_per_row: float,
    require_multi_run_overlap: bool,
) -> None:
    issues: list[str] = []
    row_count = int(build_quality["rows"])

    if row_count == 0:
        issues.append("No output rows were produced.")
    if stats["rows_with_run_negatives"] == 0:
        issues.append("0 rows received any real run-based negative.")
    if build_quality["rows_with_run_negatives_ratio"] < min_rows_with_run_negatives_ratio:
        issues.append(
            "Too few rows contain real run-based negatives: "
            f"{stats['rows_with_run_negatives']}/{row_count} "
            f"({build_quality['rows_with_run_negatives_ratio']:.1%}) < "
            f"{min_rows_with_run_negatives_ratio:.1%}."
        )
    if build_quality["rows_with_only_fallback_ratio"] > max_fallback_only_row_ratio:
        issues.append(
            "Fallback-only rows dominate the output: "
            f"{stats['rows_with_only_fallback']}/{row_count} "
            f"({build_quality['rows_with_only_fallback_ratio']:.1%}) > "
            f"{max_fallback_only_row_ratio:.1%}."
        )
    if build_quality["rows_with_excess_fallback_ratio"] > max_excess_fallback_row_ratio:
        issues.append(
            "Too many rows need fallback beyond the configuration-implied minimum: "
            f"{stats['rows_with_excess_fallback']}/{row_count} "
            f"({build_quality['rows_with_excess_fallback_ratio']:.1%}) > "
            f"{max_excess_fallback_row_ratio:.1%}."
        )
    if build_quality["avg_unique_run_candidates_per_row"] < min_avg_run_candidates_per_row:
        issues.append(
            "The real candidate pool is too poor: "
            f"avg_unique_run_candidates_per_row="
            f"{build_quality['avg_unique_run_candidates_per_row']:.2f} < "
            f"{min_avg_run_candidates_per_row:.2f}."
        )
    if len(runs) > 1 and require_multi_run_overlap and stats["rows_with_overlap_candidates"] == 0:
        issues.append(
            "Multiple runs were provided, but 0 rows had any overlap candidate supported by more than one run."
        )
    if stats["rows_underfilled"] > 0:
        issues.append(
            f"{stats['rows_underfilled']} row(s) ended up with fewer than {negatives_per_example} negatives."
        )

    if issues:
        raise ValueError("Build quality validation failed:\n- " + "\n- ".join(issues))


def validate_output_rows(
    *,
    internal_rows: list[dict[str, Any]],
    min_average_negative_tokens: float,
) -> dict[str, float]:
    issues: list[str] = []
    total_issue_count = 0
    negative_token_total = 0
    negative_count = 0

    for row_index, row in enumerate(internal_rows, start=1):
        pos_texts = [
            normalized
            for normalized in (
                normalize_text(text) for text in coerce_str_list(row.get("pos"))
            )
            if normalized
        ]
        neg_texts = [
            normalized
            for normalized in (
                normalize_text(text) for text in coerce_str_list(row.get("neg"))
            )
            if normalized
        ]
        pos_pubkeys = set(coerce_str_list(row.get("pos_pubkeys")))
        neg_pubkeys = coerce_str_list(row.get("neg_pubkeys"))

        row_issues: list[str] = []
        if not pos_texts:
            row_issues.append(f"row #{row_index} has no positive text.")
        if not neg_texts:
            row_issues.append(f"row #{row_index} has no negative text.")
        if len(neg_texts) != len(neg_pubkeys):
            row_issues.append(
                f"row #{row_index} has mismatched neg/neg_pubkeys lengths ({len(neg_texts)} vs {len(neg_pubkeys)})."
            )
        if len(set(neg_pubkeys)) != len(neg_pubkeys):
            row_issues.append(f"row #{row_index} contains duplicate negative pubkeys.")

        overlapping_pubkeys = sorted(pos_pubkeys.intersection(neg_pubkeys))
        if overlapping_pubkeys:
            row_issues.append(
                f"row #{row_index} reuses positive pubkey(s) as negatives: {', '.join(overlapping_pubkeys[:5])}."
            )

        positive_text_set = set(pos_texts)
        seen_negative_texts: set[str] = set()
        for neg_text in neg_texts:
            negative_token_total += len(neg_text.split())
            negative_count += 1

            if neg_text in positive_text_set:
                row_issues.append(f"row #{row_index} contains a negative text equal to a positive text.")
                break
            if neg_text in seen_negative_texts:
                row_issues.append(f"row #{row_index} contains duplicate negative texts.")
                break
            seen_negative_texts.add(neg_text)

        for issue in row_issues:
            total_issue_count += 1
            if len(issues) < MAX_REPORTED_VALIDATION_ISSUES:
                issues.append(issue)

    avg_negative_tokens = safe_ratio(negative_token_total, negative_count)
    print(
        "Output validation snapshot: "
        f"negative_count={negative_count}, "
        f"avg_negative_tokens={avg_negative_tokens:.2f}"
    )

    if negative_count == 0:
        total_issue_count += 1
        if len(issues) < MAX_REPORTED_VALIDATION_ISSUES:
            issues.append("No negative texts were produced in the final dataset.")

    if avg_negative_tokens < min_average_negative_tokens:
        total_issue_count += 1
        if len(issues) < MAX_REPORTED_VALIDATION_ISSUES:
            issues.append(
                "Average negative length is implausibly low: "
                f"{avg_negative_tokens:.2f} tokens < {min_average_negative_tokens:.2f}."
            )

    if total_issue_count:
        raise ValueError(
            "Output validation failed "
            f"({total_issue_count} issue(s); first {len(issues)} shown):\n- "
            + "\n- ".join(issues)
        )

    return {
        "negative_count": float(negative_count),
        "avg_negative_tokens": avg_negative_tokens,
    }


def main() -> None:
    args = parse_args()
    if args.top_k_per_run <= 0:
        raise ValueError("--top-k-per-run must be greater than 0.")
    if args.max_negatives_per_run <= 0:
        raise ValueError("--max-negatives-per-run must be greater than 0.")
    if args.negatives_per_example <= 0:
        raise ValueError("--negatives-per-example must be greater than 0.")
    if args.rank_constant < 0:
        raise ValueError("--rank-constant must be greater than or equal to 0.")
    if args.max_doc_frequency < 0:
        raise ValueError("--max-doc-frequency must be greater than or equal to 0.")
    if args.min_avg_run_candidates_per_row < 0:
        raise ValueError("--min-avg-run-candidates-per-row must be greater than or equal to 0.")
    if args.min_average_negative_tokens <= 0:
        raise ValueError("--min-average-negative-tokens must be greater than 0.")

    validate_ratio_arg("--min-run-qid-overlap", args.min_run_qid_overlap)
    validate_ratio_arg("--min-run-usable-qid-overlap", args.min_run_usable_qid_overlap)
    validate_ratio_arg(
        "--min-rows-with-run-negatives-ratio",
        args.min_rows_with_run_negatives_ratio,
    )
    validate_ratio_arg("--max-fallback-only-row-ratio", args.max_fallback_only_row_ratio)
    validate_ratio_arg("--max-excess-fallback-row-ratio", args.max_excess_fallback_row_ratio)

    base_rows = load_jsonl(args.base_internal)
    if not base_rows:
        raise ValueError("Base internal training file is empty.")

    corpus_texts = load_corpus_texts(args.corpus)
    all_docids = list(corpus_texts.keys())
    prepared_base_rows, training_qids = prepare_base_rows(base_rows, corpus_texts)
    rng = random.Random(args.seed)

    runs: list[LoadedRun] = []
    for index, run_path in enumerate(args.runs, start=1):
        path = Path(run_path)
        run = load_run(path, build_run_name(index, path))
        inspect_run_alignment(
            run=run,
            training_qids=training_qids,
            corpus_texts=corpus_texts,
            top_k_per_run=args.top_k_per_run,
        )
        runs.append(run)

    print_run_diagnostics(runs, len(training_qids))
    validate_run_diagnostics(
        runs=runs,
        training_qid_count=len(training_qids),
        min_run_qid_overlap=args.min_run_qid_overlap,
        min_run_usable_qid_overlap=args.min_run_usable_qid_overlap,
    )
    fallback_expectations = compute_fallback_expectations(
        run_count=len(runs),
        top_k_per_run=args.top_k_per_run,
        max_negatives_per_run=args.max_negatives_per_run,
        negatives_per_example=args.negatives_per_example,
    )

    internal_rows: list[dict[str, Any]] = []
    export_rows: list[dict[str, Any]] = []
    source_counter: Counter[str] = Counter()
    doc_usage_counter: Counter[str] = Counter()

    stats: Counter[str] = Counter()
    stats["base_rows"] = len(prepared_base_rows)
    stats["unique_training_qids"] = len(training_qids)
    stats["run_files"] = len(runs)
    stats["rows_without_run_candidates"] = 0
    stats["rows_with_run_negatives"] = 0
    stats["rows_with_overlap_candidates"] = 0
    stats["rows_with_overlap_negatives"] = 0
    stats["rows_with_any_fallback"] = 0
    stats["rows_with_excess_fallback"] = 0
    stats["rows_with_only_fallback"] = 0
    stats["rows_reusing_base_negatives"] = 0
    stats["rows_with_random_fill"] = 0
    stats["rows_underfilled"] = 0
    stats["skipped_duplicate_text_candidates"] = 0

    candidate_total = 0
    overlap_candidate_total = 0
    selected_run_negative_total = 0
    fallback_negative_total = 0
    excess_fallback_negative_total = 0

    for row in prepared_base_rows:
        used_docids = set(row.pos_pubkeys)
        blocked_texts = set(row.positive_texts)

        candidate_pool = build_candidate_pool(
            qid=row.qid,
            runs=runs,
            corpus_texts=corpus_texts,
            excluded_docids=used_docids,
            top_k_per_run=args.top_k_per_run,
            max_negatives_per_run=args.max_negatives_per_run,
            rank_constant=args.rank_constant,
        )

        ranked_candidates = rank_candidates(candidate_pool)
        candidate_total += len(ranked_candidates)
        overlap_candidates = sum(1 for evidence in ranked_candidates if evidence.support_count > 1)
        overlap_candidate_total += overlap_candidates
        if not ranked_candidates:
            stats["rows_without_run_candidates"] += 1
        if overlap_candidates:
            stats["rows_with_overlap_candidates"] += 1

        chosen_negatives: list[str] = []
        selected_run_evidence: list[CandidateEvidence] = []

        for evidence in ranked_candidates:
            if len(chosen_negatives) >= args.negatives_per_example:
                break
            if not can_use_doc(
                docid=evidence.docid,
                corpus_texts=corpus_texts,
                used_docids=used_docids,
                blocked_texts=blocked_texts,
                doc_usage_counter=doc_usage_counter,
                max_doc_frequency=args.max_doc_frequency,
            ):
                if (
                    evidence.docid in corpus_texts
                    and normalize_text(corpus_texts[evidence.docid]) in blocked_texts
                ):
                    stats["skipped_duplicate_text_candidates"] += 1
                continue

            chosen_negatives.append(evidence.docid)
            selected_run_evidence.append(evidence)
            register_doc(
                docid=evidence.docid,
                corpus_texts=corpus_texts,
                used_docids=used_docids,
                blocked_texts=blocked_texts,
            )

        overlap_selected = sum(
            1 for evidence in selected_run_evidence if evidence.support_count > 1
        )
        if selected_run_evidence:
            stats["rows_with_run_negatives"] += 1
        if overlap_selected:
            stats["rows_with_overlap_negatives"] += 1

        selected_run_negative_total += len(selected_run_evidence)
        for evidence in selected_run_evidence:
            if evidence.support_count > 1:
                source_counter["selected_overlap_candidates"] += 1
            else:
                source_counter["selected_single_run_candidates"] += 1
            for run_name in sorted(evidence.run_ranks):
                source_counter[f"selected_from_{run_name}"] += 1

        used_any_fallback = False
        fallback_added_for_row = 0

        if args.reuse_base_negatives and len(chosen_negatives) < args.negatives_per_example:
            reused_any_base = False
            for docid in row.base_neg_pubkeys:
                if len(chosen_negatives) >= args.negatives_per_example:
                    break
                if not can_use_doc(
                    docid=docid,
                    corpus_texts=corpus_texts,
                    used_docids=used_docids,
                    blocked_texts=blocked_texts,
                    doc_usage_counter=doc_usage_counter,
                    max_doc_frequency=args.max_doc_frequency,
                ):
                    continue
                chosen_negatives.append(docid)
                reused_any_base = True
                fallback_added_for_row += 1
                source_counter["base_reuse"] += 1
                register_doc(
                    docid=docid,
                    corpus_texts=corpus_texts,
                    used_docids=used_docids,
                    blocked_texts=blocked_texts,
                )
            if reused_any_base:
                used_any_fallback = True
                stats["rows_reusing_base_negatives"] += 1

        if args.random_fill and len(chosen_negatives) < args.negatives_per_example:
            random_negatives = sample_random_negatives(
                rng=rng,
                all_docids=all_docids,
                corpus_texts=corpus_texts,
                used_docids=used_docids,
                blocked_texts=blocked_texts,
                doc_usage_counter=doc_usage_counter,
                max_doc_frequency=args.max_doc_frequency,
                count=args.negatives_per_example - len(chosen_negatives),
            )
            if random_negatives:
                used_any_fallback = True
                stats["rows_with_random_fill"] += 1
                fallback_added_for_row += len(random_negatives)
                source_counter["random_fill"] += len(random_negatives)
                for docid in random_negatives:
                    chosen_negatives.append(docid)
                    register_doc(
                        docid=docid,
                        corpus_texts=corpus_texts,
                        used_docids=used_docids,
                        blocked_texts=blocked_texts,
                    )

        if used_any_fallback:
            stats["rows_with_any_fallback"] += 1
        if not selected_run_evidence and fallback_added_for_row > 0:
            stats["rows_with_only_fallback"] += 1

        # When multiple runs overlap heavily, the real unique candidate pool can be
        # smaller than the theoretical per-run capacity. Count fallback as "excess"
        # only when it exceeds the actual run-candidate shortfall for this row.
        required_fallback_for_row = max(
            0,
            args.negatives_per_example - len(selected_run_evidence),
        )
        excess_fallback_for_row = max(0, fallback_added_for_row - required_fallback_for_row)
        if excess_fallback_for_row > 0:
            stats["rows_with_excess_fallback"] += 1

        chosen_negatives = chosen_negatives[: args.negatives_per_example]
        if len(chosen_negatives) < args.negatives_per_example:
            stats["rows_underfilled"] += 1

        fallback_negative_total += fallback_added_for_row
        excess_fallback_negative_total += excess_fallback_for_row
        for docid in chosen_negatives:
            doc_usage_counter[docid] += 1

        negative_texts = [corpus_texts[docid] for docid in chosen_negatives]
        internal_row = {
            "qid": row.qid,
            "variant": row.variant,
            "gold_pubkey": row.gold_pubkey,
            "query": row.query,
            "pos": row.positive_texts,
            "neg": negative_texts,
            "pos_pubkeys": row.pos_pubkeys,
            "neg_pubkeys": chosen_negatives,
        }
        export_row = {
            "query": row.query,
            "pos": row.positive_texts,
            "neg": negative_texts,
        }

        internal_rows.append(internal_row)
        export_rows.append(export_row)

    build_quality = compute_build_quality_metrics(
        internal_rows=internal_rows,
        stats=stats,
        candidate_total=candidate_total,
        overlap_candidate_total=overlap_candidate_total,
        selected_run_negative_total=selected_run_negative_total,
        fallback_negative_total=fallback_negative_total,
        excess_fallback_negative_total=excess_fallback_negative_total,
        fallback_expectations=fallback_expectations,
    )
    print_build_quality_snapshot(stats=stats, build_quality=build_quality)
    validate_build_quality(
        runs=runs,
        stats=stats,
        build_quality=build_quality,
        negatives_per_example=args.negatives_per_example,
        min_rows_with_run_negatives_ratio=args.min_rows_with_run_negatives_ratio,
        max_fallback_only_row_ratio=args.max_fallback_only_row_ratio,
        max_excess_fallback_row_ratio=args.max_excess_fallback_row_ratio,
        min_avg_run_candidates_per_row=args.min_avg_run_candidates_per_row,
        require_multi_run_overlap=args.require_multi_run_overlap,
    )
    output_validation = validate_output_rows(
        internal_rows=internal_rows,
        min_average_negative_tokens=args.min_average_negative_tokens,
    )

    summary = {
        "configuration": {
            "base_internal": str(args.base_internal),
            "corpus": str(args.corpus),
            "runs": [
                {
                    "name": run.name,
                    "path": str(run.path),
                    "queries": len(run.rows_by_qid),
                    "diagnostics": run.diagnostics.as_dict(),
                }
                for run in runs
            ],
            "top_k_per_run": args.top_k_per_run,
            "max_negatives_per_run": args.max_negatives_per_run,
            "negatives_per_example": args.negatives_per_example,
            "rank_constant": args.rank_constant,
            "reuse_base_negatives": args.reuse_base_negatives,
            "random_fill": args.random_fill,
            "max_doc_frequency": args.max_doc_frequency,
            "seed": args.seed,
        },
        "quality_thresholds": {
            "min_run_qid_overlap": args.min_run_qid_overlap,
            "min_run_usable_qid_overlap": args.min_run_usable_qid_overlap,
            "min_rows_with_run_negatives_ratio": args.min_rows_with_run_negatives_ratio,
            "max_fallback_only_row_ratio": args.max_fallback_only_row_ratio,
            "max_excess_fallback_row_ratio": args.max_excess_fallback_row_ratio,
            "min_avg_run_candidates_per_row": args.min_avg_run_candidates_per_row,
            "min_average_negative_tokens": args.min_average_negative_tokens,
            "require_multi_run_overlap": args.require_multi_run_overlap,
        },
        "dataset": {
            "rows": len(internal_rows),
            "unique_training_qids": len(training_qids),
            "corpus_documents": len(corpus_texts),
            "config_run_capacity_per_run": build_quality["config_run_capacity_per_run"],
            "config_run_negative_capacity_per_row": build_quality["config_run_negative_capacity_per_row"],
            "config_min_fallback_negatives_per_row": build_quality["config_min_fallback_negatives_per_row"],
            "avg_unique_run_candidates_per_row": build_quality["avg_unique_run_candidates_per_row"],
            "avg_overlap_candidates_per_row": build_quality["avg_overlap_candidates_per_row"],
            "avg_selected_negatives_per_row": build_quality["avg_selected_negatives_per_row"],
            "avg_selected_run_negatives_per_row": build_quality["avg_selected_run_negatives_per_row"],
            "avg_fallback_negatives_per_row": build_quality["avg_fallback_negatives_per_row"],
            "avg_excess_fallback_negatives_per_row": build_quality["avg_excess_fallback_negatives_per_row"],
            "rows_with_run_negatives_ratio": build_quality["rows_with_run_negatives_ratio"],
            "rows_with_overlap_candidates_ratio": build_quality["rows_with_overlap_candidates_ratio"],
            "rows_with_overlap_negatives_ratio": build_quality["rows_with_overlap_negatives_ratio"],
            "rows_with_any_fallback_ratio": build_quality["rows_with_any_fallback_ratio"],
            "rows_with_excess_fallback_ratio": build_quality["rows_with_excess_fallback_ratio"],
            "rows_with_only_fallback_ratio": build_quality["rows_with_only_fallback_ratio"],
            "avg_negative_tokens": output_validation["avg_negative_tokens"],
            "unique_negative_documents_used": len(doc_usage_counter),
        },
        "stats": dict(stats),
        "negative_sources": dict(source_counter),
        "doc_reuse": {
            "max_frequency": max(doc_usage_counter.values(), default=0),
            "top_documents": doc_usage_counter.most_common(10),
        },
        "outputs": {
            "internal": str(args.output_internal),
            "flagembedding": str(args.output),
        },
    }

    write_jsonl(args.output_internal, internal_rows)
    write_jsonl(args.output, export_rows)
    if args.summary_output is not None:
        write_json(args.summary_output, summary)

    print(f"Wrote hard-negative internal data to {args.output_internal}")
    print(f"Wrote FlagEmbedding training data to {args.output}")
    if args.summary_output is not None:
        print(f"Wrote hard-negative summary to {args.summary_output}")
    print(
        "Rows with run negatives: "
        f"{stats['rows_with_run_negatives']}/{len(internal_rows)}, "
        "rows with overlap negatives: "
        f"{stats['rows_with_overlap_negatives']}/{len(internal_rows)}, "
        "rows needing any fallback: "
        f"{stats['rows_with_any_fallback']}/{len(internal_rows)}, "
        "rows with fallback above config minimum: "
        f"{stats['rows_with_excess_fallback']}/{len(internal_rows)}"
    )
    print(
        "Negative sources: "
        + ", ".join(f"{name}={count}" for name, count in sorted(source_counter.items()))
    )


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
