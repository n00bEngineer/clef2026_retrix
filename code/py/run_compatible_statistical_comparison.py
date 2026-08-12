#!/usr/bin/env python3
"""
Discover compatible ranking files, generate StatisticalEvaluator JSON files,
and produce comparison plots for all selected systems.

(by default this targets English TRAIN, because the broadest available active
system set in the repo uses `code/data/Train_set/en_train.json` and
contains the BGE hybrid runs plus the Nemotron reranked variants)

Example:
    python3 code/py/run_compatible_statistical_comparison.py \
      --query-file code/data/Train_set/en_train.json \
      --run-glob "runs/train/**/*.json" \
      --outdir results/statistics/en_train_all_systems \
      --title "English TRAIN"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import shlex


DEFAULT_QUERY_FILE = "code/data/Train_set/en_train.json"
DEFAULT_RUN_GLOBS = ["runs/train/**/*.json"]
DEFAULT_METRICS = [
    "mrr5",
    "ap",
    "ndcg10",
    "recall5",
    "recall10",
    "recall100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every ranking JSON compatible with a query set and "
            "generate statistical comparison plots."
        )
    )
    parser.add_argument(
        "--query-file",
        default=DEFAULT_QUERY_FILE,
        help="Query JSON file with gold pubkeys.",
    )
    parser.add_argument(
        "--run-glob",
        action="append",
        default=None,
        help=(
            "Glob for ranking JSON files. Can be repeated. "
            "Default: runs/train/**/*.json"
        ),
    )
    parser.add_argument(
        "--outdir",
        default="results/statistics/en_train_all_systems",
        help="Directory for generated stats JSON files, plots, and manifests.",
    )
    parser.add_argument(
        "--title",
        default="English TRAIN",
        help="Title prefix for generated plots.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=1.0,
        help=(
            "Minimum fraction of query ids that must exist in the ranking file. "
            "Use 1.0 for strict same-qid comparisons."
        ),
    )
    parser.add_argument(
        "--no-context-filter",
        action="store_true",
        help=(
            "Disable filename/path split-language filtering. By default the "
            "script avoids selecting, for example, enFINAL runs for deFINAL "
            "queries when numeric qids overlap."
        ),
    )
    parser.add_argument(
        "--include-archive",
        action="store_true",
        help="Include files under archive folders. By default archives are skipped.",
    )
    parser.add_argument(
        "--max-systems",
        type=int,
        default=None,
        help="Optional cap after sorting discovered systems by filename.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help=(
            "Metrics passed to statistical_comparison_plots.py. "
            "Use --all-metrics to plot every supported metric."
        ),
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Plot every supported metric instead of the default focused set.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Baseline system name for delta plots. If omitted, the first "
            "selected ranking file is used."
        ),
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Reuse existing stats JSON files and only regenerate plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only generate StatisticalEvaluator JSON files and the manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected systems without running Maven or plotting.",
    )
    return parser.parse_args()


def load_query_ids(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Query file must be a JSON list: {path}")

    qids = {
        str(item["index"])
        for item in payload
        if isinstance(item, dict) and "index" in item
    }
    if not qids:
        raise SystemExit(f"No query ids found in query file: {path}")
    return qids


def is_ranking_payload(payload: object) -> bool:
    if not isinstance(payload, dict) or not payload:
        return False
    sample_values = list(payload.values())[:20]
    return all(isinstance(value, list) for value in sample_values)


def compact_system_name(path: Path) -> str:
    name = path.stem
    replacements = [
        ("reranked_results_", ""),
        ("res_rer_", "rer_"),
        ("res_", ""),
        ("FTAarsen20252026_Lora_", "lora_"),
        ("FTAarsen2526_Lora_", "lora_"),
        ("BASELINE", "baseline"),
        ("onbiEncoder", "on_biencoder"),
    ]
    for old, new in replacements:
        name = name.replace(old, new)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name.strip("_")


def infer_query_context(path: Path) -> tuple[str | None, str | None]:
    full_path = str(path).lower()
    stem = path.stem.lower()

    split = None
    if "train_set" in full_path or "train" in stem:
        split = "train"
    elif "dev_set" in full_path or "dev" in stem:
        split = "dev"
    elif "test_set" in full_path or "test" in stem or "final" in stem:
        split = "test"

    lang = None
    if re.search(r"(^|[_-])de(train|dev|final|_|$)", stem) or "dedev" in stem or "definal" in stem:
        lang = "de"
    elif re.search(r"(^|[_-])fr(train|dev|final|_|$)", stem) or "frdev" in stem or "frfinal" in stem:
        lang = "fr"
    elif re.search(r"(^|[_-])en(train|dev|final|_|$)", stem) or stem in {
        "expanded_queries_en",
    } or "endev" in stem or "enfinal" in stem:
        lang = "en"

    return split, lang


def infer_run_context(path: Path) -> tuple[str | None, str | None]:
    full_path = str(path).lower()
    stem = path.stem.lower()
    parts = {part.lower() for part in path.parts}

    split = None
    if "train" in parts or "train" in stem:
        split = "train"
    elif "dev" in parts or "dev" in stem:
        split = "dev"
    elif "test" in parts or "final" in stem:
        split = "test"

    lang = None
    if "dedev" in stem or "definal" in stem:
        lang = "de"
    elif "frdev" in stem or "frfinal" in stem:
        lang = "fr"
    elif "endev" in stem or "enfinal" in stem:
        lang = "en"
    elif split == "dev" and "baseline_dev" in stem:
        lang = "en"
    elif split == "train" and ("bge_m3_hybrid" in stem or "rer_bge_m3_hybrid" in stem):
        lang = "en"

    return split, lang


def context_matches(
    query_context: tuple[str | None, str | None],
    run_context: tuple[str | None, str | None],
) -> bool:
    query_split, query_lang = query_context
    run_split, run_lang = run_context

    if query_split is not None and run_split is not None and query_split != run_split:
        return False
    if query_lang is not None and run_lang is not None and query_lang != run_lang:
        return False
    return True


def discover_runs(
    run_globs: list[str],
    query_ids: set[str],
    min_coverage: float,
    query_context: tuple[str | None, str | None],
    use_context_filter: bool,
    include_archive: bool,
) -> list[dict]:
    selected = []
    seen_system_names: set[str] = set()
    candidate_paths = sorted(
        {
            Path(match)
            for pattern in run_globs
            for match in glob.glob(pattern, recursive=True)
        }
    )

    for path in candidate_paths:
        if not path.is_file():
            continue
        if not include_archive and "archive" in {part.lower() for part in path.parts}:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not is_ranking_payload(payload):
            continue
        run_context = infer_run_context(path)
        if use_context_filter and not context_matches(query_context, run_context):
            continue

        run_qids = {str(key) for key in payload}
        coverage_count = len(query_ids & run_qids)
        coverage = coverage_count / len(query_ids)
        if coverage < min_coverage:
            continue

        system_name = compact_system_name(path)
        if system_name in seen_system_names:
            continue
        seen_system_names.add(system_name)
        selected.append(
            {
                "system": system_name,
                "path": path,
                "ranking_qids": len(run_qids),
                "matched_qids": coverage_count,
                "coverage": coverage,
                "split": run_context[0] or "",
                "language": run_context[1] or "",
            }
        )

    return selected


def stats_output_path(stats_dir: Path, system_name: str) -> Path:
    return stats_dir / f"{system_name}.json"


def write_manifest(manifest_path: Path, rows: list[dict], stats_dir: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "system",
                "ranking_file",
                "ranking_qids",
                "matched_qids",
                "coverage",
                "split",
                "language",
                "stats_json",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["system"],
                    row["path"],
                    row["ranking_qids"],
                    row["matched_qids"],
                    f"{row['coverage']:.6f}",
                    row["split"],
                    row["language"],
                    stats_output_path(stats_dir, row["system"]),
                ]
            )


def run_command(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def generate_stats(query_file: Path, rows: list[dict], stats_dir: Path) -> list[Path]:
    stats_dir.mkdir(parents=True, exist_ok=True)
    run_command(["mvn", "-q", "-DskipTests", "compile"])

    output_paths = []
    for row in rows:
        output_json = stats_output_path(stats_dir, row["system"])
        output_paths.append(output_json)
        args = [
            str(query_file),
            str(row["path"]),
            str(output_json),
            row["system"],
        ]
        quoted_args = " ".join(shlex.quote(arg) for arg in args)
        run_command(
            [
                "mvn",
                "-q",
                "exec:java",
                "-Dexec.mainClass=unipd.se.StatisticalEvaluator",
                f"-Dexec.args={quoted_args}",
            ]
        )
    return output_paths


def existing_stats_paths(rows: list[dict], stats_dir: Path) -> list[Path]:
    paths = [stats_output_path(stats_dir, row["system"]) for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(
            "Missing stats JSON files while --skip-eval is active:\n"
            + "\n".join(str(path) for path in missing)
        )
    return paths


def run_plots(
    stats_paths: list[Path],
    outdir: Path,
    title: str,
    metrics: list[str],
    all_metrics: bool,
    baseline: str | None,
) -> None:
    plot_dir = outdir / "plots"
    tmp_home = outdir / ".tmp-plot-home"
    (tmp_home / ".cache").mkdir(parents=True, exist_ok=True)
    (tmp_home / ".config" / "matplotlib").mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "code/py/statistical_comparison_plots.py",
        "--inputs",
        *[str(path) for path in stats_paths],
        "--outdir",
        str(plot_dir),
        "--title",
        title,
    ]
    if all_metrics:
        cmd.append("--all-metrics")
    else:
        cmd.extend(["--metrics", *metrics])
    if baseline:
        cmd.extend(["--baseline", baseline])

    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("HOME", str(tmp_home))
    env.setdefault("XDG_CACHE_HOME", str(tmp_home / ".cache"))
    env.setdefault("MPLCONFIGDIR", str(tmp_home / ".config" / "matplotlib"))
    run_command(cmd, env=env)


def main() -> None:
    args = parse_args()
    query_file = Path(args.query_file)
    outdir = Path(args.outdir)
    stats_dir = outdir / "statistical_evaluator_json"
    manifest_path = outdir / "selected_systems.csv"
    run_globs = args.run_glob or DEFAULT_RUN_GLOBS

    query_ids = load_query_ids(query_file)
    query_context = infer_query_context(query_file)
    rows = discover_runs(
        run_globs,
        query_ids,
        args.min_coverage,
        query_context,
        not args.no_context_filter,
        args.include_archive,
    )
    if args.max_systems is not None:
        rows = rows[: args.max_systems]

    if not rows:
        raise SystemExit(
            "No compatible ranking files found. Try a different --query-file, "
            "--run-glob, or lower --min-coverage."
        )

    print(f"Selected {len(rows)} systems for {query_file}:")
    print(
        "Query context: "
        f"split={query_context[0] or 'unknown'}, "
        f"language={query_context[1] or 'unknown'}"
    )
    for row in rows:
        print(
            f"  {row['system']} | {row['matched_qids']}/{len(query_ids)} qids | "
            f"split={row['split'] or 'unknown'} lang={row['language'] or 'unknown'} | "
            f"{row['path']}"
        )

    if args.dry_run:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, rows, stats_dir)
    print(f"Manifest: {manifest_path}")

    if args.skip_eval:
        stats_paths = existing_stats_paths(rows, stats_dir)
    else:
        stats_paths = generate_stats(query_file, rows, stats_dir)

    if not args.no_plots:
        run_plots(
            stats_paths=stats_paths,
            outdir=outdir,
            title=args.title,
            metrics=args.metrics,
            all_metrics=args.all_metrics,
            baseline=args.baseline,
        )

    print(f"Done. Outputs written under: {outdir}")


if __name__ == "__main__":
    main()
