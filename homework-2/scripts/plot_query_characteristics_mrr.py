#!/usr/bin/env python3
"""Plot N1000 MRR@5 by simple query-characteristic buckets.

The script intentionally uses only the Python standard library. It computes
per-query MRR@5 from a ranked-output JSON file, groups queries by token length
and by a lightweight named-entity cue heuristic, then writes both a CSV summary
and an SVG figure.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import statistics
from pathlib import Path


STOP_TITLECASE = {
    "A", "An", "The", "This", "That", "These", "Those", "While", "With",
    "Without", "For", "From", "Into", "Onto", "Over", "Under", "Of", "In",
    "On", "At", "To", "And", "Or", "But", "If", "It", "Its", "They", "We",
    "Our", "Their", "There", "Here", "What", "Why", "How", "When", "Where",
    "Not", "No", "Yes", "Fact", "Study", "Studies", "Results", "Finding",
    "Findings", "Background", "Objective", "Methods", "Conclusion",
    "Conclusions", "Evidence", "Report", "Research", "Open", "Access",
    "Significant", "Recent", "New", "First",
}

ORG_OR_VENUE_WORDS = {
    "University", "College", "Institute", "Institutes", "Hospital",
    "Hospitals", "Center", "Centre", "Centers", "Centres", "Foundation",
    "Association", "Society", "School", "Schools", "Department",
    "Departments", "Laboratory", "Laboratories", "Lab", "Labs", "Clinic",
    "Clinics", "Agency", "Ministry", "Office", "Council", "Academy",
    "Administration", "Commission", "Committee", "Consortium", "Trial",
    "Group", "Network", "Project", "Programme", "Program", "Oxford",
    "Harvard", "Stanford", "Cambridge", "Yale", "MIT", "Imperial", "Johns",
    "Hopkins", "Mayo", "Cleveland", "Karolinska", "Lancet", "Nature",
    "Science", "JAMA", "BMJ", "NEJM", "CDC", "FDA", "NIH", "WHO", "NHS",
    "OECD", "UNICEF", "UNESCO",
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def mrr_at_5(gold_pubkey: object, predictions: list[object]) -> float:
    gold = str(gold_pubkey)
    for rank, pubkey in enumerate(predictions[:5], start=1):
        if str(pubkey) == gold:
            return 1.0 / rank
    return 0.0


def entity_cue_score(text: str) -> int:
    """Return a lightweight score for named-entity-like query evidence.

    The heuristic counts multiple independent cues rather than a single
    capitalized word: organization/venue keywords, author-like constructions,
    acronyms, titlecase tokens away from the sentence start, and consecutive
    titlecase tokens. This is descriptive, not a replacement for NER.
    """

    tokens = re.findall(
        r"[@#]?[A-Za-z][A-Za-z0-9.'-]*(?:-[A-Za-z0-9.'-]+)?",
        text,
    )
    score = 0

    for word in ORG_OR_VENUE_WORDS:
        if re.search(r"\b" + re.escape(word) + r"\b", text):
            score += 2

    if re.search(
        r"\b[A-Z][a-z]+\s+et\s+al\b|\bby\s+[A-Z][a-z]+|\bfrom\s+[A-Z][a-z]+",
        text,
    ):
        score += 2

    acronyms = re.findall(r"\b[A-Z]{2,}(?:[-/]?\d+)?\b|\b[A-Z]\.[A-Z](?:\.)?\b", text)
    score += len([acronym for acronym in acronyms if acronym != "RT"])

    for index, token in enumerate(tokens):
        clean = token.strip("@#")
        if clean in STOP_TITLECASE:
            continue
        if clean[:1].isupper() and not clean.isupper() and len(clean) > 2 and index > 0:
            score += 1

    for left, right in zip(tokens, tokens[1:]):
        clean_left = left.strip("@#")
        clean_right = right.strip("@#")
        if (
            clean_left[:1].isupper()
            and clean_right[:1].isupper()
            and not clean_left.isupper()
            and not clean_right.isupper()
            and clean_left not in STOP_TITLECASE
            and clean_right not in STOP_TITLECASE
        ):
            score += 2

    return score


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summarize(records: list[dict[str, float | int]]) -> list[dict[str, object]]:
    bucket_defs = [
        (
            "Token length",
            [
                ("Short <=20", lambda row: row["tokens"] <= 20),
                ("Medium 21-34", lambda row: 21 <= row["tokens"] <= 34),
                ("Long >=35", lambda row: row["tokens"] >= 35),
            ],
        ),
        (
            "Entity cues",
            [
                ("Entity-sparse score <4", lambda row: row["entity_score"] < 4),
                ("Entity-rich score >=4", lambda row: row["entity_score"] >= 4),
            ],
        ),
    ]

    rows: list[dict[str, object]] = []
    for family, buckets in bucket_defs:
        for label, predicate in buckets:
            bucket = [row for row in records if predicate(row)]
            values = [float(row["mrr5"]) for row in bucket]
            token_lengths = [int(row["tokens"]) for row in bucket]
            mean = sum(values) / len(values)
            margin = ci95(values)
            rows.append(
                {
                    "family": family,
                    "bucket": label,
                    "queries": len(bucket),
                    "median_tokens": statistics.median(token_lengths),
                    "mean_mrr5": mean,
                    "ci95_low": mean - margin,
                    "ci95_high": mean + margin,
                    "ci95_margin": margin,
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt_int(value: object) -> str:
    return f"{int(value):,}"


def svg_text(
    x: float,
    y: float,
    text: object,
    size: int = 18,
    anchor: str = "middle",
    weight: str = "400",
    color: str = "#333333",
    rotate: float | None = None,
) -> str:
    rotation = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"'
        f' font-family="DejaVu Sans, Arial, sans-serif" font-size="{size}"'
        f' font-weight="{weight}" fill="{color}"{rotation}>{html.escape(str(text))}</text>'
    )


def render_svg(rows: list[dict[str, object]], output: Path) -> None:
    width = 1180
    height = 560
    y_max = 0.80
    plot_top = 84
    plot_height = 330
    panels = [
        ("Token length buckets", [row for row in rows if row["family"] == "Token length"], 90, 520),
        ("Named-entity cue buckets", [row for row in rows if row["family"] == "Entity cues"], 690, 380),
    ]
    colors = ["#4a4a4a", "#777777", "#a0a0a0", "#5c7f8e", "#b05d4d"]

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        ]

    color_index = 0
    for title, panel_rows, left, plot_width in panels:
        baseline = plot_top + plot_height
        parts.append(svg_text(left + plot_width / 2, 66, title, 19, weight="700"))
        parts.append(svg_text(left - 60, plot_top + plot_height / 2, "Mean MRR@5", 16, rotate=-90))

        for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
            y = baseline - (tick / y_max) * plot_height
            parts.append(
                f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e7e7e7" stroke-width="1"/>'
            )
            parts.append(svg_text(left - 12, y + 5, f"{tick:.1f}", 14, anchor="end", color="#555555"))

        parts.append(
            f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{baseline}" stroke="#bdbdbd" stroke-width="1.2"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{baseline}" x2="{left + plot_width}" y2="{baseline}" stroke="#bdbdbd" stroke-width="1.2"/>'
        )

        slot = plot_width / len(panel_rows)
        bar_width = min(92, slot * 0.48)
        for offset, row in enumerate(panel_rows):
            center = left + slot * (offset + 0.5)
            mean = float(row["mean_mrr5"])
            low = max(0.0, float(row["ci95_low"]))
            high = min(y_max, float(row["ci95_high"]))
            bar_top = baseline - (mean / y_max) * plot_height
            bar_height = baseline - bar_top
            x = center - bar_width / 2
            color = colors[color_index]
            color_index += 1

            parts.append(
                f'<rect x="{x:.1f}" y="{bar_top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
            )
            high_y = baseline - (high / y_max) * plot_height
            low_y = baseline - (low / y_max) * plot_height
            parts.append(
                f'<line x1="{center:.1f}" y1="{high_y:.1f}" x2="{center:.1f}" y2="{low_y:.1f}" stroke="#222222" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{center - 11:.1f}" y1="{high_y:.1f}" x2="{center + 11:.1f}" y2="{high_y:.1f}" stroke="#222222" stroke-width="2"/>'
            )
            parts.append(
                f'<line x1="{center - 11:.1f}" y1="{low_y:.1f}" x2="{center + 11:.1f}" y2="{low_y:.1f}" stroke="#222222" stroke-width="2"/>'
            )
            parts.append(svg_text(center, bar_top - 10, f"{mean:.3f}", 16, weight="700"))

            label_lines = [
                row["bucket"],
                f"n={fmt_int(row['queries'])}",
                f"median tokens={row['median_tokens']:g}",
            ]
            for line_number, line in enumerate(label_lines):
                parts.append(svg_text(center, baseline + 28 + 22 * line_number, line, 14, color="#444444"))

    output.parent.mkdir(parents=True, exist_ok=True)
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def build_records(query_path: Path, ranking_path: Path) -> list[dict[str, float | int]]:
    queries = load_json(query_path)
    rankings = load_json(ranking_path)
    records: list[dict[str, float | int]] = []
    for query in queries:
        text = query["text"]
        records.append(
            {
                "tokens": len(text.split()),
                "entity_score": entity_cue_score(text),
                "mrr5": mrr_at_5(query["pubkey"], rankings.get(str(query["index"]), [])),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        default="code/data/Train_set/en_train.json",
        type=Path,
        help="Query JSON file with index, pubkey, and text fields.",
    )
    parser.add_argument(
        "--ranking",
        default="runs/train/reranked/res_rer_bge_m3_hybrid_513_nemotron_tk1000.json",
        type=Path,
        help="Ranked-output JSON mapping query IDs to ordered pubkey lists.",
    )
    parser.add_argument(
        "--output-svg",
        default="homework-2/figure/query_characteristics_mrr.svg",
        type=Path,
        help="Output SVG figure path.",
    )
    parser.add_argument(
        "--output-csv",
        default="homework-2/figure/query_characteristics_mrr_summary.csv",
        type=Path,
        help="Output CSV summary path.",
    )
    args = parser.parse_args()

    records = build_records(args.queries, args.ranking)
    rows = summarize(records)
    write_csv(rows, args.output_csv)
    render_svg(rows, args.output_svg)

    for row in rows:
        print(
            f"{row['family']}: {row['bucket']} "
            f"n={row['queries']} median_tokens={row['median_tokens']:g} "
            f"MRR@5={row['mean_mrr5']:.4f} "
            f"95% CI=[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
        )
    print(f"SVG written to {args.output_svg}")
    print(f"CSV written to {args.output_csv}")


if __name__ == "__main__":
    main()
