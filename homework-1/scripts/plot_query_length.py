"""
Plot token-length histograms for Train, Dev, and Test query splits.


Usage:
    python plot_query_lengths.py \
        --train  data/en_train.json data/fr_train.json data/de_train.json \
        --dev    data/en_dev.json   data/fr_deb.json   data/de_dev.json \
        --test   data/final_en_test.json data/final_fr_test.json data/final_de_test.json \
        --field  text \
        --output figure/query_token_lengths.pdf

Arguments:
    --train     One or more JSON query files for the training split.
    --dev       One or more JSON query files for the development split.
    --test      One or more JSON query files for the test split.
    --field     JSON field to tokenize (default: "text").
                Use "expanded" to plot the expanded query lengths instead.
    --tokenizer Hugging Face tokenizer ID (default: meta-llama/Llama-3.2-1B).
                Falls back to whitespace splitting when the model is not
                available locally (no GPU or authentication required).
    --output    Output file path. The format is inferred from the extension
                (.pdf, .png, .svg). Default: query_token_lengths.pdf
    --bins      Number of histogram bins (default: 30).
    --dpi       Resolution for raster formats (default: 300).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Styling constants (matches the paper figure) ────────────────────────────
COLORS      = ["#3d3d3d", "#6b6b6b", "#b0b0b0"]   # dark → light grey
LABEL_SIZE  = 11
TITLE_SIZE  = 12
TICK_SIZE   = 9
SPINE_COLOR = "#cccccc"
GRID_COLOR  = "#e8e8e8"
FONT_FAMILY = "DejaVu Sans"   # available everywhere without extra installs


def load_texts(paths: list[str], field: str) -> list[str]:
    """Load query texts from one or more JSON files.

    Args:
        paths: Paths to JSON files. Each file must be a JSON array of objects.
        field: The key to extract from each object.

    Returns:
        A flat list of non-empty strings from all files.
    """
    texts = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"  WARNING: file not found, skipping — {path}", file=sys.stderr)
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            val = item.get(field, "")
            if val and val.strip():
                texts.append(val.strip())
        print(f"  Loaded {len(data)} queries from {path.name}")
    return texts


def build_tokenizer(model_id: str):
    """Return a callable that maps a string to a token count.

    Tries to load the Hugging Face tokenizer. Falls back to whitespace
    splitting if the model is not cached or requires authentication.

    Args:
        model_id: Hugging Face model ID.

    Returns:
        A function (str) -> int.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        print(f"  Using tokenizer: {model_id} (cached)")
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    except Exception:
        pass

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"  Using tokenizer: {model_id} (downloaded)")
        return lambda text: len(tok.encode(text, add_special_tokens=False))
    except Exception as e:
        print(f"  WARNING: could not load tokenizer ({e}).", file=sys.stderr)
        print("  Falling back to whitespace token count.", file=sys.stderr)
        return lambda text: len(text.split())


def token_lengths(texts: list[str], tokenize) -> np.ndarray:
    """Compute token lengths for every text.

    Args:
        texts: List of query strings.
        tokenize: Callable (str) -> int.

    Returns:
        NumPy array of integer token counts.
    """
    return np.array([tokenize(t) for t in texts], dtype=np.int32)


def plot_histograms(
    splits: list[tuple[str, np.ndarray]],
    bins: int,
    output: str,
    dpi: int,
) -> None:
    """Render and save a three-panel histogram figure.

    Args:
        splits: List of (label, lengths) pairs, one per panel.
        bins: Number of histogram bins.
        output: Output file path.
        dpi: DPI for raster formats.
    """
    n = len(splits)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]

    plt.rcParams["font.family"] = FONT_FAMILY

    for ax, (label, lengths), color in zip(axes, splits, COLORS):
        ax.hist(lengths, bins=bins, color=color, edgecolor="none", linewidth=0)

        # Axis labels
        ax.set_xlabel("Tokens", fontsize=LABEL_SIZE)
        ax.set_title(label, fontsize=TITLE_SIZE, pad=8)

        # Y label only on the leftmost panel
        if ax is axes[0]:
            ax.set_ylabel("Frequency", fontsize=LABEL_SIZE)

        # Tick styling
        ax.tick_params(axis="both", labelsize=TICK_SIZE, length=3, color=SPINE_COLOR)

        # Light horizontal grid, no vertical grid
        ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.xaxis.grid(False)

        # Minimal spines (bottom + left only)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["bottom", "left"]:
            ax.spines[spine].set_color(SPINE_COLOR)
            ax.spines[spine].set_linewidth(0.8)

        # Annotation: n and median
        med = int(np.median(lengths))
        ax.text(
            0.97, 0.97,
            f"n={len(lengths):,}\nmedian={med}",
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=8, color="#555555",
        )

    fig.tight_layout(w_pad=2.5)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"\nFigure saved → {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Token-length histograms for RETRIX query splits."
    )
    parser.add_argument("--train",     nargs="+", default=[],
                        metavar="FILE",
                        help="Training-split query JSON files.")
    parser.add_argument("--dev",       nargs="+", default=[],
                        metavar="FILE",
                        help="Development-split query JSON files.")
    parser.add_argument("--test",      nargs="+", default=[],
                        metavar="FILE",
                        help="Test-split query JSON files.")
    parser.add_argument("--field",     default="text",
                        help="JSON field to tokenize (default: text).")
    parser.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B",
                        help="HF tokenizer ID (default: meta-llama/Llama-3.2-1B).")
    parser.add_argument("--output",    default="query_token_lengths.pdf",
                        help="Output path (default: query_token_lengths.pdf).")
    parser.add_argument("--bins",      type=int, default=30,
                        help="Histogram bins (default: 30).")
    parser.add_argument("--dpi",       type=int, default=300,
                        help="DPI for raster output (default: 300).")
    args = parser.parse_args()

    split_defs = [
        ("Train Queries", args.train),
        ("Dev Queries",   args.dev),
        ("Test Queries",  args.test),
    ]
    # Keep only splits that have at least one file specified.
    split_defs = [(label, files) for label, files in split_defs if files]

    if not split_defs:
        parser.error(
            "Provide at least one of --train, --dev, --test with file paths."
        )

    print(f"Field: '{args.field}'")
    tokenize = build_tokenizer(args.tokenizer)

    splits: list[tuple[str, np.ndarray]] = []
    for label, files in split_defs:
        print(f"\n{label}:")
        texts = load_texts(files, args.field)
        if not texts:
            print(f"  WARNING: no texts loaded for {label}, skipping.", file=sys.stderr)
            continue
        lengths = token_lengths(texts, tokenize)
        print(f"  Total: {len(lengths):,} queries | "
              f"min={lengths.min()} median={int(np.median(lengths))} "
              f"max={lengths.max()} mean={lengths.mean():.1f}")
        splits.append((label, lengths))

    if not splits:
        sys.exit("No data to plot.")

    plot_histograms(splits, args.bins, args.output, args.dpi)


if __name__ == "__main__":
    main()
