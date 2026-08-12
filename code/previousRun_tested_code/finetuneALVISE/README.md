# BGE-M3 Fine-Tuning Package

This package contains an end-to-end workflow to fine-tune `BAAI/bge-m3` on the
repository dataset using the official FlagEmbedding fine-tuning stack.

The package is organized around four stages:

1. Prepare a clean train/dev/test split and export data in FlagEmbedding format.
2. Build stronger hard negatives from existing retrieval runs.
3. Optionally add reranker teacher scores for knowledge distillation.
4. Train and evaluate a fine-tuned `bge-m3` checkpoint.

## Files

- `prepare_flagembedding_data.py`
  Builds train/dev/test splits from:
  - `code/data/collection_data.json`
  - `code/data/expanded_queries_bge_large.json`

  It writes:
  - repo-compatible query subsets
  - FlagEmbedding corpus/query/qrels files
  - a valid base training set with random negatives
  - a candidate-pool file for official hard-negative mining

- `back_translation.py`
  Helper used by `prepare_flagembedding_data.py` to create optional
  back-translated query variants with Hugging Face translation models and a
  local JSON cache.

- `build_hard_negative_train.py`
  Replaces easy random negatives with harder negatives fused from one or more
  repo retrieval runs such as `results/bm25_results.json` or a bi-encoder run.
  The script prefers documents supported by multiple runs, falls back to base
  or random negatives when needed, and can write a JSON summary with coverage
  and fallback statistics.

- `evaluate_bge_m3_checkpoint.py`
  Runs dense retrieval with a checkpoint and computes the same main metrics used
  by the Java evaluator.

- `mine_flagembedding_hard_negatives.sh`
  Wrapper around the official FlagEmbedding `hn_mine.py`.

- `add_teacher_scores.sh`
  Wrapper around the official FlagEmbedding `add_reranker_score.py`.

- `train_bge_m3.sh`
  Wrapper around the official `FlagEmbedding.finetune.embedder.encoder_only.m3`
  training entry point with settings tuned for this repository.

- `ds_stage0.json`
  Optional DeepSpeed config used by `train_bge_m3.sh`.

## Recommended Workflow

### 1. Build fine-tuning data

```bash
python3 code/src/main/java/unipd/se/finetune/prepare_flagembedding_data.py
```

By default this writes to `code/data/finetune/`.

Important outputs:

- `code/data/finetune/repo/train_queries.json`
- `code/data/finetune/repo/dev_queries.json`
- `code/data/finetune/repo/test_queries.json`
- `code/data/finetune/flagembedding/train_base.jsonl`
- `code/data/finetune/flagembedding/train_base.internal.jsonl`
- `code/data/finetune/flagembedding/corpus.jsonl`
- `code/data/finetune/flagembedding/candidate_pool.jsonl`

The base training file is already valid for FlagEmbedding training, but it uses
mostly easy random negatives. It is a fallback, not the best option.

If you want to add text data augmentation with back translation, enable it in
the same preparation step. Example with English -> Spanish -> English:

```bash
python3 code/src/main/java/unipd/se/finetune/prepare_flagembedding_data.py \
  --augment-back-translation \
  --back-translation-pivot-langs es \
  --back-translation-cache code/data/finetune/augmentation/back_translation_cache.json
```

Practical notes:

- Back translation is applied to the training split only.
- The gold document stays the same; only the query wording changes.
- Augmented examples reuse the same positives and negatives as the source query
  variant, which makes ablations cleaner.
- Exact duplicates are removed automatically.
- The default model template expects MarianMT-style Hugging Face checkpoints
  such as `Helsinki-NLP/opus-mt-en-es` and `Helsinki-NLP/opus-mt-es-en`.
- The first run may download translation models; later runs reuse the local
  cache file.
- If you want to compare pivot languages, run one language at a time first,
  for example `es`, `vi`, `ar`, or `zh`.

### 2. Build stronger hard negatives from repo runs

Run or reuse at least one retrieval system first. The best practical setup is a
mix of lexical and semantic retrieval, but BM25 alone is still valid if that is
the only run you have available.

A good mix is:

- BM25: `results/bm25_results.json`
- your current semantic run, for example:
  `results/res_bge_m3_hybrid_424.json`

Then build a stronger training file:

```bash
python3 code/src/main/java/unipd/se/finetune/build_hard_negative_train.py \
  --base-internal code/data/finetune/flagembedding/train_base.internal.jsonl \
  --corpus code/data/collection_data.json \
  --run results/bm25_results.json \
  --run results/res_bge_m3_hybrid_424.json \
  --top-k-per-run 100 \
  --max-negatives-per-run 8 \
  --negatives-per-example 15 \
  --rank-constant 60 \
  --summary-output code/data/finetune/flagembedding/train_hardneg.summary.json \
  --output-internal code/data/finetune/flagembedding/train_hardneg.internal.jsonl \
  --output code/data/finetune/flagembedding/train_hardneg.jsonl
```

This is the recommended training file if you do not use the official
FlagEmbedding hard-negative miner.

How the builder works:

- It reads one or more retrieval runs and gathers a limited number of valid
  candidates from each run for every query.
- It then fuses candidates with a rank-aware strategy that favors documents
  appearing in multiple runs and documents ranked highly inside those runs.
- If not enough valid hard negatives remain, it reuses negatives from the base
  training file and finally fills the rest with random corpus documents.
- Exact duplicate positive/negative texts inside the same training example are
  filtered out automatically.

Supported run formats:

- JSON mapping: `{qid: [docid, ...]}`
- JSON mapping with scores: `{qid: {docid: score}}`
- JSON or JSONL records with fields such as `qid`, `docid`, `rank`, `score`
- Standard TREC text runs

Useful options:

- `--summary-output`
  Writes a JSON report with candidate coverage, fallback usage, and top reused
  negative documents.
- `--rank-constant`
  Controls the reciprocal-rank-fusion smoothing constant. The default `60`
  works well for typical top-100 runs.
- `--max-doc-frequency`
  Optional global cap on how many examples can reuse the same negative
  document. Keep it disabled unless you explicitly want to reduce repetition.

If you save the summary JSON, these fields are the most useful quick checks:

- `rows_with_run_negatives`
  How many training rows received at least one negative from the retrieval runs.
- `rows_with_overlap_negatives`
  How many rows received at least one selected negative supported by more than
  one run. This is expected to be `0` when you use only a single run.
- `rows_with_any_fallback`
  How many rows needed either base-negative reuse or random fill because the
  run-based candidate pool was not enough.

### 3. Optional: official hard-negative mining

If you have the FlagEmbedding repo available locally:

```bash
FLAGEMBEDDING_REPO=/path/to/FlagEmbedding \
INPUT_FILE=code/data/finetune/flagembedding/train_base.jsonl \
OUTPUT_FILE=code/data/finetune/flagembedding/train_mined_hn.jsonl \
code/src/main/java/unipd/se/finetune/mine_flagembedding_hard_negatives.sh
```

This uses `candidate_pool.jsonl` generated by the preparation step.

### 4. Optional: add teacher scores

Teacher scores are useful if you want a stronger M3 training setup with
knowledge distillation.

```bash
FLAGEMBEDDING_REPO=/path/to/FlagEmbedding \
INPUT_FILE=code/data/finetune/flagembedding/train_hardneg.jsonl \
OUTPUT_FILE=code/data/finetune/flagembedding/train_hardneg_scored.jsonl \
code/src/main/java/unipd/se/finetune/add_teacher_scores.sh
```

Recommended reranker:

- `BAAI/bge-reranker-v2-m3`

### 5. Train the model

Without teacher scores:

```bash
TRAIN_DATA=code/data/finetune/flagembedding/train_hardneg.jsonl \
OUTPUT_DIR=models/ft_bge_m3_hardneg \
code/src/main/java/unipd/se/finetune/train_bge_m3.sh
```

With teacher scores:

```bash
TRAIN_DATA=code/data/finetune/flagembedding/train_hardneg_scored.jsonl \
KNOWLEDGE_DISTILLATION=True \
OUTPUT_DIR=models/ft_bge_m3_hardneg_kd \
code/src/main/java/unipd/se/finetune/train_bge_m3.sh
```

### 6. Evaluate the checkpoint on dev/test

Example on the test split with original queries:

```bash
python3 code/src/main/java/unipd/se/finetune/evaluate_bge_m3_checkpoint.py \
  --model models/ft_bge_m3_hardneg \
  --queries code/data/finetune/repo/test_queries.json \
  --corpus code/data/collection_data.json \
  --query-field original \
  --output results/finetune_test_results.json
```

Example with expanded queries:

```bash
python3 code/src/main/java/unipd/se/finetune/evaluate_bge_m3_checkpoint.py \
  --model models/ft_bge_m3_hardneg \
  --queries code/data/finetune/repo/test_queries.json \
  --corpus code/data/collection_data.json \
  --query-field expanded
```

Pass `--metrics-output results/finetune_test_metrics.json` only if you also want a
separate JSON file with the aggregate metrics. Otherwise the script saves only the
retrieval results and prints the metrics to stdout.

## Data Design Choices

- Positives:
  Each query is paired with the document whose `pubkey` matches the query gold
  label.

- Query variants:
  The preparation script uses both `original` and `expanded` query texts for the
  training split by default. This creates two retrieval views of the same label.
  If back translation is enabled, each selected training variant can produce one
  additional augmented query per pivot language.

- Negatives:
  The best practical setup for this repo is to mix lexical and semantic hard
  negatives. In practice that means combining BM25 negatives with negatives from
  your current semantic retriever. The hard-negative builder fuses evidence
  across runs with a rank-aware scoring rule, filters duplicate texts within the
  same example, and can optionally limit how often the same negative document is
  reused across the dataset.

- Evaluation:
  Fine-tuning should be evaluated on held-out queries only. Do not fine-tune on
  the full 14,977 queries and then report those same queries as final test
  performance.

## Recommended Defaults

- Train query variants: `original expanded`
- Back translation first pass:
  start with one pivot language such as `es`, then compare against the
  non-augmented baseline
- Hard-negative source mix:
  `BM25 + current semantic run` when available, otherwise `BM25` only
- Query max length: `128`
- Passage max length: `512`
- Negatives per example: `15`
- Hard-negative fusion:
  `top_k_per_run=100`, `max_negatives_per_run=8`, `rank_constant=60`
- First training pass:
  hard negatives, no teacher scores
- Second training pass:
  hard negatives plus teacher scores

## Cluster Notes

- Prefer at least 1 GPU for training. More GPUs help through `torchrun`.
- If VRAM is tight, lower `PER_DEVICE_BATCH_SIZE` and keep gradient
  checkpointing enabled.
- If training on multiple GPUs, keep `NEGATIVES_CROSS_DEVICE=True`.
