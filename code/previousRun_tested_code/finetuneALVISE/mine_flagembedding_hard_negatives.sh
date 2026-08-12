#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FLAGEMBEDDING_REPO="${FLAGEMBEDDING_REPO:-}"
INPUT_FILE="${INPUT_FILE:-$REPO_ROOT/code/data/finetune/flagembedding/train_base.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$REPO_ROOT/code/data/finetune/flagembedding/train_mined_hn.jsonl}"
CANDIDATE_POOL="${CANDIDATE_POOL:-$REPO_ROOT/code/data/finetune/flagembedding/candidate_pool.jsonl}"
EMBEDDER_NAME_OR_PATH="${EMBEDDER_NAME_OR_PATH:-BAAI/bge-m3}"
EMBEDDER_MODEL_CLASS="${EMBEDDER_MODEL_CLASS:-encoder-only-m3}"
NEGATIVE_NUMBER="${NEGATIVE_NUMBER:-15}"
RANGE_FOR_SAMPLING="${RANGE_FOR_SAMPLING:-5-200}"
SEARCH_BATCH_SIZE="${SEARCH_BATCH_SIZE:-64}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-128}"
PASSAGE_MAX_LENGTH="${PASSAGE_MAX_LENGTH:-512}"
USE_GPU_FOR_SEARCHING="${USE_GPU_FOR_SEARCHING:-True}"

if [[ -z "$FLAGEMBEDDING_REPO" ]]; then
  echo "Set FLAGEMBEDDING_REPO to a local clone of FlagEmbedding." >&2
  exit 1
fi

HN_MINE_SCRIPT="$FLAGEMBEDDING_REPO/scripts/hn_mine.py"
if [[ ! -f "$HN_MINE_SCRIPT" ]]; then
  echo "hn_mine.py not found at: $HN_MINE_SCRIPT" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN"
  "$HN_MINE_SCRIPT"
  --input_file "$INPUT_FILE"
  --output_file "$OUTPUT_FILE"
  --range_for_sampling "$RANGE_FOR_SAMPLING"
  --negative_number "$NEGATIVE_NUMBER"
  --candidate_pool "$CANDIDATE_POOL"
  --search_batch_size "$SEARCH_BATCH_SIZE"
  --embedder_name_or_path "$EMBEDDER_NAME_OR_PATH"
  --embedder_model_class "$EMBEDDER_MODEL_CLASS"
  --embedder_query_max_length "$QUERY_MAX_LENGTH"
  --embedder_passage_max_length "$PASSAGE_MAX_LENGTH"
)

if [[ "$USE_GPU_FOR_SEARCHING" == "True" ]]; then
  cmd+=(--use_gpu_for_searching)
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
