#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
FLAGEMBEDDING_REPO="${FLAGEMBEDDING_REPO:-}"
INPUT_FILE="${INPUT_FILE:-$REPO_ROOT/code/data/finetune/flagembedding/train_hardneg.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-$REPO_ROOT/code/data/finetune/flagembedding/train_hardneg_scored.jsonl}"
RERANKER_NAME_OR_PATH="${RERANKER_NAME_OR_PATH:-BAAI/bge-reranker-v2-m3}"
RERANKER_QUERY_MAX_LENGTH="${RERANKER_QUERY_MAX_LENGTH:-512}"
RERANKER_MAX_LENGTH="${RERANKER_MAX_LENGTH:-1024}"
RERANKER_BATCH_SIZE="${RERANKER_BATCH_SIZE:-512}"
DEVICES="${DEVICES:-}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache/flagembedding/model}"

if [[ -z "$FLAGEMBEDDING_REPO" ]]; then
  echo "Set FLAGEMBEDDING_REPO to a local clone of FlagEmbedding." >&2
  exit 1
fi

ADD_SCORE_SCRIPT="$FLAGEMBEDDING_REPO/scripts/add_reranker_score.py"
if [[ ! -f "$ADD_SCORE_SCRIPT" ]]; then
  echo "add_reranker_score.py not found at: $ADD_SCORE_SCRIPT" >&2
  exit 1
fi

cmd=(
  "$PYTHON_BIN"
  "$ADD_SCORE_SCRIPT"
  --input_file "$INPUT_FILE"
  --output_file "$OUTPUT_FILE"
  --reranker_name_or_path "$RERANKER_NAME_OR_PATH"
  --cache_dir "$CACHE_DIR"
  --reranker_batch_size "$RERANKER_BATCH_SIZE"
  --reranker_query_max_length "$RERANKER_QUERY_MAX_LENGTH"
  --reranker_max_length "$RERANKER_MAX_LENGTH"
)

if [[ -n "$DEVICES" ]]; then
  read -r -a device_array <<< "$DEVICES"
  cmd+=(--devices "${device_array[@]}")
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
