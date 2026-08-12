#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"

CONFIG_FILE="${CONFIG_FILE:-${1:-}}"
if [[ -n "$CONFIG_FILE" ]]; then
  if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
fi

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_NAME="${MODEL_NAME:-BAAI/bge-m3}"
TRAIN_DATA="${TRAIN_DATA:-$REPO_ROOT/code/data/finetune/flagembedding/train_hardneg.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/ft_bge_m3_hardneg}"
CACHE_DIR="${CACHE_DIR:-$REPO_ROOT/.cache/flagembedding/model}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-$REPO_ROOT/.cache/flagembedding/data}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
QUERY_MAX_LEN="${QUERY_MAX_LEN:-128}"
PASSAGE_MAX_LEN="${PASSAGE_MAX_LEN:-512}"
TRAIN_GROUP_SIZE="${TRAIN_GROUP_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-500}"
KNOWLEDGE_DISTILLATION="${KNOWLEDGE_DISTILLATION:-False}"
USE_DEEPSPEED="${USE_DEEPSPEED:-auto}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-$SCRIPT_DIR/ds_stage0.json}"
NEGATIVES_CROSS_DEVICE="${NEGATIVES_CROSS_DEVICE:-True}"
USE_SELF_DISTILL="${USE_SELF_DISTILL:-True}"
SELF_DISTILL_START_STEP="${SELF_DISTILL_START_STEP:-0}"
FIX_ENCODER="${FIX_ENCODER:-False}"
RUN_EVAL_AFTER_TRAIN="${RUN_EVAL_AFTER_TRAIN:-False}"
COMPARE_TO_BASELINE_AFTER_TRAIN="${COMPARE_TO_BASELINE_AFTER_TRAIN:-False}"
BASELINE_MODEL="${BASELINE_MODEL:-BAAI/bge-m3}"
BASELINE_LABEL="${BASELINE_LABEL:-baseline}"
CANDIDATE_LABEL="${CANDIDATE_LABEL:-finetuned}"
EVAL_QUERIES="${EVAL_QUERIES:-$REPO_ROOT/code/data/finetune/repo/dev_queries.json}"
EVAL_CORPUS="${EVAL_CORPUS:-$REPO_ROOT/code/data/collection_data.json}"
EVAL_QUERY_FIELD="${EVAL_QUERY_FIELD:-expanded}"
EVAL_RETRIEVAL_MODE="${EVAL_RETRIEVAL_MODE:-hybrid}"
EVAL_TOP_K="${EVAL_TOP_K:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-}"
EVAL_CPU_THREADS="${EVAL_CPU_THREADS:-6}"
EVAL_QUERY_PREFIX="${EVAL_QUERY_PREFIX:-}"
EVAL_MAX_QUERIES="${EVAL_MAX_QUERIES:-}"
EVAL_CANDIDATE_MULTIPLIER="${EVAL_CANDIDATE_MULTIPLIER:-2}"
EVAL_HYBRID_MAX_LENGTH="${EVAL_HYBRID_MAX_LENGTH:-256}"
EVAL_COLBERT_SCORE_BATCH_SIZE="${EVAL_COLBERT_SCORE_BATCH_SIZE:-512}"
EVAL_DENSE_WEIGHT="${EVAL_DENSE_WEIGHT:-0.4}"
EVAL_SPARSE_WEIGHT="${EVAL_SPARSE_WEIGHT:-0.2}"
EVAL_COLBERT_WEIGHT="${EVAL_COLBERT_WEIGHT:-0.4}"
EVAL_PRECOMPUTE_CORPUS_HYBRID="${EVAL_PRECOMPUTE_CORPUS_HYBRID:-False}"
EVAL_CORPUS_COLBERT_DTYPE="${EVAL_CORPUS_COLBERT_DTYPE:-float16}"
EVAL_LOG_EVERY_QUERIES="${EVAL_LOG_EVERY_QUERIES:-100}"

MODEL_RUN_NAME="${MODEL_RUN_NAME:-$(basename "$OUTPUT_DIR")}"
EVAL_RESULTS_OUTPUT="${EVAL_RESULTS_OUTPUT:-$REPO_ROOT/results/${MODEL_RUN_NAME}_dev_results.json}"
EVAL_METRICS_OUTPUT="${EVAL_METRICS_OUTPUT:-$REPO_ROOT/results/${MODEL_RUN_NAME}_dev_metrics.json}"
COMPARE_OUTPUT_DIR="${COMPARE_OUTPUT_DIR:-$REPO_ROOT/results/${MODEL_RUN_NAME}_comparison}"

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Training data not found: $TRAIN_DATA" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$DATA_CACHE_DIR"

cmd=(
  "$TORCHRUN_BIN"
  --nproc_per_node "$NPROC_PER_NODE"
  -m FlagEmbedding.finetune.embedder.encoder_only.m3
  --model_name_or_path "$MODEL_NAME"
  --cache_dir "$CACHE_DIR"
  --train_data "$TRAIN_DATA"
  --cache_path "$DATA_CACHE_DIR"
  --train_group_size "$TRAIN_GROUP_SIZE"
  --query_max_len "$QUERY_MAX_LEN"
  --passage_max_len "$PASSAGE_MAX_LEN"
  --pad_to_multiple_of 8
  --same_dataset_within_batch True
  --small_threshold 0
  --drop_threshold 0
  --output_dir "$OUTPUT_DIR"
  --overwrite_output_dir
  --learning_rate "$LEARNING_RATE"
  --fp16
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --dataloader_drop_last True
  --warmup_ratio "$WARMUP_RATIO"
  --gradient_checkpointing
  --logging_steps "$LOGGING_STEPS"
  --save_steps "$SAVE_STEPS"
  --temperature 0.02
  --sentence_pooling_method cls
  --normalize_embeddings True
  --unified_finetuning True
  --use_self_distill "$USE_SELF_DISTILL"
  --fix_encoder "$FIX_ENCODER"
  --self_distill_start_step "$SELF_DISTILL_START_STEP"
)

if [[ "$NEGATIVES_CROSS_DEVICE" == "True" ]]; then
  cmd+=(--negatives_cross_device)
fi

if [[ "$KNOWLEDGE_DISTILLATION" == "True" ]]; then
  cmd+=(--knowledge_distillation True --kd_loss_type m3_kd_loss)
else
  cmd+=(--knowledge_distillation False)
fi

if [[ "$USE_DEEPSPEED" == "True" ]]; then
  cmd+=(--deepspeed "$DEEPSPEED_CONFIG")
elif [[ "$USE_DEEPSPEED" == "auto" && -f "$DEEPSPEED_CONFIG" ]]; then
  cmd+=(--deepspeed "$DEEPSPEED_CONFIG")
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
"${cmd[@]}"

build_eval_cmd() {
  local model_path="$1"
  local results_output="$2"
  local metrics_output="$3"
  local -a eval_cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/evaluate_bge_m3_checkpoint.py"
    --model "$model_path"
    --queries "$EVAL_QUERIES"
    --corpus "$EVAL_CORPUS"
    --output "$results_output"
    --metrics-output "$metrics_output"
    --retrieval-mode "$EVAL_RETRIEVAL_MODE"
    --query-field "$EVAL_QUERY_FIELD"
    --top-k "$EVAL_TOP_K"
    --batch-size "$EVAL_BATCH_SIZE"
    --cpu-threads "$EVAL_CPU_THREADS"
    --candidate-multiplier "$EVAL_CANDIDATE_MULTIPLIER"
    --hybrid-max-length "$EVAL_HYBRID_MAX_LENGTH"
    --colbert-score-batch-size "$EVAL_COLBERT_SCORE_BATCH_SIZE"
    --dense-weight "$EVAL_DENSE_WEIGHT"
    --sparse-weight "$EVAL_SPARSE_WEIGHT"
    --colbert-weight "$EVAL_COLBERT_WEIGHT"
    --corpus-colbert-dtype "$EVAL_CORPUS_COLBERT_DTYPE"
    --log-every-queries "$EVAL_LOG_EVERY_QUERIES"
  )

  if [[ -n "$EVAL_MAX_LENGTH" ]]; then
    eval_cmd+=(--max-length "$EVAL_MAX_LENGTH")
  fi
  if [[ -n "$EVAL_QUERY_PREFIX" ]]; then
    eval_cmd+=(--query-prefix "$EVAL_QUERY_PREFIX")
  fi
  if [[ -n "$EVAL_MAX_QUERIES" ]]; then
    eval_cmd+=(--max-queries "$EVAL_MAX_QUERIES")
  fi
  if [[ "$EVAL_PRECOMPUTE_CORPUS_HYBRID" == "True" ]]; then
    eval_cmd+=(--precompute-corpus-hybrid)
  fi

  printf '%s\n' "${eval_cmd[@]}"
}

if [[ "$COMPARE_TO_BASELINE_AFTER_TRAIN" == "True" ]]; then
  compare_cmd=(
    "$PYTHON_BIN"
    "$SCRIPT_DIR/compare_bge_m3_checkpoints.py"
    --baseline-model "$BASELINE_MODEL"
    --candidate-model "$OUTPUT_DIR"
    --baseline-label "$BASELINE_LABEL"
    --candidate-label "$CANDIDATE_LABEL"
    --queries "$EVAL_QUERIES"
    --corpus "$EVAL_CORPUS"
    --output-dir "$COMPARE_OUTPUT_DIR"
    --retrieval-mode "$EVAL_RETRIEVAL_MODE"
    --query-field "$EVAL_QUERY_FIELD"
    --top-k "$EVAL_TOP_K"
    --batch-size "$EVAL_BATCH_SIZE"
    --cpu-threads "$EVAL_CPU_THREADS"
    --candidate-multiplier "$EVAL_CANDIDATE_MULTIPLIER"
    --hybrid-max-length "$EVAL_HYBRID_MAX_LENGTH"
    --colbert-score-batch-size "$EVAL_COLBERT_SCORE_BATCH_SIZE"
    --dense-weight "$EVAL_DENSE_WEIGHT"
    --sparse-weight "$EVAL_SPARSE_WEIGHT"
    --colbert-weight "$EVAL_COLBERT_WEIGHT"
    --corpus-colbert-dtype "$EVAL_CORPUS_COLBERT_DTYPE"
    --log-every-queries "$EVAL_LOG_EVERY_QUERIES"
  )

  if [[ -n "$EVAL_MAX_LENGTH" ]]; then
    compare_cmd+=(--max-length "$EVAL_MAX_LENGTH")
  fi
  if [[ -n "$EVAL_QUERY_PREFIX" ]]; then
    compare_cmd+=(--query-prefix "$EVAL_QUERY_PREFIX")
  fi
  if [[ -n "$EVAL_MAX_QUERIES" ]]; then
    compare_cmd+=(--max-queries "$EVAL_MAX_QUERIES")
  fi
  if [[ "$EVAL_PRECOMPUTE_CORPUS_HYBRID" == "True" ]]; then
    compare_cmd+=(--precompute-corpus-hybrid)
  fi

  printf 'Running post-train comparison:\n%s\n' "${compare_cmd[*]}"
  "${compare_cmd[@]}"
elif [[ "$RUN_EVAL_AFTER_TRAIN" == "True" ]]; then
  mapfile -t eval_cmd < <(build_eval_cmd "$OUTPUT_DIR" "$EVAL_RESULTS_OUTPUT" "$EVAL_METRICS_OUTPUT")
  printf 'Running post-train evaluation:\n%s\n' "${eval_cmd[*]}"
  "${eval_cmd[@]}"
fi
