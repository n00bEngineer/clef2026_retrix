#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../../../../.." && pwd)"

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
MAX_STEPS="${MAX_STEPS:--1}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
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
USE_FP16="${USE_FP16:-True}"
USE_CPU="${USE_CPU:-False}"

if [[ "$USE_CPU" == "True" && "$USE_FP16" == "True" ]]; then
  echo "USE_CPU=True forces USE_FP16=False because fp16 is not supported for CPU training." >&2
  USE_FP16="False"
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
  echo "Training data not found: $TRAIN_DATA" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$DATA_CACHE_DIR"

supports_overwrite_output_dir="$("$PYTHON_BIN" - <<'PY'
from transformers import TrainingArguments
print("1" if "overwrite_output_dir" in TrainingArguments.__dataclass_fields__ else "0")
PY
)"

if [[ "$supports_overwrite_output_dir" != "1" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "Output directory already exists and this transformers version does not support --overwrite_output_dir: $OUTPUT_DIR" >&2
  echo "Use a fresh OUTPUT_DIR or empty the directory manually before retrying." >&2
  exit 1
fi

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
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --max_steps "$MAX_STEPS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
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

if [[ "$USE_FP16" == "True" ]]; then
  cmd+=(--fp16)
fi

if [[ "$USE_CPU" == "True" ]]; then
  cmd+=(--use_cpu)
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

if [[ "$supports_overwrite_output_dir" == "1" ]]; then
  cmd+=(--overwrite_output_dir)
fi

printf 'Running command:\n%s\n' "${cmd[*]}"
PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" "${cmd[@]}"
