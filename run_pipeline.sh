#!/usr/bin/env bash

# Run the full end-to-end pipeline: prepare data, train, evaluate, and prediction checks.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG_FILE="${CONFIG_FILE:-configs/hyperparameters.json}"

if [[ ! -f "$CONFIG_FILE" ]]; then
	echo "Config file not found: $CONFIG_FILE" >&2
	exit 1
fi

eval "$(python3 - "$CONFIG_FILE" <<'PY'
import json
import shlex
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as file:
	config = json.load(file)


def get(path, default):
	current = config
	for part in path.split("."):
		if isinstance(current, dict) and part in current:
			current = current[part]
		else:
			return default
	return current


values = {
	"CFG_MODEL_DIR": get("pipeline.model_dir", "output/model"),
	"CFG_MODEL_NAME": get("train.model_name", "FacebookAI/xlm-roberta-large"),
	"CFG_BATCH_SIZE": get("train.batch_size", 32),
	"CFG_EPOCHS": get("train.epochs", 3.0),
	"CFG_LEARNING_RATE": get("train.learning_rate", 5e-6),
	"CFG_DROPOUT": get("train.dropout", 0.2),
	"CFG_MAX_LENGTH": get("train.max_length", 128),
	"CFG_DATA_DIR": get("prepare_data.data_dir", "data"),
	"CFG_NUM_TEST": get("prepare_data.num_test", 250),
	"CFG_NUM_VAL": get("prepare_data.num_val", 250),
	"CFG_NEUTRAL_FRAC": get("prepare_data.neutral_frac", 0.6),
	"CFG_SAVE_PLOTS": get("prepare_data.save_plots", True),
	"CFG_SEED": get("pipeline.seed", 42),
	"CFG_RESUME_FROM_CHECKPOINT": get("pipeline.resume_from_checkpoint", False),
}


for key, value in values.items():
	if isinstance(value, bool):
		value = "1" if value else "0"
	print(f"{key}={shlex.quote(str(value))}")
PY
)"

MODEL_DIR="${MODEL_DIR:-$CFG_MODEL_DIR}"
MODEL_NAME="${MODEL_NAME:-$CFG_MODEL_NAME}"
BATCH_SIZE="${BATCH_SIZE:-$CFG_BATCH_SIZE}"
EPOCHS="${EPOCHS:-$CFG_EPOCHS}"
LEARNING_RATE="${LEARNING_RATE:-$CFG_LEARNING_RATE}"
DROPOUT="${DROPOUT:-$CFG_DROPOUT}"
SEED="${SEED:-$CFG_SEED}"
MAX_LENGTH="${MAX_LENGTH:-$CFG_MAX_LENGTH}"
DATA_DIR="${DATA_DIR:-$CFG_DATA_DIR}"
NUM_TEST="${NUM_TEST:-$CFG_NUM_TEST}"
NUM_VAL="${NUM_VAL:-$CFG_NUM_VAL}"
NEUTRAL_FRAC="${NEUTRAL_FRAC:-$CFG_NEUTRAL_FRAC}"
SAVE_PLOTS="${SAVE_PLOTS:-$CFG_SAVE_PLOTS}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-$CFG_RESUME_FROM_CHECKPOINT}"

TRAIN_ARGS=()
if [[ "${RESUME_FROM_CHECKPOINT,,}" == "1" || "${RESUME_FROM_CHECKPOINT,,}" == "true" ]]; then
	TRAIN_ARGS+=(--resume-from-checkpoint)
fi

PREPARE_ARGS=()
PREPARE_ARGS+=(--data-dir "$DATA_DIR")
PREPARE_ARGS+=(--num-test "$NUM_TEST")
PREPARE_ARGS+=(--num-val "$NUM_VAL")
PREPARE_ARGS+=(--neutral-frac "$NEUTRAL_FRAC")

if [[ "${SAVE_PLOTS,,}" == "1" || "${SAVE_PLOTS,,}" == "true" ]]; then
	PREPARE_ARGS+=(--save-plots)
fi

echo "[1/5] Preparing data..."
python3 prepare_data.py --seed "$SEED" "${PREPARE_ARGS[@]}"

echo "[2/5] Training model..."
python3 train.py \
	--model-name "$MODEL_NAME" \
	--output-dir "$MODEL_DIR" \
	--batch-size "$BATCH_SIZE" \
	--epochs "$EPOCHS" \
	--learning-rate "$LEARNING_RATE" \
	--dropout "$DROPOUT" \
	--max-length "$MAX_LENGTH" \
	--seed "$SEED" \
	"${TRAIN_ARGS[@]}"

echo "[3/5] Evaluating model on test split..."
python3 evaluate.py --model-dir "$MODEL_DIR" --max-length "$MAX_LENGTH"

echo "[4/5] Running benchmark evaluation..."
python3 evaluate.py --model-dir "$MODEL_DIR" --max-length "$MAX_LENGTH" --run-benchmark

echo "[5/5] Running prediction benchmark..."
python3 predict.py --model-dir "$MODEL_DIR" --max-length "$MAX_LENGTH" --run-benchmark

echo "Pipeline completed successfully."
echo "Config file: $CONFIG_FILE"
echo "Model directory: $MODEL_DIR"
