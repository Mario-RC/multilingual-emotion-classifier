#!/usr/bin/env bash

# Run the full end-to-end pipeline: prepare data, train, evaluate, and quick benchmark.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODEL_DIR="${MODEL_DIR:-output/model}"
MODEL_NAME="${MODEL_NAME:-FacebookAI/xlm-roberta-large}"

echo "[1/4] Preparing data..."
python3 prepare_data.py "$@"

echo "[2/4] Training model..."
python3 train.py --model-name "$MODEL_NAME" --output-dir "$MODEL_DIR"

echo "[3/4] Evaluating model..."
python3 evaluate.py --model-dir "$MODEL_DIR"

echo "[4/4] Running quick benchmark..."
python3 evaluate.py --model-dir "$MODEL_DIR" --run-benchmark

echo "Pipeline completed successfully."
