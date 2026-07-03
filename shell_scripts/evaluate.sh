#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: venv Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

# Use defaults
#./evaluate.sh --model_path "gemma4-qa-finetune" --dataset_path "sample_dataset.jsonl"

# Specify a custom output directory
#./evaluate.sh --model_path "gemma4-qa-finetune" --dataset_path "sample_dataset.jsonl" --output_dir "custom_results"

# Specify a custom model and dataset
#./evaluate.sh --model_path "path/to/model" --dataset_path "path/to/data.jsonl"

# Combine multiple parameters
#./evaluate.sh --model_path "gemma4-qa-finetune" --dataset_path "sample_dataset.jsonl" --output_dir "my_evaluation_results"

#./shell_scripts/evaluate.sh --model_path "google-gemma-4-E2B" --dataset_path "sample_dataset.jsonl" --output_dir "google-gemma-4-E2B" --model_id "google/gemma-4-E2B" --eval_mode "base"


"$PYTHON_BIN" "$ROOT_DIR/evaluate.py" "$@"

