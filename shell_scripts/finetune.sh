#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: venv Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

# Use defaults
#./finetune.sh

# Specify a custom model
#./finetune.sh --model_id "meta-llama/Meta-Llama-3-8B"

# Specify a custom dataset
#./finetune.sh --dataset_path "custom_data.jsonl"

# Specify an experiment
#./finetune.sh --experiment "my_experiment"

# Combine multiple parameters
#./finetune.sh --model_id "meta-llama/Meta-Llama-3-8B" --dataset_path "custom_data.jsonl" --experiment "my_experiment" --epochs 10

"$PYTHON_BIN" "$ROOT_DIR/finetune.py" "$@"
