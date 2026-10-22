#!/bin/bash

# Check if a virtual environment is active (Conda or standard Python venv)
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python"
elif [ -n "$CONDA_PREFIX" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
else
    echo "Error: No active virtual environment found." >&2
    echo "Please activate your virtual environment (e.g., conda activate venv_llm) before running this script." >&2
    exit 1
fi

# Fallback check
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Error: Python executable not found or not executable at $PYTHON_BIN" >&2
    exit 1
fi

# Define variables for the arguments, using provided parameters or defaults
MODEL_ID="${1:-google/gemma-4-E2B}"
DATASET_PATH="${2:-sample_dataset.jsonl}"
EXPERIMENT="${3}"
EPOCHS="${4:-5}"
PERFORM_VALIDATION="${5:-false}"

echo "Running with MODEL_ID: $MODEL_ID"
echo "Running with DATASET_PATH: $DATASET_PATH"
if [ -n "$EXPERIMENT" ]; then
    echo "Running with EXPERIMENT: $EXPERIMENT"
fi
echo "Running with EPOCHS: $EPOCHS"
echo "Running with PERFORM_VALIDATION: $PERFORM_VALIDATION"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Override from yaml if config exists
if [ -f "$REPO_ROOT/configs/default.yaml" ]; then
    YAML_ROOT=$(grep -E "^repo_root:" "$REPO_ROOT/configs/default.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -n "$YAML_ROOT" ]; then
        REPO_ROOT=$("$PYTHON_BIN" -c "import os; print(os.path.expanduser('$YAML_ROOT'))")
    fi
fi

CMD=("$PYTHON_BIN" "$REPO_ROOT/finetune.py" --model_id "$MODEL_ID" --dataset_path "$DATASET_PATH" --epochs "$EPOCHS")

if [ -n "$EXPERIMENT" ]; then
    CMD+=(--experiment "$EXPERIMENT")
fi

if [ "$PERFORM_VALIDATION" = "true" ]; then
    CMD+=(--perform_validation)
fi

# Execute the finetune.py script
"${CMD[@]}"

echo "Script execution finished."
