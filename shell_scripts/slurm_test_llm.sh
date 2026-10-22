#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cjw81@kent.ac.uk
#SBATCH --export=ALL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:ampere:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv_llm

# Example usage:
# sbatch ./shell_scripts/slurm_test_llm.sh "path/to/model" "path/to/dataset.jsonl" "path/to/output_dir" "model_id_string" "16" "true" "false"

# Define variables for the arguments, using provided parameters or defaults
MODEL_PATH="${1:-google/gemma-4-E2B}"
DATASET_PATH="${2:-sample_dataset.jsonl}"
OUTPUT_DIR="${3:-google-gemma-4-E2B}"
MODEL_ID="${4:-google/gemma-4-E2B}"
BATCH_SIZE="${5:-16}"
USE_THINKING="${6:-}"
DISABLE_THINKING="${7:-}"

echo "Running with MODEL_PATH: $MODEL_PATH"
echo "Running with DATASET_PATH: $DATASET_PATH"
echo "Running with OUTPUT_DIR: $OUTPUT_DIR"
echo "Running with MODEL_ID: $MODEL_ID"
echo "Running with BATCH_SIZE: $BATCH_SIZE"
echo "Running with USE_THINKING: $USE_THINKING"
echo "Running with DISABLE_THINKING: $DISABLE_THINKING"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Override from yaml if config exists
if [ -f "$REPO_ROOT/configs/default.yaml" ]; then
    YAML_ROOT=$(grep -E "^repo_root:" "$REPO_ROOT/configs/default.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -n "$YAML_ROOT" ]; then
        REPO_ROOT=$(python3 -c "import os; print(os.path.expanduser('$YAML_ROOT'))")
    fi
fi

# Construct the command
CMD="python \"$REPO_ROOT/test.py\" --model_path \"$MODEL_PATH\" --dataset_path \"$DATASET_PATH\" --output_dir \"$OUTPUT_DIR\" --model_id \"$MODEL_ID\" --batch_size \"$BATCH_SIZE\""

if [ "$USE_THINKING" == "true" ] || [ "$USE_THINKING" == "1" ]; then
    CMD="$CMD --use_thinking"
fi

if [ "$DISABLE_THINKING" == "true" ] || [ "$DISABLE_THINKING" == "1" ]; then
    CMD="$CMD --disable_thinking"
fi

echo "Executing command: $CMD"
eval $CMD

echo "Script execution finished."
