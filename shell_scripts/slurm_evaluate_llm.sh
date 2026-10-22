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
# sbatch ./shell_scripts/slurm_evaluate_llm.sh "path/to/model" "path/to/dataset.jsonl" "path/to/output_dir" "model_id_string" "both"

# Define variables for the arguments, using provided parameters or defaults
MODEL_PATH="${1:-google/gemma-4-E2B}"
DATASET_PATH="${2:-sample_dataset.jsonl}"
OUTPUT_DIR="${3:-google-gemma-4-E2B}"
MODEL_ID="${4:-google/gemma-4-E2B}"
EVAL_MODE="${5:-both}"

echo "Running with MODEL_PATH: $MODEL_PATH"
echo "Running with DATASET_PATH: $DATASET_PATH"
echo "Running with OUTPUT_DIR: $OUTPUT_DIR"
echo "Running with MODEL_ID: $MODEL_ID"
echo "Running with EVAL_MODE: $EVAL_MODE"

# When running via SLURM, the script runs from a temporary spool dir, so BASH_SOURCE[0] is unreliable.
# Use SLURM_SUBMIT_DIR which points to where `sbatch` was called (expected to be the repo root).
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_ROOT="$(dirname "$SCRIPT_DIR")"
fi

# Override from yaml if config exists
if [ -f "$REPO_ROOT/configs/default.yaml" ]; then
    YAML_ROOT=$(grep -E "^repo_root:" "$REPO_ROOT/configs/default.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -n "$YAML_ROOT" ]; then
        REPO_ROOT=$(python3 -c "import os; print(os.path.expanduser('$YAML_ROOT'))")
    fi
fi

python "$REPO_ROOT/evaluate.py" --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" --output_dir "$OUTPUT_DIR" --model_id "$MODEL_ID" --eval_mode "$EVAL_MODE"

echo "Script execution finished."
