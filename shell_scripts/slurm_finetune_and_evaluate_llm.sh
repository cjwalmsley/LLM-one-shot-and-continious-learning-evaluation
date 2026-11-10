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

# Define variables for the arguments, using provided parameters or defaults
MODEL_ID="${1:-Qwen/Qwen3-0.6B}"
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

CMD=(python "$REPO_ROOT/finetune.py" --model_id "$MODEL_ID" --dataset_path "$DATASET_PATH" --epochs "$EPOCHS")

if [ -n "$EXPERIMENT" ]; then
    CMD+=(--experiment "$EXPERIMENT")
fi

if [ "$PERFORM_VALIDATION" = "true" ]; then
    CMD+=(--perform_validation)
fi

# Execute the finetune.py script
"${CMD[@]}"

echo "Script execution finished."
