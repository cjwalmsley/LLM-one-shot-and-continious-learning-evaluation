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
MODEL_ID="${1:-google/gemma-4-E2B}"
DATASET_PATH="${2:-sample_dataset.jsonl}"
EXPERIMENT="${3}"
EPOCHS="${4:-5}"

echo "Running with MODEL_ID: $MODEL_ID"
echo "Running with DATASET_PATH: $DATASET_PATH"
if [ -n "$EXPERIMENT" ]; then
    echo "Running with EXPERIMENT: $EXPERIMENT"
fi
echo "Running with EPOCHS: $EPOCHS"

# Execute the finetune.py script
if [ -n "$EXPERIMENT" ]; then
    python /home/bwc/cjw81/LLM-one-shot-and-continious-learning-evaluation/finetune.py --model_id "$MODEL_ID" --dataset_path "$DATASET_PATH" --experiment "$EXPERIMENT" --epochs "$EPOCHS"
else
    python /home/bwc/cjw81/LLM-one-shot-and-continious-learning-evaluation/finetune.py --model_id "$MODEL_ID" --dataset_path "$DATASET_PATH" --epochs "$EPOCHS"
fi

echo "Script execution finished."
