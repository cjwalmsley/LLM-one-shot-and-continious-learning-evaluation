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
MODEL_PATH="${1:-google/gemma-4-E2B}"
DATASET_PATH="${2:-sample_dataset.jsonl}"
OUTPUT_DIR="${3:-google-gemma-4-E2B}"
MODEL_ID="${4:-google/gemma-4-E2B}"
EVAL_MODE="${4:-both}"

echo "Running with MODEL_PATH: $MODEL_PATH"
echo "Running with DATASET_PATH: $DATASET_PATH"
echo "Running with OUTPUT_DIR: $OUTPUT_DIR"
echo "Running with MODEL_ID: $MODEL_ID"
echo "Running with EVAL_MODE: $EVAL_MODE"

python /home/bwc/cjw81/LLM-one-shot-and-continious-learning-evaluation/evalaute.py --model_path "$MODEL_PATH" --dataset_path "$DATASET_PATH" -output_dir "$OUTPUT_DIR "--model_id "$MODEL_ID" --eval_mode "$EVAL_MODE"

echo "Script execution finished."
