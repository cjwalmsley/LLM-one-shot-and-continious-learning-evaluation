#!/bin/bash

# Arguments
MODEL_PATH=$1
DATASET_PATH=$2

# Check if required arguments are provided
if [ -z "$MODEL_PATH" ] || [ -z "$DATASET_PATH" ]; then
    echo "Usage: ./run_finetune.sh <model_path> <dataset_path>"
    echo "Example: ./run_finetune.sh google/gemma-4-E2B sample_dataset.jsonl"
    exit 1
fi

# Load environment variables from .env file
if [ -f ".env" ]; then
    export "$(grep -v '^#' .env | xargs)"
else
    echo "Warning: .env file not found."
fi

# Ensure HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN environment variable is not set."
    echo "Please add it to your .env file:"
    echo "  HF_TOKEN=your_huggingface_token"
    exit 1
fi

echo "Starting fine-tuning with:"
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"

# Run the python script with the provided parameters
python finetune_gemma4.py --model_id "$MODEL_PATH" --dataset_path "$DATASET_PATH"