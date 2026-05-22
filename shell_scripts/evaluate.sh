#!/bin/bash

# Use defaults
#./evaluate.sh --model_path "gemma4-qa-finetune" --dataset_path "sample_dataset.jsonl"

# Specify a custom output directory
#./evaluate.sh --model_path "gemma4-qa-finetune" --dataset_path "sample_dataset.jsonl" --output_dir "custom_results"

# Specify a custom model and dataset
#./evaluate.sh --model_path "path/to/model" --dataset_path "path/to/data.jsonl"

# Combine multiple parameters
#./evaluate.sh --model_path "gemma4-qa-finetune" --dataset_path "sample_dataset.jsonl" --output_dir "my_evaluation_results"

~/PycharmProjects/LLM-one-shot-and-continious-learning-evaluation/.venv/bin/python ~/PycharmProjects/LLM-one-shot-and-continious-learning-evaluation/evaluate.py "$@"

