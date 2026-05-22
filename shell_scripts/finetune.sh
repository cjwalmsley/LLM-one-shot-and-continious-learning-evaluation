#!/bin/bash

# Use defaults
#./finetune.sh

# Specify a custom model
#./finetune.sh --model_id "meta-llama/Meta-Llama-3-8B"

# Specify a custom dataset
#./finetune.sh --dataset_path "custom_data.jsonl"

# Specify an experiment
#./finetune.sh --experiment "my_experiment"

# Combine multiple parameters
#./finetune.sh --model_id "meta-llama/Meta-Llama-3-8B" --dataset_path "custom_data.jsonl" --experiment "my_experiment"

~/PycharmProjects/LLM-one-shot-and-continious-learning-evaluation/.venv/bin/python ~/PycharmProjects/LLM-one-shot-and-continious-learning-evaluation/finetune.py "$@"
