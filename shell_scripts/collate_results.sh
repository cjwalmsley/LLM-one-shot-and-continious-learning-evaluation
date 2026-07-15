#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv_llm

python /home/bwc/cjw81/LLM-one-shot-and-continious-learning-evaluation/collate_results.py

echo "Script execution finished."