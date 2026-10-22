#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate venv_llm

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Override from yaml if config exists
if [ -f "$REPO_ROOT/configs/default.yaml" ]; then
    YAML_ROOT=$(grep -E "^repo_root:" "$REPO_ROOT/configs/default.yaml" | awk '{print $2}' | tr -d '"' | tr -d "'")
    if [ -n "$YAML_ROOT" ]; then
        # Use python to expand ~
        REPO_ROOT=$(python3 -c "import os; print(os.path.expanduser('$YAML_ROOT'))")
    fi
fi

python "$REPO_ROOT/collate_results.py"

echo "Script execution finished."