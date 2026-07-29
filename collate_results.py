import os
import json
import re
import pandas as pd
from pathlib import Path

DEFAULT_BASE_DIR = "/home/bwc/cjw81/LLM-one-shot-and-continious-learning-evaluation/experiments"

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def parse_experiment_info(fold_path, base_dir):
    try:
        rel_path = os.path.relpath(fold_path, base_dir)
        parts = rel_path.split(os.sep)

        if len(parts) < 3:
            return None, None, None

        experiment_number = parts[0]
        fold_dir_name = parts[-1]
        model = "/".join(parts[1:-1])

        # Extract fold number
        fold_match = re.search(r'fold_(\d+)', fold_dir_name)
        fold_number = int(fold_match.group(1)) if fold_match else None

        return experiment_number, model, fold_number
    except Exception:
        return None, None, None

def collate_results(base_dir=DEFAULT_BASE_DIR):
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Base directory {base_dir} does not exist.")
        return

    folds_data = {}

    # Traverse directories to find json files
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file in ['evaluation_summary.json', 'test_summary.json']:
                parent_dir_name = os.path.basename(root)
                if file == 'evaluation_summary.json' and parent_dir_name != 'evaluation':
                    continue
                if file == 'test_summary.json' and parent_dir_name != 'testing':
                    continue

                file_path = os.path.join(root, file)
                # The fold directory is typically the parent or grandparent
                # Let's find the fold directory by going up until we find a directory with 'fold_' in it
                # Or just assume the fold directory is the parent of 'evaluation' / 'testing'
                # Given paths like <fold_dir>/evaluation/evaluation_summary.json
                # The fold directory is os.path.dirname(root)
                # However, let's just trace up to the directory under the model name
                # Actually, finding the directory containing 'fold_' is robust

                curr_dir = root
                fold_dir = None
                while curr_dir != base_dir and curr_dir != '/':
                    if 'fold_' in os.path.basename(curr_dir):
                        fold_dir = curr_dir
                        break
                    curr_dir = os.path.dirname(curr_dir)

                if fold_dir is None:
                    # Fallback to parent of current root if 'fold_' is not in name but we want to capture it
                    fold_dir = os.path.dirname(root)

                if fold_dir not in folds_data:
                    folds_data[fold_dir] = {}

                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        continue

                    flattened = flatten_dict(data)
                    file_type = 'eval' if file == 'evaluation_summary.json' else 'test'

                    # Store with temporary prefix to handle overlaps later
                    folds_data[fold_dir][file_type] = flattened

    rows = []
    for fold_dir, data in folds_data.items():
        eval_data = data.get('eval', {})
        test_data = data.get('test', {})

        # Handle overlaps
        eval_keys = set(eval_data.keys())
        test_keys = set(test_data.keys())
        overlapping_keys = eval_keys.intersection(test_keys)

        merged_data = {}
        for k, v in eval_data.items():
            if k in overlapping_keys:
                merged_data[f"eval_{k}"] = v
            else:
                merged_data[k] = v

        for k, v in test_data.items():
            if k in overlapping_keys:
                merged_data[f"test_{k}"] = v
            else:
                merged_data[k] = v

        experiment_number, model, fold_number = parse_experiment_info(fold_dir, base_dir)

        # Construct the final row
        row = {
            "path to results": fold_dir,
            "experiment number": experiment_number,
            "model": model,
            "fold number": fold_number
        }
        row.update(merged_data)
        rows.append(row)

    df = pd.DataFrame(rows)

    output_path = os.path.join(base_dir, "llm_test_results_all_from_hydra.csv")

    # Ensure the base directory exists before saving (might be relevant if base_dir is created just now, but we checked it above)
    if not df.empty:
        df.to_csv(output_path, index=False)
        print(f"Results collated and saved to {output_path}")
    else:
        print("No data found to collate.")

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_DIR
    collate_results(base)
