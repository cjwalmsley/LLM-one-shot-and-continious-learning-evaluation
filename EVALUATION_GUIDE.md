# Model Evaluation Guide
## Overview
After fine-tuning completes, the `finetune.py` script automatically runs an evaluation of the trained model against the training dataset. The evaluation compares model-generated responses to ground truth answers using two matching strategies.
## Features
### Evaluation Metrics
The evaluation script calculates accuracy using two methods:
1. **Exact Match**: Case-insensitive exact string matching
   - Compares predictions directly to ground truth answers
   - Useful for factual answers
2. **Fuzzy Match**: Token-based similarity matching (70% threshold)
   - Uses `rapidfuzz.fuzz.token_set_ratio()` for comparison
   - Tolerates minor variations in wording
   - Useful for semantic similarity
### Generation Parameters
All responses are generated using these parameters:
- **temperature**: 1.0 (allows sampling)
- **top_p**: 0.95 (nucleus sampling)
- **top_k**: 64 (top-k sampling)
- **do_sample**: True (enables sampling)
## Automatic Evaluation Workflow
When you run `finetune.py`:
```bash
python finetune.py --experiment "89"
```
The script will:
1. Train the model
2. Save the LoRA adapter
3. **Automatically run evaluation** on the training dataset
## Output Files
Evaluation results are saved to `{output_dir}/evaluation/`:
### 1. `evaluation_results.jsonl`
Per-sample results with the following fields:
```json
{
  "sample_id": 0,
  "context": "The reactor core is safe.",
  "question": "Is the reactor core safe?",
  "ground_truth": "Yes, the reactor core is safe.",
  "predicted": "Yes, it is safe.",
  "exact_match": false,
  "fuzzy_match": true
}
```
### 2. `evaluation_summary.json`
Summary statistics:
```json
{
  "timestamp": "2024-05-22T10:30:45.123456",
  "model_path": "/path/to/model",
  "dataset_path": "/path/to/dataset.jsonl",
  "total_samples": 1488,
  "exact_match": {
    "count": 450,
    "accuracy": "30.24%"
  },
  "fuzzy_match": {
    "count": 1200,
    "accuracy": "80.65%"
  }
}
```
## Manual Evaluation
You can also run evaluation independently:
```bash
python evaluate.py \
  --model_path path/to/trained/model \
  --dataset_path path/to/dataset.jsonl \
  --output_dir evaluation_results \
  --model_id Qwen/Qwen3-0.6B
```
### Arguments:
- `--model_path`: Path to the directory containing the fine-tuned model (required)
- `--dataset_path`: Path to the JSONL dataset file (required)
- `--output_dir`: Directory to save evaluation results (default: "evaluation_results")
- `--model_id`: Base model ID used to load the adapter (default: "Qwen/Qwen3-0.6B")
## Dataset Format
The evaluation script expects a JSONL dataset with:
- `declarative_statement`: Context for the question
- `question`: Question to ask the model
- `answer`: Ground truth answer
- `is_pretraining`: Boolean flag (True = training set, False = test set)
Only samples with `is_pretraining=True` are used for evaluation.
## Example Output
```
============================================================
EVALUATION SUMMARY
============================================================
Model Path: gemma4-qa-finetune
Dataset Path: ~/gdrive/PhD/work/annabell/experiments/89/data_sets/dataset_with_generated_sentences.jsonl
Total Samples Evaluated: 1488
------------------------------------------------------------
Exact Match Accuracy: 450/1488 (30.24%)
Fuzzy Match Accuracy: 1200/1488 (80.65%)
============================================================
```
## Interpreting Results
- **Exact Match Accuracy**: Strict correctness score
  - Higher values indicate the model produces exact answers
  - Good for factual/knowledge questions
- **Fuzzy Match Accuracy**: Lenient similarity score
  - Higher values indicate semantically correct answers
  - More forgiving of wording variations
  - Better represents actual understanding
## Dependencies
The evaluation requires:
- `transformers`
- `datasets`
- `peft`
- `rapidfuzz` (for fuzzy matching)
All are included in `requirements.txt`.
