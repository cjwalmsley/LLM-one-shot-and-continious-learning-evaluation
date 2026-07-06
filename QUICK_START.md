# Quick Start: Post-Training Evaluation
## What Changed?
Your `finetune.py` now **automatically runs model evaluation** after training completes!
## How to Use
### Run Training + Evaluation (All-in-One)
```bash
python finetune.py --experiment "89"
```
That's it! After training finishes, evaluation will automatically start.
### Run Evaluation Separately
```bash
python evaluate.py \
  --model_path gemma4-qa-finetune \
  --dataset_path sample_dataset.jsonl \
  --output_dir evaluation_results
```
## What Gets Evaluated
- **Dataset**: Training samples (filtered by `is_pretraining=True`)
- **Metrics**: 
  - Exact Match: Case-insensitive string comparison
  - Fuzzy Match: 70% token-based similarity
## Where Are the Results?
Results are saved to `{output_dir}/evaluation/`:
```
evaluation/
├── evaluation_results.jsonl    # Per-sample results
└── evaluation_summary.json     # Summary statistics
```
## Example Output
```
============================================================
EVALUATION SUMMARY
============================================================
Model Path: gemma4-qa-finetune
Dataset Path: sample_dataset.jsonl
Total Samples Evaluated: 100
------------------------------------------------------------
Exact Match Accuracy: 75/100 (75.00%)
Fuzzy Match Accuracy: 92/100 (92.00%)
============================================================
```
## Understanding the Metrics
| Metric | Description | Use Case |
|--------|-------------|----------|
| **Exact Match** | Perfect word-for-word match (case-insensitive) | Factual, knowledge-based answers |
| **Fuzzy Match** | Token-based similarity >= 70% | Semantic correctness, wording flexibility |
## Files
| File | Purpose |
|------|---------|
| `evaluate.py` | Evaluation script (NEW) |
| `finetune.py` | Modified to auto-run evaluation |
| `requirements.txt` | Added `rapidfuzz` |
| `EVALUATION_GUIDE.md` | Detailed documentation |
## Troubleshooting
**"Warning: evaluate.py script not found"**
- Make sure `evaluate.py` is in the same directory as `finetune.py`
**"evaluate.py imports successfully" error**
- Run: `pip install -r requirements.txt`
**No results generated**
- Check that dataset has samples with `is_pretraining=True`
- Ensure model was saved to `{output_dir}/final-adapter/`
## Next Steps
1. Run training: `python finetune.py --experiment "89"`
2. Wait for evaluation to complete
3. Check results in `{output_dir}/evaluation/`
4. Analyze per-sample results in `evaluation_results.jsonl`
5. Review summary stats in `evaluation_summary.json`
---
📖 For more details, see **EVALUATION_GUIDE.md** and **IMPLEMENTATION_SUMMARY.md**
