# Measuring One-Shot and Continual Learning in LLMs

Research code accompanying the paper *[PAPER TITLE – add citation / DOI here]*. It is part of a PhD project at the
University of Kent comparing LLMs with the ANNABELL cognitive architecture.

The harness answers one question: **can a language model learn new facts it has never seen, and how does the way the
facts are presented affect whether it can recall them?**

1. A base LLM is LoRA fine-tuned on a set of facts (the *pretraining* split) as a question-answering (QA) model.
2. It is then given **new, unseen facts** (the *held-out* split) in several different ways: one-shot in context,
   accumulated working memory, secondary fine-tuning and continued pre-training.
3. After each, we ask questions about those facts and score the answers.

Results are compared across models, experiments and cross-validation folds.

## Contents

- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Evaluation protocol](#evaluation-protocol)
- [Replicating the paper's experiments](#replicating-the-papers-experiments)
- [Outputs and collating results](#outputs-and-collating-results)
- [Supported models](#supported-models)
- [Hyperparameters](#hyperparameters)
- [Tests](#tests)
- [Further documentation](#further-documentation)
- [Citation](#citation)

## Repository layout

| File | Role |
|------|------|
| [finetune.py](finetune.py) | Main entry point. LoRA SFT with `trl.SFTTrainer`, saves the adapter and loss curve, then runs evaluation and testing. |
| [evaluate.py](evaluate.py) | Closed-book recall on the **training** split, base model vs fine-tuned model. |
| [test.py](test.py) | Continual-learning probe on the **unseen** split (modes 1, 2.1, 2, 3, 4). *This is a pipeline module, not a unit test file.* |
| [utils.py](utils.py) | Model loading, batched generation, `exact_match` and `fuzzy_match`. |
| [model_adapters.py](model_adapters.py) | Per-model-family settings (LoRA target modules, token budgets, thinking mode). |
| [paths.py](paths.py) | Repo-root resolution (optional override in `configs/default.yaml`). |
| [collate_results.py](collate_results.py) | Flattens all result summaries into one CSV row per fold. |
| [shell_scripts/](shell_scripts) | Local wrappers and SLURM job scripts. |
| `test_*.py` | Unit tests (mocked; no GPU or network needed). |
| [sample_dataset.jsonl](sample_dataset.jsonl) | Small example dataset for smoke tests. |

```
JSONL dataset ──► finetune.py ──► {output_dir}/final-adapter/
                      │                 ├─► evaluate.py  (is_pretraining=True)  → evaluation/
                      │                 └─► test.py      (is_pretraining=False) → testing/
                      └─► loss_curve.png
                                    ...many runs... └─► collate_results.py → CSV
```

## Dataset

The raw dataset files are hosted on OneDrive:

- 🔗 **[Browse & Download Dataset](https://livekentac-my.sharepoint.com/:f:/g/personal/cjw81_kent_ac_uk/IgBsJIQ1gDYXQbp7qPeaI0eOAdo1_YWzg-lvFMofGIWvlms)**

> **Note:** The link provides read-only public access and does not require a Microsoft login.

Each dataset is a JSONL file with one fact per line. The facts were derived from SQuAD passages by the upstream
ANNABELL pipeline. Required fields:

| Field | Meaning |
|-------|---------|
| `declarative_statement` | The fact as a statement, e.g. `"Rats and rabbits are wild creatures."` |
| `question` | A question about the fact, e.g. `"What are rats and rabbits?"` |
| `answer` | Ground-truth answer, e.g. `"wild creatures"` |
| `is_pretraining` | `true` → used for fine-tuning and `evaluate.py`. `false` → held out and used only by `test.py`. |

Other fields (`id`, `raw_sentence`, `*_formatted`, `created_commands`, …) are produced by the ANNABELL pipeline.
They are ignored here but tolerated. See [sample_dataset.jsonl](sample_dataset.jsonl) for an example.

Cross-validation folds are separate files, e.g. `experiments/{exp}/data_sets/fold_N/..._fold_N.jsonl`. Keep the
`fold_N` directory/file naming, because [collate_results.py](collate_results.py) uses it to recover the fold number.

## Installation

Python 3.12 is used. Pick the requirements file for your hardware:

| Hardware | File |
|----------|------|
| NVIDIA GPU (CUDA) | [requirements_nvidia.txt](requirements_nvidia.txt) |
| AMD GPU (ROCm) | [requirements_rocm.txt](requirements_rocm.txt) |
| Apple Silicon | [requirements_mac_amd.txt](requirements_mac_amd.txt) |

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements_nvidia.txt        # or the ROCm / Apple file
# ROCm only: install torch first
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2
pip install pandas                            # needed by collate_results.py, not in the requirements files
```

[requirements.txt](requirements.txt) is only a pointer to the files above.

Gated models (Gemma, Llama) need a Hugging Face token. Create a `.env` file in the repo root (do not commit it):

```
HF_TOKEN=hf_your_token_here
```

**Hardware notes**

- On GPU, models load in 4-bit NF4 quantisation (bitsandbytes) with SDPA attention. Precision is bf16 where
  supported, otherwise fp16. On CPU, models load in fp32.
- The 24–32B models used in the paper were run on a SLURM cluster with large-VRAM GPUs. Small models such as
  `Qwen/Qwen3-0.6B` are suitable for local testing.
- On MPS/CPU, `finetune.py` only trains. Run `evaluate.py` and `test.py` manually afterwards (see below).
  Automatic evaluation and testing run only when CUDA (or ROCm) is available.

## Quick start

Smoke test with the bundled sample dataset and the default model (`Qwen/Qwen3-0.6B`):

```bash
# Train, then (on GPU) automatically evaluate and test
python finetune.py --dataset_path sample_dataset.jsonl

# Or run the stages separately against an existing run directory (must contain final-adapter/)
python evaluate.py --model_path Qwen-Qwen3-0.6B/sample_dataset --dataset_path sample_dataset.jsonl \
                   --output_dir Qwen-Qwen3-0.6B/sample_dataset/evaluation --eval_mode both
python test.py     --model_path Qwen-Qwen3-0.6B/sample_dataset --dataset_path sample_dataset.jsonl \
                   --output_dir Qwen-Qwen3-0.6B/sample_dataset/testing --batch_size 16
```

Run `python finetune.py --help` (and likewise for the other scripts) for all options. Key `finetune.py` flags:
`--model_id`, `--dataset_path`, `--experiment`, `--epochs` (default 5), `--perform_validation`,
`--inference_batch_size`, `--use_thinking` / `--disable_thinking`.

### Output directory resolution

| `--experiment` | `--dataset_path` | Output directory |
|---|---|---|
| set | unset | `~/gdrive/PhD/work/annabell/experiments/{exp}/llm_comparison/finetune/{model_id}` (author's layout) |
| set | set | `./experiments/{exp}/{model_id}` (cluster layout, recommended) |
| unset | set / unset | `./{model_id with / replaced by -}` |

The dataset file stem is always appended as a final sub-directory.

## Evaluation protocol

All generation is greedy (`do_sample=False`) with left padding.

**Training prompt.** User: `Context: {declarative_statement}\n\nQuestion: {question}`. Assistant: `{answer}`,
rendered with the model's chat template.

**`evaluate.py` (training split).** Prompt is `Question: {question}` with **no context**. It tests whether the
fine-tuned facts were absorbed into the weights, for the `base` and/or `finetuned` model.

**`test.py` (unseen split).** All modes are written to one `test_results.jsonl` / `test_summary.json`.

| Mode | Summary key | What it measures | Mechanism |
|------|-------------|------------------|-----------|
| 1 | `mode_1_isolated` | One-shot, in-context learning | `Context: …\n\nQuestion: …` per sample with the fine-tuned adapter. |
| 2.1 | `mode_2_1_continuous_base` | Working-memory baseline | Same as mode 2 but on the base model (adapter disabled). |
| 2 | `mode_2_continuous` | Working memory / continuous context | One growing chat. Pass 1: `Please remember this fact: {statement}` for every sample. Pass 2: `Question: {q}` for every sample. |
| 3 | `mode_3_parametric` | Parametric memory consolidation | Continue training the existing QA adapter on raw statements, then ask `Question: {q}`. |
| 4 | `mode_4_cpt` | Continued pre-training + adapter stacking | Train a fresh LoRA on the base model on statements, merge it, stack the original QA adapter on top, then ask `Question: {q}`. |

**Metrics** (in [utils.py](utils.py)):

- `exact_match`: case-insensitive, whitespace-stripped equality.
- `fuzzy_match`: `rapidfuzz.fuzz.token_set_ratio >= 70` on lowercased strings.

Summaries store accuracies as formatted strings such as `"75.00%"`.

See [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) for output formats.

## Replicating the paper's experiments

1. **Get the data.** Download the dataset (above) and place the files under
   `experiments/{exp}/data_sets/`, with folds in `fold_N/` sub-directories. The experiment numbers used in the paper
   are given in the paper; the commands the author ran are logged in [useful_commands.txt](useful_commands.txt).
2. **Run each model on each fold.** On a SLURM cluster:

   ```bash
   sbatch ./shell_scripts/slurm_finetune_and_evaluate_llm.sh <model_id> <dataset.jsonl> <experiment> [epochs] [true|false]

   # example
   sbatch ./shell_scripts/slurm_finetune_and_evaluate_llm.sh Qwen/QwQ-32B \
     ~/LLM-one-shot-and-continious-learning-evaluation/experiments/107/data_sets/fold_2/nyc_squad_with_pretraining_commands_fold_2.jsonl 107
   ```

   Without SLURM, call `python finetune.py --model_id <id> --dataset_path <file> --experiment <exp>` directly (or
   [shell_scripts/finetune.sh](shell_scripts/finetune.sh)).
3. **Re-run stages individually if needed:**

   ```bash
   sbatch ./shell_scripts/slurm_evaluate_llm.sh <model_path> <dataset.jsonl> <output_dir> <model_id> [both|base|finetuned]
   sbatch ./shell_scripts/slurm_test_llm.sh <model_path> <dataset.jsonl> <output_dir> <model_id> [batch] [use_thinking] [disable_thinking]
   ```
4. **Collate** all runs into a CSV (see below).

The SLURM scripts were written for the University of Kent *hydra* cluster (conda env `venv_llm`). Edit the
environment activation and partition/GPU directives for your own cluster. Paths resolve relative to the script
location. To override the repo root, create `configs/default.yaml` containing `repo_root: /path/to/repo`.

Models used: Gemma 4 (e.g. `google/gemma-4-31B`), `Qwen/QwQ-32B`, `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`,
LiquidAI LFM2.5, and others. Check the paper for the full list.

> [!IMPORTANT]
> Prompts, hyperparameters, metrics and summary keys are fixed so that runs stay comparable. Changing them
> invalidates comparison with the published results.

## Outputs and collating results

Inside each run directory (`experiments/{exp}/{org}/{model}/{dataset_stem}/`):

```
final-adapter/        LoRA adapter + tokenizer
checkpoint-*/         training checkpoints
loss_curve.png
evaluation/           evaluation_summary.json, per-sample results
testing/              test_summary.json, test_results.jsonl,
                      mode3_adapter/, mode4_cpt_adapter/
```

Collate everything into one CSV, one row per fold:

```bash
python collate_results.py [base_dir]
```

It writes `llm_test_results_all_from_hydra.csv`. Experiment, model and fold are recovered from the directory
structure, so keep it intact.

> [!NOTE]
> Errors during post-training evaluation or testing are printed as warnings rather than raised. A finished
> `finetune.py` run may still lack `evaluation/` or `testing/` outputs, so check the log.

## Supported models

[model_adapters.py](model_adapters.py) selects a model family by substring match on `--model_id`: Gemma, Llama
(`meta-llama/llama-3.1-8b`), Qwen/QwQ, DeepSeek (including DeepSeek-R1-Distill-Qwen) and `LiquidAI/LFM2` models.
Unknown models fall back to the Gemma config with a warning, so check the log when trying a new model. To add a
family, subclass `ModelConfig`, register it in `get_model_config()` and add a case to
[test_model_adapters.py](test_model_adapters.py).

Thinking mode (`--use_thinking` / `--disable_thinking`) raises the generation token budget and is honoured by
DeepSeek, Qwen and Llama only. Gemma and LFM force it off.

## Hyperparameters

Fixed in code for comparability. Only `--epochs` (default 5) and `--perform_validation` are exposed.

| Setting | Value |
|---------|-------|
| LoRA | r=16, α=32, dropout 0.05 |
| Batch size | 2 × gradient accumulation 4 |
| Learning rate | 2e-4, warmup ratio 0.1 |
| Optimiser | `adamw_torch` |
| Max length | 1024 (SFT), 256 (modes 3 and 4, always 5 epochs) |
| Validation (optional) | 25% split, seed 42, eval every 10 steps |
| Quantisation | 4-bit NF4 with double quant (GPU) |

See [METHODOLOGY_REPORT.md](METHODOLOGY_REPORT.md) for the rationale.

## Tests

The unit tests are fully mocked and need no GPU, network or HF token:

```bash
python -m unittest discover -p "test_*.py"
```

## Further documentation

- [QUICK_START.md](QUICK_START.md): end-to-end walkthrough
- [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md): output formats and metric details
- [METHODOLOGY_REPORT.md](METHODOLOGY_REPORT.md): rationale for training choices
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md): history of the evaluation integration
- [AGENTS.md](AGENTS.md): detailed notes for AI coding agents

## Citation

```bibtex
@article{TODO,
  title  = {TODO},
  author = {TODO},
  year   = {TODO}
}
```

## License

*TODO: add a license file before publishing. The repo currently has none.*
