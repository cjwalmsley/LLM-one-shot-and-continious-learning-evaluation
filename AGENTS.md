# AI Agent Notes

## Project purpose

Research codebase (PhD work, ANNABELL comparison) for **demonstrating and measuring continual learning in LLMs**.
A base LLM is LoRA fine-tuned on a set of facts (the *pretraining* split). It is then shown **new, unseen facts**
in several different ways (one-shot in context, accumulated working memory, secondary fine-tuning, continued
pre-training). After each, we measure whether it can answer questions about those facts. Results are compared across
models, experiments and cross-validation folds.

Treat this as an **experimental research harness**. Reproducibility and comparability between runs matter more than
elegance. Do not change prompts, hyperparameters, metrics or output schemas unless you are asked to. Any of these
changes invalidates comparison with earlier experiments.

## Big picture / data flow

```
JSONL dataset ──► finetune.py ──► {output_dir}/final-adapter/   (LoRA adapter + tokenizer)
                      │                 │
                      │                 ├─► evaluate.evaluate_model()  (is_pretraining=True split)  → {output_dir}/evaluation/
                      │                 └─► test.test_model()          (is_pretraining=False split) → {output_dir}/testing/
                      └─► {output_dir}/loss_curve.png
                                         ...many runs...
                                         └─► collate_results.py → llm_test_results_all.csv
```

- **Dataset**: JSONL, one record per fact. Required fields: `declarative_statement`, `question`, `answer`, and
  `is_pretraining` (bool). Other fields (`id`, `raw_sentence`, `*_formatted`, `created_commands`, …) come from the
  upstream ANNABELL pipeline. They are ignored here but must be tolerated.
    - `is_pretraining=True` → used for training and for `evaluate.py`.
    - `is_pretraining=False` → held out and used only by `test.py` (the continual-learning probe).
- **Base model + adapter**: base weights come from HF Hub (`--model_id`, default `Qwen/Qwen3-0.6B`). The PEFT
  adapter is loaded from `{model_path}/final-adapter` via `utils.load_tokenizer_and_model()`.

## Module map

| File                  | Role                                                                                                                                                                                                                           |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `finetune.py`         | CLI entry point. Resolves paths, loads 4-bit model, LoRA SFT with `trl.SFTTrainer`, saves adapter, plots loss, then calls `evaluate_model()` and `test_model()`.                                                               |
| `evaluate.py`         | `evaluate_model()` / CLI. Closed-book recall on the **training** split. Compares base model (`model.disable_adapter()`) vs fine-tuned.                                                                                         |
| `test.py`             | `test_model()` / CLI. Continual-learning probe on the **unseen** split. Runs modes 1, 2.1, 2, 3 and 4.                                                                                                                         |
| `utils.py`            | Shared: `get_device()`, `load_tokenizer_and_model()`, `generate_batch_responses()`, `generate_single_response()`, `exact_match()`, `fuzzy_match()`. Also a `WeightConverter` monkey-patch for PEFT/Transformers compatibility. |
| `model_adapters.py`   | Per-model-family strategy classes (`ModelConfig` ABC + Llama/Qwen/DeepSeek/LFM). The `get_model_config(model_id)` factory selects one by substring match.                                                                      |
| `collate_results.py`  | Walks `experiments/` and flattens every `evaluation/evaluation_summary.json` and `testing/test_summary.json` into one CSV row per fold.                                                                                        |
| `shell_scripts/`      | Local wrappers (`finetune.sh`, `evaluate.sh` use `./.venv/bin/python`). SLURM jobs for a cluster (`slurm_*.sh`, conda env `venv_llm`). Paths resolve relatively or via `configs/default.yaml` override.                        |
| `test_*.py`           | Unit tests (`unittest` + `unittest.mock`).                                                                                                                                                                                     |
| `useful_commands.txt` | Personal log of real `sbatch`/`rsync` invocations per experiment. Use it for examples, but do not edit it unless asked.                                                                                                        |

## Evaluation protocol (keep these semantics exact)

All generation is greedy (`do_sample=False`) with left padding. `max_new_tokens` comes from
`ModelConfig.get_max_new_tokens()` unless stated otherwise.

**Training prompt** (`ModelConfig.format_prompt`): user = `Context: {declarative_statement}\n\nQuestion: {question}`,
assistant = `{answer}`, rendered with `tokenizer.apply_chat_template`.

**`evaluate.py`** (training split): user = `Question: {question}` only, with **no context**. This tests whether the
fine-tuned facts were absorbed into the weights. It runs for `base` and/or `finetuned`
(`--eval_mode both|base|finetuned`).

**`test.py`** (unseen split). All modes are written to one `test_results.jsonl` / `test_summary.json`:

| Mode | Summary key                | What it measures                          | Mechanism                                                                                                                                                                                                          |
|------|----------------------------|-------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | `mode_1_isolated`          | One-shot, in-context                      | Batched. `Context: …\n\nQuestion: …` per sample, fine-tuned adapter.                                                                                                                                               |
| 2.1  | `mode_2_1_continuous_base` | Working-memory baseline                   | Same as mode 2, but inside `model.disable_adapter()` (base model).                                                                                                                                                 |
| 2    | `mode_2_continuous`        | Working memory / continuous context       | Sequential, one growing chat. Pass 1: `Please remember this fact: {statement}` for every sample (model replies, 50 tok). Pass 2: `Question: {q}` for every sample (150 tok), with answers appended to the context. |
| 3    | `mode_3_parametric`        | Parametric memory consolidation           | Keeps training the **existing QA adapter** (`is_trainable=True`) on raw `declarative_statement` text (`format_mode3_train`). Saves to `testing/mode3_adapter/`, reloads, then asks `Question: {q}` batched.        |
| 4    | `mode_4_cpt`               | Continued pre-training + adapter stacking | Fresh LoRA on the base model trained on statements (`format_cpt_train`), `merge_and_unload()`, then the original QA adapter is stacked on top. Asks `Question: {q}` batched.                                       |

**Metrics** (`utils.py`): `exact_match` is a case-insensitive, whitespace-stripped equality check. `fuzzy_match` is
`rapidfuzz.fuzz.token_set_ratio >= 70` on lowercased strings. Summaries store accuracies as **formatted strings**
(e.g. `"75.00%"`). `collate_results.py` and downstream notebooks depend on these key names and formats.

Per-sample result record: `sample_id, mode, question, ground_truth, predicted, exact_match, fuzzy_match`.
`evaluate.py` also adds `context`.

## Path conventions

`finetune.py` resolves `dataset_path` / `output_dir` like this (then always appends `/{Path(dataset_path).stem}`):

| `--experiment` | `--dataset_path` | dataset                            | output_dir                                             |
|----------------|------------------|------------------------------------|--------------------------------------------------------|
| set            | unset            | `{dataset_dir}{exp}{dataset_name}` | `{output_dir}{exp}/llm_comparison/finetune/{model_id}` |
| set            | set              | given                              | `{output_dir}{exp}/{model_id}`                         |
| unset          | set              | given                              | `{output_dir}/{model_id with / → -}`                   |
| unset          | unset            | `sample_dataset.jsonl`             | `{output_dir}/{model_id with / → -}`                   |

Inside `output_dir`: `final-adapter/`, `checkpoint-*/`, `.cache/` (HF datasets cache), `loss_curve.png`,
`evaluation/`, `testing/` (+ `testing/mode3_adapter/`, `testing/mode4_cpt_adapter/`).

Cross-validation folds are separate dataset files (`.../data_sets/fold_N/..._fold_N.jsonl`). The stem-based output
dir, which contains `fold_N`, is what `collate_results.parse_experiment_info()` uses to recover
experiment / model / fold. Keep `experiments/{exp}/{org}/{model}/{..._fold_N}/{evaluation|testing}/` intact.

## Model support (`model_adapters.py`)

- Add a new model family by subclassing `ModelConfig`. Implement `get_lora_target_modules()` and override
  `get_max_new_tokens()` / `setup_tokenizer()` if needed. Then register it in `get_model_config()` and add a case
  to `test_model_adapters.py`.
- Most models target sub-modules `q_proj` … `down_proj`. LFM uses
  `in_proj/out_proj/w1-3`.
- Factory matching is **order-sensitive substring matching**: `gemma` → Llama (only `meta-llama/llama-3.1-8b`) →
  `qwen`/`qwq` (DeepSeek-distill-Qwen → DeepSeek) → `deepseek` → `liquidai/lfm2-24b-a2b`. Anything else falls back
  to Gemma with a warning, so check the warning when you try a new model.
- Thinking mode: `--use_thinking` / `--disable_thinking` give a tri-state `use_thinking` (`True`/`False`/`None`).
  Only DeepSeek/Qwen/Llama honour it, by raising `max_new_tokens`. Gemma and LFM force it off.

## Hardware & runtime conventions

- Device detection is centralised in `utils.get_device()`: CUDA (NVIDIA **or AMD ROCm**, both report `cuda`) → MPS →
  CPU.
- On GPU, models load in 4-bit NF4 (`BitsAndBytesConfig`, double quant) with `attn_implementation="sdpa"`. Do not
  switch to Flash-Attention 2, because head-dim limits break some models. Precision is bf16 if supported, else fp16.
  On CPU, models load in fp32 with no quantisation.
- Optimiser is `adamw_torch` (not a bnb optimiser) for ROCm compatibility.
- VRAM hygiene is deliberate. Between phases the code does `del model/trainer` → `release_memory()` →
  `gc.collect()` → `torch.cuda.empty_cache()` + `ipc_collect()`. Keep this pattern whenever you add a phase that
  loads another model.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set at the top of entry modules.
- Training hyperparameters are **hard-coded** (LoRA r=16, α=32, dropout 0.05; batch 2 × grad-accum 4; lr 2e-4;
  warmup_ratio 0.1; max_length 1024 for SFT and 256 for modes 3/4). Only `--epochs` (default 5) and
  `--perform_validation` (25 % split, seed 42, eval every 10 steps) are exposed. Modes 3/4 always use 5 epochs.
- Default inference batch size after training is chosen from `max_new_tokens` (>16k → 2, >4096 → 4, else 16) unless
  `--inference_batch_size` is given.

## Build / environment

- Python 3.12, local venv at `./.venv` (the cluster uses conda env `venv_llm`). There is no `pyproject.toml`, no
  package build and no linter config. Scripts are run directly from the repo root.
- Install: `pip install -r requirements.txt`. Install PyTorch separately for the target backend, e.g. ROCm:
  `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2`. `flash-attn` and
  `bitsandbytes` may not install on macOS, so skip them locally if needed.
- `collate_results.py` needs `pandas`, which is **not** listed in `requirements.txt`.
- `HF_TOKEN` must be in `.env` (loaded by `finetune.py` via `python-dotenv`) or in the environment. Gated models (Gemma,
  Llama) need it.

## Critical workflows

```bash
# Unit tests (fast, fully mocked, no GPU / network / HF token needed). 38 tests.
.venv/bin/python -m unittest discover -p "test_*.py"
.venv/bin/python -m unittest test_test.TestRunMode2          # single class

# Train + auto evaluate + test
python finetune.py --dataset_path sample_dataset.jsonl
python finetune.py --model_id Qwen/QwQ-32B --dataset_path experiments/107/data_sets/fold_2/..._fold_2.jsonl \
                   --experiment 107 --epochs 5 --perform_validation

# Standalone evaluation / testing against an existing run dir (must contain final-adapter/)
python evaluate.py --model_path Qwen-Qwen3-0.6B/sample_dataset --dataset_path sample_dataset.jsonl \
                   --output_dir Qwen-Qwen3-0.6B/sample_dataset/evaluation --eval_mode both
python test.py     --model_path Qwen-Qwen3-0.6B/sample_dataset --dataset_path sample_dataset.jsonl \
                   --output_dir Qwen-Qwen3-0.6B/sample_dataset/testing --batch_size 16

# Collate all experiment summaries to CSV
python collate_results.py [base_dir]

# Cluster (SLURM)
sbatch ./shell_scripts/slurm_finetune_and_evaluate_llm.sh <model_id> <dataset.jsonl> <experiment> [epochs] [true|false]
sbatch ./shell_scripts/slurm_evaluate_llm.sh <model_path> <dataset.jsonl> <output_dir> <model_id> [both|base|finetuned]
sbatch ./shell_scripts/slurm_test_llm.sh <model_path> <dataset.jsonl> <output_dir> <model_id> [batch] [use_thinking] [disable_thinking]
```

## Testing guidelines

- Framework: stdlib `unittest` (pytest is not installed). Tests live flat in the repo root as `test_<module>.py`.
- Tests **must not** download models or need a GPU. Mock `AutoTokenizer`, `AutoModelForCausalLM`, `PeftModel`,
  `SFTTrainer`, `SFTConfig`, `load_dataset`, `torch`, and generation helpers. Patch them **where they are looked up**
  (e.g. `@patch("test.SFTTrainer")`, `@patch("evaluate.generate_batch_responses")`, `@patch("finetune.torch")`).
- Mode functions accept plain lists of dicts as datasets in tests. Keep them compatible with both HF `Dataset` and lists
  (iteration, and `dataset[i:i+n]` returning a dict of columns for batched modes).
- When you change prompts, `max_new_tokens`, mode order or summary keys, update the matching assertions in
  `test_test.py` / `test_evaluate.py` / `test_finetune.py` / `test_collate_results.py`.
- Run the full suite before you finish any change.

## Gotchas

- **`test.py` is the pipeline module, not a test file.** It shadows the stdlib `test` package, so it only resolves
  because the repo root is first on `sys.path`. Tests import it as `import test as test_module`. Do not rename it
  casually: `finetune.py`, the SLURM scripts and the tests all depend on the name.
- In `finetune.py`, **evaluation and testing only run when `torch.cuda.is_available()`**. They are nested under that
  `if`. On MPS/CPU, training finishes and you must then run `evaluate.py` / `test.py` manually.
- Errors in post-training evaluation/testing are caught and printed as warnings, so a "successful" `finetune.py` run
  can still be missing `evaluation/` or `testing/` outputs.
- In `finetune.py`, `PYTORCH_CUDA_ALLOC_CONF` is set *after* `import torch` (it is correct in `evaluate.py` and
  `test.py`).
- `collate_results.DEFAULT_BASE_DIR` and all `slurm_*.sh` scripts resolve paths dynamically relative to the script
  location, but can be overridden by providing `repo_root:` in `configs/default.yaml`. Configure experiment-based
  `finetune.py` dataset and output paths with `dataset_dir`, `dataset_name`, and `output_dir` in that gitignored file.
- `.env` holds `HF_TOKEN` and is **not** in `.gitignore`. Never commit it. Run artefacts (`fold_*/`, `experiments/`,
  model output dirs, `test_results.jsonl`) are also untracked and should stay out of git.

## Development guidelines

- Keep functions importable and side-effect free at import time (aside from env setup). `finetune.py` imports
  `evaluate_model` / `test_model` directly, and every module exposes a `main()` behind `if __name__ == "__main__"`.
- Put shared logic in `utils.py` and model-specific behaviour in `model_adapters.py`. Do not hard-code
  model-name checks in the pipeline scripts.
- New CLI flags: add them to the argparse in the entry script, thread them through the function signature (with a
  default that keeps current behaviour), and update the corresponding `shell_scripts/*.sh` wrapper (s).
- Use `os.path.expanduser` on any user-supplied path, because experiment paths use `~`.
- Print progress with `flush=True` in long loops (SLURM logs buffer otherwise).
- Preserve existing comments/docstrings. Small, focused PR-style commits (history uses `feat:` / `fix:` prefixes).

## Further reading

- `EVALUATION_GUIDE.md`: output formats and metric details.
- `QUICK_START.md`: end-to-end walkthrough.
