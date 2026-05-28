# AI Agent Notes

## Big picture

- Core pipeline is `finetune.py` -> LoRA adapter saved to `final-adapter/` -> automatic `evaluate_model()` (training
  split) and `test_model()` (unseen split).
- Dataset is JSONL with `declarative_statement`, `question`, `answer`, `is_pretraining`; train/eval uses
  `is_pretraining=True`, testing uses `False`.
- Model I/O flow: base model (`--model_id`, default `google/gemma-4-E2B`) + PEFT adapter (`final-adapter/`) loaded in
  `evaluate.py`/`test.py` via `PeftModel`.

## Key entry points

- `finetune.py`: LoRA fine-tune with `trl.SFTTrainer` and Gemma-style chat formatting via
  `tokenizer.apply_chat_template`.
- `evaluate.py`: deterministic generation (temperature=0, `do_sample=False`) on training split; writes
  `evaluation_results.jsonl` + `evaluation_summary.json`.
- `test.py`: three modes for unseen split: mode1 - isolated context, mode2 - continuous context, and mode3 - parametric
  memory (secondary fine-tuning).

## Project conventions & patterns

- Adapters live under `{output_dir}/final-adapter/`; evaluation writes to `{output_dir}/evaluation/`, testing writes to
  `{output_dir}/testing/`.
- Experiment mode in `finetune.py` rewires paths to `~/gdrive/PhD/work/annabell/experiments/{experiment}/...`.
- Device selection is centralized (`get_device()` in `evaluate.py`/`test.py`) and supports CUDA/ROCm, MPS, or CPU.
- Chat prompts are always wrapped as:
    - Training: `Context: ...\n\nQuestion: ...` then assistant answer in `format_prompt()`.
    - Eval/Test:
        - in mode 1, the same context/question structure passed into `generate_response()`.
        - In mode2, within a single context the declarative statements for all samples are passed first, then each
          question passed and the response is evaluated against the ground truth answer.
        - In mode3, the declarative statements for all samples are used to fine-tune a secondary model. Then in a
          separate session, each question is passed and the response is evaluated against the ground truth answer.

## Critical workflows

- Training + auto evaluation/testing:
    - `python finetune.py --experiment "89"`
    - or `python finetune.py --dataset_path sample_dataset.jsonl`
- Manual evaluation:
  -
  `python evaluate.py --model_path gemma4-qa-finetune --dataset_path sample_dataset.jsonl --output_dir evaluation_results`
- Manual testing:
    - `python test.py --model_path gemma4-qa-finetune --dataset_path sample_dataset.jsonl --output_dir test_results`

## External dependencies & integration points

- Hugging Face auth is required: `HF_TOKEN` must be set (via `.env` loaded in `finetune.py`).
- Libraries: `transformers`, `datasets`, `peft`, `trl`, `rapidfuzz` (see `requirements.txt`).
- PEFT adapter compatibility assumes Gemma-style modules (`q_proj.linear`, `k_proj.linear`, `v_proj.linear`, etc.).

## Where to look for examples

- Output format examples: `EVALUATION_GUIDE.md`.
- End-to-end flow: `QUICK_START.md`.
- Training and LoRA config: `finetune.py` (LoRA config and `SFTConfig`).
- Testing modes and memory strategy: `test.py` (modes 1–3).
