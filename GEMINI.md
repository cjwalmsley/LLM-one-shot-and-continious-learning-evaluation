# Gemini Agent Context

This project's agent instructions live in [AGENTS.md](./AGENTS.md), the single source of truth shared by all AI
coding agents. Update that file rather than this one.

@./AGENTS.md

## Quick reference

- Purpose: LoRA fine-tune LLMs, then measure how they learn **new, unseen facts** (continual learning). The probes
  are in-context (mode 1), continuous working memory (modes 2 / 2.1), secondary fine-tuning (mode 3) and continued
  pre-training + adapter stacking (mode 4).
- Pipeline: `finetune.py` → `{output_dir}/final-adapter/` → `evaluate.py` (training split) + `test.py` (unseen split)
  → `collate_results.py`.
- Run tests: `.venv/bin/python -m unittest discover -p "test_*.py"` (fully mocked, no GPU).
- Do not change prompts, hyperparameters, metrics or summary JSON keys without being asked. They must stay comparable
  across experiments.
- `test.py` is a pipeline module, not a test file.
