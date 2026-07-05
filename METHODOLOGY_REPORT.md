# Methodology Review Report

## 1. Number of Epochs (`num_train_epochs`)
- **Current Value:** `200` in `finetune.py` and `test.py` (Modes 3 & 4).
- **Issue:** 200 epochs is exceptionally high and problematic for LLM fine-tuning, especially with a small dataset (max 500 samples). This will almost certainly cause **catastrophic overfitting**. The model will memorize the exact phrasing of the training set rather than learning underlying patterns, which will destroy its general capabilities and performance on unseen data.
- **Recommendation:** Reduce `num_train_epochs` to a standard value. For small datasets (500 samples), a good range is **3 to 10 epochs**. Given that this includes continued pre-training (Mode 4) and parametric memory (Mode 3), I recommend trying **5 epochs** as a starting point.

## 2. Checkpointing / Save Strategy
- **Current Value:** `save_steps=10000` in `finetune.py`.
- **Issue:** With a dataset of 500 samples, `per_device_train_batch_size=2`, and `gradient_accumulation_steps=4`, the effective batch size is 8. This means there are roughly 62 optimization steps per epoch. If we reduce the epochs to 5, the total training run will only be ~310 steps. The model will **never save a checkpoint** during training because it won't reach 10,000 steps.
- **Recommendation:** Change the save strategy from steps to epochs:
  - `save_strategy="epoch"` (saves at the end of each epoch).
  - Alternatively, use `save_steps=50` (or similar) if you want multiple checkpoints per epoch, but `epoch` is cleaner for small datasets.

## 3. Learning Rate
- **Current Value:** `learning_rate=2e-4`.
- **Observation:** `2e-4` is a very common and effective learning rate for LoRA fine-tuning. However, when combined with 200 epochs, it contributes heavily to the overfitting issue.
- **Recommendation:** Keep `learning_rate=2e-4` for now, but monitor training loss. If you notice instability, you may consider lowering it slightly to `1e-4`.

## 4. Warmup Steps
- **Current Value:** `warmup_steps=10`.
- **Issue/Observation:** For a very short training run (e.g., 310 total steps), 10 warmup steps is about 3% of the training time, which is acceptable but a bit short.
- **Recommendation:** Switch to `warmup_ratio=0.1` (10% of total training steps) to dynamically adjust the warmup period regardless of the final epoch count, or leave it at a fixed number if you prefer.

## 5. Summary of Proposed Code Changes
In `finetune.py`:
- `num_train_epochs = 5` (down from 200)
- `save_strategy = "epoch"` (replacing `save_strategy="steps"` and `save_steps=10000`)
- `warmup_ratio = 0.1` (replacing `warmup_steps=10`)

In `test.py` (Mode 3 & 4 SFTConfigs):
- `num_train_epochs = 5` (down from 200)
- Add `warmup_ratio = 0.1` (currently missing)
