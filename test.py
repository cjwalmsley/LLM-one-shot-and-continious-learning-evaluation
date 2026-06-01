import argparse
import json
import os
import gc
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import SFTTrainer, SFTConfig
from rapidfuzz import fuzz
from accelerate.utils import release_memory


def get_device():
    """Detect and return the appropriate device (AMD ROCm compatible)."""
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
            if "AMD" in device_name or "ROCM" in device_name.upper():
                print(f"✓ Detected AMD ROCm device: {device_name}")
            else:
                print(f"✓ Detected NVIDIA CUDA device: {device_name}")
            return "cuda"
        except:
            return "cuda"
    elif torch.backends.mps.is_available():
        print("✓ Detected Apple MPS device")
        return "mps"
    else:
        print("✓ No GPU detected, using CPU")
        return "cpu"


def _load_tokenizer_and_model(adapter_path, model_id, device, is_trainable=False):
    """Loads the base model and attaches the specified adapter."""
    print(f"Loading tokenizer and model from {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # CRITICAL FOR BATCHING: Set pad token and force left-padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=("cpu" if device == "cpu" else {"": device}),  # Strict GPU mapping
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    model = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=is_trainable
    )

    if not is_trainable:
        model.eval()
    else:
        # Explicitly ensure only the LoRA parameters require gradients
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    return tokenizer, model


def generate_batch_responses(
    model, tokenizer, messages_batch, device, max_new_tokens=50
):
    """Generate responses for a batch of messages simultaneously."""
    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    if device != "cpu":
        inputs = {key: value.to(device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[:, input_length:]
    responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return [response.strip() for response in responses]


def generate_single_response(model, tokenizer, messages, device, max_new_tokens=50):
    """Generate a response sequentially (used exclusively for Mode 2 statefulness)."""
    return generate_batch_responses(
        model, tokenizer, [messages], device, max_new_tokens
    )[0]


def exact_match(predicted, ground_truth):
    return predicted.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(predicted, ground_truth, threshold=80):
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold


def run_mode_1(model, tokenizer, test_dataset, device, batch_size):
    """Mode 1: Prompt model with Context and Question simultaneously (Batched)."""
    print("\n--- Running Mode 1: Context + Question (Isolated Memory - Batched) ---")
    results = []
    exact_count, fuzzy_count = 0, 0
    total = len(test_dataset)

    for i in range(0, total, batch_size):
        batch = test_dataset[i : i + batch_size]
        contexts = batch.get("declarative_statement", [])
        questions = batch.get("question", [])
        ground_truths = batch.get("answer", [])
        sample_ids = range(i, i + len(contexts))

        messages_batch = [
            [{"role": "user", "content": f"Context: {c}\n\nQuestion: {q}"}]
            for c, q in zip(contexts, questions)
        ]

        predictions = generate_batch_responses(model, tokenizer, messages_batch, device)

        for idx, truth, question, pred in zip(
            sample_ids, ground_truths, questions, predictions
        ):
            exact = exact_match(pred, truth)
            fuzzy = fuzzy_match(pred, truth)

            if exact:
                exact_count += 1
            if fuzzy:
                fuzzy_count += 1

            results.append(
                {
                    "sample_id": idx,
                    "mode": 1,
                    "question": question,
                    "ground_truth": truth,
                    "predicted": pred,
                    "exact_match": exact,
                    "fuzzy_match": fuzzy,
                }
            )

        print(f"  Processed {min(i + batch_size, total)}/{total} samples")

    return results, exact_count, fuzzy_count


def run_mode_2(model, tokenizer, test_dataset, device):
    """Mode 2: Two-pass continuous context (Working Memory - Sequential)."""
    print("\n--- Running Mode 2: Continuous Context (Working Memory) ---")
    print("  (Running sequentially to maintain unbroken context state without OOM)")
    results = []
    exact_count, fuzzy_count = 0, 0
    total = len(test_dataset)
    messages = []

    print("  Pass 1: Loading declarative statements into context...")
    for idx, sample in enumerate(test_dataset):
        statement = sample.get("declarative_statement", "")
        messages.append(
            {"role": "user", "content": f"Please remember this fact: {statement}"}
        )
        reply = generate_single_response(
            model, tokenizer, messages, device, max_new_tokens=50
        )
        messages.append({"role": "assistant", "content": reply})

    print("  Pass 2: Prompting questions from continuous context...")
    for idx, sample in enumerate(test_dataset):
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")

        messages.append({"role": "user", "content": f"Question: {question}"})
        predicted = generate_single_response(model, tokenizer, messages, device)
        messages.append({"role": "assistant", "content": predicted})

        exact = exact_match(predicted, ground_truth)
        fuzzy = fuzzy_match(predicted, ground_truth)

        if exact:
            exact_count += 1
        if fuzzy:
            fuzzy_count += 1

        results.append(
            {
                "sample_id": idx,
                "mode": 2,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "exact_match": exact,
                "fuzzy_match": fuzzy,
            }
        )
        if (idx + 1) % 10 == 0:
            print(f"    Answered {idx + 1}/{total} questions")

    return results, exact_count, fuzzy_count


def run_mode_3(
    original_adapter_path,
    dataset_path,
    output_dir,
    model_id,
    device,
    test_dataset,
    batch_size,
):
    """Mode 3: Parametric Memory Consolidation (Secondary Fine-Tuning - Batched Inference)."""
    print("\n--- Running Mode 3: Parametric Memory (Weights Fine-Tuning) ---")

    def format_mode3_train(example):
        return {"text": example["declarative_statement"]}

    mode3_train_data = test_dataset.map(format_mode3_train)
    mode3_adapter_dir = os.path.join(output_dir, "mode3_adapter")

    print("  Pass 1: Fine-tuning weights on declarative statements...")
    tokenizer, trainable_model = _load_tokenizer_and_model(
        original_adapter_path, model_id, device, is_trainable=True
    )

    training_args = SFTConfig(
        output_dir=mode3_adapter_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=256,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=trainable_model,
        train_dataset=mode3_train_data,
        peft_config=None,
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(mode3_adapter_dir)
    tokenizer.save_pretrained(mode3_adapter_dir)

    print("  [System] Tearing down graph and clearing VRAM before batched inference...")
    del trainer
    del trainable_model
    release_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(
        "  Pass 2: Evaluating questions using updated parametric weights (Batched)..."
    )
    tokenizer, eval_model = _load_tokenizer_and_model(
        mode3_adapter_dir, model_id, device, is_trainable=False
    )

    results = []
    exact_count, fuzzy_count = 0, 0
    total = len(test_dataset)

    for i in range(0, total, batch_size):
        batch = test_dataset[i : i + batch_size]
        questions = batch.get("question", [])
        ground_truths = batch.get("answer", [])
        sample_ids = range(i, i + len(questions))

        messages_batch = [
            [{"role": "user", "content": f"Question: {q}"}] for q in questions
        ]

        predictions = generate_batch_responses(
            eval_model, tokenizer, messages_batch, device
        )

        for idx, question, truth, pred in zip(
            sample_ids, questions, ground_truths, predictions
        ):
            exact = exact_match(pred, truth)
            fuzzy = fuzzy_match(pred, truth)

            if exact:
                exact_count += 1
            if fuzzy:
                fuzzy_count += 1

            results.append(
                {
                    "sample_id": idx,
                    "mode": 3,
                    "question": question,
                    "ground_truth": truth,
                    "predicted": pred,
                    "exact_match": exact,
                    "fuzzy_match": fuzzy,
                }
            )

        print(f"    Answered {min(i + batch_size, total)}/{total} questions")

    print("  [System] Cleaning up post-evaluation VRAM...")
    del eval_model
    release_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return results, exact_count, fuzzy_count


def _detect_lora_target_modules(model, preferred=None):
    if preferred is None:
        preferred = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    supported = (
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
    )
    supported_module_names = {
        name
        for name, module in model.named_modules()
        if isinstance(module, supported)
    }

    matches = []
    for pref in preferred:
        linear_suffix = f"{pref}.linear"
        if any(name.endswith(linear_suffix) for name in supported_module_names):
            matches.append(linear_suffix)
            continue

        plain_suffix = f".{pref}"
        if any(
            name == pref or name.endswith(plain_suffix)
            for name in supported_module_names
        ):
            matches.append(pref)

    return list(dict.fromkeys(matches))


def run_mode_4(
    original_adapter_path,
    dataset_path,
    output_dir,
    model_id,
    device,
    test_dataset,
    batch_size,
):
    """Mode 4: Continued Pre-Training (CPT) on Declarative Statements + QA Stacking."""
    print("\n--- Running Mode 4: Continued Pre-Training (LoRA Stacking) ---")

    # 1. Format the dataset for CPT using the declarative statements exclusively
    def format_cpt_train(example):
        return {"text": example["declarative_statement"]}

    cpt_train_data = test_dataset.map(format_cpt_train)
    mode4_cpt_dir = os.path.join(output_dir, "mode4_cpt_adapter")

    print("  Pass 1: Pre-training base model on continuous declarative statements...")

    # Load fresh base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(original_adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=("cpu" if device == "cpu" else {"": device}),  # Strict GPU mapping
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    # Setup temporary LoRA adapter specifically for CPT
    from peft import LoraConfig, get_peft_model

    target_modules = _detect_lora_target_modules(base_model)
    if not target_modules:
        raise RuntimeError(
            "No matching LoRA target modules found for CPT. "
            "Check model architecture or update target module detection."
        )
    print(f"  [System] CPT target modules: {', '.join(target_modules)}")

    cpt_peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    cpt_model = get_peft_model(base_model, cpt_peft_config)

    def _count_trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_params = _count_trainable_params(cpt_model)
    print(f"  [System] CPT trainable parameters: {trainable_params:,}")
    if trainable_params == 0:
        raise RuntimeError(
            "CPT adapter has zero trainable parameters. "
            "Check target_modules for this model architecture."
        )

    training_args = SFTConfig(
        output_dir=mode4_cpt_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-5,
        bf16=True,  # Native AMD speed
        optim="adamw_torch",
        dataset_text_field="text",
        packing=False,  # Packs the declarative sentences into continuous blocks
        max_length=256,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=cpt_model,
        train_dataset=cpt_train_data,
        peft_config=None,  # It is already a PeftModel
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()

    print("  [System] Merging CPT knowledge into base weights...")
    # Merge the CPT adapter into the base model weights permanently for this session
    merged_base_model = trainer.model.merge_and_unload()

    print("  [System] Stacking original QA adapter on top of updated base model...")
    # Load the original QA adapter on top of the newly updated base model
    final_eval_model = PeftModel.from_pretrained(
        merged_base_model, original_adapter_path, is_trainable=False
    )
    final_eval_model.eval()

    print(
        "  [System] Tearing down training graph to free VRAM before batched inference..."
    )
    del trainer
    del cpt_model
    from accelerate.utils import release_memory

    release_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print(
        "  Pass 2: Evaluating questions using CPT-updated base + QA adapter (Batched)..."
    )
    results = []
    exact_count, fuzzy_count = 0, 0
    total = len(test_dataset)

    for i in range(0, total, batch_size):
        batch = test_dataset[i : i + batch_size]
        questions = batch.get("question", [])
        ground_truths = batch.get("answer", [])
        sample_ids = range(i, i + len(questions))

        messages_batch = [
            [{"role": "user", "content": f"Question: {q}"}] for q in questions
        ]

        predictions = generate_batch_responses(
            final_eval_model, tokenizer, messages_batch, device
        )

        for idx, question, truth, pred in zip(
            sample_ids, questions, ground_truths, predictions
        ):
            exact = exact_match(pred, truth)
            fuzzy = fuzzy_match(pred, truth)

            if exact:
                exact_count += 1
            if fuzzy:
                fuzzy_count += 1

            results.append(
                {
                    "sample_id": idx,
                    "mode": 4,
                    "question": question,
                    "ground_truth": truth,
                    "predicted": pred,
                    "exact_match": exact,
                    "fuzzy_match": fuzzy,
                }
            )

        print(f"    Answered {min(i + batch_size, total)}/{total} questions")

    print("  [System] Cleaning up post-evaluation VRAM...")
    del final_eval_model
    del merged_base_model
    release_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return results, exact_count, fuzzy_count


def test_model(
    model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B", batch_size=16
):
    """Main testing pipeline executing Modes 1, 2, and 3."""
    device = get_device()

    print("Loading test dataset...")
    dataset = load_dataset("json", data_files=dataset_path)
    test_dataset = dataset["train"].filter(
        lambda x: x.get("is_pretraining", False) == False
    )

    total_samples = len(test_dataset)
    if total_samples == 0:
        print("Warning: No test samples found. Exiting.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    original_adapter = os.path.join(model_path, "final-adapter")

    # Load base model for Modes 1 & 2
    tokenizer, model = _load_tokenizer_and_model(original_adapter, model_id, device)

    # Execute Mode 1 (Batched) & Mode 2 (Sequential)
    results_m1, exact_m1, fuzzy_m1 = run_mode_1(
        model, tokenizer, test_dataset, device, batch_size
    )
    results_m2, exact_m2, fuzzy_m2 = run_mode_2(model, tokenizer, test_dataset, device)

    print("  [System] Tearing down Mode 1 & 2 computational graph to free VRAM...")
    del model
    del tokenizer
    release_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # Execute Mode 3 (Batched Inference)
    results_m3, exact_m3, fuzzy_m3 = run_mode_3(
        original_adapter,
        dataset_path,
        output_dir,
        model_id,
        device,
        test_dataset,
        batch_size,
    )

    # Execute Mode 4 (CPT + Stacking)
    results_m4, exact_m4, fuzzy_m4 = run_mode_4(
        original_adapter,
        dataset_path,
        output_dir,
        model_id,
        device,
        test_dataset,
        batch_size,
    )

    all_results = results_m1 + results_m2 + results_m3 + results_m4
    results_file = os.path.join(output_dir, "test_results.jsonl")
    with open(results_file, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "total_test_samples": total_samples,
        "batch_size": batch_size,
        "mode_1_isolated": {
            "exact_match_accuracy": f"{(exact_m1 / total_samples) * 100:.2f}%",
            "fuzzy_match_accuracy": f"{(fuzzy_m1 / total_samples) * 100:.2f}%",
        },
        "mode_2_continuous": {
            "exact_match_accuracy": f"{(exact_m2 / total_samples) * 100:.2f}%",
            "fuzzy_match_accuracy": f"{(fuzzy_m2 / total_samples) * 100:.2f}%",
        },
        "mode_3_parametric": {
            "exact_match_accuracy": f"{(exact_m3 / total_samples) * 100:.2f}%",
            "fuzzy_match_accuracy": f"{(fuzzy_m3 / total_samples) * 100:.2f}%",
        },
        "mode_4_cpt": {
            "exact_match_accuracy": f"{(exact_m4 / total_samples) * 100:.2f}%",
            "fuzzy_match_accuracy": f"{(fuzzy_m4 / total_samples) * 100:.2f}%",
        },
    }
    summary_file = os.path.join(output_dir, "test_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TESTING SUMMARY (UNSEEN DATA)")
    print("=" * 60)
    print(f"Total Test Samples: {total_samples}")

    print("\n[Mode 1: Context + Question (Isolated - Batched)]")
    print(
        f"Exact Match: {exact_m1}/{total_samples} ({summary['mode_1_isolated']['exact_match_accuracy']})"
    )
    print(
        f"Fuzzy Match: {fuzzy_m1}/{total_samples} ({summary['mode_1_isolated']['fuzzy_match_accuracy']})"
    )

    print("\n[Mode 2: Continuous Context (Working Memory - Sequential)]")
    print(
        f"Exact Match: {exact_m2}/{total_samples} ({summary['mode_2_continuous']['exact_match_accuracy']})"
    )
    print(
        f"Fuzzy Match: {fuzzy_m2}/{total_samples} ({summary['mode_2_continuous']['fuzzy_match_accuracy']})"
    )

    print("\n[Mode 3: Parametric Memory (Secondary Fine-Tuning - Batched)]")
    print(
        f"Exact Match: {exact_m3}/{total_samples} ({summary['mode_3_parametric']['exact_match_accuracy']})"
    )
    print(
        f"Fuzzy Match: {fuzzy_m3}/{total_samples} ({summary['mode_3_parametric']['fuzzy_match_accuracy']})"
    )

    print("\n[Mode 4: Continued Pre-Training (LoRA Stacking)]")
    print(
        f"Exact Match: {exact_m4}/{total_samples} ({summary['mode_4_cpt']['exact_match_accuracy']})"
    )

    print(
        f"Fuzzy Match: {fuzzy_m4}/{total_samples} ({summary['mode_4_cpt']['fuzzy_match_accuracy']})"
    )

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Test fine-tuned LLM model on unseen data"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Directory containing the fine-tuned model",
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="JSONL dataset file"
    )
    parser.add_argument(
        "--output_dir", type=str, default="testing", help="Directory for test results"
    )
    parser.add_argument(
        "--model_id", type=str, default="google/gemma-4-E2B", help="Base model ID"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for inference"
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    dataset_path = os.path.expanduser(args.dataset_path)
    output_dir = os.path.expanduser(args.output_dir)

    test_model(
        model_path,
        dataset_path,
        output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
