import argparse
import json
import os
import gc
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from rapidfuzz import fuzz


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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=(
            "cpu" if device == "cpu" else {"": device}
        ),  # Forces strict GPU mapping
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    # REMOVED: prepare_model_for_kbit_training.
    # It causes massive float32 VRAM bloat and is unnecessary for bfloat16 LoRA.

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


def generate_response(model, tokenizer, messages, device, max_length=256):
    """Generate a deterministic response using greedy decoding."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    if device != "cpu":
        # Send inputs directly to the confirmed active device
        inputs = {key: value.to(device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=input_length + max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def exact_match(predicted, ground_truth):
    return predicted.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(predicted, ground_truth, threshold=80):
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold


def run_mode_1(model, tokenizer, test_dataset, device):
    """Mode 1: Prompt model with Context and Question simultaneously."""
    print("\n--- Running Mode 1: Context + Question (Isolated Memory) ---")
    results = []
    exact_count, fuzzy_count = 0, 0
    total = len(test_dataset)

    for idx, sample in enumerate(test_dataset):
        context = sample.get("declarative_statement", "")
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")

        messages = [
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
        predicted = generate_response(model, tokenizer, messages, device)

        exact = exact_match(predicted, ground_truth)
        fuzzy = fuzzy_match(predicted, ground_truth)

        if exact:
            exact_count += 1
        if fuzzy:
            fuzzy_count += 1

        results.append(
            {
                "sample_id": idx,
                "mode": 1,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "exact_match": exact,
                "fuzzy_match": fuzzy,
            }
        )
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total} samples")

    return results, exact_count, fuzzy_count


def run_mode_2(model, tokenizer, test_dataset, device):
    """Mode 2: Two-pass continuous context (Working Memory)."""
    print("\n--- Running Mode 2: Continuous Context (Working Memory) ---")
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
        reply = generate_response(model, tokenizer, messages, device, max_length=50)
        messages.append({"role": "assistant", "content": reply})

    print("  Pass 2: Prompting questions from continuous context...")
    for idx, sample in enumerate(test_dataset):
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")

        messages.append({"role": "user", "content": f"Question: {question}"})
        predicted = generate_response(model, tokenizer, messages, device)
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
    original_adapter_path, dataset_path, output_dir, model_id, device, test_dataset
):
    """Mode 3: Parametric Memory Consolidation (Secondary Fine-Tuning)."""
    print("\n--- Running Mode 3: Parametric Memory (Weights Fine-Tuning) ---")

    # Format dataset to only include the declarative statements for training
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
        num_train_epochs=1,  # One pass to prevent catastrophic forgetting of QA format
        learning_rate=2e-5,
        bf16=True,  # Native AMD speed
        optim="adamw_torch",
        dataset_text_field="text",
        max_length=256,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=trainable_model,
        train_dataset=mode3_train_data,
        peft_config=None,  # It is already a PeftModel
        processing_class=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.model.save_pretrained(mode3_adapter_dir)
    tokenizer.save_pretrained(mode3_adapter_dir)

    # ---------------------------------------------------------
    # CRITICAL: Teardown graph and clear VRAM before inference
    # ---------------------------------------------------------
    del trainer
    del trainable_model
    torch.cuda.empty_cache()
    gc.collect()

    print("  Pass 2: Evaluating questions using updated parametric weights...")
    # Load the newly trained adapter in evaluation mode
    tokenizer, eval_model = _load_tokenizer_and_model(
        mode3_adapter_dir, model_id, device, is_trainable=False
    )

    results = []
    exact_count, fuzzy_count = 0, 0
    total = len(test_dataset)

    for idx, sample in enumerate(test_dataset):
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")

        # Prompt model with JUST the question (no context)
        messages = [{"role": "user", "content": f"Question: {question}"}]
        predicted = generate_response(eval_model, tokenizer, messages, device)

        exact = exact_match(predicted, ground_truth)
        fuzzy = fuzzy_match(predicted, ground_truth)

        if exact:
            exact_count += 1
        if fuzzy:
            fuzzy_count += 1

        results.append(
            {
                "sample_id": idx,
                "mode": 3,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "exact_match": exact,
                "fuzzy_match": fuzzy,
            }
        )
        if (idx + 1) % 10 == 0:
            print(f"    Answered {idx + 1}/{total} questions")

    # Cleanup post-evaluation
    del eval_model
    torch.cuda.empty_cache()
    gc.collect()

    return results, exact_count, fuzzy_count


def test_model(model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B"):
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

    # Execute Mode 1 & 2
    results_m1, exact_m1, fuzzy_m1 = run_mode_1(model, tokenizer, test_dataset, device)
    results_m2, exact_m2, fuzzy_m2 = run_mode_2(model, tokenizer, test_dataset, device)

    print("  [System] Tearing down Mode 1 & 2 computational graph to free VRAM...")
    del model
    del tokenizer

    # Use accelerate's deep memory release to clear device_map hooks
    from accelerate.utils import release_memory

    release_memory()

    # Force Python and PyTorch garbage collection
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.ipc_collect()
    # -----------------------------------------

    # Execute Mode 3
    results_m3, exact_m3, fuzzy_m3 = run_mode_3(
        original_adapter, dataset_path, output_dir, model_id, device, test_dataset
    )

    # Save & Summarize
    all_results = results_m1 + results_m2 + results_m3
    results_file = os.path.join(output_dir, "test_results.jsonl")
    with open(results_file, "w") as f:
        for res in all_results:
            f.write(json.dumps(res) + "\n")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "total_test_samples": total_samples,
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
    }

    summary_file = os.path.join(output_dir, "test_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TESTING SUMMARY (UNSEEN DATA)")
    print("=" * 60)
    print(f"Total Test Samples: {total_samples}")

    print("\n[Mode 1: Context + Question (Isolated)]")
    print(
        f"Exact Match: {exact_m1}/{total_samples} ({summary['mode_1_isolated']['exact_match_accuracy']})"
    )

    print("\n[Mode 2: Continuous Context (Working Memory)]")
    print(
        f"Exact Match: {exact_m2}/{total_samples} ({summary['mode_2_continuous']['exact_match_accuracy']})"
    )

    print("\n[Mode 3: Parametric Memory (Secondary Fine-Tuning)]")
    print(
        f"Exact Match: {exact_m3}/{total_samples} ({summary['mode_3_parametric']['exact_match_accuracy']})"
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
        "--output_dir",
        type=str,
        default="testing",
        help="Directory for test results",
    )
    parser.add_argument(
        "--model_id", type=str, default="google/gemma-4-E2B", help="Base model ID"
    )
    args = parser.parse_args()

    model_path = os.path.expanduser(args.model_path)
    dataset_path = os.path.expanduser(args.dataset_path)
    output_dir = os.path.expanduser(args.output_dir)

    test_model(model_path, dataset_path, output_dir, model_id=args.model_id)


if __name__ == "__main__":
    main()
