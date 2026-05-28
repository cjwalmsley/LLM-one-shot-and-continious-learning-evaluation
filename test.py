import argparse
import json
import os
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rapidfuzz import fuzz


def get_device():
    """
    Detect and return the appropriate device.
    Explicitly checks for AMD/ROCm compatibility.
    """
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


def _load_tokenizer_and_model(model_path, model_id, device):
    adapter_path = os.path.join(model_path, "final-adapter")
    print("Loading tokenizer and model for testing...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # Ensure chat template exists and pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu" if device == "cpu" else "auto",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_response(model, tokenizer, messages, device, max_length=256):
    """
    Generate a deterministic response using a list of conversation messages.
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    if device != "cpu":
        target_device = getattr(model, "device", torch.device(device))
        if str(target_device) != "meta":
            inputs = {key: value.to(target_device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=input_length + max_length,
            do_sample=False,  # Greedy decoding for factual testing
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice the output to only extract the newly generated tokens
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return response


def exact_match(predicted, ground_truth):
    return predicted.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(predicted, ground_truth, threshold=80):
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold


def run_mode_1(model, tokenizer, test_dataset, device):
    """
    Mode 1: Prompt model with Context and Question simultaneously. (Isolated Memory)
    """
    print("\n--- Running Mode 1: Context + Question (Isolated) ---")
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
            print(f"  Processed {idx + 1}/{total} samples (Mode 1)")

    return results, exact_count, fuzzy_count


def run_mode_2(model, tokenizer, test_dataset, device):
    """
    Mode 2: Two-pass continuous context. (Working Memory)
    Pass 1: Load all declarative statements into history.
    Pass 2: Ask all questions based on the accumulated history.
    """
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
        # We must prompt the model to register the interaction in the chat history
        reply = generate_response(model, tokenizer, messages, device, max_length=50)
        messages.append({"role": "assistant", "content": reply})

        if (idx + 1) % 10 == 0:
            print(f"    Loaded {idx + 1}/{total} statements")

    print("  Pass 2: Prompting questions from continuous context...")
    for idx, sample in enumerate(test_dataset):
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")

        messages.append({"role": "user", "content": question})
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


def test_model(model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B"):
    """
    Main testing pipeline executing both Mode 1 and Mode 2.
    """
    device = get_device()
    tokenizer, model = _load_tokenizer_and_model(model_path, model_id, device)

    print("Loading test dataset...")
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset["train"]
    test_dataset = dataset.filter(lambda x: x.get("is_pretraining", False) == False)

    total_samples = len(test_dataset)
    if total_samples == 0:
        print("Warning: No test samples found. Exiting.")
        return None

    print(f"Total unseen samples for testing: {total_samples}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Execute Mode 1 ---
    results_m1, exact_m1, fuzzy_m1 = run_mode_1(model, tokenizer, test_dataset, device)

    # --- Execute Mode 2 ---
    results_m2, exact_m2, fuzzy_m2 = run_mode_2(model, tokenizer, test_dataset, device)

    # --- Save & Summarize ---
    all_results = results_m1 + results_m2
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
    print(
        f"Fuzzy Match: {fuzzy_m1}/{total_samples} ({summary['mode_1_isolated']['fuzzy_match_accuracy']})"
    )

    print("\n[Mode 2: Continuous Context (Working Memory)]")
    print(
        f"Exact Match: {exact_m2}/{total_samples} ({summary['mode_2_continuous']['exact_match_accuracy']})"
    )
    print(
        f"Fuzzy Match: {fuzzy_m2}/{total_samples} ({summary['mode_2_continuous']['fuzzy_match_accuracy']})"
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
        default="test_results",
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
