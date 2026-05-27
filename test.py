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

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu" if device == "cpu" else "auto",
        torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_response(model, tokenizer, context, question, device, max_length=256):
    """
    Generate a deterministic response using greedy decoding.
    """
    messages = [
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)

    inputs = tokenizer(prompt, return_tensors="pt")

    if device != "cpu":
        target_device = getattr(model, "device", torch.device(device))
        if str(target_device) != "meta":
            inputs = {key: value.to(target_device) for key, value in inputs.items()}

    # Deterministic generation for factual testing
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,  # Greedy decoding for exact testing
            pad_token_id=tokenizer.eos_token_id,
        )

    if outputs.device.type != "cpu":
        outputs = outputs.cpu()

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response


def exact_match(predicted, ground_truth):
    return predicted.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(predicted, ground_truth, threshold=80):
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold


def test_model(model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B"):
    """
    Run formal testing on the unseen test dataset.
    """
    device = get_device()
    tokenizer, model = _load_tokenizer_and_model(model_path, model_id, device)

    print("Loading test dataset...")
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset["train"]

    # Filter strictly for unseen test examples
    test_dataset = dataset.filter(lambda x: x.get("is_pretraining", False) == False)

    total_samples = len(test_dataset)
    if total_samples == 0:
        print("Warning: No test samples found (is_pretraining=False). Exiting.")
        return None

    print(f"Testing on {total_samples} unseen samples...")
    results = []
    exact_match_count = 0
    fuzzy_match_count = 0

    for idx, sample in enumerate(test_dataset):
        context = sample.get("declarative_statement", "")
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")

        predicted = generate_response(model, tokenizer, context, question, device)

        exact = exact_match(predicted, ground_truth)
        fuzzy = fuzzy_match(predicted, ground_truth)

        if exact:
            exact_match_count += 1
        if fuzzy:
            fuzzy_match_count += 1

        results.append(
            {
                "sample_id": idx,
                "context": context,
                "question": question,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "exact_match": exact,
                "fuzzy_match": fuzzy,
            }
        )

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{total_samples} test samples")

    exact_accuracy = (exact_match_count / total_samples) * 100
    fuzzy_accuracy = (fuzzy_match_count / total_samples) * 100

    summary = {
        "timestamp": datetime.now().isoformat(),
        "phase": "test",
        "device": str(device),
        "model_path": model_path,
        "dataset_path": dataset_path,
        "total_test_samples": total_samples,
        "exact_match": {
            "count": exact_match_count,
            "accuracy": f"{exact_accuracy:.2f}%",
        },
        "fuzzy_match": {
            "count": fuzzy_match_count,
            "accuracy": f"{fuzzy_accuracy:.2f}%",
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, "test_results.jsonl")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    summary_file = os.path.join(output_dir, "test_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TESTING SUMMARY (UNSEEN DATA)")
    print("=" * 60)
    print(f"Total Test Samples: {total_samples}")
    print(
        f"Exact Match Accuracy: {exact_match_count}/{total_samples} ({exact_accuracy:.2f}%)"
    )
    print(
        f"Fuzzy Match Accuracy: {fuzzy_match_count}/{total_samples} ({fuzzy_accuracy:.2f}%)"
    )
    print("=" * 60 + "\n")

    return summary


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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    test_model(model_path, dataset_path, output_dir, model_id=args.model_id)


if __name__ == "__main__":
    main()
