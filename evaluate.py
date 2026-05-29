import argparse
import json
import os
import gc
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rapidfuzz import fuzz
from accelerate.utils import release_memory


def get_device():
    """
    Detect and return the appropriate device for the system.
    Supports CUDA (both NVIDIA and AMD ROCm), MPS (Apple), and CPU.
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
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    # CRITICAL FOR BATCHING: Set pad token and force left-padding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu" if device == "cpu" else {"": device},  # Strict GPU mapping
        torch_dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model


def generate_batch_responses(
    model, tokenizer, contexts, questions, device, max_new_tokens=50
):
    """
    Generate responses for a batch of contexts and questions simultaneously.
    """
    messages_batch = [
        [{"role": "user", "content": f"Context: {c}\n\nQuestion: {q}"}]
        for c, q in zip(contexts, questions)
    ]

    prompts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    # Tokenize the entire batch, padding to the longest sequence in the batch
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)

    if device != "cpu":
        inputs = {key: value.to(device) for key, value in inputs.items()}

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,  # Prevents runaway generation
            do_sample=False,  # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    # Slice off the prompt tokens and decode only the newly generated tokens
    new_tokens = outputs[:, input_length:]
    responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    return [response.strip() for response in responses]


def generate_response(model, tokenizer, context, question, device, max_new_tokens=50):
    """Generate a response for a single context/question pair."""
    messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt", padding=True)
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
    response = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return response.strip()


def exact_match(predicted, ground_truth):
    """Check for exact string match (case-insensitive)."""
    return predicted.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(predicted, ground_truth, threshold=80):
    """Check for fuzzy string match using token_set_ratio."""
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold


def evaluate_model(
    model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B", batch_size=16
):
    """
    Evaluate the fine-tuned model on the training dataset using batched inference.
    """
    device = get_device()
    tokenizer, model = _load_tokenizer_and_model(model_path, model_id, device)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path)
    train_dataset = dataset["train"].filter(
        lambda x: x.get("is_pretraining", False) == True
    )

    total_samples = len(train_dataset)
    print(
        f"Evaluating on {total_samples} training samples using batch size {batch_size}..."
    )

    results = []
    exact_match_count = 0
    fuzzy_match_count = 0

    # Process the dataset in chunks (batches)
    for i in range(0, total_samples, batch_size):
        # Extract batch columns safely
        batch = train_dataset[i : i + batch_size]
        contexts = batch.get("declarative_statement", [])
        questions = batch.get("question", [])
        ground_truths = batch.get("answer", [])
        sample_ids = range(i, i + len(contexts))

        predictions = [
            generate_response(model, tokenizer, context, question, device)
            for context, question in zip(contexts, questions)
        ]

        # Calculate metrics for the batch
        for idx, context, question, truth, pred in zip(
            sample_ids, contexts, questions, ground_truths, predictions
        ):
            exact = exact_match(pred, truth)
            fuzzy = fuzzy_match(pred, truth)

            if exact:
                exact_match_count += 1
            if fuzzy:
                fuzzy_match_count += 1

            results.append(
                {
                    "sample_id": idx,
                    "context": context,
                    "question": question,
                    "ground_truth": truth,
                    "predicted": pred,
                    "exact_match": exact,
                    "fuzzy_match": fuzzy,
                }
            )

        print(
            f"  Evaluated {min(i + batch_size, total_samples)}/{total_samples} samples"
        )

    exact_accuracy = (
        (exact_match_count / total_samples) * 100 if total_samples > 0 else 0
    )
    fuzzy_accuracy = (
        (fuzzy_match_count / total_samples) * 100 if total_samples > 0 else 0
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "model_path": model_path,
        "dataset_path": dataset_path,
        "total_samples": total_samples,
        "batch_size": batch_size,
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

    results_file = os.path.join(output_dir, "evaluation_results.jsonl")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model Path: {model_path}")
    print(f"Total Samples Evaluated: {total_samples} (Batch Size: {batch_size})")
    print("-" * 60)
    print(
        f"Exact Match Accuracy: {exact_match_count}/{total_samples} ({exact_accuracy:.2f}%)"
    )
    print(
        f"Fuzzy Match Accuracy: {fuzzy_match_count}/{total_samples} ({fuzzy_accuracy:.2f}%)"
    )
    print("=" * 60 + "\n")

    # -------- VRAM TEARDOWN BLOCK --------
    print("Tearing down evaluation computational graph to free VRAM...")
    del model
    del tokenizer
    release_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    # -------------------------------------

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to JSONL dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Directory for results",
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

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    evaluate_model(
        model_path,
        dataset_path,
        output_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
