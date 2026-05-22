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

    if device == "cpu":
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return tokenizer, model



def generate_response(model, tokenizer, context, question, device, max_length=256):
    """
    Generate a response from the model given context and question.
    """
    messages = [
        {
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        }
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    # Tokenize - when using device_map="auto", keep inputs on CPU initially
    # and let transformers handle device placement
    inputs = tokenizer(prompt, return_tensors="pt")
    if device != "cpu":
        target_device = getattr(model, "device", torch.device(device))
        if str(target_device) != "meta":
            inputs = {key: value.to(target_device) for key, value in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Move output to CPU for decoding if needed
    if outputs.device.type != "cpu":
        outputs = outputs.cpu()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's response (remove the context and question)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    return response



def exact_match(predicted, ground_truth):
    """
    Check for exact string match (case-insensitive).
    """
    return predicted.strip().lower() == ground_truth.strip().lower()



def fuzzy_match(predicted, ground_truth, threshold=80):
    """
    Check for fuzzy string match using token_set_ratio.
    Returns True if fuzzy match score is above threshold.
    """
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold



def evaluate_model(model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B"):
    """
    Evaluate the fine-tuned model on the training dataset.
    """
    # Detect device
    device = get_device()
    tokenizer, model = _load_tokenizer_and_model(model_path, model_id, device)

    print("Loading dataset...")
    dataset = load_dataset("json", data_files=dataset_path)
    dataset = dataset["train"]
    # Filter for training examples (is_pretraining=True)
    train_dataset = dataset.filter(lambda x: x.get("is_pretraining", False) == True)
    print(f"Evaluating on {len(train_dataset)} training samples...")
    results = []
    exact_match_count = 0
    fuzzy_match_count = 0
    for idx, sample in enumerate(train_dataset):
        context = sample.get("declarative_statement", "")
        question = sample.get("question", "")
        ground_truth = sample.get("answer", "")
        # Generate response
        predicted = generate_response(model, tokenizer, context, question, device)
        # Check matches
        exact = exact_match(predicted, ground_truth)
        fuzzy = fuzzy_match(predicted, ground_truth)
        if exact:
            exact_match_count += 1
        if fuzzy:
            fuzzy_match_count += 1
        result = {
            "sample_id": idx,
            "context": context,
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "exact_match": exact,
            "fuzzy_match": fuzzy
        }
        results.append(result)
        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"  Evaluated {idx + 1}/{len(train_dataset)} samples")
    # Calculate accuracy
    total_samples = len(train_dataset)
    exact_accuracy = (exact_match_count / total_samples) * 100 if total_samples > 0 else 0
    fuzzy_accuracy = (fuzzy_match_count / total_samples) * 100 if total_samples > 0 else 0
    # Create summary statistics
    summary = {
        "timestamp": datetime.now().isoformat(),
        "device": str(device),
        "model_path": model_path,
        "dataset_path": dataset_path,
        "total_samples": total_samples,
        "exact_match": {
            "count": exact_match_count,
            "accuracy": f"{exact_accuracy:.2f}%"
        },
        "fuzzy_match": {
            "count": fuzzy_match_count,
            "accuracy": f"{fuzzy_accuracy:.2f}%"
        }
    }
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    # Write per-sample results to file
    results_file = os.path.join(output_dir, "evaluation_results.jsonl")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\nPer-sample results saved to: {results_file}")
    # Write summary statistics to file
    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics saved to: {summary_file}")
    # Display summary statistics
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Device: {device}")
    print(f"Model Path: {model_path}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Total Samples Evaluated: {total_samples}")
    print("-"*60)
    print(f"Exact Match Accuracy: {exact_match_count}/{total_samples} ({exact_accuracy:.2f}%)")
    print(f"Fuzzy Match Accuracy: {fuzzy_match_count}/{total_samples} ({fuzzy_accuracy:.2f}%)")
    print("="*60 + "\n")
    return summary



def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing the fine-tuned model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the JSONL dataset file")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E2B", help="Base model ID used for the adapter")
    args = parser.parse_args()
    # Expand user paths
    model_path = os.path.expanduser(args.model_path)
    dataset_path = os.path.expanduser(args.dataset_path)
    output_dir = os.path.expanduser(args.output_dir)
    # Validate paths
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    evaluate_model(model_path, dataset_path, output_dir, model_id=args.model_id)


if __name__ == "__main__":
    main()
