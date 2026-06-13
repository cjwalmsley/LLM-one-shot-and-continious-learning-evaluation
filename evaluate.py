import argparse
import json
import os
import gc
import torch
from datetime import datetime
from datasets import load_dataset
from accelerate.utils import release_memory

from utils import (
    get_device,
    load_tokenizer_and_model,
    generate_batch_responses,
    exact_match,
    fuzzy_match,
)


def evaluate_model(
    model_path, dataset_path, output_dir, model_id="google/gemma-4-E2B", batch_size=16, use_thinking=None
):
    """
    Evaluate the fine-tuned model on the training dataset using batched inference.
    """
    device = get_device()
    adapter_path = os.path.join(model_path, "final-adapter")
    tokenizer, model, model_config = load_tokenizer_and_model(adapter_path, model_id, device, use_thinking=use_thinking, return_config=True)

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

        messages_batch = [
            [{"role": "user", "content": f"Context: {c}\n\nQuestion: {q}"}]
            for c, q in zip(contexts, questions)
        ]

        predictions = generate_batch_responses(model, tokenizer, messages_batch, device, max_new_tokens=model_config.get_max_new_tokens())

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
    parser.add_argument(
        "--use_thinking",
        action="store_true",
        default=None,
        help="Enable thinking mode (if supported by the model)",
    )
    parser.add_argument(
        "--disable_thinking",
        action="store_true",
        default=False,
        help="Explicitly disable thinking mode",
    )
    args = parser.parse_args()

    use_thinking = False if args.disable_thinking else (True if args.use_thinking else None)

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
        use_thinking=use_thinking,
    )


if __name__ == "__main__":
    main()
