import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import argparse
import os
from pathlib import Path

# Set PyTorch CUDA allocation configuration to avoid fragmentation before importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from dotenv import load_dotenv
from evaluate import evaluate_model
from test import test_model
from model_adapters import get_model_config
import datetime


def main():

    #   python finetune_gemma4.py --model_id "meta-llama/Meta-Llama-3-8B" --dataset_path "custom_data.jsonl"

    parser = argparse.ArgumentParser(description="Finetune Gemma model")
    parser.add_argument(
        "--model_id",
        type=str,
        default="google/gemma-4-E2B",
        help="The model to be finetuned",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="The filepath of the dataset file to load",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="The name of the experiment that the finetuning dataset is from",
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
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=None,
        help="Batch size to use for evaluation and testing. If not provided, it is dynamically determined.",
    )
    args = parser.parse_args()

    use_thinking = (
        False if args.disable_thinking else (True if args.use_thinking else None)
    )

    # Load environment variables from .env file
    if os.path.exists(".env"):
        load_dotenv(".env")
    else:
        print("Warning: .env file not found.")

    # Ensure HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        raise EnvironmentError(
            "Error: HF_TOKEN environment variable is not set.\n"
            "Please add it to your .env file:\n"
            "  HF_TOKEN=your_huggingface_token"
        )

    model_id = args.model_id

    experiment = args.experiment

    safe_dir_name = model_id.replace("/", "-")

    # if an experiment name is provided us the default paths for that experiment name. Otherwise, use the provided dataset path.
    if experiment and not args.dataset_path:
        dataset_path = f"~/gdrive/PhD/work/annabell/experiments/{experiment}/data_sets/nyc_squad_with_pretraining_commands.jsonl"
        output_dir = f"~/gdrive/PhD/work/annabell/experiments/{experiment}/llm_comparison/finetune/{model_id}"
    elif experiment and args.dataset_path:
        dataset_path = args.dataset_path
        output_dir = f"./experiments/{experiment}/{model_id}"
    elif not experiment and args.dataset_path:
        dataset_path = args.dataset_path
        output_dir = f"./{safe_dir_name}"
    else:
        dataset_path = "sample_dataset.jsonl"
        output_dir = f"./{safe_dir_name}"

    output_dir = f"{output_dir}/{Path(dataset_path).stem}"

    # check if the dataset_path exists
    expanded_dataset_path = os.path.expanduser(dataset_path)
    if not os.path.exists(expanded_dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {expanded_dataset_path}")

    # check if the output_dir exists, create it if it doesn't
    expanded_output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(expanded_output_dir):
        os.makedirs(expanded_output_dir)
        print(f"Created output directory: {expanded_output_dir}")

    model_config = get_model_config(model_id, use_thinking=use_thinking)

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer = model_config.setup_tokenizer(tokenizer)

    print("Loading model...")
    quantization_config = None
    attn_implementation = None
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # Note: SDPA is highly optimized and avoids Flash Attention 2 head dimension limits (max 256)
        if torch.cuda.is_available():
            attn_implementation = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=model_config.get_lora_target_modules(),
    )

    print("Loading and formatting dataset...")
    # Load dataset without specifying split to get all examples
    dataset = load_dataset("json", data_files=expanded_dataset_path)
    dataset = dataset["train"]  # Get the train split from the loaded dataset

    # Split dataset based on is_pretraining flag
    print("Splitting dataset based on is_pretraining flag...")
    train_dataset = dataset.filter(lambda x: x.get("is_pretraining", False) == True)
    test_dataset = dataset.filter(lambda x: x.get("is_pretraining", False) == False)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Map the formatting function to both datasets
    train_dataset = train_dataset.map(
        lambda x: model_config.format_prompt(x, tokenizer)
    )
    test_dataset = test_dataset.map(lambda x: model_config.format_prompt(x, tokenizer))

    print("Setting up training arguments...")

    # Determine precision based on hardware support
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
        else:
            use_fp16 = True
    elif torch.backends.mps.is_available():
        use_fp16 = True

    print(f"Using bf16: {use_bf16}, fp16: {use_fp16}")

    training_args = SFTConfig(
        output_dir=expanded_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        # max_steps=200,
        num_train_epochs=5,
        learning_rate=2e-4,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=10,
        optim="adamw_torch",  # Uses native PyTorch AdamW instead of CUDA bnb optimizer
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=4,  # Pre-fetches data tokens via CPU threads
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
    )

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete. Saving the LoRA adapter...")
    trainer.model.save_pretrained(f"{expanded_output_dir}/final-adapter")
    tokenizer.save_pretrained(f"{expanded_output_dir}/final-adapter")
    print("Done!")

    print("Tearing down training graph to free VRAM for evaluation...")
    del trainer
    del model

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        # Determine a safe inference batch size based on the model's max_new_tokens
        inference_batch_size = args.inference_batch_size
        if inference_batch_size is None:
            max_tokens = model_config.get_max_new_tokens()
            # If generating massive outputs, lower batch size severely to avoid KV cache OOM
            if max_tokens > 16000:
                inference_batch_size = 2
            elif max_tokens > 4096:
                inference_batch_size = 4
            else:
                inference_batch_size = 16

        print(f"\nUsing inference batch size: {inference_batch_size}")

        # Run evaluation natively
        print("\n" + "=" * 60)
        print("Starting model evaluation...")
        print("=" * 60)
        eval_output_dir = f"{expanded_output_dir}/evaluation"

        try:
            evaluate_model(
                model_path=expanded_output_dir,
                dataset_path=expanded_dataset_path,
                output_dir=eval_output_dir,
                model_id=model_id,
                batch_size=inference_batch_size,
                use_thinking=use_thinking,
            )
        except Exception as e:
            print(f"Warning: Evaluation failed with error: {e}")

        # Run testing natively
        print("\n" + "=" * 60)
        print("Starting formal testing phase...")
        print("=" * 60)
        test_output_dir = f"{expanded_output_dir}/testing"

        try:
            test_model(
                model_path=expanded_output_dir,
                dataset_path=expanded_dataset_path,
                output_dir=test_output_dir,
                model_id=model_id,
                batch_size=inference_batch_size,
                use_thinking=use_thinking,
            )
        except Exception as e:
            print(f"Warning: Testing failed with error: {e}")


if __name__ == "__main__":
    main()
