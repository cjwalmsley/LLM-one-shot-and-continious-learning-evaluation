from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from evaluate import evaluate_model
import argparse
import os
from dotenv import load_dotenv


def format_prompt(example, tokenizer):
    """
    Structures the context, question, and answer into Gemma 4's native chat format.
    """
    messages = [
        {
            "role": "user",
            "content": f"Context: {example['declarative_statement']}\n\nQuestion: {example['question']}",
        },
        {"role": "assistant", "content": example["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


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
        default="sample_dataset.jsonl",
        help="The filepath of the dataset file to load",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="The name of the experiment that the finetuning dataset is from",
    )
    args = parser.parse_args()

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

    # if an experiment name is provided us the default paths for that experiment name. Otherwise, use the provided dataset path.
    if experiment:
        dataset_path = f"~/gdrive/PhD/work/annabell/experiments/{experiment}/data_sets/nyc_squad_with_pretraining_commands.jsonl"
        output_dir = f"~/gdrive/PhD/work/annabell/experiments/{experiment}/llm_comparison/finetune/{model_id}"
    else:
        dataset_path = args.dataset_path
        output_dir = "gemma4-qa-finetune"

    # check if the dataset_path exists
    expanded_dataset_path = os.path.expanduser(dataset_path)
    if not os.path.exists(expanded_dataset_path):
        raise FileNotFoundError(f"Dataset file not found at: {expanded_dataset_path}")

    # check if the output_dir exists, create it if it doesn't
    expanded_output_dir = os.path.expanduser(output_dir)
    if not os.path.exists(expanded_output_dir):
        os.makedirs(expanded_output_dir)
        print(f"Created output directory: {expanded_output_dir}")

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Gemma models generally do not define a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Target modules common to Gemma architectures
        target_modules=[
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ],
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
    train_dataset = train_dataset.map(lambda x: format_prompt(x, tokenizer))
    test_dataset = test_dataset.map(lambda x: format_prompt(x, tokenizer))

    print("Setting up training arguments...")
    training_args = SFTConfig(
        output_dir=expanded_output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        bf16=True,  # Switched from fp16=True for native AMD speed
        logging_steps=10,
        optim="adamw_torch",  # Uses native PyTorch AdamW instead of CUDA bnb optimizer
        save_strategy="steps",
        save_steps=50,
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

    # Run evaluation
    print("\n" + "=" * 60)
    print("Starting model evaluation...")
    print("=" * 60)
    eval_output_dir = f"{expanded_output_dir}/evaluation"
    try:
        evaluate_model(
            expanded_output_dir,
            expanded_dataset_path,
            eval_output_dir,
            model_id=model_id,
        )
    except Exception as e:
        print(f"Warning: Evaluation failed with error: {e}")


if __name__ == "__main__":
    main()
