from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import argparse


def format_prompt(example, tokenizer):
    """
    Structures the context, question, and answer into Gemma 4's native chat format.
    """
    messages = [
        {
            "role": "user",
            "content": f"Context: {example['declarative_statement']}\n\nQuestion: {example['question']}"
        },
        {
            "role": "assistant",
            "content": example['answer']
        }
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


def main():

#   python finetune_gemma4.py --model_id "meta-llama/Meta-Llama-3-8B" --dataset_path "custom_data.jsonl"

    parser = argparse.ArgumentParser(description="Finetune Gemma model")
    parser.add_argument("--model_id", type=str, default="google/gemma-4-E2B", help="The model to be finetuned")
    parser.add_argument("--dataset_path", type=str, default="sample_dataset.jsonl", help="The filepath of the dataset file to load")
    args = parser.parse_args()

    model_id = args.model_id
    dataset_path = args.dataset_path
    output_dir = "./gemma4-qa-finetune"

    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Gemma models generally do not define a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto"
    )

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
        target_modules=["q_proj.linear", "k_proj.linear", "v_proj.linear", "o_proj.linear", "gate_proj.linear", "up_proj.linear", "down_proj.linear"]
    )

    print("Loading and formatting dataset...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Map the formatting function to the dataset
    dataset = dataset.map(lambda x: format_prompt(x, tokenizer))

    print("Setting up training arguments...")
    training_args = SFTConfig(
        output_dir=output_dir,
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
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print("Training complete. Saving the LoRA adapter...")
    trainer.model.save_pretrained(f"{output_dir}/final-adapter")
    tokenizer.save_pretrained(f"{output_dir}/final-adapter")
    print("Done!")


if __name__ == "__main__":
    main()