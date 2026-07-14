import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from rapidfuzz import fuzz
from model_adapters import get_model_config


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


def load_tokenizer_and_model(
    adapter_path,
    model_id,
    device,
    is_trainable=False,
    use_thinking=None,
    return_config=False,
):
    """Loads the base model and attaches the specified adapter."""
    print(f"Loading tokenizer and model from {adapter_path}...", flush=True)
    model_config = get_model_config(model_id, use_thinking=use_thinking)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer = model_config.setup_tokenizer(tokenizer)
    print("Tokenizer loaded.", flush=True)

    quantization_config = None
    attn_implementation = None
    if device != "cpu":
        # Check for bf16 support dynamically to maintain AMD ROCm and CUDA compatibility
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        attn_implementation = "sdpa"

    if device == "cpu":
        target_dtype = torch.float32
    else:
        target_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    print(f"Loading base model weights for {model_id}...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=("cpu" if device == "cpu" else "auto"),
        torch_dtype=target_dtype,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
    )
    print("Base model loaded successfully.", flush=True)

    print(f"Loading PEFT adapter from {adapter_path}...", flush=True)
    model = PeftModel.from_pretrained(
        base_model, adapter_path, is_trainable=is_trainable
    )
    print("PEFT adapter loaded and applied successfully.", flush=True)

    if not is_trainable:
        print("Setting model to eval mode...", flush=True)
        model.eval()
    else:
        # Explicitly ensure only the LoRA parameters require gradients
        for name, param in model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True

    if return_config:
        return tokenizer, model, model_config
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
    """Check for exact string match (case-insensitive)."""
    return predicted.strip().lower() == ground_truth.strip().lower()


def fuzzy_match(predicted, ground_truth, threshold=70):
    """Check for fuzzy string match using token_set_ratio."""
    score = fuzz.token_set_ratio(predicted.lower(), ground_truth.lower())
    return score >= threshold
