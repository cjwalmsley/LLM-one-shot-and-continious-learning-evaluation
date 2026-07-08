from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class ModelConfig(ABC):
    """Abstract base class for model configurations."""

    def __init__(self, use_thinking: Optional[bool] = None):
        self.use_thinking = use_thinking if use_thinking is not None else False

    @abstractmethod
    def get_lora_target_modules(self) -> List[str]:
        """Return the target modules for LoRA."""
        pass

    def get_max_new_tokens(self) -> int:
        """Return the max new tokens for generation."""
        return 50  # Default value

    def format_prompt(self, example: Dict[str, Any], tokenizer: Any) -> Dict[str, str]:
        """Format the dataset example for SFT training."""
        messages = [
            {
                "role": "user",
                "content": f"Context: {example.get('declarative_statement', '')}\n\nQuestion: {example.get('question', '')}",
            },
            {"role": "assistant", "content": example.get("answer", "")},
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    def format_mode3_train(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format the dataset example for Mode 3 training."""
        return {"text": example.get("declarative_statement", "")}

    def format_cpt_train(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format the dataset example for CPT training (Mode 4)."""
        return {"text": example.get("declarative_statement", "")}

    def setup_tokenizer(self, tokenizer: Any) -> Any:
        """Apply model-specific tokenizer configurations."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer


class GemmaModelConfig(ModelConfig):
    """Configuration for Gemma models."""
    def __init__(self, use_thinking: Optional[bool] = None):
        super().__init__(use_thinking=False) # Thinking mode not supported

    def get_lora_target_modules(self) -> List[str]:
        # Target the text tower layers explicitly. Gemma 4 text layers are usually named
        # `model.layers.X...` but vision layers are `model.vision_tower...`.
        # The text layers use standard Linear (or 4bit Linear) which PEFT supports as q_proj,
        # but vision layers use `Gemma4ClippableLinear` requiring the `.linear` suffix.
        # By targeting the text layers with regex and standard projection names, we avoid the error.
        return [
            ".*model\\.layers\\.\\d+\\.self_attn\\.(?:q_proj|k_proj|v_proj|o_proj)",
            ".*model\\.layers\\.\\d+\\.mlp\\.(?:gate_proj|up_proj|down_proj)",
        ]

    def setup_tokenizer(self, tokenizer: Any) -> Any:
        tokenizer = super().setup_tokenizer(tokenizer)
        if getattr(tokenizer, 'chat_template', None) is None:
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        return tokenizer


class DeepSeekModelConfig(ModelConfig):
    """Configuration for DeepSeek models."""
    def __init__(self, use_thinking: Optional[bool] = None):
        # DeepSeek R1 models have thinking builtin without a flag, but we allow configuration
        super().__init__(use_thinking=use_thinking if use_thinking is not None else True)

    def get_max_new_tokens(self) -> int:
        return 8192 if self.use_thinking else 1024

    def get_lora_target_modules(self) -> List[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

class QwenModelConfig(ModelConfig):
    """Configuration for Qwen models."""
    def __init__(self, use_thinking: Optional[bool] = None):
        super().__init__(use_thinking=use_thinking if use_thinking is not None else True)

    def get_max_new_tokens(self) -> int:
        return 32768 if self.use_thinking else 1024

    def get_lora_target_modules(self) -> List[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]


class LlamaModelConfig(ModelConfig):
    """Configuration for Llama models."""
    def __init__(self, use_thinking: Optional[bool] = None):
        super().__init__(use_thinking=use_thinking if use_thinking is not None else False)

    def get_max_new_tokens(self) -> int:
        return 1024

    def get_lora_target_modules(self) -> List[str]:
        return [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

    def setup_tokenizer(self, tokenizer: Any) -> Any:
        tokenizer = super().setup_tokenizer(tokenizer)
        if getattr(tokenizer, 'chat_template', None) is None:
            # Standard Llama 3 Instruct chat template
            tokenizer.chat_template = (
                "{% set loop_messages = messages %}"
                "{% for message in loop_messages %}"
                "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
                "{% if loop.index0 == 0 %}"
                "{% set content = bos_token + content %}"
                "{% endif %}"
                "{{ content }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )
        return tokenizer


def get_model_config(model_id: str, use_thinking: Optional[bool] = None) -> ModelConfig:
    """Factory function to get the appropriate model configuration."""
    model_id_lower = model_id.lower()
    if "gemma" in model_id_lower:
        return GemmaModelConfig(use_thinking=use_thinking)
    elif "meta-llama/llama-3.1-8b" in model_id_lower:
        return LlamaModelConfig(use_thinking=use_thinking)
    elif "qwq" in model_id_lower or "qwen" in model_id_lower:
        # Check Qwen first because DeepSeek R1 Distill Qwen will match both "deepseek" and "qwen"
        # Since it is a Qwen architecture underneath, QwenModelConfig is appropriate for its chat template
        # However, for backwards compatibility with DeepSeek configs we will check DeepSeek first
        if "deepseek" in model_id_lower:
            return DeepSeekModelConfig(use_thinking=use_thinking)
        return QwenModelConfig(use_thinking=use_thinking)
    elif "deepseek" in model_id_lower:
        return DeepSeekModelConfig(use_thinking=use_thinking)
    else:
        # Fallback to a default config or raise an error
        # Assuming Gemma as the default fallback for existing codebase behavior
        print(f"Warning: No specific config found for {model_id}. Defaulting to Gemma config.")
        return GemmaModelConfig(use_thinking=use_thinking)
