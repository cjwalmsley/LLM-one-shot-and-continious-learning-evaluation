from abc import ABC, abstractmethod
from typing import Dict, Any, List

class ModelConfig(ABC):
    """Abstract base class for model configurations."""

    @abstractmethod
    def get_lora_target_modules(self) -> List[str]:
        """Return the target modules for LoRA."""
        pass

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

    def get_lora_target_modules(self) -> List[str]:
        return [
            "q_proj.linear",
            "k_proj.linear",
            "v_proj.linear",
            "o_proj.linear",
            "gate_proj.linear",
            "up_proj.linear",
            "down_proj.linear",
        ]

    def setup_tokenizer(self, tokenizer: Any) -> Any:
        tokenizer = super().setup_tokenizer(tokenizer)
        if getattr(tokenizer, 'chat_template', None) is None:
            tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
        return tokenizer


class DeepSeekModelConfig(ModelConfig):
    """Configuration for DeepSeek models."""

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

def get_model_config(model_id: str) -> ModelConfig:
    """Factory function to get the appropriate model configuration."""
    model_id_lower = model_id.lower()
    if "gemma" in model_id_lower:
        return GemmaModelConfig()
    elif "deepseek" in model_id_lower:
        return DeepSeekModelConfig()
    else:
        # Fallback to a default config or raise an error
        # Assuming Gemma as the default fallback for existing codebase behavior
        print(f"Warning: No specific config found for {model_id}. Defaulting to Gemma config.")
        return GemmaModelConfig()
