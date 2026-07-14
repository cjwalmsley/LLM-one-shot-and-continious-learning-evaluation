import unittest
from unittest.mock import MagicMock
from model_adapters import (
    get_model_config,
    GemmaModelConfig,
    DeepSeekModelConfig,
    QwenModelConfig,
    LlamaModelConfig,
    LFMModelConfig,
)


class TestModelAdapters(unittest.TestCase):
    def test_get_gemma_config(self):
        config = get_model_config("google/gemma-4-E2B")
        self.assertIsInstance(config, GemmaModelConfig)
        self.assertFalse(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 50)

    def test_get_deepseek_config(self):
        config = get_model_config("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        self.assertIsInstance(config, DeepSeekModelConfig)
        self.assertFalse(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 150)

    def test_get_deepseek_distill_qwen_config(self):
        config = get_model_config("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        self.assertIsInstance(config, DeepSeekModelConfig)

    def test_get_qwq_config(self):
        config = get_model_config("Qwen/QwQ-32B")
        self.assertIsInstance(config, QwenModelConfig)
        self.assertFalse(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 150)

        # Test overriding use_thinking to True
        config_think = get_model_config("Qwen/QwQ-32B", use_thinking=True)
        self.assertTrue(config_think.use_thinking)
        self.assertEqual(config_think.get_max_new_tokens(), 8192)

    def test_get_llama_config(self):
        config = get_model_config("meta-llama/Llama-3.1-8B")
        self.assertIsInstance(config, LlamaModelConfig)
        self.assertFalse(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 150)

        # Test target modules exist
        self.assertIn("q_proj", config.get_lora_target_modules())

        # Test tokenizer setup sets default chat template
        mock_tokenizer = MagicMock()
        mock_tokenizer.chat_template = None
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<|eot_id|>"
        config.setup_tokenizer(mock_tokenizer)
        self.assertIsNotNone(mock_tokenizer.chat_template)
        self.assertIn("<|start_header_id|>", mock_tokenizer.chat_template)

    def test_get_lfm_config(self):
        config = get_model_config("LiquidAI/LFM2-24B-A2B")
        self.assertIsInstance(config, LFMModelConfig)
        self.assertFalse(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 1024)
        target_modules = config.get_lora_target_modules()
        self.assertIn("q_proj", target_modules)
        self.assertIn("in_proj", target_modules)
        self.assertIn("w1", target_modules)


if __name__ == "__main__":
    unittest.main()
