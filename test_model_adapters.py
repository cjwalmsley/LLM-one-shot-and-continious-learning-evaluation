import unittest
from unittest.mock import MagicMock
from model_adapters import get_model_config, GemmaModelConfig, DeepSeekModelConfig, QwenModelConfig

class TestModelAdapters(unittest.TestCase):
    def test_get_gemma_config(self):
        config = get_model_config("google/gemma-4-E2B")
        self.assertIsInstance(config, GemmaModelConfig)
        self.assertFalse(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 50)

    def test_get_deepseek_config(self):
        config = get_model_config("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
        self.assertIsInstance(config, DeepSeekModelConfig)
        self.assertTrue(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 8192)

    def test_get_deepseek_distill_qwen_config(self):
        config = get_model_config("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
        self.assertIsInstance(config, DeepSeekModelConfig)

    def test_get_qwq_config(self):
        config = get_model_config("Qwen/QwQ-32B")
        self.assertIsInstance(config, QwenModelConfig)
        self.assertTrue(config.use_thinking)
        self.assertEqual(config.get_max_new_tokens(), 32768)

        # Test overriding use_thinking
        config_no_think = get_model_config("Qwen/QwQ-32B", use_thinking=False)
        self.assertFalse(config_no_think.use_thinking)
        self.assertEqual(config_no_think.get_max_new_tokens(), 1024)

if __name__ == "__main__":
    unittest.main()
