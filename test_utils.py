import unittest
from unittest.mock import MagicMock, patch
import torch

from utils import (
    get_device,
    load_tokenizer_and_model,
    generate_batch_responses,
    generate_single_response,
    exact_match,
    fuzzy_match,
)


class TestGetDevice(unittest.TestCase):
    def test_get_device_prefers_cuda(self):
        with patch("utils.torch.cuda.is_available", return_value=True), \
             patch("utils.torch.cuda.get_device_name", return_value="NVIDIA A100"), \
             patch("utils.torch.backends.mps.is_available", return_value=False):
            self.assertEqual(get_device(), "cuda")

    def test_get_device_uses_mps_when_no_cuda(self):
        with patch("utils.torch.cuda.is_available", return_value=False), \
             patch("utils.torch.backends.mps.is_available", return_value=True):
            self.assertEqual(get_device(), "mps")

    def test_get_device_falls_back_to_cpu(self):
        with patch("utils.torch.cuda.is_available", return_value=False), \
             patch("utils.torch.backends.mps.is_available", return_value=False):
            self.assertEqual(get_device(), "cpu")


class TestLoadTokenizerAndModel(unittest.TestCase):
    @patch("utils.PeftModel")
    @patch("utils.AutoModelForCausalLM")
    @patch("utils.AutoTokenizer")
    def test_load_tokenizer_and_model_cpu(self, mock_tokenizer_cls, mock_model_cls, mock_peft_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "eos"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_base_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_base_model

        mock_peft = MagicMock()
        mock_peft_cls.from_pretrained.return_value = mock_peft

        tokenizer, model = load_tokenizer_and_model("/tmp/model", "base-id", "cpu")

        self.assertIs(tokenizer, mock_tokenizer)
        self.assertEqual(tokenizer.pad_token, "eos")
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertIs(model, mock_peft)

        mock_model_cls.from_pretrained.assert_called_with(
            "base-id",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        mock_peft.eval.assert_called_once_with()

    @patch("utils.PeftModel")
    @patch("utils.AutoModelForCausalLM")
    @patch("utils.AutoTokenizer")
    def test_load_tokenizer_and_model_trainable(self, mock_tokenizer_cls, mock_model_cls, mock_peft_cls):
        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_base_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_base_model

        mock_peft = MagicMock()

        param_mock1 = MagicMock()
        param_mock2 = MagicMock()
        mock_peft.named_parameters.return_value = [
            ("lora_A.weight", param_mock1),
            ("base_weight", param_mock2)
        ]
        mock_peft_cls.from_pretrained.return_value = mock_peft

        tokenizer, model = load_tokenizer_and_model("/tmp/model", "base-id", "cuda", is_trainable=True)

        self.assertTrue(param_mock1.requires_grad)
        mock_peft.eval.assert_not_called()


class TestGenerateResponse(unittest.TestCase):
    def test_generate_batch_responses_returns_decoded_text(self):
        tokenizer = MagicMock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        tokenizer.batch_decode.return_value = ["final answer"]
        tokenizer.eos_token_id = 2

        model = MagicMock()
        model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])

        response = generate_batch_responses(
            model,
            tokenizer,
            messages_batch=[[{"role": "user", "content": "Question?"}]],
            device="cpu",
            max_new_tokens=16,
        )

        self.assertEqual(response, ["final answer"])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "Question?"}],
            tokenize=False,
            add_generation_prompt=True,
        )

    @patch("utils.generate_batch_responses")
    def test_generate_single_response_wraps_batching(self, mock_generate_batch):
        mock_generate_batch.return_value = ["single answer"]

        model = MagicMock()
        tokenizer = MagicMock()
        messages = [{"role": "user", "content": "Question?"}]

        response = generate_single_response(
            model, tokenizer, messages, "cpu", max_new_tokens=16
        )

        self.assertEqual(response, "single answer")
        mock_generate_batch.assert_called_once_with(
            model, tokenizer, [messages], "cpu", 16
        )


class TestExactMatch(unittest.TestCase):
    def test_exact_match_identical_strings(self):
        self.assertTrue(exact_match("yes", "yes"))

    def test_exact_match_is_case_insensitive(self):
        self.assertTrue(exact_match("YES", "yes"))

    def test_exact_match_trims_whitespace(self):
        self.assertTrue(exact_match("  yes  ", "yes"))

    def test_exact_match_returns_false_for_different_strings(self):
        self.assertFalse(exact_match("yes", "no"))


class TestFuzzyMatch(unittest.TestCase):
    def test_fuzzy_match_identical_strings(self):
        self.assertTrue(fuzzy_match("reactor core safe", "reactor core safe"))

    def test_fuzzy_match_similar_strings(self):
        self.assertTrue(fuzzy_match("the reactor core is safe", "reactor core is safe"))

    def test_fuzzy_match_returns_false_for_unrelated_strings(self):
        self.assertFalse(fuzzy_match("apple", "banana"))

    def test_fuzzy_match_threshold_boundary(self):
        self.assertTrue(fuzzy_match("the", "the"))


if __name__ == "__main__":
    unittest.main()
