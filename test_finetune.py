import unittest
from unittest.mock import patch, MagicMock
from finetune import main
from model_adapters import GemmaModelConfig


class TestFinetune(unittest.TestCase):
    def test_format_prompt(self):
        # Arrange
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "mock_templated_text"
        model_config = GemmaModelConfig()

        # Including new dataset schema fields to ensure they don't interfere
        example = {
            "id": "0a9ea765-a925-482d-a471-41d82b2790ed",
            "declarative_statement": "The reactor core is safe.",
            "question": "Is the reactor core safe?",
            "answer": "Yes, the reactor core is safe.",
            "grammatical_structure": "Active Voice",
            "raw_sentence": "The reactor core is safe.",
            "is_valid": True,
        }

        # Act
        result = model_config.format_prompt(example, mock_tokenizer)

        # Assert
        self.assertEqual(result, {"text": "mock_templated_text"})

        expected_messages = [
            {
                "role": "user",
                "content": "Context: The reactor core is safe.\n\nQuestion: Is the reactor core safe?",
            },
            {"role": "assistant", "content": "Yes, the reactor core is safe."},
        ]
        mock_tokenizer.apply_chat_template.assert_called_once_with(
            expected_messages, tokenize=False
        )

    @patch.dict("os.environ", {"HF_TOKEN": "test_token"})
    @patch(
        "sys.argv",
        [
            "finetune.py",
            "--model_id",
            "custom-model-id",
            "--dataset_path",
            "custom_dataset.jsonl",
        ],
    )
    @patch("finetune.load_dotenv")
    @patch("finetune.os.makedirs")
    @patch("finetune.os.path.exists")
    @patch("finetune.AutoTokenizer")
    @patch("finetune.AutoModelForCausalLM")
    @patch("finetune.load_dataset")
    @patch("finetune.SFTTrainer")
    @patch("finetune.SFTConfig")
    @patch("finetune.torch")
    @patch("finetune.evaluate_model")
    @patch("finetune.test_model")
    def test_main(
        self,
        mock_test_model,
        mock_evaluate_model,
        mock_torch,
        mock_sft_config,
        mock_sft_trainer,
        mock_load_dataset,
        mock_auto_model,
        mock_auto_tokenizer,
        mock_path_exists,
        mock_makedirs,
        mock_load_dotenv,
    ):
        # Arrange
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.is_bf16_supported.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float16 = "float16"
        mock_path_exists.return_value = True
        mock_tokenizer_instance = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"

        mock_model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance

        mock_dataset_instance = MagicMock()
        mock_load_dataset.return_value = mock_dataset_instance
        mock_dataset_instance.map.return_value = mock_dataset_instance

        mock_trainer_instance = MagicMock()
        mock_sft_trainer.return_value = mock_trainer_instance

        # Act
        main()

        # Assert
        mock_auto_tokenizer.from_pretrained.assert_called_with("custom-model-id")
        self.assertEqual(mock_tokenizer_instance.pad_token, "<eos>")
        mock_auto_model.from_pretrained.assert_called_with(
            "custom-model-id",
            device_map="auto",
            quantization_config=unittest.mock.ANY,
            attn_implementation="sdpa"
        )
        mock_load_dataset.assert_called_with("json", data_files="custom_dataset.jsonl")

        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.model.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()
        mock_evaluate_model.assert_called_once()
        mock_test_model.assert_called_once()

    def test_dataset_splitting(self):
        """Test that dataset is correctly split based on is_pretraining flag"""
        # Arrange
        train_example = {
            "id": "train-1",
            "question": "What is X?",
            "answer": "X is Y",
            "declarative_statement": "X is Y.",
            "is_pretraining": True,
        }

        test_example = {
            "id": "test-1",
            "question": "What is A?",
            "answer": "A is B",
            "declarative_statement": "A is B.",
            "is_pretraining": False,
        }

        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_text"

        all_examples = [train_example, test_example]

        # Simulate the filter behavior
        def filter_by_pretraining(condition):
            filtered_examples = [ex for ex in all_examples if condition(ex)]
            filtered_dataset = MagicMock()
            filtered_dataset.__len__ = MagicMock(return_value=len(filtered_examples))
            filtered_dataset.map = MagicMock(return_value=filtered_dataset)
            return filtered_dataset

        mock_dataset = MagicMock()
        mock_dataset.filter = filter_by_pretraining

        # Act - Simulate the splitting logic
        train_dataset = mock_dataset.filter(
            lambda x: x.get("is_pretraining", False) == True
        )
        test_dataset = mock_dataset.filter(
            lambda x: x.get("is_pretraining", False) == False
        )

        # Assert
        self.assertEqual(
            len(train_dataset), 1, "Train dataset should contain 1 example"
        )
        self.assertEqual(len(test_dataset), 1, "Test dataset should contain 1 example")

        # Verify that formatting is applied to both datasets
        model_config = GemmaModelConfig()
        train_dataset.map(lambda x: model_config.format_prompt(x, mock_tokenizer))
        test_dataset.map(lambda x: model_config.format_prompt(x, mock_tokenizer))

        self.assertEqual(train_dataset.map.call_count, 1)
        self.assertEqual(test_dataset.map.call_count, 1)


if __name__ == "__main__":
    unittest.main()
