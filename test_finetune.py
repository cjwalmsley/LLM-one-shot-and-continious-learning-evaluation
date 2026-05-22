import unittest
from unittest.mock import patch, MagicMock
from finetune import format_prompt, main


class TestFinetune(unittest.TestCase):
    def test_format_prompt(self):
        # Arrange
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "mock_templated_text"

        # Including new dataset schema fields to ensure they don't interfere
        example = {
            'id': '0a9ea765-a925-482d-a471-41d82b2790ed',
            'declarative_statement': 'The reactor core is safe.',
            'question': 'Is the reactor core safe?',
            'answer': 'Yes, the reactor core is safe.',
            'grammatical_structure': 'Active Voice',
            'raw_sentence': 'The reactor core is safe.',
            'is_valid': True
        }

        # Act
        result = format_prompt(example, mock_tokenizer)

        # Assert
        self.assertEqual(result, {"text": "mock_templated_text"})

        expected_messages = [
            {
                "role": "user",
                "content": "Context: The reactor core is safe.\n\nQuestion: Is the reactor core safe?"
            },
            {
                "role": "assistant",
                "content": "Yes, the reactor core is safe."
            }
        ]
        mock_tokenizer.apply_chat_template.assert_called_once_with(expected_messages, tokenize=False)

    @patch.dict('os.environ', {'HF_TOKEN': 'test_token'})
    @patch('sys.argv', ['finetune.py', '--model_id', 'custom-model-id', '--dataset_path', 'custom_dataset.jsonl'])
    @patch('finetune.load_dotenv')
    @patch('finetune.os.makedirs')
    @patch('finetune.os.path.exists')
    @patch('finetune.AutoTokenizer')
    @patch('finetune.AutoModelForCausalLM')
    @patch('finetune.prepare_model_for_kbit_training')
    @patch('finetune.load_dataset')
    @patch('finetune.SFTTrainer')
    @patch('finetune.SFTConfig')
    def test_main(self, mock_sft_config, mock_sft_trainer, mock_load_dataset,
                  mock_prepare_model, mock_auto_model, mock_auto_tokenizer, mock_path_exists, mock_makedirs, mock_load_dotenv):
        # Arrange
        mock_path_exists.return_value = True
        mock_tokenizer_instance = MagicMock()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"

        mock_model_instance = MagicMock()
        mock_auto_model.from_pretrained.return_value = mock_model_instance
        mock_prepare_model.return_value = mock_model_instance

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
        mock_auto_model.from_pretrained.assert_called_with("custom-model-id", device_map="auto")
        mock_load_dataset.assert_called_with("json", data_files="custom_dataset.jsonl", split="train")

        mock_trainer_instance.train.assert_called_once()
        mock_trainer_instance.model.save_pretrained.assert_called_once()
        mock_tokenizer_instance.save_pretrained.assert_called_once()


if __name__ == '__main__':
    unittest.main()
