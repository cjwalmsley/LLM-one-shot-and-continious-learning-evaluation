import unittest
from unittest.mock import MagicMock, patch
import json
import os
import tempfile
import test as test_module


class TestRunMode2(unittest.TestCase):
    def test_run_mode_2_records_results_and_pass2_questions(self):
        sample_records = [
            {
                "declarative_statement": "Context A",
                "question": "Question A?",
                "answer": "Answer A",
            },
            {
                "declarative_statement": "Context B",
                "question": "Question B?",
                "answer": "Answer B",
            },
        ]
        model = MagicMock()
        tokenizer = MagicMock()

        model_config = MagicMock()
        model_config.get_max_new_tokens.return_value = 50

        with patch("test.generate_single_response") as mock_generate:
            mock_generate.side_effect = ["ok", "ok", "Answer A", "Answer B"]
            results, exact_count, fuzzy_count = test_module.run_mode_2(
                model, tokenizer, sample_records, "cpu", model_config
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(exact_count, 2)
        self.assertEqual(fuzzy_count, 2)
        self.assertEqual(results[0]["mode"], 2)
        self.assertEqual(results[1]["predicted"], "Answer B")

        self.assertEqual(mock_generate.call_args_list[0].kwargs["max_new_tokens"], 50)
        self.assertEqual(mock_generate.call_args_list[1].kwargs["max_new_tokens"], 50)
        self.assertEqual(mock_generate.call_args_list[2].kwargs["max_new_tokens"], 50)
        self.assertEqual(mock_generate.call_args_list[3].kwargs["max_new_tokens"], 50)


class TestRunMode2_1(unittest.TestCase):
    def test_run_mode_2_1_records_results_and_pass2_questions(self):
        sample_records = [
            {
                "declarative_statement": "Context A",
                "question": "Question A?",
                "answer": "Answer A",
            },
            {
                "declarative_statement": "Context B",
                "question": "Question B?",
                "answer": "Answer B",
            },
        ]
        model = MagicMock()
        tokenizer = MagicMock()

        model_config = MagicMock()
        model_config.get_max_new_tokens.return_value = 50

        with patch("test.generate_single_response") as mock_generate:
            mock_generate.side_effect = ["ok", "ok", "Answer A", "Answer B"]
            results, exact_count, fuzzy_count = test_module.run_mode_2_1(
                model, tokenizer, sample_records, "cpu", model_config
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(exact_count, 2)
        self.assertEqual(fuzzy_count, 2)
        self.assertEqual(results[0]["mode"], 2.1)
        self.assertEqual(results[1]["predicted"], "Answer B")

        self.assertEqual(mock_generate.call_args_list[0].kwargs["max_new_tokens"], 50)
        self.assertEqual(mock_generate.call_args_list[1].kwargs["max_new_tokens"], 50)
        self.assertEqual(mock_generate.call_args_list[2].kwargs["max_new_tokens"], 50)
        self.assertEqual(mock_generate.call_args_list[3].kwargs["max_new_tokens"], 50)


class TestRunMode4(unittest.TestCase):
    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    @patch("test.generate_batch_responses")
    @patch("test.SFTTrainer")
    @patch("test.SFTConfig")
    @patch("test.AutoModelForCausalLM")
    @patch("test.AutoTokenizer")
    @patch("test.PeftModel")
    def test_run_mode_4_returns_results_and_counts(
        self,
        mock_peft_model_cls,
        mock_tokenizer_cls,
        mock_base_model_cls,
        mock_sft_config_cls,
        mock_trainer_cls,
        mock_generate_batch,
        mock_lora_config_cls,
        mock_get_peft_model,
    ):
        test_dataset = [
            {
                "declarative_statement": "Context A",
                "question": "Question A?",
                "answer": "Answer A",
            },
            {
                "declarative_statement": "Context B",
                "question": "Question B?",
                "answer": "Answer B",
            },
        ]

        class FakeDataset:
            def __init__(self, records):
                self._records = records

            def __len__(self):
                return len(self._records)

            def map(self, _):
                return self

            def __getitem__(self, index):
                if isinstance(index, slice):
                    chunk = self._records[index]
                    return {
                        "question": [item.get("question") for item in chunk],
                        "answer": [item.get("answer") for item in chunk],
                    }
                return self._records[index]

        mapped_dataset = FakeDataset(test_dataset)

        mock_tokenizer = MagicMock()
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer

        mock_base_model = MagicMock()
        mock_base_model.named_modules.return_value = [
            ("q_proj.linear", test_module.torch.nn.Linear(4, 4))
        ]
        mock_base_model_cls.from_pretrained.return_value = mock_base_model

        mock_trainer = MagicMock()
        mock_trainer.model.merge_and_unload.return_value = MagicMock()
        mock_trainer_cls.return_value = mock_trainer

        mock_peft_model = MagicMock()
        mock_peft_model_cls.from_pretrained.return_value = mock_peft_model

        trainable_param = MagicMock()
        trainable_param.requires_grad = True
        trainable_param.numel.return_value = 128
        mock_cpt_model = MagicMock()
        mock_cpt_model.parameters.return_value = [trainable_param]
        mock_get_peft_model.return_value = mock_cpt_model

        mock_generate_batch.return_value = ["Answer A", "Answer B"]

        results, exact_count, fuzzy_count = test_module.run_mode_4(
            "/tmp/final-adapter",
            "/tmp/data.jsonl",
            "/tmp/out",
            "google/gemma-4-E2B",
            "cpu",
            mapped_dataset,
            batch_size=2,
            use_thinking=None,
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(exact_count, 2)
        self.assertEqual(fuzzy_count, 2)
        self.assertEqual(results[0]["mode"], 4)
        self.assertEqual(results[1]["predicted"], "Answer B")
        expected_messages = [
            [{"role": "user", "content": "Question: Question A?"}],
            [{"role": "user", "content": "Question: Question B?"}],
        ]
        mock_generate_batch.assert_called_once_with(
            mock_peft_model, mock_tokenizer, expected_messages, "cpu", max_new_tokens=50
        )


class TestTestModel(unittest.TestCase):
    @patch("test.get_device", return_value="cpu")
    @patch("test.load_tokenizer_and_model")
    @patch("test.load_dataset")
    def test_model_returns_none_when_no_samples(
        self, mock_load_dataset, mock_load_model, mock_device
    ):
        mock_load_model.return_value = (MagicMock(), MagicMock())
        mock_dataset = MagicMock()
        mock_dataset_train = MagicMock()
        mock_dataset.__getitem__.return_value = mock_dataset_train
        mock_load_dataset.return_value = mock_dataset
        mock_dataset_train.filter.return_value = []

        result = test_module.test_model("/tmp/model", "/tmp/data.jsonl", "/tmp/out")

        self.assertIsNone(result)
        mock_load_model.assert_not_called()

    @patch("test.run_mode_4")
    @patch("test.run_mode_3")
    @patch("test.run_mode_2")
    @patch("test.run_mode_2_1")
    @patch("test.run_mode_1")
    @patch("test.get_device", return_value="cpu")
    @patch("test.load_tokenizer_and_model")
    @patch("test.load_dataset")
    def test_model_writes_outputs_and_summary(
        self,
        mock_load_dataset,
        mock_load_model,
        mock_device,
        mock_run_mode_1,
        mock_run_mode_2_1,
        mock_run_mode_2,
        mock_run_mode_3,
        mock_run_mode_4,
    ):
        mock_model = MagicMock()
        mock_model_config = MagicMock()
        mock_load_model.return_value = (MagicMock(), mock_model, mock_model_config)
        mock_dataset = MagicMock()
        mock_dataset_train = MagicMock()
        mock_dataset.__getitem__.return_value = mock_dataset_train
        mock_load_dataset.return_value = mock_dataset

        sample_records = [
            {
                "declarative_statement": "Context A",
                "question": "Question A?",
                "answer": "Answer A",
                "is_pretraining": False,
            },
            {
                "declarative_statement": "Context B",
                "question": "Question B?",
                "answer": "Answer B",
                "is_pretraining": False,
            },
        ]
        mock_dataset_train.filter.return_value = sample_records

        mock_run_mode_1.return_value = (
            [
                {
                    "sample_id": 0,
                    "mode": 1,
                    "ground_truth": "Answer A",
                    "predicted": "Answer A",
                    "exact_match": True,
                    "fuzzy_match": True,
                }
            ],
            1,
            1,
        )
        mock_run_mode_2_1.return_value = (
            [
                {
                    "sample_id": 1,
                    "mode": 2.1,
                    "ground_truth": "Answer B",
                    "predicted": "Answer B",
                    "exact_match": True,
                    "fuzzy_match": True,
                }
            ],
            1,
            1,
        )
        mock_run_mode_2.return_value = (
            [
                {
                    "sample_id": 2,
                    "mode": 2,
                    "ground_truth": "Answer C",
                    "predicted": "Answer C",
                    "exact_match": True,
                    "fuzzy_match": True,
                }
            ],
            1,
            1,
        )
        mock_run_mode_3.return_value = (
            [
                {
                    "sample_id": 3,
                    "mode": 3,
                    "ground_truth": "Answer D",
                    "predicted": "Answer D",
                    "exact_match": True,
                    "fuzzy_match": True,
                }
            ],
            1,
            1,
        )
        mock_run_mode_4.return_value = (
            [
                {
                    "sample_id": 4,
                    "mode": 4,
                    "ground_truth": "Answer E",
                    "predicted": "Answer E",
                    "exact_match": True,
                    "fuzzy_match": True,
                }
            ],
            1,
            1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = test_module.test_model(
                tmpdir, os.path.join(tmpdir, "data.jsonl"), tmpdir
            )

            self.assertIsNone(result)

            # Assert that the context manager was entered and exited
            mock_model.disable_adapter.return_value.__enter__.assert_called_once()
            mock_model.disable_adapter.return_value.__exit__.assert_called_once()

            # Ensure the mocked mode functions were called
            mock_run_mode_2_1.assert_called_once()
            mock_run_mode_3.assert_called_once()
            mock_run_mode_4.assert_called_once()

            results_file = os.path.join(tmpdir, "test_results.jsonl")
            summary_file = os.path.join(tmpdir, "test_summary.json")

            self.assertTrue(os.path.exists(results_file))
            self.assertTrue(os.path.exists(summary_file))

            with open(results_file, "r") as handle:
                results_lines = handle.readlines()

            # Update expected line count to 5 since we added Mode 2.1
            self.assertEqual(len(results_lines), 5)

            with open(summary_file, "r") as handle:
                saved_summary = json.load(handle)

            self.assertEqual(saved_summary["total_test_samples"], 2)
            self.assertEqual(
                saved_summary["mode_1_isolated"]["exact_match_accuracy"], "50.00%"
            )
            # Verify the new Mode 2.1 is properly written to the JSON summary
            self.assertEqual(
                saved_summary["mode_2_1_continuous_base"]["exact_match_accuracy"],
                "50.00%",
            )
            self.assertEqual(
                saved_summary["mode_2_continuous"]["fuzzy_match_accuracy"], "50.00%"
            )
            self.assertEqual(
                saved_summary["mode_4_cpt"]["exact_match_accuracy"], "50.00%"
            )


if __name__ == "__main__":
    unittest.main()
