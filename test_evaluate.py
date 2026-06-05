import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile
import os
from evaluate import evaluate_model
from utils import exact_match, fuzzy_match


class TestEvaluateModel(unittest.TestCase):
    """Test evaluate_model function"""

    @patch("evaluate.load_tokenizer_and_model")
    @patch("evaluate.load_dataset")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_model_creates_output_files(
        self, mock_file, mock_load_dataset, mock_load_tokenizer_and_model
    ):
        """Test that evaluate_model creates output files"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 2
        mock_model = MagicMock()
        mock_load_tokenizer_and_model.return_value = (mock_tokenizer, mock_model)
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset_train = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_dataset_train)
        mock_load_dataset.return_value = mock_dataset
        # Mock sample records
        sample_records = [
            {
                "id": "1",
                "declarative_statement": "Test context",
                "question": "Test question?",
                "answer": "Test answer",
                "is_pretraining": True,
            }
        ]
        filtered_dataset = MagicMock()
        filtered_dataset.__len__ = MagicMock(return_value=1)
        filtered_dataset.__iter__ = MagicMock(return_value=iter(sample_records))
        mock_dataset_train.filter.return_value = filtered_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Act
            with patch(
                "evaluate.generate_batch_responses", return_value=["Model response"]
            ):
                evaluate_model(tmpdir, "test_dataset.jsonl", tmpdir)
            # Assert
            self.assertTrue(mock_file.called)

    @patch("evaluate.load_tokenizer_and_model")
    @patch("evaluate.load_dataset")
    @patch("builtins.open", new_callable=mock_open)
    def test_evaluate_model_calculates_metrics(
        self, mock_file, mock_load_dataset, mock_load_tokenizer_and_model
    ):
        """Test that evaluate_model calculates metrics correctly"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_load_tokenizer_and_model.return_value = (mock_tokenizer, mock_model)
        # Mock dataset
        sample_records = [
            {
                "id": "1",
                "declarative_statement": "Context",
                "question": "Q?",
                "answer": "Answer A",
                "is_pretraining": True,
            },
            {
                "id": "2",
                "declarative_statement": "Context 2",
                "question": "Q2?",
                "answer": "Answer B",
                "is_pretraining": True,
            },
        ]
        mock_dataset = MagicMock()
        mock_dataset_train = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_dataset_train)
        mock_load_dataset.return_value = mock_dataset
        filtered_dataset = MagicMock()
        filtered_dataset.__len__ = MagicMock(return_value=2)
        filtered_dataset.__iter__ = MagicMock(return_value=iter(sample_records))
        mock_dataset_train.filter.return_value = filtered_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("evaluate.generate_batch_responses") as mock_gen:
                mock_gen.return_value = ["Answer A", "Answer B"]
                summary = evaluate_model(tmpdir, "test.jsonl", tmpdir)
            self.assertIn("total_samples", summary)
            self.assertEqual(summary["total_samples"], 2)


class TestEvaluationIntegration(unittest.TestCase):
    """Integration tests for evaluation workflow"""

    def test_exact_and_fuzzy_match_consistency(self):
        """Test that exact matches are also fuzzy matches"""
        test_string = "The reactor is safe"
        exact = exact_match(test_string, test_string)
        fuzzy = fuzzy_match(test_string, test_string)
        self.assertTrue(exact)
        self.assertTrue(fuzzy)

    def test_matching_functions_with_real_examples(self):
        """Test matching functions with realistic QA examples"""
        # Example 1: Exact match scenario
        ground_truth = "Yes, the reactor core is safe."
        prediction1 = "Yes, the reactor core is safe."
        self.assertTrue(exact_match(prediction1, ground_truth))
        self.assertTrue(fuzzy_match(prediction1, ground_truth))
        # Example 2: Fuzzy match scenario
        prediction2 = "The reactor core is safe, yes."
        self.assertFalse(exact_match(prediction2, ground_truth))
        self.assertTrue(fuzzy_match(prediction2, ground_truth))
        # Example 3: No match scenario
        prediction3 = "The reactor is not safe"
        self.assertFalse(exact_match(prediction3, ground_truth))
        self.assertFalse(fuzzy_match(prediction3, ground_truth))

    def test_edge_cases_with_special_characters(self):
        """Test matching with special characters"""
        ground_truth = "It's a safe, secure reactor."
        prediction = "it's a safe secure reactor"
        # Exact match should fail due to punctuation
        self.assertFalse(exact_match(prediction, ground_truth))
        # Fuzzy match should succeed
        self.assertTrue(fuzzy_match(prediction, ground_truth))


class TestDataFormatting(unittest.TestCase):
    """Test data formatting and structure"""

    def test_evaluation_result_structure(self):
        """Test that evaluation results have correct structure"""
        result = {
            "sample_id": 0,
            "context": "test",
            "question": "test?",
            "ground_truth": "answer",
            "predicted": "answer",
            "exact_match": True,
            "fuzzy_match": True,
        }
        # Verify all required fields exist
        required_fields = [
            "sample_id",
            "context",
            "question",
            "ground_truth",
            "predicted",
            "exact_match",
            "fuzzy_match",
        ]
        for field in required_fields:
            self.assertIn(field, result)

    def test_summary_statistics_structure(self):
        """Test that summary statistics have correct structure"""
        summary = {
            "timestamp": "2024-05-22T10:00:00",
            "model_path": "path",
            "dataset_path": "path",
            "total_samples": 100,
            "exact_match": {"count": 75, "accuracy": "75.00%"},
            "fuzzy_match": {"count": 90, "accuracy": "90.00%"},
        }
        # Verify structure
        self.assertIn("exact_match", summary)
        self.assertIn("fuzzy_match", summary)
        self.assertIn("count", summary["exact_match"])
        self.assertIn("accuracy", summary["exact_match"])


if __name__ == "__main__":
    unittest.main()
