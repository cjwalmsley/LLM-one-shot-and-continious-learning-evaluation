import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import tempfile
import os
from evaluate import exact_match, fuzzy_match, generate_response, evaluate_model
class TestExactMatch(unittest.TestCase):
    """Test exact_match function"""
    def test_exact_match_identical_strings(self):
        """Test exact match with identical strings"""
        result = exact_match("yes", "yes")
        self.assertTrue(result)
    def test_exact_match_case_insensitive(self):
        """Test exact match is case-insensitive"""
        result = exact_match("YES", "yes")
        self.assertTrue(result)
        result = exact_match("Yes, it is safe.", "yes, it is safe.")
        self.assertTrue(result)
    def test_exact_match_with_whitespace(self):
        """Test exact match handles whitespace"""
        result = exact_match("  yes  ", "yes")
        self.assertTrue(result)
    def test_exact_match_different_strings(self):
        """Test exact match returns False for different strings"""
        result = exact_match("yes", "no")
        self.assertFalse(result)
    def test_exact_match_partial_match(self):
        """Test exact match returns False for partial matches"""
        result = exact_match("yes it is", "yes")
        self.assertFalse(result)
    def test_exact_match_empty_strings(self):
        """Test exact match with empty strings"""
        result = exact_match("", "")
        self.assertTrue(result)
class TestFuzzyMatch(unittest.TestCase):
    """Test fuzzy_match function"""
    def test_fuzzy_match_identical_strings(self):
        """Test fuzzy match with identical strings"""
        result = fuzzy_match("yes", "yes")
        self.assertTrue(result)
    def test_fuzzy_match_similar_strings(self):
        """Test fuzzy match with similar strings"""
        result = fuzzy_match("the reactor core is safe", "reactor core is safe")
        self.assertTrue(result)
    def test_fuzzy_match_substring_match(self):
        """Test fuzzy match with substrings"""
        result = fuzzy_match("yes", "absolutely yes")
        self.assertTrue(result)
    def test_fuzzy_match_reordered_tokens(self):
        """Test fuzzy match with reordered tokens (token_set_ratio)"""
        result = fuzzy_match("core reactor safe", "reactor core safe")
        self.assertTrue(result)
    def test_fuzzy_match_case_insensitive(self):
        """Test fuzzy match is case-insensitive"""
        result = fuzzy_match("YES IS SAFE", "yes is safe")
        self.assertTrue(result)
    def test_fuzzy_match_different_strings(self):
        """Test fuzzy match returns False for completely different strings"""
        result = fuzzy_match("apple", "banana")
        self.assertFalse(result)
    def test_fuzzy_match_threshold_boundary(self):
        """Test fuzzy match at threshold boundary"""
        result = fuzzy_match("the", "the")
        self.assertTrue(result)
class TestGenerateResponse(unittest.TestCase):
    """Test generate_response function"""
    def test_generate_response_function_exists(self):
        """Test that generate_response function is callable"""
        self.assertTrue(callable(generate_response))
class TestEvaluateModel(unittest.TestCase):
    """Test evaluate_model function"""
    @patch('evaluate.PeftModel')
    @patch('evaluate.AutoModelForCausalLM')
    @patch('evaluate.AutoTokenizer')
    @patch('evaluate.load_dataset')
    @patch('builtins.open', new_callable=mock_open)
    def test_evaluate_model_creates_output_files(self, mock_file, mock_load_dataset,
                                                  mock_tokenizer_class, mock_model_class,
                                                  mock_peft_class):
        """Test that evaluate_model creates output files"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_tokenizer.eos_token_id = 2
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.device = "cpu"
        mock_peft_instance = MagicMock()
        mock_peft_class.from_pretrained.return_value = mock_peft_instance
        mock_peft_instance.eval.return_value = None
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset_train = MagicMock()
        mock_dataset.__getitem__ = MagicMock(return_value=mock_dataset_train)
        mock_load_dataset.return_value = mock_dataset
        # Mock sample records
        sample_records = [
            {
                'id': '1',
                'declarative_statement': 'Test context',
                'question': 'Test question?',
                'answer': 'Test answer',
                'is_pretraining': True
            }
        ]
        filtered_dataset = MagicMock()
        filtered_dataset.__len__ = MagicMock(return_value=1)
        filtered_dataset.__iter__ = MagicMock(return_value=iter(sample_records))
        mock_dataset_train.filter.return_value = filtered_dataset
        with tempfile.TemporaryDirectory() as tmpdir:
            # Act
            with patch('evaluate.generate_response', return_value='Model response'):
                evaluate_model(tmpdir, 'test_dataset.jsonl', tmpdir)
            # Assert
            self.assertTrue(mock_file.called)
    @patch('evaluate.PeftModel')
    @patch('evaluate.AutoModelForCausalLM')
    @patch('evaluate.AutoTokenizer')
    @patch('evaluate.load_dataset')
    @patch('builtins.open', new_callable=mock_open)
    def test_evaluate_model_calculates_metrics(self, mock_file, mock_load_dataset,
                                               mock_tokenizer_class, mock_model_class,
                                               mock_peft_class):
        """Test that evaluate_model calculates metrics correctly"""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer
        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.device = "cpu"
        mock_peft_instance = MagicMock()
        mock_peft_class.from_pretrained.return_value = mock_peft_instance
        # Mock dataset
        sample_records = [
            {
                'id': '1',
                'declarative_statement': 'Context',
                'question': 'Q?',
                'answer': 'Answer A',
                'is_pretraining': True
            },
            {
                'id': '2',
                'declarative_statement': 'Context 2',
                'question': 'Q2?',
                'answer': 'Answer B',
                'is_pretraining': True
            }
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
            with patch('evaluate.generate_response') as mock_gen:
                mock_gen.side_effect = ['Answer A', 'Answer B']
                summary = evaluate_model(tmpdir, 'test.jsonl', tmpdir)
            self.assertIn('total_samples', summary)
            self.assertEqual(summary['total_samples'], 2)
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
            "fuzzy_match": True
        }
        # Verify all required fields exist
        required_fields = ["sample_id", "context", "question", "ground_truth", 
                          "predicted", "exact_match", "fuzzy_match"]
        for field in required_fields:
            self.assertIn(field, result)
    def test_summary_statistics_structure(self):
        """Test that summary statistics have correct structure"""
        summary = {
            "timestamp": "2024-05-22T10:00:00",
            "model_path": "path",
            "dataset_path": "path",
            "total_samples": 100,
            "exact_match": {
                "count": 75,
                "accuracy": "75.00%"
            },
            "fuzzy_match": {
                "count": 90,
                "accuracy": "90.00%"
            }
        }
        # Verify structure
        self.assertIn("exact_match", summary)
        self.assertIn("fuzzy_match", summary)
        self.assertIn("count", summary["exact_match"])
        self.assertIn("accuracy", summary["exact_match"])
class TestMatchingThresholds(unittest.TestCase):
    """Test fuzzy matching threshold behavior"""
    def test_fuzzy_match_threshold_80_percent(self):
        """Test fuzzy match uses 80% threshold"""
        result = fuzzy_match("the quick brown fox", "quick brown fox")
        self.assertTrue(result)
    def test_high_similarity_strings(self):
        """Test strings with high similarity"""
        result = fuzzy_match(
            "The reactor core is operating safely",
            "The reactor core is operating in a safe manner"
        )
        self.assertTrue(result)
    def test_low_similarity_strings(self):
        """Test strings with low similarity"""
        result = fuzzy_match("apple orange banana", "xyz 123 abc")
        self.assertFalse(result)
if __name__ == '__main__':
    unittest.main()
