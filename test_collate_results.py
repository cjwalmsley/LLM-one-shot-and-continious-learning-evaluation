import unittest
import os
import shutil
import tempfile
import json
import pandas as pd
from collate_results import collate_results

class TestCollateResults(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory structure
        self.test_dir = tempfile.mkdtemp()

        # Setup experiment 104
        self.exp_dir = os.path.join(self.test_dir, "104", "Qwen", "QwQ-32B", "nyc_squad_with_pretraining_commands_fold_1")
        self.eval_dir = os.path.join(self.exp_dir, "evaluation")
        self.test_subdir = os.path.join(self.exp_dir, "testing")

        os.makedirs(self.eval_dir, exist_ok=True)
        os.makedirs(self.test_subdir, exist_ok=True)

        # Mock evaluation JSON data
        eval_data = {
          "timestamp": "2026-07-06T15:07:13.158903",
          "device": "cuda",
          "model_path": "./experiments/104/Qwen/QwQ-32B/nyc_squad_with_pretraining_commands_fold_1",
          "dataset_path": "./experiments/104/data_sets/fold_1/nyc_squad_with_pretraining_commands_fold_1.jsonl",
          "total_samples": 366,
          "batch_size": 2,
          "base_model": {
            "exact_match": {
              "count": 0,
              "accuracy": "0.00%"
            },
            "fuzzy_match": {
              "count": 342,
              "accuracy": "93.44%"
            }
          },
          "finetuned_model": {
            "exact_match": {
              "count": 366,
              "accuracy": "100.00%"
            },
            "fuzzy_match": {
              "count": 366,
              "accuracy": "100.00%"
            }
          }
        }

        # Mock testing JSON data
        test_data = {
          "timestamp": "2026-07-06T23:22:33.457981",
          "device": "cuda",
          "total_test_samples": 123,
          "batch_size": 2,
          "mode_1_isolated": {
            "exact_match_accuracy": "97.56%",
            "fuzzy_match_accuracy": "100.00%"
          },
          "mode_2_1_continuous_base": {
            "exact_match_accuracy": "0.00%",
            "fuzzy_match_accuracy": "91.06%"
          },
          "mode_2_continuous": {
            "exact_match_accuracy": "66.67%",
            "fuzzy_match_accuracy": "95.93%"
          },
          "mode_3_parametric": {
            "exact_match_accuracy": "0.00%",
            "fuzzy_match_accuracy": "0.00%"
          },
          "mode_4_cpt": {
            "exact_match_accuracy": "7.32%",
            "fuzzy_match_accuracy": "8.13%"
          }
        }

        with open(os.path.join(self.eval_dir, "evaluation_summary.json"), "w") as f:
            json.dump(eval_data, f)

        with open(os.path.join(self.test_subdir, "test_summary.json"), "w") as f:
            json.dump(test_data, f)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_collate_results(self):
        # Run the function
        collate_results(self.test_dir)

        # Check if CSV was created
        csv_path = os.path.join(self.test_dir, "llm_test_results_all_from_hydra.csv")
        self.assertTrue(os.path.exists(csv_path), "CSV file was not created")

        # Load the CSV and verify contents
        df = pd.read_csv(csv_path)

        # Verify 1 row was created
        self.assertEqual(len(df), 1)

        row = df.iloc[0]

        # Check metadata
        self.assertEqual(str(row['experiment number']), "104")
        self.assertEqual(row['model'], "Qwen/QwQ-32B")
        self.assertEqual(row['fold number'], 1)
        self.assertEqual(row['path to results'], self.exp_dir)

        # Check some flattened data
        self.assertEqual(row['finetuned_model_exact_match_accuracy'], "100.00%")
        self.assertEqual(row['mode_2_continuous_exact_match_accuracy'], "66.67%")

        # Check overlapped keys
        self.assertIn("eval_timestamp", df.columns)
        self.assertIn("test_timestamp", df.columns)
        self.assertIn("eval_batch_size", df.columns)
        self.assertIn("test_batch_size", df.columns)

        # Check value of overlapping keys
        self.assertEqual(row['eval_batch_size'], 2)
        self.assertEqual(row['test_batch_size'], 2)
        self.assertEqual(row['eval_device'], 'cuda')
        self.assertEqual(row['test_device'], 'cuda')

if __name__ == '__main__':
    unittest.main()
