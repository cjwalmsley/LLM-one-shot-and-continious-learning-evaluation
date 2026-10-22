import unittest
import os
import tempfile
import importlib
from pathlib import Path
from unittest.mock import patch

class TestPaths(unittest.TestCase):
    def test_default_repo_root(self):
        # We need to reload the module to ensure it reads the state cleanly
        # We mock CONFIG_PATH to point to a non-existent file so no override happens
        with patch('paths.CONFIG_PATH', Path('/non/existent/path/configs/default.yaml')):
            import paths
            importlib.reload(paths)

            # By default, REPO_ROOT should be the directory containing paths.py
            expected_root = Path(paths.__file__).resolve().parent
            self.assertEqual(paths.REPO_ROOT, expected_root)

    def test_override_repo_root(self):
        # Create a temporary directory to act as the override path
        temp_override_dir = tempfile.mkdtemp()
        dummy_config_dir = tempfile.mkdtemp()

        try:
            configs_dir = Path(dummy_config_dir) / "configs"
            configs_dir.mkdir()
            config_file = configs_dir / "default.yaml"

            with open(config_file, "w") as f:
                f.write(f"repo_root: '{temp_override_dir}'\n")

            # Patch CONFIG_PATH in the paths module *before* evaluating the module logic
            # However, since the logic runs at import time, we can mock `Path.exists` and `open`
            # or just write to a temporary configs/default.yaml in the actual repo.
            # But writing to the actual repo during tests can be problematic.

            # Instead, we can read the module code and exec it in a namespace,
            # or mock REPO_ROOT during importlib.reload.

            with patch('paths.REPO_ROOT', Path(dummy_config_dir)):
                import paths
                # When paths is imported, it resolves REPO_ROOT based on __file__.
                # We can mock Path.resolve and Path.exists.

                with patch('pathlib.Path.exists', return_value=True):
                    with patch('builtins.open', unittest.mock.mock_open(read_data=f"repo_root: '{temp_override_dir}'\n")):
                        importlib.reload(paths)

                        expected_override = Path(temp_override_dir).resolve()
                        self.assertEqual(paths.REPO_ROOT, expected_override)

        finally:
            import shutil
            shutil.rmtree(temp_override_dir)
            shutil.rmtree(dummy_config_dir)

if __name__ == '__main__':
    unittest.main()
