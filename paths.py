from pathlib import Path
import yaml
import os

# Resolves to the directory containing this file (repository root)
REPO_ROOT = Path(__file__).resolve().parent

# Check for configs/default.yaml to see if there is an override
CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"

if CONFIG_PATH.exists():
    try:
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            if config and isinstance(config, dict) and "repo_root" in config:
                override_path = config["repo_root"]
                if override_path:
                    # Expand user (~) in case they provided a path with it
                    REPO_ROOT = Path(os.path.expanduser(override_path)).resolve()
    except Exception as e:
        print(f"Warning: Failed to parse {CONFIG_PATH}: {e}")
