"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import yaml

def load_config(config_path: str = "../configs/example.yaml") -> dict:
    """
    Load YAML configuration from the specified path.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file {config_path} not found")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config