"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/evoinference/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os

import yaml
from logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """
    Load and validate the configuration from a YAML file.
    Expects the YAML to have an 'evoinference' key.
    """
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if "evoinference" not in config:
        raise ValueError("Configuration file missing 'evoinference' section.")
    # Additional validations can be added here.
    return config["evoinference"]
