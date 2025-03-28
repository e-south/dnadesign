"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/config.py

Loads and validates the libshuffle configuration from a YAML file.


Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import yaml
from pathlib import Path

class LibShuffleConfig:
    def __init__(self, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with config_path.open("r") as f:
            config = yaml.safe_load(f)
        if "libshuffle" not in config:
            raise ValueError("libshuffle configuration key not found in config file.")
        self.config = config["libshuffle"]

    def get(self, key, default=None):
        return self.config.get(key, default)
