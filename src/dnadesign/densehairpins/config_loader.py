"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/congif_loader.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = Path(config_path)  # Use provided path directly.
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {self.config_path} does not exist.")
        self.config = self._load_config()

    def _load_config(self):
        with self.config_path.open("r") as f:
            config = yaml.safe_load(f)
        loaded_config = config.get("densehairpins", {})
        assert isinstance(loaded_config, dict), "Configuration for 'densehairpins' must be a dictionary."
        return loaded_config

    def get(self, key, default=None):
        return self.config.get(key, default)


