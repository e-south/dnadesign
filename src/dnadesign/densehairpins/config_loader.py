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
        self.config_path = Path(__file__).resolve().parent.parent / "configs" / "example.yaml"
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
        self.config = self._load_config()

    def _load_config(self):
        with self.config_path.open("r") as f:
            config = yaml.safe_load(f)
        # Expect configuration under the key "densehairpins"
        return config.get("densehairpins", {})

    def get(self, key, default=None):
        return self.config.get(key, default)

