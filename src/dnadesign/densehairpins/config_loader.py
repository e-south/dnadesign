"""
--------------------------------------------------------------------------------
<dnadesign project>
/densehairpins/config_loader.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import yaml
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file {self.config_path} does not exist.")
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self):
        with self.config_path.open("r") as f:
            config = yaml.safe_load(f)
        loaded_config = config.get("densehairpins", {})
        assert isinstance(loaded_config, dict), "Configuration for 'densehairpins' must be a dictionary."
        return loaded_config

    def _validate_config(self):
        """Validate configuration parameters for consistency."""
        # Validate analysis_style if present
        if "analysis_style" in self.config:
            style = self.config["analysis_style"]
            if style not in ["per_batch", "aggregate"]:
                raise ValueError(f"Invalid analysis_style: {style}. Must be 'per_batch' or 'aggregate'.")
            
            # If per_batch and run_post_solve is true, ensure target_batch is specified
            if style == "per_batch" and self.config.get("run_post_solve", False) and "target_batch" not in self.config:
                raise ValueError("When analysis_style is 'per_batch' with run_post_solve=true, target_batch must be specified.")
                
        # Validate score_weights
        if "score_weights" in self.config:
            weights = self.config["score_weights"]
            if not isinstance(weights, dict):
                raise ValueError("score_weights must be a dictionary.")
            
            # Validate tf_diversity weight is present when analysis_style is aggregate
            if self.config.get("analysis_style") == "aggregate" and "tf_diversity" not in weights:
                print("WARNING: Using 'aggregate' analysis without 'tf_diversity' weight in score_weights.")
                print("Consider adding 'tf_diversity' weight for better diversity scoring.")

    def get(self, key, default=None):
        return self.config.get(key, default)
