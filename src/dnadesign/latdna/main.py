"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
from pathlib import Path

import yaml

from dnadesign.latdna import analysis, generation


def load_config(config_path: Path):
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Resolve config path (assumes config is at project_root/configs/example.yaml)
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"
    logging.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    latdna_config = config.get("latdna")
    if not latdna_config:
        raise ValueError("Missing 'latdna' section in configuration file.")

    mode = latdna_config.get("mode")
    if mode == "generation":
        logging.info("Running in GENERATION mode.")
        generation.run_generation_pipeline(latdna_config)
    elif mode == "analysis":
        logging.info("Running in ANALYSIS mode.")
        analysis.run_analysis_pipeline(latdna_config)
    else:
        raise ValueError("latdna.mode must be either 'generation' or 'analysis'.")


if __name__ == "__main__":
    main()
