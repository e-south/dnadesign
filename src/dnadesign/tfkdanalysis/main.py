"""
--------------------------------------------------------------------------------
<dnadesign project>

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Import analysis functions
from analysis import (
    compute_histogram_xlim,
    export_regulator_csvs,  # New CSV export function
    generate_regulator_scatter_plot,
    generate_volcano_plot,
)

# Import the load_dataset function from utils.py (assumed to be in the parent directory)
from dnadesign.utils import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} not found.")
    with config_path.open("r") as f:
        config_all = yaml.safe_load(f)
    # We assume that 'tfkdanalysis' is the key for our project in the YAML.
    config = config_all.get("tfkdanalysis", {})
    if "batch_name" not in config:
        raise ValueError("The configuration must contain a 'batch_name'.")
    return config


def main():
    # Since main.py is in tfkdanalysis, batch_results is created as a subdirectory.
    project_root = Path(__file__).resolve().parent
    config_path = project_root.parent / "configs" / "example.yaml"
    output_base = project_root / "batch_results"  # now inside tfkdanalysis

    config = load_config(config_path)

    # Load the datasets with expected parameters
    try:
        logging.info("Loading han_et_al_known_interactions dataset...")
        known_df = load_dataset("han_et_al_known_interactions", sheet_name="Supplementary Data 7", header=1)
        expected_known_columns = [
            "tf_gene",
            "operon",
            "knownEffect",
            "FC_Glu",
            "class_Glu",
            "FC_LB",
            "class_LB",
            "FC_Gly",
            "class_Gly",
        ]
        for col in expected_known_columns:
            if col not in known_df.columns:
                raise ValueError(f"Expected column '{col}' not found in han_et_al_known_interactions dataset.")

        logging.info("Loading han_et_al_pptp_seq_interactions dataset...")
        pptp_df = load_dataset("han_et_al_pptp_seq_interactions", sheet_name="Supplementary Data 6", header=1)
        expected_pptp_columns = [
            "tf_gene",
            "operon",
            "FC_Glu",
            "-logP_Glu",
            "class_Glu",
            "FC_LB",
            "-logP_LB",
            "class_LB",
            "FC_Gly",
            "-logP_Gly",
            "class_Gly",
            "gene expression effect",
            "known TFBS",
            "DAP Fold change (peak intensity/background intensity)",
            "DAP_seq",
            "gSELEX peak intensity/maximum peak intensity %",
            "gSELEX",
            "ChIP_seq",
            "ChIP_exo",
            "Total number of evidence",
            "known Interaction? (YES OR NO)",
        ]
        for col in expected_pptp_columns:
            if col not in pptp_df.columns:
                raise ValueError(f"Expected column '{col}' not found in han_et_al_pptp_seq_interactions dataset.")
    except Exception as e:
        logging.error(e)
        sys.exit(1)

    # Create output directories with a batch-specific structure.
    timestamp = datetime.now().strftime("%Y%m%d")
    regulators_str = "_".join(config.get("regulators", [])) if config.get("regulators") else "all"
    batch_dir_name = f"{timestamp}_{config['batch_name']}_{regulators_str}"
    batch_dir = output_base / batch_dir_name
    plots_dir = batch_dir / "plots"
    csvs_dir = batch_dir / "csvs"
    plots_dir.mkdir(parents=True, exist_ok=True)
    csvs_dir.mkdir(parents=True, exist_ok=True)

    # Determine media-specific column names (e.g., 'glu' → FC_Glu and -logP_Glu)
    media = config.get("media", "glu").lower()
    fc_column = f"FC_{media.capitalize()}" if f"FC_{media.capitalize()}" in pptp_df.columns else f"FC_{media}"
    logp_column = (
        f"-logP_{media.capitalize()}" if f"-logP_{media.capitalize()}" in pptp_df.columns else f"-logP_{media}"
    )

    # Generate volcano plot if enabled in the configuration
    if config.get("volcano_plot", False):
        logging.info("Generating volcano plot...")
        volcano_df = generate_volcano_plot(
            df=pptp_df,
            fc_column=fc_column,
            logp_column=logp_column,
            config=config,
            output_path=plots_dir / "volcano_plot.png",
        )
        # Export CSV files for each user-defined regulator.
        export_regulator_csvs(
            df=volcano_df,
            regulators=config.get("regulators", []),
            fc_column=fc_column,
            threshold=float(config.get("threshold", 1.5)),
            output_dir=csvs_dir,
        )

    # Generate a combined regulator scatter plot
    regulators = config.get("regulators", [])
    if regulators:
        logging.info("Generating regulator scatter plot...")
        global_xlim = compute_histogram_xlim(pptp_df, fc_column, regulators)
        scatter_output_path = plots_dir / "regulator_scatter.png"
        generate_regulator_scatter_plot(
            df=pptp_df,
            regulators=regulators,
            fc_column=fc_column,
            global_xlim=global_xlim,
            config=config,
            output_path=scatter_output_path,
        )

    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()
