"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/main.py

CLI entry point for the libshuffle pipeline.
Loads the configuration, resolves the input .pt file,
performs subsampling and metric computations (including CDS if selected),
generates a scatter plot, and writes global and sublibrary YAML output files.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
import time
import torch
import datetime
from pathlib import Path
import logging

from dnadesign.libshuffle.config import LibShuffleConfig
from dnadesign.libshuffle.subsampler import Subsampler
from dnadesign.libshuffle.visualization import plot_scatter, plot_kde_coremetrics, plot_pairplot_coremetrics
from dnadesign.libshuffle.output import save_summary, save_results, save_selected_subsamples
from dnadesign.libshuffle.metrics import apply_composite_transformation
from dnadesign.libshuffle.hitzone_visualization import plot_hitzone_details


def load_pt_file(input_dir: Path):
    pt_files = list(input_dir.glob("*.pt"))
    if len(pt_files) != 1:
        raise FileNotFoundError(f"Expected exactly one .pt file in {input_dir}, found {len(pt_files)}.")
    return pt_files[0]

def load_sequences(pt_file: Path):
    sequences = torch.load(pt_file, map_location=torch.device('cpu'))
    if not isinstance(sequences, list):
        raise ValueError("Loaded .pt file must contain a list of sequence entries.")
    return sequences

def create_output_directory(output_dir_prefix):
    # Creates output directory: libshuffle/batch_results/libshuffle_<timestamp>/
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    output_dir = Path(__file__).resolve().parent / "batch_results" / f"{output_dir_prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir, timestamp

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger = logging.getLogger("libshuffle")

    try:
        start_time = time.time()
        run_start_time = datetime.datetime.now()

        # Load configuration.
        config_path = Path(__file__).parent.parent / "configs" / "example.yaml"
        config_loader = LibShuffleConfig(config_path)
        config = config_loader.config

        # Resolve input .pt file.
        input_pt_dir = Path(__file__).parent.parent / config.get("input_pt_path")
        if not input_pt_dir.exists():
            logger.error(f"Input directory {input_pt_dir} does not exist.")
            sys.exit(1)
        pt_file = load_pt_file(input_pt_dir)
        logger.info(f"Using input .pt file: {pt_file}")

        sequences = load_sequences(pt_file)
        num_sequences = len(sequences)
        sequence_batch_id = input_pt_dir.name

        # Create output directory.
        output_dir, _ = create_output_directory(config.get("output_dir_prefix"))
        logger.info(f"Output directory created: {output_dir}")

        # Run subsampling.
        subsampler = Subsampler(sequences, config)
        subsamples = subsampler.run()

        # Apply composite transformation across all subsamples.
        # This updates each subsample's "billboard_metric" field (and adds "cds_components" if using CDS)
        subsamples, pca_model_info = apply_composite_transformation(subsamples, config)

        # Generate visualization.
        run_info = {
            "num_draws": config.get("num_draws", 200),
            "subsample_size": config.get("subsample_size", 100)
        }
        plot_path = plot_scatter(subsamples, config, output_dir, run_info)
        logger.info(f"Scatter plot saved to: {plot_path}")

        # Generate new diagnostic visualizations.
        kde_raw_path, kde_zscore_path = plot_kde_coremetrics(subsamples, config, output_dir)
        logger.info(f"KDE plots saved to: {kde_raw_path} and {kde_zscore_path}")
        pairplot_path = plot_pairplot_coremetrics(subsamples, config, output_dir)
        logger.info(f"Pairwise scatter plot saved to: {pairplot_path}")

        # New hit-zone detailed plot
        hitzone_path = plot_hitzone_details(subsamples, sequences, config, output_dir, run_info)
        logger.info(f"Hit‑zone details plot saved to: {hitzone_path}")

        # Save global summary and sublibrary details.
        duration = time.time() - start_time
        summary_path = save_summary(output_dir, config, pt_file, sequence_batch_id, num_sequences, run_start_time, duration)
        logger.info(f"Global summary saved to: {summary_path}")

        results_path = save_results(output_dir, subsamples, pca_model_info)
        logger.info(f"Sublibraries details saved to: {results_path}")

        # Optionally, save selected subsamples back to the sequences directory.
        if config.get("save_selected_pt", False):
            sequences_dir = input_pt_dir.parent
            saved_files = save_selected_subsamples(subsamples, config, sequences, sequences_dir)
            if saved_files:
                logger.info(f"Selected subsamples saved: {saved_files}")

        end_time = time.time()
        logger.info(f"Libshuffle completed in {end_time - start_time:.2f} seconds.")

    except Exception:
        logging.exception("An error occurred during libshuffle execution.")
        sys.exit(1)

if __name__ == "__main__":
    main()