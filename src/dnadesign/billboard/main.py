"""
--------------------------------------------------------------------------------
<dnadesign project>
billboard/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import os
import datetime
import yaml
from pathlib import Path
import numpy as np

from dnadesign.billboard.core import process_sequences
from dnadesign.billboard.summary import generate_csv_summaries, generate_entropy_summary_csv, print_summary
from dnadesign.billboard.plot_helpers import (
    save_tf_frequency_barplot,
    save_occupancy_plot,
    save_motif_length_histogram,
    save_motif_stats_barplot
)

def main():
    """
    Entry point for analyzing sequence batches.
    
    Reads the YAML config (e.g., configs/example.yaml) to obtain pt_files (which may be directory names),
    processes the sequences, generates CSVs, and (if not in DRY_RUN mode) produces the following plots:
      - TF frequency bar plot (bars colored by positional entropy),
      - Combined occupancy heatmap (two subplots sharing a y-axis, sorted by frequency, with square cells and a common colorbar),
      - Motif length density plot,
      - Motif statistics bar plot (vertical subplots: pass ratio on top and average length on bottom with inverted axis).
      
    The output directory is named using an optional custom prefix plus today's date (YYYYMMDD).
    
    DRY_RUN Mode:
      - When enabled (dry_run: true), the program announces dry-run mode,
        skips plot generation, and generates only the entropy_summary.csv.
    """
    # Locate the configuration file.
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    billboard_config = config.get("billboard", {})

    # Read dry_run flag (default False)
    dry_run = billboard_config.get("dry_run", False)
    if dry_run:
        print("DRY RUN enabled: Only the entropy summary CSV will be generated. No plots will be produced.")

    # Process pt_files from the config.
    pt_files_config = billboard_config.get("pt_files", [])
    pt_paths = []
    for p in pt_files_config:
        path_candidate = Path(p)
        if not path_candidate.is_absolute():
            project_root = Path(__file__).parent.parent
            path_candidate = project_root / p
        if path_candidate.is_dir():
            pt_candidates = list(path_candidate.glob("*.pt"))
            if len(pt_candidates) != 1:
                raise ValueError(f"Expected exactly one .pt file in directory '{path_candidate}', found {len(pt_candidates)}")
            pt_paths.append(str(pt_candidates[0]))
        else:
            pt_paths.append(str(path_candidate))
    
    # Create output directory using custom prefix plus today's date.
    today = datetime.datetime.now().strftime("%Y%m%d")
    custom_prefix = billboard_config.get("output_dir_prefix", "").strip()
    if custom_prefix:
        out_dir_name = f"{custom_prefix}_{today}"
    else:
        out_dir_name = today
    output_dir = Path(__file__).parent / "batch_results" / out_dir_name
    os.makedirs(output_dir / "csvs", exist_ok=True)
    os.makedirs(output_dir / "plots", exist_ok=True)
    
    # Process sequences and compute metrics.
    results = process_sequences(pt_paths, billboard_config)
    
    if dry_run:
        # In DRY_RUN mode, only generate the entropy summary CSV.
        generate_entropy_summary_csv(results, str(output_dir))
    else:
        # Full run: generate all CSV summaries.
        generate_csv_summaries(results, str(output_dir))
    
        # Build dictionary for positional entropy.
        tf_entropy = {}
        for d in results["per_tf_entropy"]:
            tf = d["tf"]
            tf_entropy[tf] = (d["entropy_forward"] + d["entropy_reverse"]) / 2.0
        save_tf_frequency_barplot(
            results["tf_frequency"],
            tf_entropy,
            "TF Frequency Bar Plot",
            str(output_dir / "plots" / "tf_frequency_barplot.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(14,7)
        )
    
        # For occupancy plot: sort occupancy_tf_list by descending frequency.
        occupancy_tf_list = results["occupancy_tf_list"]
        occupancy_tf_list = sorted(occupancy_tf_list, key=lambda tf: results["tf_frequency"].get(tf, 0), reverse=True)
        order = [results["occupancy_tf_list"].index(tf) for tf in occupancy_tf_list]
        occ_forward = results["occupancy_forward_matrix"][order, :]
        occ_reverse = results["occupancy_reverse_matrix"][order, :]
    
        save_occupancy_plot(
            occ_forward,
            occ_reverse,
            occupancy_tf_list,
            "Library-wide sequence diversity through motif coverage and arrangement variability",
            str(output_dir / "plots" / "tf_occupancy_combined.png"),
            dpi=1000,
            figsize=(12,10)
        )
    
        save_motif_length_histogram(
            results["motif_info"],
            str(output_dir / "plots" / "motif_length_density.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(8,6)
        )
    
        save_motif_stats_barplot(
            results["motif_stats"],
            str(output_dir / "plots" / "motif_stats_barplot.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(14,7)
        )
    
    print_summary(results)

if __name__ == "__main__":
    main()
