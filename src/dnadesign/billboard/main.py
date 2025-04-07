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

from dnadesign.billboard.core import process_sequences, compute_core_metrics, compute_composite_metrics
from dnadesign.billboard.summary import generate_csv_summaries, generate_entropy_summary_csv, print_summary
from dnadesign.billboard.plot_helpers import (
    save_tf_frequency_barplot,
    save_occupancy_plot,
    save_motif_length_histogram,
    save_tf_entropy_kde_plot,
    save_gini_lorenz_plot,
    save_jaccard_histogram,
    save_motif_levenshtein_boxplot
)

def main():
    # Load configuration.
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"
    with config_path.open("r") as f:
        config = yaml.safe_load(f)
    billboard_config = config.get("billboard", {})
    
    dry_run = billboard_config.get("dry_run", False)
    if dry_run:
        print("DRY RUN enabled: Only the diversity summary CSV will be generated. No plots will be produced.")
    
    # Assumes PT file is the name of a subdirectory within the project's "sequences" directory.
    pt_files_config = billboard_config.get("pt_files", [])
    pt_paths = []
    for p in pt_files_config:
        path_candidate = Path(p)
        if not path_candidate.is_absolute():
            project_root = Path(__file__).parent.parent  # Adjust as needed.
            base_dir = project_root / "sequences"
            path_candidate = base_dir / p
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
    out_dir_name = f"{custom_prefix}_{today}" if custom_prefix else today
    output_dir = Path(__file__).parent / "batch_results" / out_dir_name
    os.makedirs(output_dir / "csvs", exist_ok=True)
    os.makedirs(output_dir / "plots", exist_ok=True)
    
    # Process sequences and compute basic results.
    results = process_sequences(pt_paths, billboard_config)
    
    # Compute core diversity metrics using the full results dictionary.
    core_metrics = compute_core_metrics(results, billboard_config)
    
    # Compute composite metrics (using a weighted sum).
    composite_metrics = compute_composite_metrics(core_metrics, billboard_config)
    
    # Add computed metrics to results for export.
    results["core_metrics"] = core_metrics
    results["composite_metrics"] = composite_metrics
    
    if dry_run:
        generate_entropy_summary_csv(results, str(output_dir))
    else:
        generate_csv_summaries(results, str(output_dir))
        
        # TF Frequency Bar Plot
        tf_metric = {}
        for tf in results["tf_frequency"]:
            tf_metric[tf] = composite_metrics.get("billboard_weighted_sum", 0)
        save_tf_frequency_barplot(
            results["tf_frequency"],
            tf_metric,
            "TF Frequency Bar Plot",
            str(output_dir / "plots" / "tf_frequency_barplot.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(14,7)
        )
        
        # TF Occupancy Plot
        occupancy_tf_list = results.get("occupancy_tf_list", [])
        if occupancy_tf_list:
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
        
        # Motif Length Density Plot.
        save_motif_length_histogram(
            results["motif_info"],
            str(output_dir / "plots" / "motif_length_density.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(8,6)
        )
                
        # TF Entropy KDE Plot: Overlay KDE plots for top N TFs based on positional occupancy.
        save_tf_entropy_kde_plot(
            results["occupancy_forward_matrix"],
            results["occupancy_reverse_matrix"],
            results["occupancy_tf_list"],
            results["sequence_length"],
            str(output_dir / "plots" / "tf_entropy_kde.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(10,6)
        )
        
        # Gini Lorenz Plot: Plot the Lorenz curve for TF frequency and annotate with the Gini coefficient.
        save_gini_lorenz_plot(
            results["tf_frequency"],
            str(output_dir / "plots" / "tf_gini_lorenz.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(8,6)
        )
                
        # Jaccard Histogram: Plot a histogram of pairwise Jaccard dissimilarities.
        save_jaccard_histogram(
            results["sequences"],
            str(output_dir / "plots" / "tf_jaccard_histogram.png"),
            dpi=billboard_config.get("dpi", 600),
            figsize=(8,6),
            sample_size=1000
        )
                
        # Motif Levenshtein Boxplot: visualize the distribution of pairwise edit distances.
        if "motif_string_levenshtein" in billboard_config.get("diversity_metrics", []):
            from dnadesign.billboard.plot_helpers import save_motif_levenshtein_boxplot
            save_motif_levenshtein_boxplot(
                results["motif_strings"],
                billboard_config,
                str(output_dir / "plots" / "motif_levenshtein_boxplot.png"),
                dpi=billboard_config.get("dpi", 600),
                figsize=(8,6)
            )
                
    print_summary(results)

if __name__ == "__main__":
    main()