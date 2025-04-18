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
import logging
from pathlib import Path

from dnadesign.billboard.core import process_sequences, compute_core_metrics
from dnadesign.billboard.summary import (
    generate_csv_summaries,
    generate_entropy_summary_csv,
    print_summary
)
from dnadesign.billboard.plot_helpers import (
    save_tf_frequency_barplot,
    save_occupancy_plot,
    save_motif_length_histogram,
    save_tf_entropy_kde_plot,
    save_gini_lorenz_plot,
    save_jaccard_histogram,
)
from dnadesign.billboard.by_cluster import (
    compute_cluster_metrics,
    save_cluster_characterization_scatter
)

logger = logging.getLogger(__name__)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )
    logger.info("Starting billboard pipeline")

    # Load config
    cfg_path = Path(__file__).parent.parent / "configs" / "example.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["billboard"]
    logger.info(f"Loaded configuration from {cfg_path}")

    # Validate config
    assert cfg.get("pt_files"), "Must specify at least one entry in pt_files."
    metrics = cfg.get("diversity_metrics", [])
    assert metrics, "diversity_metrics list cannot be empty."
    logger.info(f"Metrics to compute: {metrics}")

    dry_run = cfg.get("dry_run", False)
    skip_nw = cfg.get("skip_aligner_call", False)

    # Assemble PT file paths
    project_root = Path(__file__).parent.parent
    pt_paths = []
    for entry in cfg["pt_files"]:
        p = Path(entry)
        if not p.is_absolute():
            p = project_root / "sequences" / entry
        if p.is_dir():
            found = list(p.glob("*.pt"))
            assert len(found) == 1, f"Expected one .pt file in {p}, found {len(found)}"
            p = found[0]
        pt_paths.append(str(p))
    logger.info(f"Found {len(pt_paths)} .pt file(s) to process")

    # Prepare output directories
    today = datetime.datetime.now().strftime("%Y%m%d")
    prefix = cfg.get("output_dir_prefix", "").strip()
    out_name = f"{prefix}_{today}" if prefix else today
    out_dir = project_root / "billboard" / "batch_results" / out_name
    csv_dir = out_dir / "csvs"
    plot_dir = out_dir / "plots"
    csv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will go to {out_dir}")

    # Process sequences
    results = process_sequences(pt_paths, cfg)
    logger.info(f"Loaded and processed {results['num_sequences']} sequences")

    # Compute core metrics
    core_metrics = compute_core_metrics(results, cfg)
    results["core_metrics"] = core_metrics
    results["skip_aligner_call"] = skip_nw
    logger.info(f"Computed core metrics: {core_metrics}")

    # Generate summaries & plots
    if dry_run:
        logger.info("Dry run enabled: skipping plots, writing only diversity_summary.csv")
        generate_entropy_summary_csv(results, csv_dir)
    else:
        logger.info("Writing CSV summaries")
        generate_csv_summaries(results, csv_dir)

        logger.info("Generating TF frequency bar plot")
        save_tf_frequency_barplot(
            results["tf_frequency"],
            title="TF Frequency Bar Plot",
            path=plot_dir / "tf_frequency.png",
            dpi=cfg["dpi"],
        )

        logger.info("Generating occupancy plot")
        save_occupancy_plot(
            results["occupancy_forward_matrix"],
            results["occupancy_reverse_matrix"],
            results["occupancy_tf_list"],
            title="Library-wide motif coverage",
            path=plot_dir / "occupancy.png",
            dpi=cfg["dpi"],
        )

        logger.info("Generating motif length histogram")
        save_motif_length_histogram(
            results["motif_info"],
            path=plot_dir / "motif_length.png",
            dpi=cfg["dpi"],
        )

        logger.info("Generating TF entropy KDE plot")
        save_tf_entropy_kde_plot(
            results["occupancy_forward_matrix"],
            results["occupancy_reverse_matrix"],
            results["occupancy_tf_list"],
            results["sequence_length"],
            path=plot_dir / "tf_entropy_kde.png",
            dpi=cfg["dpi"],
        )

        logger.info("Generating Lorenz (Gini) plot")
        save_gini_lorenz_plot(
            results["tf_frequency"],
            path=plot_dir / "gini_lorenz.png",
            dpi=cfg["dpi"],
        )

        logger.info("Generating Jaccard histogram")
        save_jaccard_histogram(
            results["sequences"],
            path=plot_dir / "jaccard_hist.png",
            dpi=cfg["dpi"],
        )

        if cfg.get("characterize_by_leiden_cluster", {}).get("enabled", False):
            logger.info("Characterizing by Leiden cluster")
            dfc = compute_cluster_metrics(results, cfg)
            save_cluster_characterization_scatter(
                dfc, cfg,
                path=plot_dir / "cluster_char.png",
                dpi=cfg["dpi"],
            )

    # Final summary
    print_summary(results)
    logger.info("Pipeline complete")

if __name__ == "__main__":
    main()