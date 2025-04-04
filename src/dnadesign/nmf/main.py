"""
--------------------------------------------------------------------------------
<dnadesign project>
nmf/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path

from dnadesign.nmf.config import load_config
from dnadesign.nmf.parser import load_pt_file
from dnadesign.nmf.encoder import encode_sequence_matrix
from dnadesign.nmf.nmf_train import train_nmf
from dnadesign.nmf.diagnostics import (
    plot_reconstruction_loss,
    plot_heatmaps,
    plot_top_motifs_grid,
    plot_sequence_program_composition,
    plot_stability_metrics_grid,
    plot_signature_stability_riverplot,
    export_program_tf_contributions,
    compute_stability_metrics,
    compute_program_flow,
    plot_motif_occurrence_heatmap,
    plot_program_association_stacked_barplot,
    plot_composite_diversity_curve
)
from dnadesign.nmf.persistence import (
    export_feature_names,
    export_row_ids,
    annotate_sequences_with_nmf
)

def build_motif2tf_map(sequences):
    """
    Build a mapping from motif (uppercased) to transcription factor name
    using meta_tfbs_parts in the sequences.
    """
    motif2tf = {}
    from dnadesign.nmf.parser import robust_parse_tfbs
    for seq in sequences:
        tfbs_parts = seq.get("meta_tfbs_parts", [])
        for part in tfbs_parts:
            try:
                tf_name, motif = robust_parse_tfbs(part, seq_id=seq.get("id", "unknown"))
                motif2tf[motif.upper()] = tf_name
            except Exception as e:
                continue
    return motif2tf

def main():
    config_path = Path(__file__).parent.parent / "configs" / "example.yaml"
    config = load_config(str(config_path))
    nmf_config = config["nmf"]
    batch_name = nmf_config.get("batch_name", "default")
    
    # Locate the PT file.
    pt_dir = Path(__file__).parent.parent / "sequences" / batch_name
    if not pt_dir.exists() or not pt_dir.is_dir():
        logging.error(f"Sequence directory {pt_dir} not found or is not a directory.")
        sys.exit(1)
    
    pt_files = list(pt_dir.glob("*.pt"))
    if not pt_files:
        logging.error(f"No .pt files found in {pt_dir}.")
        sys.exit(1)
    if len(pt_files) > 1:
        logging.warning(f"Multiple .pt files found; using the first: {pt_files[0]}")
    pt_file = pt_files[0]
    
    logging.info(f"Loading sequences from {pt_file}...")
    sequences = load_pt_file(str(pt_file))
    if not sequences:
        logging.error("No sequences loaded. Exiting.")
        sys.exit(1)
    
    logging.info("Encoding sequence data into feature matrix...")
    X, feature_names = encode_sequence_matrix(sequences, config)
    logging.info(f"Feature matrix shape: {X.shape}")
    
    batch_results_dir = Path(__file__).parent / "batch_results" / batch_name
    batch_results_dir.mkdir(parents=True, exist_ok=True)
    export_feature_names(feature_names, str(batch_results_dir / "feature_names.txt"))
    row_ids = [seq.get("id", str(i)) for i, seq in enumerate(sequences)]
    export_row_ids(row_ids, str(batch_results_dir / "row_ids.txt"))
    
    logging.info("Starting NMF training...")
    start_time = time.time()
    results = train_nmf(X, config)
    elapsed = time.time() - start_time
    logging.info(f"NMF training completed in {elapsed:.2f} seconds.")
    
    # Determine best_k: if specified in config, use that; otherwise, use computed best_k.
    if "best_k" in nmf_config:
        best_k = nmf_config["best_k"]
        logging.info(f"Using best_k from config: {best_k}")
        # Load best_W from the corresponding subdirectory.
        best_k_dir = batch_results_dir / f"k_{best_k}"
        if best_k_dir.exists():
            W_file = best_k_dir / "W.csv"
            if W_file.exists():
                import pandas as pd
                best_W = pd.read_csv(W_file, index_col=0).values
            else:
                logging.error(f"W.csv not found in {best_k_dir}")
                sys.exit(1)
        else:
            logging.error(f"Directory {best_k_dir} does not exist")
            sys.exit(1)
    else:
        best_k = results.get("best_k")
        best_W = results.get("best_W")
        logging.info(f"Computed best_k: {best_k}")
    
    # Annotate sequences with NMF metadata using the correct best_W.
    annotate_sequences_with_nmf(sequences, best_W, best_k, str(pt_file))
    
    plots_dir = batch_results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    recon_loss_plot = plots_dir / "reconstruction_loss.png"
    plot_reconstruction_loss(results["metrics"], str(recon_loss_plot))
    
    # Generate motif-centric plots.
    motif_occurrence_heatmap_path = plots_dir / "motif_occurrence_heatmap.png"
    plot_motif_occurrence_heatmap(
        X,
        str(motif_occurrence_heatmap_path),
        max_occ=nmf_config.get("max_motif_occupancy", 5),
        transpose=True
    )
    
    # Build motif->TF map.
    motif2tf_map = build_motif2tf_map(sequences)
        
    # Load H from the best_k subdirectory.
    best_k_dir = batch_results_dir / f"k_{best_k}"
    if best_k_dir.exists():
        H_file = best_k_dir / "H.csv"
        if H_file.exists():
            import pandas as pd
            H_best = pd.read_csv(H_file, index_col=0).values
        else:
            logging.error(f"H.csv not found in {best_k_dir}")
    else:
        logging.error(f"Directory {best_k_dir} does not exist")
    
    program_assoc_stacked_path = plots_dir / "program_association_stacked_barplot.png"
    plot_program_association_stacked_barplot(
        H_best,
        feature_names,
        str(program_assoc_stacked_path),
        motif2tf_map=motif2tf_map,
        normalize=True
    )
        
    if best_k is not None:
        best_k_dir = batch_results_dir / f"k_{best_k}"
        try:
            import pandas as pd
            W_file = best_k_dir / "W.csv"
            H_file = best_k_dir / "H.csv"
            if not W_file.exists() or not H_file.exists():
                logging.error(f"Best k directory {best_k_dir} is missing W.csv or H.csv.")
            else:
                W = pd.read_csv(W_file, index_col=0).values
                H = pd.read_csv(H_file, index_col=0).values
                
                plot_heatmaps(W, H, str(plots_dir), encoding_mode="motif_occupancy", feature_names=feature_names)
                plot_top_motifs_grid(
                    H,
                    feature_names,
                    str(plots_dir),
                    motif2tf_map=motif2tf_map,
                    max_top_tfs=nmf_config.get("diagnostics", {}).get("max_top_tfs", 5),
                    ncols=nmf_config.get("diagnostics", {}).get("barplot_ncols", 4),
                    encoding_mode="motif_occupancy"
                )
                export_program_tf_contributions(H, feature_names, str(plots_dir))
                plot_sequence_program_composition(
                    W,
                    str(plots_dir / "sequence_program_composition.png")
                )
                if nmf_config.get("stability_metrics", {}).get("enable", False):
                    k_range = list(range(nmf_config["k_range"][0], nmf_config["k_range"][1]+1))
                    stability_data = compute_stability_metrics(str(batch_results_dir), k_range)
                    stability_metrics_output = plots_dir / "stability_metrics_grid.png"
                    plot_stability_metrics_grid(stability_data, str(stability_metrics_output))
                if nmf_config.get("plots", {}).get("generate_riverplot", False):
                    k_range = list(range(nmf_config["k_range"][0], nmf_config["k_range"][1]+1))
                    program_flow = compute_program_flow(
                        str(batch_results_dir),
                        k_range,
                        nmf_config.get("stability_metrics", {}).get("similarity_threshold", 0.6)
                    )
                    riverplot_output = plots_dir / "signature_stability_riverplot.png"
                    k_range = list(range(nmf_config["k_range"][0], nmf_config["k_range"][1] + 1))
                    plot_signature_stability_riverplot(program_flow, str(riverplot_output), k_range)
        except Exception as e:
            logging.error(f"Error generating diagnostics plots: {str(e)}")
    
    diversity_cfg = nmf_config.get("diversity_score", {"enable": False})
    if diversity_cfg.get("enable", False):
        alpha = diversity_cfg.get("alpha", 0.5)
        # Assuming your k_range is available
        k_range = list(range(nmf_config["k_range"][0], nmf_config["k_range"][1] + 1))
        diversity_output = plots_dir / "composite_diversity_curve.png"
        plot_composite_diversity_curve(str(batch_results_dir), k_range, alpha, str(diversity_output))
    
    logging.info("NMF pipeline completed successfully.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    main()