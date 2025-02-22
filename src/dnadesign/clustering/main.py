"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/clustering/main.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import sys
import yaml
import torch
import numpy as np
import scanpy as sc
import datetime
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dnadesign.clustering import cluster_select, plotter, normalization, \
    cluster_composition, diversity_analysis, differential_feature_analysis

def load_config(config_path="configs/example.yaml"):
    """
    Load the configuration file.
    """
    base_dir = Path(__file__).resolve().parent.parent
    config_file = base_dir / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    with config_file.open("r") as f:
        config = yaml.safe_load(f)
    return config

def load_pt_files(input_dirs, base_path):
    """
    Load .pt files from each input directory on CPU.
    """
    data_entries = []
    base_dir = Path(base_path)
    for d in input_dirs:
        dir_path = base_dir / d
        if not dir_path.is_dir():
            print(f"Warning: directory {dir_path} not found")
            continue
        for file_path in dir_path.iterdir():
            if file_path.suffix == ".pt":
                entries = torch.load(str(file_path), map_location="cpu")
                for entry in entries:
                    entry["meta_input_source"] = d
                data_entries.extend(entries)
    return data_entries

def extract_features(data_entries, key="evo2_logits_mean_pooled"):
    """
    Extract feature vectors from the provided key.
    """
    features = []
    for entry in data_entries:
        tensor_val = entry.get(key, None)
        if tensor_val is None:
            raise ValueError(f"Entry missing key {key}")
        if tensor_val.dtype == torch.bfloat16:
            tensor_val = tensor_val.float()
        np_val = tensor_val.cpu().numpy().squeeze()
        features.append(np_val)
    return np.array(features)

def save_cluster_output(cluster_id, entries, output_prefix, batch_name, base_output, extra_summary):
    """
    Save cluster data and summary YAML.
    Directory and file names do not include timestamps; the current timestamp is saved in the YAML.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(base_output)
    out_dir_name = f"{output_prefix}{batch_name}_{cluster_id}"
    out_dir = base_dir / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    output_pt = out_dir / f"cluster_{cluster_id}.pt"
    torch.save(entries, str(output_pt))
    
    summary = {
        "cluster_id": cluster_id,
        "num_entries": len(entries),
        "timestamp": timestamp,
        "description": f"Cluster {cluster_id} output from clustering pipeline"
    }
    summary.update(extra_summary)
    
    output_yaml = out_dir / "summary.yaml"
    with output_yaml.open("w") as f:
        yaml.dump(summary, f)
    print(f"Saved cluster {cluster_id} to {out_dir}")

def main():
    parser = argparse.ArgumentParser(description="Clustering Pipeline Main")
    parser.add_argument("--config", type=str, default="configs/example.yaml",
                        help="Path to configuration file (relative to project root)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Perform a dry run without writing outputs")
    args = parser.parse_args()

    print("Loading configuration...")
    config = load_config(args.config)
    clustering_config = config.get("clustering", {})
    analysis_config = clustering_config.get("analysis", {})

    base_dir = Path(__file__).resolve().parent.parent
    base_seq_dir = base_dir / "sequences"
    
    # Use batch_name from config.
    batch_name = clustering_config.get("batch_name", "batch")
    
    # Determine the hue method (for appending to the results folder name).
    umap_config = clustering_config.get("umap", {})
    hue_method = umap_config.get("hue", {}).get("method", "leiden")
    
    # Create a results directory including batch_name and hue method.
    results_dir = base_dir / "clustering" / "batch_results" / f"{batch_name}_{hue_method}"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Determine input directories.
    input_sources = clustering_config.get("input_sources", [])
    input_dirs = []
    for source in input_sources:
        if "dir" in source:
            input_dirs.append(source["dir"])
        elif "pattern" in source:
            try:
                all_dirs = [d.name for d in base_seq_dir.iterdir() if d.is_dir()]
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Sequences directory not found: {base_seq_dir}") from e
            pattern = source["pattern"].replace("*", "")
            matched = [d for d in all_dirs if pattern in d]
            input_dirs.extend(matched)
    
    if not input_dirs:
        print("No input directories found. Exiting.")
        sys.exit(1)
    
    print(f"Input directories found: {input_dirs}")
    print("Loading data entries from input directories...")
    data_entries = load_pt_files(input_dirs, base_path=str(base_seq_dir))
    if not data_entries:
        print("No data entries found in the provided input sources.")
        sys.exit(1)
    
    print("Extracting features from data entries...")
    features = extract_features(data_entries, key="evo2_logits_mean_pooled")
    
    print("Creating AnnData object for Scanpy processing...")
    adata = sc.AnnData(features)
    
    n_neighbors = umap_config.get("n_neighbors", 15)
    min_dist = umap_config.get("min_dist", 0.1)
    
    print(f"Computing neighbors (n_neighbors={n_neighbors}, use_rep='X') and UMAP (min_dist={min_dist})...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.umap(adata, min_dist=min_dist)
    
    print("Performing Leiden clustering (resolution=0.2)...")
    sc.tl.leiden(adata, resolution=0.2)
    clusters = adata.obs["leiden"].to_numpy()
    for i, entry in enumerate(data_entries):
        entry["leiden_cluster"] = clusters[i]
    
    if clustering_config.get("overlay_cluster_numbers", False):
        unique_clusters = sorted(set(clusters))
        centroids = {}
        umap_coords = adata.obsm["X_umap"]
        for cl in unique_clusters:
            idx = [i for i, entry in enumerate(data_entries) if entry["leiden_cluster"] == cl]
            coords = umap_coords[idx, :]
            centroids[cl] = np.mean(coords, axis=0)
        cluster_label_map = {cl: str(i) for i, cl in enumerate(unique_clusters, 1)}
        adata.uns["cluster_centroids"] = centroids
        adata.uns["cluster_labels"] = cluster_label_map
        print("Computed cluster centroids for overlaying cluster numbers.")
    
    print("Selecting clusters based on configuration...")
    clusters_to_save = cluster_select.select_clusters(
        data_entries, clustering_config.get("cluster_selection", {}))
    print(f"Selected clusters: {clusters_to_save}")
    
    extra_summary = {
        "num_leiden_clusters": int(len(set(clusters))),
        "umap_parameters": {"n_neighbors": n_neighbors, "min_dist": min_dist},
        "leiden_parameters": {"resolution": 0.2},
        "input_sources": input_dirs,
    }
    
    if not args.dry_run:
        output_prefix = clustering_config.get("output_prefix", "clusterbatch_")
        for cl in clusters_to_save:
            cl_entries = [e for e in data_entries if e["leiden_cluster"] == cl]
            save_cluster_output(cl, cl_entries, output_prefix, batch_name, base_output=str(base_seq_dir), extra_summary=extra_summary)
    else:
        print("Dry run enabled: Not saving cluster outputs.")
    
    if umap_config.get("enabled", False):
        print("Generating UMAP plot...")
        umap_plot_name = f"umap_{batch_name}.png"
        plot_path = results_dir / umap_plot_name
        plotter.plot_umap(adata, umap_config, data_entries, batch_name=batch_name, save_path=str(plot_path))
        print(f"UMAP plot saved to {plot_path}")
    
    if analysis_config.get("cluster_composition", False):
        print("Performing Cluster Composition Analysis...")
        cc_dims = analysis_config.get("cluster_composition_plot_dimensions", [10, 7])
        comp_csv = results_dir / f"cluster_composition_{batch_name}.csv"
        comp_png = results_dir / f"cluster_composition_{batch_name}.png"
        cluster_composition.analyze(data_entries, batch_name=batch_name, save_csv=str(comp_csv), save_png=str(comp_png), plot_dims=cc_dims)
        print(f"Cluster composition results saved: CSV to {comp_csv}, plot to {comp_png}")
    
    if analysis_config.get("diversity_assessment", False):
        print("Performing Diversity Assessment...")
        da_dims = analysis_config.get("diversity_assessment_plot_dimensions", [10, 6])
        div_csv = results_dir / f"diversity_assessment_{batch_name}.csv"
        div_png = results_dir / f"diversity_assessment_{batch_name}.png"
        diversity_analysis.assess(data_entries, batch_name=batch_name, save_csv=str(div_csv), save_png=str(div_png), plot_dims=da_dims)
        print(f"Diversity assessment saved: CSV to {div_csv}, plot to {div_png}")
    
    if analysis_config.get("differential_feature_analysis", False):
        print("Performing Differential Feature Analysis...")
        diff_csv = results_dir / f"differential_feature_analysis_{batch_name}.csv"
        differential_feature_analysis.identify_markers(data_entries, save_csv=str(diff_csv))
        print(f"Differential feature analysis CSV saved to {diff_csv}")
    
    print("Clustering pipeline completed successfully.")

if __name__ == "__main__":
    main()
