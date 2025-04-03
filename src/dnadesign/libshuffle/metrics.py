"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/metrics.py

Provides functions to compute diversity metrics for LibShuffle. This module:
  - Instantiates a fresh temporary Billboard configuration by merging the global
    Billboard config (from config["billboard"]) with LibShuffle overrides (from config["billboard_metric"]).
  - Runs the full Billboard processing pipeline (process_sequences, compute_core_metrics,
    compute_composite_metrics) on a given subsample.
  - Computes the Evo2 diversity metric with flexible options (l2, log1p_l2, and cosine similarity).
  - Applies composite transformation (via LibShuffle's own apply_composite_transformation later)
    across subsamples.

This implementation is decoupled from Billboard’s internal logic by using a helper
to build a complete temporary config.
  
Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import torch
import tempfile
import os
import math
import numpy as np
import torch.nn.functional as F

from dnadesign.billboard.core import process_sequences, compute_core_metrics, compute_composite_metrics
from dnadesign.billboard.summary import generate_entropy_summary_csv

def make_temp_billboard_config(config, temp_pt_path):
    """
    Creates a fresh temporary Billboard configuration by merging the global Billboard
    config (from config["billboard"]) with LibShuffle overrides (from config["billboard_metric"]).
    This configuration is used solely for computing the diversity summary.
    
    It forces dry_run mode (which should compute the diversity summary without generating plots)
    and sets pt_files and output_dir_prefix appropriately.
    """
    # Start with a copy of the global Billboard config.
    global_billboard = config.get("billboard", {}).copy()
    # Retrieve LibShuffle overrides.
    lib_overrides = config.get("billboard_metric", {})
    # Force DRY_RUN mode.
    global_billboard["dry_run"] = True
    # Override pt_files and output directory.
    global_billboard["pt_files"] = [temp_pt_path]
    global_billboard["output_dir_prefix"] = "temp_billboard_" + next(tempfile._get_candidate_names())
    # Ensure that the key diversity_metrics is present.
    if "diversity_metrics" not in global_billboard or not global_billboard["diversity_metrics"]:
        global_billboard["diversity_metrics"] = lib_overrides.get("core_metrics", [])
    # Ensure composite_weights is present.
    if "composite_weights" not in global_billboard or not global_billboard["composite_weights"]:
        global_billboard["composite_weights"] = lib_overrides.get("weights", {})
    return global_billboard

def compute_billboard_metric(subsample, config):
    """
    Computes the raw Billboard metrics for a subsample by running the full Billboard pipeline:
      1. Save the subsample to a temporary .pt file.
      2. Build a fresh temporary Billboard configuration.
      3. Run process_sequences() to compute results.
      4. Compute core diversity metrics via compute_core_metrics().
      5. Compute composite metrics via compute_composite_metrics() (for diagnostic purposes).
    
    Prints the core and composite metrics for debugging and returns the core metrics dictionary.
    
    If composite_score is false and only one core metric is expected, returns that single value.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write the current subsample to a temporary .pt file.
        temp_pt_path = os.path.join(tmpdir, "subsample.pt")
        torch.save(subsample, temp_pt_path)
        
        # Build a fresh temporary Billboard configuration.
        billboard_config = make_temp_billboard_config(config, temp_pt_path)
        
        # Create output directories (in case Billboard needs them).
        temp_output_dir = os.path.join(tmpdir, "billboard_output")
        os.makedirs(os.path.join(temp_output_dir, "csvs"), exist_ok=True)
        os.makedirs(os.path.join(temp_output_dir, "plots"), exist_ok=True)
        
        # Run Billboard processing.
        results = process_sequences([temp_pt_path], billboard_config)
        
        # Compute core metrics using Billboard's functions.
        core_metrics = compute_core_metrics(results, billboard_config)
        composite_metrics = compute_composite_metrics(core_metrics, billboard_config)
        results["core_metrics"] = core_metrics
        results["composite_metrics"] = composite_metrics
        
        # print("Core metrics for this subsample:", core_metrics)
        # print("Composite metrics for this subsample:", composite_metrics)
        
        # In LibShuffle we use the core metrics (to be later transformed across subsamples).
        bm_config = config.get("billboard_metric", {})
        composite_enabled = bm_config.get("composite_score", False)
        if not composite_enabled:
            if len(bm_config.get("core_metrics", [])) != 1:
                raise ValueError("When composite_score is false, exactly one core metric must be specified.")
            return core_metrics[bm_config["core_metrics"][0]]
        else:
            return core_metrics

def compute_evo2_metric(subsample, config):
    """
    Compute the mean pairwise distance among the evo2 vectors in the subsample.
    Each subsample entry is expected to have a field "evo2_logits_mean_pooled".
    
    The metric_type in the config supports:
      - "l2": Mean pairwise Euclidean (L2) distance.
      - "log1p_l2": log1p of the mean pairwise L2 distance.
      - "cosine": Mean pairwise cosine distance (1 - cosine similarity).
    """
    vectors = []
    for entry in subsample:
        vector = entry.get("evo2_logits_mean_pooled")
        if vector is None:
            raise ValueError("Entry missing 'evo2_logits_mean_pooled' field required for evo2 metric.")
        if isinstance(vector, torch.Tensor):
            vector = vector.to(torch.float32).flatten()
        elif isinstance(vector, list):
            vector = torch.tensor(vector, dtype=torch.float32).flatten()
        else:
            vector = torch.tensor(vector, dtype=torch.float32).flatten()
        vectors.append(vector)
    if len(vectors) < 2:
        return 0.0
    mat = torch.stack(vectors)
    metric_type = config.get("evo2_metric", {}).get("type", "l2")
    if metric_type == "l2":
        distances = torch.cdist(mat, mat, p=2)
        n = distances.size(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise = distances[mask]
        return pairwise.mean().item()
    elif metric_type == "log1p_l2":
        distances = torch.cdist(mat, mat, p=2)
        n = distances.size(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise = distances[mask]
        mean_distance = pairwise.mean().item()
        return math.log1p(mean_distance)
    elif metric_type == "cosine":
        normed = F.normalize(mat, p=2, dim=1)
        cosine_sim = normed @ normed.t()
        cosine_distance = 1 - cosine_sim
        n = cosine_distance.size(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pairwise = cosine_distance[mask]
        return pairwise.mean().item()
    else:
        raise ValueError(f"Unsupported evo2_metric type: {metric_type}")

def apply_composite_transformation(subsamples, config):
    """
    Applies composite transformation on the raw Billboard (core) metric values across all subsamples.
    For each subsample, the raw values (obtained from compute_billboard_metric) are combined
    into a composite score via the specified method:
      - "zscore_pca": Normalize (using z-score or min-max) then project onto the first principal component.
      - "minmax_weighted": Normalize (min-max or zscore as specified) then compute a weighted sum.
    If composite_score is false, it verifies that each subsample has a single raw value.
    
    Returns:
      updated_subsamples: The list of subsample dicts with "billboard_metric" updated to the composite score.
      pca_model_info: Dictionary containing PCA artifacts (if PCA is used), else None.
    """
    bm_config = config.get("billboard_metric", {})
    composite_enabled = bm_config.get("composite_score", False)
    if not composite_enabled:
        for sub in subsamples:
            sub["billboard_metric"] = sub.pop("raw_billboard_vector", sub.get("billboard_metric"))
        return subsamples, None

    core_metrics = bm_config.get("core_metrics", [])
    data = []
    for sub in subsamples:
        raw = sub.pop("raw_billboard_vector", None)
        if raw is None:
            raise ValueError("Composite mode enabled but raw_billboard_vector not found in subsample.")
        vector = [raw.get(metric, 0.0) for metric in core_metrics]
        data.append(vector)
        sub["raw_billboard_vector"] = vector
    data = np.array(data)

    norm_method = bm_config.get("normalize", None)
    if norm_method == "zscore":
        means = np.mean(data, axis=0)
        stds = np.std(data, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        data_norm = (data - means) / stds
    elif norm_method == "minmax":
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
        data_norm = (data - mins) / ranges
    else:
        data_norm = data.copy()

    method = bm_config.get("method", "zscore_pca")
    pca_model_info = None
    if method == "zscore_pca":
        if norm_method != "zscore":
            data_centered = data_norm - np.mean(data_norm, axis=0)
        else:
            data_centered = data_norm
        cov = np.cov(data_centered, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eig_vals)[::-1]
        eig_vals = eig_vals[sorted_idx]
        eig_vecs = eig_vecs[:, sorted_idx]
        pc1 = eig_vecs[:, 0]
        composite_scores = data_centered.dot(pc1)
        total_variance = np.sum(eig_vals)
        explained_variance_ratio = eig_vals[0] / total_variance if total_variance > 0 else 0.0
        pca_model_info = {
            "explained_variance_ratio": [float(explained_variance_ratio)],
            "components": [pc1.tolist()],
            "used_metrics": core_metrics
        }
    elif method == "minmax_weighted":
        weights = bm_config.get("weights", {})
        if not all(metric in weights for metric in core_metrics):
            weights = {metric: 1.0/len(core_metrics) for metric in core_metrics}
        weight_vec = np.array([weights.get(metric, 0.0) for metric in core_metrics])
        composite_scores = data_norm.dot(weight_vec)
    else:
        raise ValueError(f"Unsupported composite method: {method}")

    for i, sub in enumerate(subsamples):
        sub["billboard_metric"] = float(composite_scores[i])
    return subsamples, pca_model_info