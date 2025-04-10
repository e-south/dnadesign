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
  - Integrates the new CDS method when billboard_metric.method is set to "cds".
  
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
    Creates a temporary Billboard configuration that is decoupled from the global Billboard config.
    Instead, it uses the diversity metrics specified under 'libshuffle_core_metrics' in the LibShuffle configuration.
    """
    global_billboard = config.get("billboard", {}).copy()
    global_billboard["dry_run"] = True
    global_billboard["pt_files"] = [temp_pt_path]
    global_billboard["output_dir_prefix"] = "temp_billboard_" + next(tempfile._get_candidate_names())
    global_billboard["weights_only"] = False
    libshuffle_core = config.get("libshuffle_core_metrics", [])
    if libshuffle_core:
        global_billboard["diversity_metrics"] = libshuffle_core
    elif "diversity_metrics" not in global_billboard or not global_billboard["diversity_metrics"]:
        global_billboard["diversity_metrics"] = config.get("billboard_metric", {}).get("core_metrics", [])
    return global_billboard

def compute_billboard_metric(subsample, config):
    import torch.serialization
    import numpy as np
    torch.serialization.add_safe_globals([np.generic, np._core.multiarray.scalar, np.dtype])
    
    bm_config = config.get("billboard_metric", {})
    composite_enabled = bm_config.get("composite_score", False)

    if composite_enabled:
        if bm_config.get("method") == "cds":
            from dnadesign.libshuffle.cds_score import compute_cds_from_sequences
            return compute_cds_from_sequences(subsample, bm_config.get("alpha", 0.5))
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_pt_path = os.path.join(tmpdir, "subsample.pt")
                torch.save(subsample, temp_pt_path)
                billboard_config = make_temp_billboard_config(config, temp_pt_path)
                temp_output_dir = os.path.join(tmpdir, "billboard_output")
                os.makedirs(os.path.join(temp_output_dir, "csvs"), exist_ok=True)
                os.makedirs(os.path.join(temp_output_dir, "plots"), exist_ok=True)
                results = process_sequences([temp_pt_path], billboard_config)
                core_metrics = compute_core_metrics(results, billboard_config)
                return core_metrics
    else:
        core_metrics_list = bm_config.get("core_metrics", [])
        if len(core_metrics_list) != 1:
            raise ValueError("When composite_score is false, exactly one core metric must be specified.")
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_pt_path = os.path.join(tmpdir, "subsample.pt")
            torch.save(subsample, temp_pt_path)
            billboard_config = make_temp_billboard_config(config, temp_pt_path)
            temp_output_dir = os.path.join(tmpdir, "billboard_output")
            os.makedirs(os.path.join(temp_output_dir, "csvs"), exist_ok=True)
            os.makedirs(os.path.join(temp_output_dir, "plots"), exist_ok=True)
            results = process_sequences([temp_pt_path], billboard_config)
            core_metrics = compute_core_metrics(results, billboard_config)
            key = core_metrics_list[0]
            if key not in core_metrics:
                alt_key = key + "_mean"
                if alt_key in core_metrics:
                    key = alt_key
                else:
                    raise KeyError(f"Key '{core_metrics_list[0]}' not found in computed core metrics: {list(core_metrics.keys())}")
            return core_metrics[key]

def compute_evo2_metric(subsample, config):
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
    bm_config = config.get("billboard_metric", {})
    composite_enabled = bm_config.get("composite_score", False)

    if not composite_enabled:
        core_metrics_list = bm_config.get("core_metrics", [])
        if len(core_metrics_list) != 1:
            raise ValueError("When composite_score is false, exactly one core metric must be specified.")
        for sub in subsamples:
            sub["billboard_metric"] = sub.pop("raw_billboard_vector", sub.get("billboard_metric"))
        return subsamples, None

    if bm_config.get("method") == "cds":
        for sub in subsamples:
            raw = sub.pop("raw_billboard_vector", None)
            if raw is None:
                raise ValueError("CDS mode enabled but raw_billboard_vector not found in subsample.")
            sub["billboard_metric"] = float(raw["cds_score"])
            sub["cds_components"] = raw
        return subsamples, None

    core_metrics = bm_config.get("core_metrics", [])
    data = []
    for sub in subsamples:
        raw = sub.pop("raw_billboard_vector", None)
        if raw is None:
            raise ValueError("Composite mode enabled but raw_billboard_vector not found in subsample.")
        vector = []
        for metric in core_metrics:
            val = raw.get(metric, None)
            if val is None:
                alt_key = metric + "_mean"
                val = raw.get(alt_key, 0.0)
            vector.append(val)
        data.append(vector)
        sub["raw_billboard_vector"] = vector
    data = np.array(data)

    if bm_config.get("method") == "inverse_rank_sum":
        n_samples, n_metrics = data.shape
        rank_data = np.zeros_like(data, dtype=float)
        for j in range(n_metrics):
            col = data[:, j]
            ranks = (n_samples - 1) - np.argsort(np.argsort(col))
            rank_data[:, j] = ranks
        composite_scores = rank_data.sum(axis=1)
        pca_model_info = None

    elif bm_config.get("method") == "percentile_avg":
        n_samples, n_metrics = data.shape
        percentile_data = np.zeros_like(data, dtype=float)
        for j in range(n_metrics):
            col = data[:, j]
            ranks = np.argsort(np.argsort(col))
            if n_samples > 1:
                percentile_data[:, j] = ranks / (n_samples - 1)
            else:
                percentile_data[:, j] = 0.5
        composite_scores = percentile_data.mean(axis=1)
        pca_model_info = None

    elif bm_config.get("method") == "zscore_pca":
        norm_method = bm_config.get("normalize", None)
        if norm_method != "zscore":
            data_centered = data - np.mean(data, axis=0)
        else:
            data_centered = data
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
    elif bm_config.get("method") == "minmax_weighted":
        norm_method = bm_config.get("normalize", None)
        if norm_method in [None, "null"]:
            data_norm = data.copy()
        elif norm_method == "zscore":
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
        weights = bm_config.get("weights", {})
        if not all(metric in weights for metric in core_metrics):
            weights = {metric: 1.0/len(core_metrics) for metric in core_metrics}
        weight_vec = np.array([weights.get(metric, 0.0) for metric in core_metrics])
        composite_scores = data_norm.dot(weight_vec)
        pca_model_info = None
    else:
        raise ValueError(f"Unsupported composite method: {bm_config.get('method')}")
        
    for i, sub in enumerate(subsamples):
        sub["billboard_metric"] = float(composite_scores[i])
    return subsamples, pca_model_info