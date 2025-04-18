"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/metrics.py

Utility functions that let **libshuffle** call **billboard** on-the-fly.

A *temporary* Billboard configuration is built from scratch for every
sub-sample, containing **only** the metric names requested in the libshuffle
YAML.  Any per-metric option blocks that Billboard needs (currently just the
`motif_string_levenshtein` penalties) are injected automatically, so users do
*not* have to embed a full “billboard:” section in their libshuffle config.

Module Author(s): Eric J. South  
Dunlop Lab
"""

from __future__ import annotations

import math
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from dnadesign.aligner.metrics import mean_pairwise
from dnadesign.billboard.core import compute_core_metrics, process_sequences


# Helper: temporarily silence **all** dnadesign.billboard loggers
@contextmanager
def _silence_billboard():
    """
    Context‑manager that raises the logging level of every logger whose name
    starts with ``dnadesign.billboard`` to WARNING, restoring the original
    levels on exit.
    """
    saved_levels: Dict[str, int] = {}
    for name, lg in logging.root.manager.loggerDict.items():
        if name.startswith("dnadesign.billboard"):
            saved_levels[name] = lg.level
            lg.setLevel(logging.WARNING)

    try:
        yield
    finally:
        for name, lvl in saved_levels.items():
            logging.getLogger(name).setLevel(lvl)

def _inject(cfg: Dict, key: str, default_block: Dict) -> None:
    """Insert *default_block* if **key** is missing from *cfg*."""
    if key not in cfg:
        cfg[key] = default_block.copy()


def make_temp_billboard_config(config: Dict, temp_pt_path: str) -> Dict:
    """
    Build a *minimal* Billboard configuration purely from the libshuffle YAML.

    *   Uses only the metric names present under
        `libshuffle_core_metrics` **or** `billboard_metric.core_metrics`.

    *   Injects metric‑specific option blocks that Billboard expects
        (`motif_string_levenshtein`, etc.).

    *   Always sets:
        - `dry_run            = True`   (skip figures / CSVs)
        - `skip_aligner_call  = False`  (so NW similarities are computed)
        - `pt_files           = [temp_pt_path]`
    """
    bb_cfg: Dict = {}

    
    # Mandatory runtime flags
    bb_cfg["dry_run"] = True
    bb_cfg["skip_aligner_call"] = False
    bb_cfg["pt_files"] = [temp_pt_path]
    bb_cfg["output_dir_prefix"] = (
        "temp_billboard_" + next(tempfile._get_candidate_names())
    )

    # Collect metrics from libshuffle config
    lib_core: List[str] = config.get("libshuffle_core_metrics", [])
    if not lib_core:
        lib_core = config.get("billboard_metric", {}).get("core_metrics", [])
    bb_cfg["diversity_metrics"] = lib_core

    
    # Inject per‑metric option blocks that Billboard expects
    if "min_motif_string_levenshtein" in lib_core:
        _inject(
            bb_cfg,
            "motif_string_levenshtein",
            {
                "tf_penalty": 1.0,
                "strand_penalty": 0.5,
                "partial_penalty": 0.8,
            },
        )

    return bb_cfg


def compute_billboard_metric(subsample: List[dict], config: Dict):
    """
    Run Billboard on a single *subsample* (list of sequence dicts).

    Returns either:
    *   the full **core‑metrics dict** (if `billboard_metric.composite_score`
        is **True**), **or**
    *   the single requested metric value (if `composite_score` is **False**).

    Needleman–Wunsch similarity / dissimilarity are *always* appended to
    the core‑metrics dict so they can be used downstream.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_pt = Path(tmpdir) / "subsample.pt"
        torch.save(subsample, temp_pt)

        bb_cfg = make_temp_billboard_config(config, str(temp_pt))

        # Billboard *requires* the output folder structure, even in dry‑run.
        (Path(tmpdir) / "batch_results" / "csvs").mkdir(parents=True, exist_ok=True)
        (Path(tmpdir) / "batch_results" / "plots").mkdir(parents=True, exist_ok=True)

        # ── Run Billboard
        results = process_sequences([str(temp_pt)], bb_cfg)
        core = compute_core_metrics(results, bb_cfg)

        # ── Inject NW metrics
        seqs = results["sequences"]
        nw_sim = mean_pairwise(seqs, sequence_key="sequence", use_cache=False)
        core["nw_similarity"] = nw_sim
        core["nw_dissimilarity"] = 1.0 - nw_sim

        # ── Decide what to return
        bm_conf = config.get("billboard_metric", {})
        if bm_conf.get("composite_score", False):
            return core

        requested_metrics = bm_conf.get("core_metrics", [])
        if len(requested_metrics) != 1:
            raise ValueError(
                "When composite_score is False, exactly **one** core metric "
                "must be listed in billboard_metric.core_metrics."
            )

        key = requested_metrics[0]
        if key not in core and f"{key}_mean" in core:
            key = f"{key}_mean"  # allow _mean fallback
        if key not in core:
            raise KeyError(
                f"Metric '{key}' not present in Billboard results: {list(core)}"
            )
        return core[key]


def compute_evo2_metric(subsample: List[dict], config: Dict) -> float:
    """Mean pairwise distance in Evo2 latent space (L2 / log1p‑L2 / cosine)."""
    vectors = []
    for entry in subsample:
        vec = entry.get("evo2_logits_mean_pooled")
        if vec is None:
            raise ValueError(
                "Entry missing 'evo2_logits_mean_pooled' field required for Evo2 metric."
            )
        if isinstance(vec, torch.Tensor):
            vec = vec.to(torch.float32).flatten()
        else:
            vec = torch.tensor(vec, dtype=torch.float32).flatten()
        vectors.append(vec)

    if len(vectors) < 2:  # single‑sequence edge‑case
        return 0.0

    mat = torch.stack(vectors)
    metric_type = config.get("evo2_metric", {}).get("type", "l2")

    if metric_type == "l2":
        distances = torch.cdist(mat, mat, p=2)
        pairwise = distances[torch.triu(torch.ones_like(distances, dtype=bool), 1)]
        return pairwise.mean().item()

    if metric_type == "log1p_l2":
        distances = torch.cdist(mat, mat, p=2)
        pairwise = distances[torch.triu(torch.ones_like(distances, dtype=bool), 1)]
        return math.log1p(pairwise.mean().item())

    if metric_type == "cosine":
        normed = F.normalize(mat, p=2, dim=1)
        cos_dist = 1.0 - (normed @ normed.T)
        pairwise = cos_dist[torch.triu(torch.ones_like(cos_dist, dtype=bool), 1)]
        return pairwise.mean().item()

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