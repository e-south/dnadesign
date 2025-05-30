"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/metrics.py

Utility functions that let **libshuffle** call **billboard** on-the-fly.

- compute_pairwise_stats: returns mean/min for cosine & Euclidean.
- compute_evo2_pairwise_matrix: full NxN cosine-dissimilarity for selection.
- compute_billboard_metric: runs Billboard and filters metrics, mapping nw.

Module Author(s): Eric J. South  
Dunlop Lab
--------------------------------------------------------------------------------
"""

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from dnadesign.aligner.metrics import mean_pairwise
from dnadesign.billboard.core import compute_core_metrics, process_sequences


@contextmanager
def _silence_billboard():
    keys = [n for n in logging.root.manager.loggerDict if n.startswith("dnadesign.billboard")]
    orig_levels = {n: logging.getLogger(n).level for n in keys}
    for n in keys:
        logging.getLogger(n).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for n, lvl in orig_levels.items():
            logging.getLogger(n).setLevel(lvl)


def compute_pairwise_stats(subsample: List[dict]) -> Dict[str, float]:
    """
    Compute both mean and min of pairwise distances:
      - cosine dissimilarity
      - Euclidean distance
    Returns keys: mean_cosine, min_cosine, mean_euclidean, min_euclidean
    """
    vecs = []
    for e in subsample:
        v = e.get("evo2_logits_mean_pooled")
        if v is None:
            raise KeyError("Missing Evo2 embedding in subsample.")
        t = v.flatten().float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32)
        vecs.append(t)
    M = torch.stack(vecs)
    if M.size(0) < 2:
        return {k: 0.0 for k in ["mean_cosine", "min_cosine", "mean_euclidean", "min_euclidean"]}
    D_euc = torch.cdist(M, M, p=2)
    N = F.normalize(M, dim=1)
    D_cos = 1.0 - (N @ N.T)
    mask = torch.triu(torch.ones_like(D_euc, dtype=bool), diagonal=1)
    vals_euc = D_euc[mask]
    vals_cos = D_cos[mask]
    return {
        "mean_cosine": float(vals_cos.mean().item()),
        "min_cosine": float(vals_cos.min().item()),
        "mean_euclidean": float(vals_euc.mean().item()),
        "min_euclidean": float(vals_euc.min().item()),
    }


def compute_evo2_pairwise_matrix(sequences: List[dict], indices: List[int], cfg: Any) -> np.ndarray:
    """
    For selection: full NxN matrix of cosine dissimilarity for given indices.
    """
    vecs = []
    for i in indices:
        emb = sequences[i].get("evo2_logits_mean_pooled")
        if emb is None:
            raise KeyError(f"Missing Evo2 embedding for sequence index {i}")
        t = emb.flatten().float() if isinstance(emb, torch.Tensor) else torch.tensor(emb, dtype=torch.float32)
        vecs.append(t)
    M = torch.stack(vecs)
    if M.size(0) < 2:
        return np.zeros((M.size(0), M.size(0)), dtype=float)
    N = F.normalize(M, dim=1)
    D = 1.0 - (N @ N.T)
    return D.cpu().numpy()


def compute_billboard_metric(subsample: List[dict], cfg: Any) -> dict:
    """
    Run Billboard on a subsample, then compute & inject a fresh
    NW-based dissimilarity (min_nw_dissimilarity) with caching disabled.
    """
    # 1) run Billboard (dry run) to get core metrics
    with tempfile.TemporaryDirectory() as td, _silence_billboard():
        temp_pt = Path(td) / "subsample.pt"
        torch.save(subsample, temp_pt)

        bb_cfg = {
            "diversity_metrics": cfg.billboard_core_metrics,
            "pt_files": [str(temp_pt)],
            "dry_run": True,
            "skip_aligner_call": False,
        }
        # ensure motif_string penalty injected if needed
        if "min_motif_string_levenshtein" in cfg.billboard_core_metrics:
            bb_cfg["motif_string_levenshtein"] = {
                "tf_penalty": 1.0,
                "strand_penalty": 0.5,
                "partial_penalty": 0.8,
            }

        results = process_sequences([str(temp_pt)], bb_cfg)
        full_core = compute_core_metrics(results, bb_cfg)

    # 2) compute NW similarity *without* on‑disk caching
    nw_sim = mean_pairwise(
        subsample, use_cache=False, cache_dir=None  # ← turn off disk cache  # ← avoids any default cache directory
    )

    # 3) assemble final core dict, mapping Billboard + our NW dissimilarity
    core: dict = {}
    for m in cfg.billboard_core_metrics:
        if m in full_core:
            core[m] = full_core[m]
        elif m == "min_nw_dissimilarity" and "nw_dissimilarity" in full_core:
            # if user asked for min_nw_dissimilarity but we only got nw_dissimilarity
            core[m] = full_core["nw_dissimilarity"]

    # overwrite / add the true fresh min_nw_dissimilarity
    core["min_nw_dissimilarity"] = 1.0 - nw_sim

    return core
