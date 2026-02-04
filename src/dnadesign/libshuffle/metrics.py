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

from __future__ import annotations

import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from dnadesign.aligner.metrics import compute_alignment_scores
from dnadesign.billboard.core import compute_core_metrics, process_sequences


# ──────────────────────────────────────────────────────────────────────────────
@contextmanager
def _silence_billboard():
    keys = [n for n in logging.root.manager.loggerDict if n.startswith("dnadesign.billboard")]
    levels = {n: logging.getLogger(n).level for n in keys}
    try:
        for n in keys:
            logging.getLogger(n).setLevel(logging.WARNING)
        yield
    finally:
        for n, lvl in levels.items():
            logging.getLogger(n).setLevel(lvl)


# ──────────────────────────────────────────────────────────────────────────────
def _min_hamming_distance(seqs: List[str]) -> int:
    """
    Compute the minimum pair-wise Hamming distance for a list of equal-length
    sequences.
    """
    if len(seqs) < 2:
        return 0
    L = len(seqs[0])
    # fast path: vectorise with NumPy
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype="|S1").reshape(len(seqs), L)
    min_hd = L  # upper bound
    for i in range(len(seqs)):
        diffs = (arr[i] != arr[i + 1 :]).sum(axis=1)
        hd_i = int(diffs.min()) if diffs.size else L
        if hd_i < min_hd:
            min_hd = hd_i
            if min_hd == 0:
                break
    return min_hd


def compute_pairwise_stats(subsample: List[dict]) -> Dict[str, float]:
    """
    Average & minimum cosine dissimilarity and Euclidean distance between Evo2
    vectors.
    """
    vecs = []
    for entry in subsample:
        emb = entry.get("evo2_logits_mean_pooled")
        if emb is None:
            raise KeyError("Missing Evo2 embedding in subsample.")
        t = emb.flatten().float() if isinstance(emb, torch.Tensor) else torch.tensor(emb, dtype=torch.float32)
        vecs.append(t)

    M = torch.stack(vecs)
    if M.size(0) < 2:
        return {k: 0.0 for k in ("mean_cosine", "min_cosine", "mean_euclidean", "min_euclidean")}

    D_euc = torch.cdist(M, M, p=2)
    N = F.normalize(M, dim=1)
    D_cos = 1.0 - (N @ N.T)
    mask = torch.triu(torch.ones_like(D_euc, dtype=bool), diagonal=1)

    return {
        "mean_cosine": float(D_cos[mask].mean()),
        "min_cosine": float(D_cos[mask].min()),
        "mean_euclidean": float(D_euc[mask].mean()),
        "min_euclidean": float(D_euc[mask].min()),
    }


def compute_evo2_pairwise_matrix(sequences: List[dict], indices: List[int], cfg: Any) -> np.ndarray:
    """
    Return the full cosine-dissimilarity matrix for the given indices.
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
        return np.zeros((M.size(0), M.size(0)))
    N = F.normalize(M, dim=1)
    return (1.0 - (N @ N.T)).cpu().numpy()


def compute_billboard_metric(subsample: List[dict], cfg: Any) -> dict:
    """
    Run Billboard in “dry-run” mode to obtain core metrics, then inject:
      • `min_nw_dissimilarity`  (unchanged, normalised 0-1)
      • `min_hamming_distance`  (integer bp count)
    """
    # ── 1. Billboard core metrics ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td, _silence_billboard():
        temp_pt = Path(td) / "subsample.pt"
        torch.save(subsample, temp_pt)

        bb_cfg = {
            "diversity_metrics": cfg.billboard_core_metrics,
            "pt_files": [str(temp_pt)],
            "dry_run": True,
            "skip_aligner_call": False,
        }
        if "min_motif_string_levenshtein" in cfg.billboard_core_metrics:
            bb_cfg["motif_string_levenshtein"] = {
                "tf_penalty": 1.0,
                "strand_penalty": 0.5,
                "partial_penalty": 0.8,
            }

        results = process_sequences([str(temp_pt)], bb_cfg)
        core = compute_core_metrics(results, bb_cfg)

    # ── 2. Needleman–Wunsch (similarity → dissimilarity) ─────────────────────
    seqs = [s["sequence"] for s in subsample]
    aln = compute_alignment_scores(seqs, return_formats=("condensed",), verbose=False)
    sim_vec = np.asarray(aln["condensed"], dtype=float)
    core["min_nw_dissimilarity"] = float((1.0 - sim_vec).min())

    # ── 3. Hamming distance (integer bp) ─────────────────────────────────────
    core["min_hamming_distance"] = _min_hamming_distance(seqs)
    core["sequence_length"] = len(seqs[0]) if seqs else 0

    return core
