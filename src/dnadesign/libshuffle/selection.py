"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/selection.py

Select the best subsample based on
  1.  per-sequence TF richness ≥ cfg.min_tf_richness
  2.  semantic diversity          (mean-cosine threshold)
  3.  literal diversity           (Hamming / Levenshtein / Jaccard / NW)
  4.  Leiden-cluster uniqueness
then maximise the minimum Euclidean gap.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np

from dnadesign.billboard.core import robust_parse_tfbs
from dnadesign.libshuffle.visualization import Plotter

logger = logging.getLogger("libshuffle.selection")


# ───────────────────────────────────────────────────────── helpers ──
def _sequence_tf_set(seq_dict: dict) -> set[str]:
    """Return the set of TFs present in *one* sequence dict."""
    motif_to_tf: dict[str, str] = {}
    for part in seq_dict["meta_tfbs_parts"]:
        try:
            tf, motif = robust_parse_tfbs(part, seq_dict.get("id"))
            motif_to_tf[motif] = tf
        except ValueError:
            continue
    return {
        motif_to_tf[m] for m in seq_dict["meta_tfbs_parts_in_array"] if m in motif_to_tf
    }


def _passes_tf_richness(sample: dict, sequences: Sequence[dict], k: int) -> bool:
    """Every sequence in the subsample must have ≥ k unique TFs."""
    return all(len(_sequence_tf_set(sequences[i])) >= k for i in sample["indices"])


def _passes_literal_filters(sample: dict, cfg) -> bool:
    """Apply literal diversity filters requested in YAML."""
    filters: List[str] = getattr(cfg, "literal_filters", []) or []
    if not filters:
        return True

    core = sample["raw_billboard"]
    min_bp = getattr(cfg, "literal_min_bp_diff", 0)

    for f in filters:
        if f in ("hamming", "nw"):
            if core.get("min_hamming_distance", 0) < min_bp:
                return False
        elif f == "levenshtein" and core.get("min_motif_string_levenshtein", 1) == 0:
            return False
        elif f == "jaccard" and core.get("min_jaccard_dissimilarity", 1) == 0:
            return False
        else:
            logger.warning(f"Unknown literal filter: {f}")
    return True


# ───────────────────────────────────────────────────────── pipeline ──
def select_best_subsample(subsamples, cfg, sequences):
    # 1) TF-richness screen (MOST STRINGENT) -------------------------------
    survivors = [
        s for s in subsamples if _passes_tf_richness(s, sequences, cfg.min_tf_richness)
    ]
    logger.info(
        f"After per-sequence TF-richness ≥ {cfg.min_tf_richness}: " f"{len(survivors)}"
    )

    if not survivors:
        raise ValueError("No subsample satisfies the TF-richness requirement.")

    # 2) semantic diversity (mean-cosine threshold) ------------------------
    mc = np.array([s.get("mean_cosine", np.nan) for s in survivors])
    thr = Plotter.compute_threshold(mc[~np.isnan(mc)], cfg.plot.scatter.threshold)
    logger.info(f"Semantic threshold (mean_cosine) = {thr:.3e}")

    survivors = [s for s in survivors if s.get("mean_cosine", 0.0) >= thr]
    logger.info(f"Survivors after semantic filter: {len(survivors)}")
    if not survivors:
        raise ValueError("All TF-rich subsamples failed the semantic filter.")

    # 3) literal diversity --------------------------------------------------
    survivors = [s for s in survivors if _passes_literal_filters(s, cfg)]
    logger.info(f"After literal filters: {len(survivors)}")
    if not survivors:
        raise ValueError("No subsample passes literal-diversity filters.")

    # 4) Leiden-cluster uniqueness -----------------------------------------
    # survivors = [
    #     s for s in survivors if s.get("unique_cluster_count", 0) == cfg.subsample_size
    # ]
    # logger.info(f"After Leiden uniqueness: {len(survivors)}")
    # if not survivors:
    #     raise ValueError("No subsample passes cluster-uniqueness check.")

    # 5) maximise minimum Euclidean gap ------------------------------------
    best = max(survivors, key=lambda s: s.get("min_euclidean", -np.inf))
    best["passed_selection"] = True
    logger.info(
        f"Winner: {best['subsample_id']} "
        f"(min_euclidean = {best['min_euclidean']:.3e})"
    )
    return best
