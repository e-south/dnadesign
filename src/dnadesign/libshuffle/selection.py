"""
--------------------------------------------------------------------------------
<dnadesign project>
libshuffle/selection.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""
import numpy as np
import logging
from dnadesign.libshuffle.visualization import Plotter

logger = logging.getLogger('libshuffle.selection')

def select_best_subsample(subsamples, cfg, sequences=None):
    """
    1) Compute the same mean_cosine threshold as in the scatter.
    2) Filter to subsamples with mean_cosine >= threshold.
    3) Drop any literal‑dropped (min_jaccard_dissimilarity == 0 or min_motif_string_levenshtein == 0).
    3.1) Drop subsamples where not every sequence occupies its own unique Leiden cluster.
    4) Among those survivors, pick the one with highest raw min_euclidean.
    """
    # 1) extract mean_cosine and compute threshold
    mc = np.array([s.get('mean_cosine', np.nan) for s in subsamples], dtype=float)
    valid = mc[~np.isnan(mc)]
    thr_cfg = cfg.plot.scatter.threshold
    thr = Plotter.compute_threshold(valid, thr_cfg)
    logger.info(f"Selection applying mean_cosine threshold: {thr:.3e}")

    # 2) filter by threshold
    pre = [
        s for s, m in zip(subsamples, mc)
        if not np.isnan(m) and m >= thr
    ]
    logger.info(f"Selection after threshold (>=): {len(pre)}")

    # 3) drop literal‑dropped
    survivors = [
        s for s in pre
        if s['raw_billboard'].get('min_jaccard_dissimilarity', 1) > 0
        and s['raw_billboard'].get('min_motif_string_levenshtein', 1) > 0
    ]
    logger.info(f"Selection survivors count (post‑literal‑drop): {len(survivors)}")
    if not survivors:
        raise ValueError("No survivors after threshold + literal‑drop filters.")

    # 3.1) drop any subsample where cluster uniqueness is violated
    survivors = [
        s for s in survivors
        if s.get('unique_cluster_count', 0) == cfg.subsample_size
    ]
    logger.info(f"Selection survivors count (post‑cluster‑uniqueness): {len(survivors)}")
    if not survivors:
        raise ValueError("No survivors after requiring unique Leiden clusters.")

    # 4) among those, pick best by raw min_euclidean
    best = max(survivors, key=lambda s: s.get('min_euclidean', -np.inf))
    best['passed_selection'] = True
    logger.info(
        f"Selection chosen: {best['subsample_id']} "
        f"with min_euclidean={best['min_euclidean']:.3e}"
    )

    return best