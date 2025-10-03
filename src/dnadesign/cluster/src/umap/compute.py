"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/umap/compute.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np


def _imports():
    try:
        import scanpy as sc
    except Exception as e:
        raise RuntimeError("scanpy is required for UMAP. Install scanpy==1.10.x") from e
    return sc


def compute(
    X: np.ndarray,
    neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    seed: int | None = 42,
) -> np.ndarray:
    sc = _imports()
    ad = sc.AnnData(X.astype(np.float32, copy=False))
    sc.pp.neighbors(
        ad, n_neighbors=neighbors, use_rep="X", metric=metric, random_state=seed
    )
    sc.tl.umap(ad, min_dist=min_dist, random_state=seed)
    return ad.obsm["X_umap"]
