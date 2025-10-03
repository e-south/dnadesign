"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/cluster/src/algo/leiden.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np


def _imports():
    try:
        import scanpy as sc
    except Exception as e:
        raise RuntimeError(
            "scanpy is required for Leiden clustering. Install scanpy==1.10.x"
        ) from e
    return sc


def run(
    X: np.ndarray,
    neighbors: int = 15,
    resolution: float = 0.3,
    scale: bool = False,
    metric: str = "euclidean",
    seed: int | None = 42,
    backend: str = "igraph",
) -> np.ndarray:
    sc = _imports()
    ad = sc.AnnData(X.astype(np.float32, copy=False))
    # Optional scaling
    if scale:
        sc.pp.scale(ad)
    sc.pp.neighbors(
        ad, n_neighbors=neighbors, use_rep="X", metric=metric, random_state=seed
    )
    if backend not in {"leidenalg", "igraph"}:
        raise ValueError("backend must be 'leidenalg' or 'igraph'")
    # directed=False works for both backends; n_iterations is recommended for igraph
    kwargs = {
        "resolution": resolution,
        "random_state": seed,
        "directed": False,
        "flavor": backend,
    }
    if backend == "igraph":
        kwargs["n_iterations"] = 2
    sc.tl.leiden(ad, **kwargs)
    labels = ad.obs["leiden"].to_numpy()
    return labels
