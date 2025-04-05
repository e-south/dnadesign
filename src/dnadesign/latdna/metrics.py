"""
--------------------------------------------------------------------------------
<dnadesign project>
latdna/metrics.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_distances

def cosine(lat_vecs: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise cosine dissimilarity (1 - cosine similarity) for latent vectors.
    Input: lat_vecs with shape (N, D)
    Output: Pairwise distance matrix.
    """
    return cosine_distances(lat_vecs)

def euclidean(lat_vecs: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances for latent vectors.
    """
    diff = lat_vecs[:, None, :] - lat_vecs[None, :, :]
    return np.linalg.norm(diff, axis=2)

def log1p_euclidean(lat_vecs: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distances and apply log1p transformation.
    """
    euclid = euclidean(lat_vecs)
    return np.log1p(euclid)

# Registry for easy dispatch
METRIC_REGISTRY = {
    "cosine": cosine,
    "euclidean": euclidean,
    "log1p_euclidean": log1p_euclidean,
}