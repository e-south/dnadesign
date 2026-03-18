"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/methods/kmeans.py

MiniBatch K-Means clustering implementation for faster local and large-scale runs.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

KMEANS_FIT_PARAM_NAMES = frozenset(
    {
        "batch_size",
        "init",
        "max_iter",
        "n_clusters",
        "n_init",
        "random_state",
        "reassignment_ratio",
        "tol",
    }
)


def _imports():
    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:
        raise RuntimeError(
            "scikit-learn is required for K-Means clustering. Install scikit-learn to use method='kmeans'."
        ) from exc
    return MiniBatchKMeans


def _coerce_positive_int(value: Any, *, param_name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"K-Means fit param '{param_name}' must be > 0.")
    return resolved


def _coerce_non_negative_float(value: Any, *, param_name: str) -> float:
    resolved = float(value)
    if resolved < 0:
        raise ValueError(f"K-Means fit param '{param_name}' must be >= 0.")
    return resolved


def _coerce_init(value: Any) -> str:
    text = str(value).strip()
    if text not in {"k-means++", "random"}:
        raise ValueError("K-Means fit param 'init' must be one of: k-means++, random.")
    return text


def _coerce_n_init(value: Any) -> int | str:
    if isinstance(value, str) and value.strip() == "auto":
        return "auto"
    return _coerce_positive_int(value, param_name="n_init")


def resolve_fit_params(
    preset: Mapping[str, Any] | None = None,
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, int | float | str]:
    merged = dict(preset or {})
    if raw_params:
        unexpected = sorted(set(raw_params).difference(KMEANS_FIT_PARAM_NAMES))
        if unexpected:
            raise ValueError(
                "Unsupported K-Means fit params: "
                + ", ".join(unexpected)
                + ". Pass only method-specific params supported by the selected method."
            )
        merged.update(raw_params)
    return {
        "n_clusters": _coerce_positive_int(merged.get("n_clusters", 8), param_name="n_clusters"),
        "batch_size": _coerce_positive_int(merged.get("batch_size", 1024), param_name="batch_size"),
        "max_iter": _coerce_positive_int(merged.get("max_iter", 100), param_name="max_iter"),
        "random_state": int(merged.get("random_state", 42)),
        "n_init": _coerce_n_init(merged.get("n_init", "auto")),
        "reassignment_ratio": _coerce_non_negative_float(
            merged.get("reassignment_ratio", 0.01),
            param_name="reassignment_ratio",
        ),
        "tol": _coerce_non_negative_float(merged.get("tol", 0.0), param_name="tol"),
        "init": _coerce_init(merged.get("init", "k-means++")),
    }


def slug_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {"k": int(params["n_clusters"])}


def run(
    X: np.ndarray,
    *,
    n_clusters: int = 8,
    batch_size: int = 1024,
    max_iter: int = 100,
    random_state: int = 42,
    n_init: int | str = "auto",
    reassignment_ratio: float = 0.01,
    tol: float = 0.0,
    init: str = "k-means++",
) -> np.ndarray:
    MiniBatchKMeans = _imports()
    model = MiniBatchKMeans(
        n_clusters=int(n_clusters),
        batch_size=int(batch_size),
        max_iter=int(max_iter),
        random_state=int(random_state),
        n_init=n_init,
        reassignment_ratio=float(reassignment_ratio),
        tol=float(tol),
        init=str(init),
    )
    return model.fit_predict(X.astype(np.float32, copy=False))
