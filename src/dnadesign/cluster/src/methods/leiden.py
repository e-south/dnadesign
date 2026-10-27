"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/methods/leiden.py

Leiden clustering implementation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

LEIDEN_FIT_PARAM_NAMES = frozenset({"backend", "metric", "neighbors", "random_state", "resolution", "scale"})


def _imports():
    try:
        import scanpy as sc
    except Exception as e:
        raise RuntimeError("scanpy is required for Leiden clustering. Install scanpy==1.10.x") from e
    return sc


def _coerce_bool(value: Any, *, param_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Leiden fit param '{param_name}' must be boolean-like, got {value!r}.")


def _coerce_choice(value: Any, *, param_name: str, allowed: set[str]) -> str:
    text = str(value).strip()
    if text not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"Leiden fit param '{param_name}' must be one of: {choices}.")
    return text


def resolve_fit_params(
    preset: Mapping[str, Any] | None = None,
    raw_params: Mapping[str, Any] | None = None,
) -> dict[str, int | float | bool | str]:
    merged = dict(preset or {})
    if raw_params:
        unexpected = sorted(set(raw_params).difference(LEIDEN_FIT_PARAM_NAMES))
        if unexpected:
            raise ValueError(
                "Unsupported Leiden fit params: "
                + ", ".join(unexpected)
                + ". Pass only method-specific params supported by the selected method."
            )
        merged.update(raw_params)
    return {
        "neighbors": int(merged.get("neighbors", 15)),
        "resolution": float(merged.get("resolution", 0.30)),
        "scale": _coerce_bool(merged.get("scale", False), param_name="scale"),
        "metric": str(merged.get("metric", "euclidean")),
        "random_state": int(merged.get("random_state", 42)),
        "backend": _coerce_choice(
            merged.get("backend", "igraph"),
            param_name="backend",
            allowed={"igraph", "leidenalg"},
        ),
    }


def slug_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n": int(params["neighbors"]),
        "r": float(params["resolution"]),
    }


def run(
    X: np.ndarray,
    *,
    neighbors: int = 15,
    resolution: float = 0.3,
    scale: bool = False,
    metric: str = "euclidean",
    random_state: int | None = 42,
    backend: str = "igraph",
) -> np.ndarray:
    sc = _imports()
    ad = sc.AnnData(X.astype(np.float32, copy=False))
    if scale:
        sc.pp.scale(ad)
    sc.pp.neighbors(ad, n_neighbors=neighbors, use_rep="X", metric=metric, random_state=random_state)
    if backend not in {"leidenalg", "igraph"}:
        raise ValueError("backend must be 'leidenalg' or 'igraph'")
    kwargs = {
        "resolution": resolution,
        "random_state": random_state,
        "directed": False,
        "flavor": backend,
    }
    if backend == "igraph":
        kwargs["n_iterations"] = 2
    sc.tl.leiden(ad, **kwargs)
    return ad.obs["leiden"].to_numpy()
