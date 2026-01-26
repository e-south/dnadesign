"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/objectives/sfxi_math.py

Pure SFXI math helpers shared by objectives and diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

STATE_ORDER = ("00", "10", "01", "11")


def assert_state_order(order: Sequence[str]) -> None:
    if tuple(order) != STATE_ORDER:
        raise ValueError(f"Invalid state order: {order}. Expected {STATE_ORDER}.")


def parse_setpoint_vector(params: dict) -> np.ndarray:
    setpoint = np.asarray(params.get("setpoint_vector", [0, 0, 0, 1]), dtype=float).ravel()
    if setpoint.size != 4:
        raise ValueError("[sfxi_math] setpoint_vector must have length 4.")
    if not np.all(np.isfinite(setpoint)):
        raise ValueError("[sfxi_math] setpoint_vector must be finite.")
    if np.any(setpoint < 0.0) or np.any(setpoint > 1.0):
        raise ValueError("[sfxi_math] setpoint_vector entries must be in [0, 1].")
    return setpoint


def worst_corner_distance(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    if p.size != 4:
        raise ValueError("[sfxi_math] setpoint must have length 4.")
    a = np.maximum(p * p, (1.0 - p) * (1.0 - p))
    return float(np.sqrt(np.sum(a)))


def logic_fidelity(v_hat: np.ndarray, p: np.ndarray) -> np.ndarray:
    v = np.asarray(v_hat, dtype=float)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    if v.shape[1] != 4:
        raise ValueError("[sfxi_math] v_hat must have shape (n, 4).")
    p = np.asarray(p, dtype=float).ravel()
    if p.size != 4:
        raise ValueError("[sfxi_math] setpoint must have length 4.")
    v = np.clip(v, 0.0, 1.0)
    D = worst_corner_distance(p)
    if not np.isfinite(D) or D <= 0.0:
        return np.ones(v.shape[0], dtype=float)
    dist = np.linalg.norm(v - p[None, :], axis=1)
    out = 1.0 - (dist / D)
    return np.clip(out, 0.0, 1.0)


def weights_from_setpoint(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=float).ravel()
    if p.size != 4:
        raise ValueError("[sfxi_math] setpoint must have length 4.")
    total = float(np.sum(p))
    if not np.isfinite(total) or total <= eps:
        return np.zeros_like(p)
    return p / total


def recover_linear_intensity(y_star: np.ndarray, delta: float) -> np.ndarray:
    y = np.asarray(y_star, dtype=float)
    return np.maximum(0.0, np.power(2.0, y) - float(delta))


def effect_raw(y_linear: np.ndarray, weights: np.ndarray) -> np.ndarray:
    y = np.asarray(y_linear, dtype=float)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    if y.shape[1] != 4:
        raise ValueError("[sfxi_math] y_linear must have shape (n, 4).")
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != 4:
        raise ValueError("[sfxi_math] weights must have length 4.")
    return (y * w[None, :]).sum(axis=1)


def effect_scaled(effect_raw_vals: np.ndarray, denom: float) -> np.ndarray:
    if not np.isfinite(denom) or denom <= 0.0:
        raise ValueError(f"[sfxi_math] denom must be positive and finite; got {denom}.")
    raw = np.asarray(effect_raw_vals, dtype=float).ravel()
    return np.clip(raw / float(denom), 0.0, 1.0)


def denom_from_pool(pool: Sequence[float], *, percentile: int, min_n: int, eps: float) -> float:
    arr = np.asarray(pool, dtype=float)
    if arr.size < int(min_n):
        raise ValueError(f"[sfxi_math] Need at least min_n={int(min_n)} labels to compute denom; got {arr.size}.")
    if not (1 <= int(percentile) <= 100):
        raise ValueError(f"[sfxi_math] percentile must be in [1, 100]; got {percentile}.")
    v = float(np.percentile(arr, int(percentile)))
    if not np.isfinite(v):
        raise ValueError(f"[sfxi_math] Invalid denom computed (value={v}). Check labels and scaling config.")
    if v < 0.0:
        raise ValueError(f"[sfxi_math] Denom percentile is negative (value={v}). Check labels and scaling config.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"[sfxi_math] eps must be positive and finite; got {eps}.")
    return max(v, float(eps))


def denom_from_labels(
    y_star_labels: np.ndarray,
    setpoint: np.ndarray,
    *,
    delta: float,
    percentile: int,
    min_n: int,
    eps: float,
) -> float:
    y_star = np.asarray(y_star_labels, dtype=float)
    if y_star.ndim == 1:
        y_star = y_star.reshape(1, -1)
    if y_star.shape[1] != 4:
        raise ValueError("[sfxi_math] y_star must have shape (n, 4).")
    w = weights_from_setpoint(setpoint, eps=1e-12)
    if not np.any(w):
        return 1.0
    y_lin = recover_linear_intensity(y_star, delta=delta)
    pool = effect_raw(y_lin, w)
    return denom_from_pool(pool, percentile=percentile, min_n=min_n, eps=eps)
