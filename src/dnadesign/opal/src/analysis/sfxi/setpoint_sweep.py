"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/analysis/sfxi/setpoint_sweep.py

Setpoint sweep utilities for SFXI objective diagnostics.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from ...objectives import sfxi_math
from .gates import GATE_LIBRARY
from .intensity_scaling import summarize_intensity_scaling
from .state_order import require_state_order


def format_setpoint_label(setpoint: Sequence[float], *, precision: int = 2) -> str:
    vec = np.asarray(setpoint, dtype=float).ravel()
    if vec.size != 4:
        raise ValueError("setpoint must have length 4.")
    vals = []
    for v in vec.tolist():
        text = f"{float(v):.{int(precision)}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        vals.append(text)
    return "[" + ",".join(vals) + "]"


@dataclass(frozen=True)
class SetpointSpec:
    name: str
    vector: np.ndarray
    source: str


def build_setpoint_library(
    current_setpoint: Sequence[float],
    *,
    include_truth_tables: bool = True,
    include_current: bool = True,
    state_order: Sequence[str] | None = None,
) -> list[SetpointSpec]:
    require_state_order(state_order)
    library: list[SetpointSpec] = []
    if include_truth_tables:
        for gate in GATE_LIBRARY:
            library.append(SetpointSpec(name=gate.code, vector=gate.vector.copy(), source="truth_table"))
    if include_current:
        cur = np.asarray(current_setpoint, dtype=float).ravel()
        if cur.size != 4 or not np.all(np.isfinite(cur)):
            raise ValueError("current_setpoint must be a finite length-4 vector.")
        library.append(SetpointSpec(name="current", vector=cur.copy(), source="current"))
    return library


def sweep_setpoints(
    labels_vec8: np.ndarray,
    *,
    current_setpoint: Sequence[float],
    percentile: int,
    min_n: int,
    eps: float,
    delta: float,
    beta: float,
    gamma: float,
    state_order: Sequence[str] | None = None,
    pool_vec8: np.ndarray | None = None,
) -> pl.DataFrame:
    order = require_state_order(state_order)
    beta_val = float(beta)
    gamma_val = float(gamma)
    if not np.isfinite(beta_val):
        raise ValueError("beta must be finite.")
    if not np.isfinite(gamma_val):
        raise ValueError("gamma must be finite.")
    labels = np.asarray(labels_vec8, dtype=float)
    if labels.ndim == 1:
        labels = labels.reshape(1, -1)
    if labels.shape[1] < 8:
        raise ValueError("labels_vec8 must have shape (n, 8+).")
    if not np.all(np.isfinite(labels)):
        raise ValueError("labels_vec8 must be finite.")

    v_obs = labels[:, 0:4]
    y_star = labels[:, 4:8]

    pool = None
    if pool_vec8 is not None:
        pool = np.asarray(pool_vec8, dtype=float)
        if pool.ndim == 1:
            pool = pool.reshape(1, -1)
        if pool.shape[1] < 8:
            raise ValueError("pool_vec8 must have shape (n, 8+).")
        if not np.all(np.isfinite(pool)):
            raise ValueError("pool_vec8 must be finite.")

    library = build_setpoint_library(current_setpoint, state_order=order)
    rows: list[dict] = []
    for spec in library:
        p = np.asarray(spec.vector, dtype=float).ravel()
        F_logic = sfxi_math.logic_fidelity(v_obs, p)

        scaling = summarize_intensity_scaling(
            y_star,
            setpoint=p,
            delta=delta,
            percentile=percentile,
            min_n=min_n,
            eps=eps,
            state_order=order,
        )
        effect_scaled = np.asarray(scaling.effect_scaled, dtype=float)
        score = np.power(F_logic, beta_val)
        if not scaling.intensity_disabled:
            score = score * np.power(effect_scaled, gamma_val)
        row = {
            "setpoint_name": spec.name,
            "setpoint_source": spec.source,
            "setpoint_vector": p.tolist(),
            "setpoint_label": format_setpoint_label(p),
            "logic_fidelity": float(np.nanmedian(F_logic)) if F_logic.size else float("nan"),
            "effect_scaled": float(np.nanmedian(effect_scaled)) if effect_scaled.size else float("nan"),
            "score": float(np.nanmedian(score)) if score.size else float("nan"),
            "denom_used": float(scaling.denom),
            "clip_lo_fraction": float(scaling.clip_lo_fraction),
            "clip_hi_fraction": float(scaling.clip_hi_fraction),
            "intensity_disabled": bool(scaling.intensity_disabled),
        }
        if pool is not None:
            pool_y_star = pool[:, 4:8]
            pool_effect_raw, _ = sfxi_math.effect_raw_from_y_star(
                pool_y_star,
                p,
                delta=delta,
                eps=eps,
                state_order=order,
            )
            if scaling.intensity_disabled:
                pool_effect_scaled = np.ones(pool_effect_raw.shape[0], dtype=float)
            else:
                pool_effect_scaled = sfxi_math.effect_scaled(pool_effect_raw, float(scaling.denom))
            pool_clip_lo = float(np.mean(pool_effect_scaled <= 0.0 + 1e-12)) if pool_effect_scaled.size else 0.0
            pool_clip_hi = float(np.mean(pool_effect_scaled >= 1.0 - 1e-12)) if pool_effect_scaled.size else 0.0
            row["pool_clip_lo_fraction"] = pool_clip_lo
            row["pool_clip_hi_fraction"] = pool_clip_hi
        rows.append(row)

    return pl.DataFrame(rows)
