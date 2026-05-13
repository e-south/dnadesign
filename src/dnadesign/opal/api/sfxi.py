"""
Public SFXI scoring API.

This module is the supported cross-package boundary for consumers that need
OPAL-compatible SFXI scores without importing OPAL internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from dnadesign.opal.src.objectives import sfxi_math

SFXI_API_VERSION = "1"
SFXI_OBJECTIVE_NAME = "sfxi_v1"
SFXI_STATE_ORDER = sfxi_math.STATE_ORDER


@dataclass(frozen=True)
class SFXIScoringConfig:
    setpoint_vector: Sequence[float] = (0.0, 0.0, 0.0, 1.0)
    scaling_percentile: int = 95
    scaling_min_n: int = 5
    scaling_eps: float = 1.0e-8
    logic_exponent_beta: float = 1.0
    intensity_exponent_gamma: float = 1.0
    intensity_log2_offset_delta: float = 0.0


@dataclass(frozen=True)
class SFXIScoringResult:
    logic_fidelity: np.ndarray
    effect_raw: np.ndarray
    effect_scaled: np.ndarray
    sfxi: np.ndarray
    denom_used: float
    denom_percentile: int
    setpoint_vector: tuple[float, float, float, float]
    clip_lo_mask: np.ndarray
    clip_hi_mask: np.ndarray
    intensity_disabled: bool
    objective_name: str = SFXI_OBJECTIVE_NAME
    api_version: str = SFXI_API_VERSION
    state_order: tuple[str, str, str, str] = SFXI_STATE_ORDER

    def to_records(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for idx in range(len(self.sfxi)):
            rows.append(
                {
                    "objective_name": self.objective_name,
                    "api_version": self.api_version,
                    "state_order": list(self.state_order),
                    "setpoint_vector": list(self.setpoint_vector),
                    "denom_percentile": int(self.denom_percentile),
                    "denom_used": float(self.denom_used),
                    "logic_fidelity": float(self.logic_fidelity[idx]),
                    "effect_raw": float(self.effect_raw[idx]),
                    "effect_scaled": float(self.effect_scaled[idx]),
                    "sfxi": float(self.sfxi[idx]),
                    "clip_lo_mask": bool(self.clip_lo_mask[idx]),
                    "clip_hi_mask": bool(self.clip_hi_mask[idx]),
                    "intensity_disabled": bool(self.intensity_disabled),
                }
            )
        return rows


def _coerce_vec8(value: np.ndarray | Sequence[Sequence[float]], *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 8:
        raise ValueError(f"{name} must have shape (n, 8+).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    return arr


def _parse_config(config: SFXIScoringConfig) -> tuple[np.ndarray, int, int, float, float, float, float]:
    if not isinstance(config, SFXIScoringConfig):
        raise TypeError("config must be an SFXIScoringConfig.")
    setpoint = sfxi_math.parse_setpoint_vector({"setpoint_vector": list(config.setpoint_vector)})
    percentile = int(config.scaling_percentile)
    min_n = int(config.scaling_min_n)
    eps = float(config.scaling_eps)
    beta = float(config.logic_exponent_beta)
    gamma = float(config.intensity_exponent_gamma)
    delta = float(config.intensity_log2_offset_delta)
    if not (1 <= percentile <= 100):
        raise ValueError(f"sfxi_v1: scaling.percentile must be in [1, 100]; got {percentile}.")
    if min_n < 1:
        raise ValueError(f"sfxi_v1: scaling.min_n must be >= 1; got {min_n}.")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"sfxi_v1: scaling.eps must be positive and finite; got {eps}.")
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError(f"sfxi_v1: logic_exponent_beta must be >= 0; got {beta}.")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError(f"sfxi_v1: intensity_exponent_gamma must be >= 0; got {gamma}.")
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError(f"sfxi_v1: intensity_log2_offset_delta must be >= 0; got {delta}.")
    return setpoint, percentile, min_n, eps, beta, gamma, delta


def score_vec8(
    vec8: np.ndarray | Sequence[Sequence[float]],
    config: SFXIScoringConfig,
    *,
    scaling_vec8: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> SFXIScoringResult:
    candidates = _coerce_vec8(vec8, name="vec8")
    scaling_pool = candidates if scaling_vec8 is None else _coerce_vec8(scaling_vec8, name="scaling_vec8")
    setpoint, percentile, min_n, eps, beta, gamma, delta = _parse_config(config)

    v_hat = np.clip(candidates[:, 0:4].astype(float), 0.0, 1.0)
    y_star = candidates[:, 4:8].astype(float)
    pool_y_star = scaling_pool[:, 4:8].astype(float)

    logic_fidelity = sfxi_math.logic_fidelity(v_hat, setpoint)
    effect_raw, weights = sfxi_math.effect_raw_from_y_star(
        y_star,
        setpoint,
        delta=delta,
        eps=eps,
        state_order=SFXI_STATE_ORDER,
    )
    intensity_disabled = bool(not np.any(weights))
    if intensity_disabled:
        denom = 1.0
        effect_scaled = np.ones(candidates.shape[0], dtype=float)
    else:
        denom = sfxi_math.denom_from_labels(
            pool_y_star,
            setpoint,
            delta=delta,
            percentile=percentile,
            min_n=min_n,
            eps=eps,
            state_order=SFXI_STATE_ORDER,
        )
        effect_scaled = sfxi_math.effect_scaled(effect_raw, denom)

    sfxi = np.power(logic_fidelity, beta) * np.power(effect_scaled, gamma)
    return SFXIScoringResult(
        logic_fidelity=np.asarray(logic_fidelity, dtype=float).ravel(),
        effect_raw=np.asarray(effect_raw, dtype=float).ravel(),
        effect_scaled=np.asarray(effect_scaled, dtype=float).ravel(),
        sfxi=np.asarray(sfxi, dtype=float).ravel(),
        denom_used=float(denom),
        denom_percentile=int(percentile),
        setpoint_vector=tuple(float(x) for x in setpoint.tolist()),  # type: ignore[arg-type]
        clip_lo_mask=np.asarray(effect_scaled <= 0.0 + 1.0e-12, dtype=bool).ravel(),
        clip_hi_mask=np.asarray(effect_scaled >= 1.0 - 1.0e-12, dtype=bool).ravel(),
        intensity_disabled=intensity_disabled,
    )


__all__ = [
    "SFXI_API_VERSION",
    "SFXI_OBJECTIVE_NAME",
    "SFXI_STATE_ORDER",
    "SFXIScoringConfig",
    "SFXIScoringResult",
    "score_vec8",
]
