"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/api/sfxi.py

Public SFXI scoring API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..src.objectives import sfxi_math

SFXI_API_VERSION = "1"
SFXI_OBJECTIVE_NAME = "sfxi_v1"
SFXI_STATE_ORDER = sfxi_math.STATE_ORDER
SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION = "opal.sfxi_reference_overlay.v1"
SFXI_REFERENCE_OVERLAY_NAMESPACE = "sfxi_ref"
SFXI_REFERENCE_OVERLAY_PREFIX = f"{SFXI_REFERENCE_OVERLAY_NAMESPACE}__"
SFXI_REFERENCE_OVERLAY_FIELDS = (
    "reference_instance_id",
    "collection_id",
    "batch_id",
    "campaign_id",
    "reader_experiment_id",
    "reader_experiment_date",
    "metric_id",
    "metric_value",
    "metric_provenance",
    "source_ref",
    "score_ref",
    "api_version",
    "objective_name",
    "state_order",
    "setpoint_name",
    "setpoint_vector",
    "denom_used",
    "denom_percentile",
    "logic_fidelity",
    "effect_raw",
    "effect_scaled",
    "sfxi",
    "r_logic",
    "time_selected_h",
    "reference_design_id",
    "sequence_source_id",
    "clip_lo_mask",
    "clip_hi_mask",
    "intensity_disabled",
    "flat_logic",
)
_MAX_LOG2_FOR_SCORE = float(np.log2(np.finfo(float).max)) - 1.0


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

    pool_y_star = scaling_pool[:, 4:8].astype(float)
    _validate_intensity_log2_range(pool_y_star, context="scaling_vec8")
    weights = sfxi_math.weights_from_setpoint(setpoint, eps=eps)
    intensity_disabled = bool(not np.any(weights))
    if intensity_disabled:
        denom = 1.0
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
    return _score_vec8_with_resolved_denom(
        candidates,
        setpoint=setpoint,
        percentile=percentile,
        eps=eps,
        beta=beta,
        gamma=gamma,
        delta=delta,
        denom=denom,
    )


def score_vec8_with_denom(
    vec8: np.ndarray | Sequence[Sequence[float]],
    config: SFXIScoringConfig,
    *,
    denom: float,
) -> SFXIScoringResult:
    """Score vec8 rows against an explicitly persisted SFXI denominator.

    Use this for ledger audits and deterministic reranks where the objective
    denominator is already part of the run context. New OPAL rounds should use
    :func:`score_vec8` so the denominator is derived from observed labels.
    """

    candidates = _coerce_vec8(vec8, name="vec8")
    setpoint, percentile, _min_n, eps, beta, gamma, delta = _parse_config(config)
    if not np.isfinite(denom) or float(denom) <= 0.0:
        raise ValueError(f"sfxi_v1: denom must be positive and finite; got {denom}.")
    return _score_vec8_with_resolved_denom(
        candidates,
        setpoint=setpoint,
        percentile=percentile,
        eps=eps,
        beta=beta,
        gamma=gamma,
        delta=delta,
        denom=float(denom),
    )


def _score_vec8_with_resolved_denom(
    candidates: np.ndarray,
    *,
    setpoint: np.ndarray,
    percentile: int,
    eps: float,
    beta: float,
    gamma: float,
    delta: float,
    denom: float,
) -> SFXIScoringResult:
    v_hat = np.clip(candidates[:, 0:4].astype(float), 0.0, 1.0)
    y_star = candidates[:, 4:8].astype(float)
    _validate_intensity_log2_range(y_star, context="vec8")
    logic_fidelity = sfxi_math.logic_fidelity(v_hat, setpoint)
    effect_raw, weights = sfxi_math.effect_raw_from_y_star(
        y_star,
        setpoint,
        delta=delta,
        eps=eps,
        state_order=SFXI_STATE_ORDER,
    )
    intensity_disabled = bool(not np.any(weights))
    effect_scaled = (
        np.ones(candidates.shape[0], dtype=float) if intensity_disabled else sfxi_math.effect_scaled(effect_raw, denom)
    )
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


def _validate_intensity_log2_range(values: np.ndarray, *, context: str) -> None:
    if np.any(np.asarray(values, dtype=float) > _MAX_LOG2_FOR_SCORE):
        observed = float(np.max(values))
        raise ValueError(
            "sfxi_v1: intensity log2 values exceed stable score range "
            f"in {context} (max allowed {_MAX_LOG2_FOR_SCORE:.1f}, observed {observed:.1f})."
        )


def to_sfxi_reference_overlay_records(
    result: SFXIScoringResult,
    *,
    metric_id: str | None = None,
    metric_provenance: str = "dnadesign.opal.api.sfxi.score_vec8",
    source_ref: str | None = None,
    score_ref: str = "dnadesign.opal.api.sfxi.score_vec8",
    reference_instance_id: Sequence[str] | None = None,
    collection_id: str | Sequence[str] | None = None,
    batch_id: str | Sequence[str] | None = None,
    campaign_id: str | Sequence[str] | None = None,
    reader_experiment_id: str | Sequence[str] | None = None,
    reader_experiment_date: int | Sequence[int] | None = None,
    setpoint_name: str | Sequence[str] | None = None,
    r_logic: Sequence[float] | None = None,
    time_selected_h: Sequence[float] | None = None,
    reference_design_id: Sequence[str] | None = None,
    sequence_source_id: Sequence[str] | None = None,
    design_id: Sequence[str] | None = None,
    source_id: Sequence[str] | None = None,
    flat_logic: Sequence[bool] | None = None,
    namespace: str = SFXI_REFERENCE_OVERLAY_NAMESPACE,
) -> list[dict[str, object]]:
    """Return registry-compatible, namespaced SFXI reference-overlay records.

    ``reference_instance_id`` and ``sequence_source_id`` are the canonical USR
    overlay fields. ``design_id`` and ``source_id`` remain accepted aliases so
    callers can migrate without changing the scoring math.
    """

    if not isinstance(result, SFXIScoringResult):
        raise TypeError("result must be an SFXIScoringResult.")
    ns = str(namespace).strip()
    if not ns or "__" in ns:
        raise ValueError("namespace must be a non-empty column namespace without '__'.")
    prefix = f"{ns}__"
    base_records = result.to_records()
    n_rows = len(base_records)
    metric_id_value = (
        str(metric_id) if metric_id is not None else _metric_id_from_setpoint_name(setpoint_name, n_rows=n_rows)
    )
    reference_ids = _overlay_values(
        reference_instance_id if reference_instance_id is not None else design_id,
        n_rows=n_rows,
        field="reference_instance_id",
    )
    collection_ids = _overlay_values(collection_id, n_rows=n_rows, field="collection_id")
    batch_ids = _overlay_values(batch_id, n_rows=n_rows, field="batch_id")
    campaign_ids = _overlay_values(campaign_id, n_rows=n_rows, field="campaign_id")
    reader_experiment_ids = _overlay_values(
        reader_experiment_id,
        n_rows=n_rows,
        field="reader_experiment_id",
    )
    reader_experiment_dates = _overlay_values(
        reader_experiment_date,
        n_rows=n_rows,
        field="reader_experiment_date",
        coerce=str,
    )
    setpoint_names = _overlay_values(setpoint_name, n_rows=n_rows, field="setpoint_name")
    r_logic_values = _overlay_values(r_logic, n_rows=n_rows, field="r_logic", coerce=float)
    time_selected_values = _overlay_values(time_selected_h, n_rows=n_rows, field="time_selected_h", coerce=float)
    reference_design_ids = _overlay_values(reference_design_id, n_rows=n_rows, field="reference_design_id")
    sequence_source_ids = _overlay_values(
        sequence_source_id if sequence_source_id is not None else source_id,
        n_rows=n_rows,
        field="sequence_source_id",
    )
    flat_logic_values = _overlay_values(flat_logic, n_rows=n_rows, field="flat_logic", coerce=bool)

    rows: list[dict[str, object]] = []
    for idx, record in enumerate(base_records):
        rows.append(
            {
                f"{prefix}reference_instance_id": reference_ids[idx],
                f"{prefix}collection_id": collection_ids[idx],
                f"{prefix}batch_id": batch_ids[idx],
                f"{prefix}campaign_id": campaign_ids[idx],
                f"{prefix}reader_experiment_id": reader_experiment_ids[idx],
                f"{prefix}reader_experiment_date": _maybe_int(reader_experiment_dates[idx]),
                f"{prefix}metric_id": metric_id_value,
                f"{prefix}metric_value": record["sfxi"],
                f"{prefix}metric_provenance": str(metric_provenance),
                f"{prefix}source_ref": source_ref,
                f"{prefix}score_ref": str(score_ref),
                f"{prefix}objective_name": record["objective_name"],
                f"{prefix}api_version": record["api_version"],
                f"{prefix}state_order": record["state_order"],
                f"{prefix}setpoint_name": setpoint_names[idx],
                f"{prefix}setpoint_vector": record["setpoint_vector"],
                f"{prefix}denom_used": record["denom_used"],
                f"{prefix}denom_percentile": record["denom_percentile"],
                f"{prefix}logic_fidelity": record["logic_fidelity"],
                f"{prefix}effect_raw": record["effect_raw"],
                f"{prefix}effect_scaled": record["effect_scaled"],
                f"{prefix}sfxi": record["sfxi"],
                f"{prefix}r_logic": r_logic_values[idx],
                f"{prefix}time_selected_h": time_selected_values[idx],
                f"{prefix}reference_design_id": reference_design_ids[idx],
                f"{prefix}sequence_source_id": sequence_source_ids[idx],
                f"{prefix}clip_lo_mask": record["clip_lo_mask"],
                f"{prefix}clip_hi_mask": record["clip_hi_mask"],
                f"{prefix}intensity_disabled": record["intensity_disabled"],
                f"{prefix}flat_logic": flat_logic_values[idx],
            }
        )
    return rows


def validate_sfxi_reference_overlay_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_setpoint_vector: Sequence[float] | None = None,
    metric_id: str | None = None,
    namespace: str = SFXI_REFERENCE_OVERLAY_NAMESPACE,
) -> dict[str, object]:
    """Validate that SFXI reference overlay rows use OPAL-compatible math.

    The validator is intentionally strict on the fields that would create math
    drift in plots: API version, objective name, state order, setpoint vector,
    finite metric values, and non-empty metric provenance.
    """

    ns = str(namespace).strip()
    if not ns or "__" in ns:
        raise ValueError("namespace must be a non-empty column namespace without '__'.")
    prefix = f"{ns}__"
    required = (
        "objective_name",
        "api_version",
        "state_order",
        "setpoint_vector",
        "metric_id",
        "metric_value",
        "metric_provenance",
        "denom_used",
        "denom_percentile",
        "logic_fidelity",
        "effect_raw",
        "effect_scaled",
        "sfxi",
    )
    row_list = list(rows)
    for index, row in enumerate(row_list):
        missing = [field for field in required if f"{prefix}{field}" not in row]
        if missing:
            raise ValueError(f"SFXI reference overlay row {index} missing fields: {missing}.")
        if str(row[f"{prefix}objective_name"]) != SFXI_OBJECTIVE_NAME:
            raise ValueError(f"SFXI reference overlay row {index} has unsupported objective_name.")
        if str(row[f"{prefix}api_version"]) != SFXI_API_VERSION:
            raise ValueError(f"SFXI reference overlay row {index} has unsupported api_version.")
        state_order = tuple(str(item) for item in list(row[f"{prefix}state_order"]))
        if state_order != tuple(SFXI_STATE_ORDER):
            raise ValueError(f"SFXI reference overlay row {index} has unsupported state_order.")
        if metric_id is not None and str(row[f"{prefix}metric_id"]) != str(metric_id):
            raise ValueError(f"SFXI reference overlay row {index} has metric_id outside requested scope.")
        provenance = str(row[f"{prefix}metric_provenance"] or "").strip()
        if not provenance:
            raise ValueError(f"SFXI reference overlay row {index} has empty metric_provenance.")
        for numeric_field in (
            "metric_value",
            "denom_used",
            "logic_fidelity",
            "effect_raw",
            "effect_scaled",
            "sfxi",
        ):
            value = float(row[f"{prefix}{numeric_field}"])
            if not np.isfinite(value):
                raise ValueError(f"SFXI reference overlay row {index} has non-finite {numeric_field}.")
        setpoint = _coerce_length4(row[f"{prefix}setpoint_vector"], field="setpoint_vector")
        if expected_setpoint_vector is not None:
            expected = _coerce_length4(expected_setpoint_vector, field="expected_setpoint_vector")
            if not np.allclose(setpoint, expected, atol=1.0e-12, rtol=0.0):
                raise ValueError(
                    f"SFXI reference overlay row {index} setpoint_vector does not match the active campaign setpoint."
                )
    metric_ids = sorted({str(row[f"{prefix}metric_id"]) for row in row_list})
    return {
        "schema_version": SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION,
        "namespace": ns,
        "row_count": len(row_list),
        "metric_ids": metric_ids,
    }


def _overlay_values(
    value: Any,
    *,
    n_rows: int,
    field: str,
    coerce: Any = str,
) -> list[object | None]:
    if value is None:
        return [None] * n_rows
    if isinstance(value, str) or not isinstance(value, SequenceABC):
        return [_coerce_optional(value, coerce=coerce)] * n_rows
    values = list(value)
    if len(values) != n_rows:
        raise ValueError(f"{field} length {len(values)} does not match SFXI row count {n_rows}.")
    return [_coerce_optional(item, coerce=coerce) for item in values]


def _coerce_optional(value: object, *, coerce: Any) -> object | None:
    if value is None:
        return None
    return coerce(value)


def _maybe_int(value: object | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _metric_id_from_setpoint_name(value: object, *, n_rows: int) -> str:
    values = _overlay_values(value, n_rows=n_rows, field="setpoint_name")
    names = sorted({str(item).strip() for item in values if item is not None and str(item).strip()})
    if len(names) == 1:
        return f"{SFXI_OBJECTIVE_NAME}/{names[0]}/sfxi"
    return f"{SFXI_OBJECTIVE_NAME}/sfxi"


def _coerce_length4(value: object, *, field: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float).ravel()
    if arr.size != 4 or not np.all(np.isfinite(arr)):
        raise ValueError(f"{field} must be a finite length-4 vector.")
    return arr


__all__ = [
    "SFXI_API_VERSION",
    "SFXI_REFERENCE_OVERLAY_FIELDS",
    "SFXI_REFERENCE_OVERLAY_NAMESPACE",
    "SFXI_REFERENCE_OVERLAY_PREFIX",
    "SFXI_REFERENCE_OVERLAY_SCHEMA_VERSION",
    "SFXI_OBJECTIVE_NAME",
    "SFXI_STATE_ORDER",
    "SFXIScoringConfig",
    "SFXIScoringResult",
    "score_vec8",
    "score_vec8_with_denom",
    "to_sfxi_reference_overlay_records",
    "validate_sfxi_reference_overlay_records",
]
