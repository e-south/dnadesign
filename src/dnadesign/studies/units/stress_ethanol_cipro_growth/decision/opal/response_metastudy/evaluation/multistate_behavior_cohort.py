"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/multistate_behavior_cohort.py

Canonical exact-cohort validation and receipts for behavior evidence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .multistate_behavior_protocol import MultistateBehaviorShadowProtocol


@dataclass(frozen=True)
class VerifiedBehaviorCohortReceipt:
    """Receipt produced from the exhaustive verified Reader cohort projection."""

    cohort_id: str
    primary_reduction_id: str
    unit_count: int
    candidate_count: int
    reader_experiment_count: int
    excluded_nonexact_unit_count: int
    reader_bundle_manifest_sha256: str
    candidate_bindings_manifest_sha256: str
    unit_ids_sha256: str
    source_rows_sha256: str


def validated_behavior_evidence(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sorted exact primary rows after validating joint-draw support."""

    components = behavior_component_columns(protocol)
    label_required = {
        "id",
        "candidate_id",
        "reader_experiment_id",
        "reduction_id",
        *components,
        *(f"{component}_bound_kind" for component in components),
    }
    draw_required = {"id", "draw_index", *components}
    if missing := sorted(label_required - set(labels.columns)):
        raise ValueError(f"behavior normalization labels lack fields: {missing}")
    if missing := sorted(draw_required - set(draws.columns)):
        raise ValueError(f"behavior normalization draws lack fields: {missing}")
    label_rows = labels.copy()
    draw_rows = draws.copy()
    if label_rows.empty:
        raise ValueError("behavior normalization requires at least one candidate-experiment unit.")
    if label_rows["id"].astype(str).duplicated().any():
        raise ValueError("behavior normalization label unit ids must be unique.")
    if not label_rows["reduction_id"].astype(str).eq(protocol.primary_reduction_id).all():
        raise ValueError(f"behavior normalization labels must use primary reduction {protocol.primary_reduction_id!r}.")
    bound_columns = [f"{component}_bound_kind" for component in components]
    if not label_rows[bound_columns].astype(str).eq("exact").all(axis=None):
        raise ValueError("behavior normalization requires exact component evidence for every cohort unit.")
    label_ids = set(label_rows["id"].astype(str))
    draw_ids = set(draw_rows["id"].astype(str))
    if draw_ids != label_ids:
        raise ValueError(
            "behavior normalization label and draw unit ids disagree: "
            f"missing={sorted(label_ids - draw_ids)}, extra={sorted(draw_ids - label_ids)}."
        )
    if draw_rows.duplicated(subset=["id", "draw_index"]).any():
        raise ValueError("behavior normalization draw indexes must be unique within each unit.")
    if draw_rows["draw_index"].map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise ValueError("behavior normalization draw indexes must be nonnegative integers.")
    draw_indexes = pd.to_numeric(draw_rows["draw_index"], errors="coerce").to_numpy(dtype=float)
    if (
        not np.isfinite(draw_indexes).all()
        or (draw_indexes < 0).any()
        or not np.equal(draw_indexes, np.floor(draw_indexes)).all()
    ):
        raise ValueError("behavior normalization draw indexes must be nonnegative integers.")
    draw_rows["draw_index"] = draw_indexes.astype(np.int64)
    counts = draw_rows.groupby("id", sort=False)["draw_index"].nunique()
    if counts.nunique() != 1:
        raise ValueError("behavior normalization requires identical bootstrap draw counts across units.")
    draw_count = int(counts.iloc[0])
    expected_draws = set(range(draw_count))
    if any(set(group["draw_index"].astype(int)) != expected_draws for _, group in draw_rows.groupby("id", sort=False)):
        raise ValueError("behavior normalization draw indexes must be contiguous and aligned across units.")
    if draw_count < protocol.normalization.minimum_bootstrap_draws:
        raise ValueError(
            "behavior normalization has too few bootstrap draws: "
            f"observed={draw_count}, required>={protocol.normalization.minimum_bootstrap_draws}."
        )
    for frame, columns, label in (
        (label_rows, components, "labels"),
        (draw_rows, components, "draws"),
    ):
        if not np.isfinite(frame.loc[:, list(columns)].to_numpy(dtype=float)).all():
            raise ValueError(f"behavior normalization {label} contain non-finite component values.")
    return (
        label_rows.sort_values("id", kind="mergesort").reset_index(drop=True),
        draw_rows.sort_values(["id", "draw_index"], kind="mergesort").reset_index(drop=True),
    )


def verify_behavior_cohort_receipt(
    labels: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
    receipt: VerifiedBehaviorCohortReceipt,
) -> None:
    """Require counts and identities to reproduce the exhaustive projection."""

    if receipt.cohort_id != protocol.normalization.cohort_id:
        raise ValueError("behavior cohort receipt identity disagrees with the shadow protocol.")
    if receipt.primary_reduction_id != protocol.primary_reduction_id:
        raise ValueError("behavior cohort receipt reduction disagrees with the shadow protocol.")
    observed = {
        "unit_count": len(labels),
        "candidate_count": int(labels["candidate_id"].nunique()),
        "reader_experiment_count": int(labels["reader_experiment_id"].nunique()),
    }
    expected = {
        "unit_count": receipt.unit_count,
        "candidate_count": receipt.candidate_count,
        "reader_experiment_count": receipt.reader_experiment_count,
    }
    if observed != expected:
        raise ValueError(f"behavior cohort receipt counts disagree: receipt={expected}, observed={observed}.")
    if behavior_cohort_unit_ids_sha256(labels) != receipt.unit_ids_sha256:
        raise ValueError("behavior cohort receipt unit identities disagree with the exhaustive projection.")


def behavior_normalization_source_rows_sha256(
    labels: pd.DataFrame,
    draws: pd.DataFrame,
    *,
    protocol: MultistateBehaviorShadowProtocol,
) -> str:
    """Return the canonical digest for exact, sorted cohort evidence."""

    label_rows, draw_rows = validated_behavior_evidence(labels, draws, protocol=protocol)
    components = list(behavior_component_columns(protocol))
    label_columns = ["id", "candidate_id", "reader_experiment_id", "reduction_id", *components]
    payload = {
        "labels": _canonical_evidence_records(
            label_rows,
            text_columns=label_columns[:4],
            float_columns=components,
        ),
        "draws": _canonical_evidence_records(
            draw_rows,
            text_columns=["id"],
            integer_columns=["draw_index"],
            float_columns=components,
        ),
    }
    rendered = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def behavior_cohort_unit_ids_sha256(labels: pd.DataFrame) -> str:
    """Digest the exact ordered candidate-experiment identity set."""

    required = {"id", "candidate_id", "reader_experiment_id"}
    if missing := sorted(required - set(labels.columns)):
        raise ValueError(f"behavior cohort identity rows lack fields: {missing}")
    rows = labels.loc[:, sorted(required)].astype(str).sort_values("id", kind="mergesort")
    return hashlib.sha256(rows.to_json(orient="records").encode("utf-8")).hexdigest()


def behavior_component_columns(protocol: MultistateBehaviorShadowProtocol) -> tuple[str, ...]:
    """Return Reader response then fluorescence columns in declared state order."""

    return tuple(f"r{state}" for state in protocol.state_ids) + tuple(f"b{state}" for state in protocol.state_ids)


def _canonical_evidence_records(
    frame: pd.DataFrame,
    *,
    text_columns: list[str],
    float_columns: list[str],
    integer_columns: list[str] | None = None,
) -> list[dict[str, object]]:
    """Encode identity and IEEE-754 values without decimal precision loss."""

    integer_columns = integer_columns or []
    records: list[dict[str, object]] = []
    for row in frame.loc[:, [*text_columns, *integer_columns, *float_columns]].itertuples(index=False, name=None):
        values = iter(row)
        record: dict[str, object] = {column: str(next(values)) for column in text_columns}
        record.update({column: int(next(values)) for column in integer_columns})
        record.update({column: float(next(values)).hex() for column in float_columns})
        records.append(record)
    return records


__all__ = [
    "VerifiedBehaviorCohortReceipt",
    "behavior_cohort_unit_ids_sha256",
    "behavior_component_columns",
    "behavior_normalization_source_rows_sha256",
    "validated_behavior_evidence",
    "verify_behavior_cohort_receipt",
]
