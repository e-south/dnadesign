"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/evaluation/model_representations.py

Label representations for the response-metric model screen.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from dnadesign.opal import validated_response_magnitude as _validated_rmf_input

from ..core.response_contracts import STRESS_STATE_IDS


@dataclass(frozen=True)
class LabelRepresentation:
    """Model target plus a decoder to the eight-value response/magnitude space."""

    id: str
    target: np.ndarray
    response_magnitude_truth: np.ndarray
    decoder: Literal["identity_response_magnitude", "factorial_contrast7"]
    promotion_eligible: bool


def build_label_representations(
    *,
    ids: Sequence[str],
    response_summaries: pd.DataFrame,
    primary_reduction_id: str,
    promotion_reduction_ids: frozenset[str],
) -> tuple[LabelRepresentation, ...]:
    """Build aligned state-summary and contrast targets."""

    candidate_ids = [str(value) for value in ids]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("label representation ids must be unique.")
    representations: list[LabelRepresentation] = []
    for reduction_id, frame in response_summaries.groupby("reduction_id", sort=True):
        ordered = aligned_response_magnitude(frame, ids=candidate_ids, reduction_id=str(reduction_id))
        representations.append(
            LabelRepresentation(
                id=str(reduction_id),
                target=ordered,
                response_magnitude_truth=ordered,
                decoder="identity_response_magnitude",
                promotion_eligible=str(reduction_id) in promotion_reduction_ids,
            )
        )
        if str(reduction_id) == primary_reduction_id:
            representations.append(
                LabelRepresentation(
                    id=f"{reduction_id}__factorial_contrast7",
                    target=response_magnitude_to_factorial_contrast7(ordered),
                    response_magnitude_truth=ordered,
                    decoder="factorial_contrast7",
                    promotion_eligible=str(reduction_id) in promotion_reduction_ids,
                )
            )
    if primary_reduction_id not in {value.id for value in representations}:
        raise ValueError(f"primary response reduction {primary_reduction_id!r} is missing.")
    observed_reduction_ids = {value.id for value in representations if value.decoder == "identity_response_magnitude"}
    if not promotion_reduction_ids or not promotion_reduction_ids.issubset(observed_reduction_ids):
        raise ValueError("promotion reduction ids must be a non-empty subset of modeled reductions.")
    return tuple(representations)


def response_magnitude_to_factorial_contrast7(values: np.ndarray) -> np.ndarray:
    """Map response states to input-A, input-B, interaction, and magnitude."""

    matrix = _validated_response_magnitude(values, expected_rows=None, context="response_magnitude")
    r00, r10, r01, r11 = matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3]
    input_a = (-r00 + r10 - r01 + r11) / 4.0
    input_b = (-r00 - r10 + r01 + r11) / 4.0
    interaction = (r00 - r10 - r01 + r11) / 4.0
    return np.column_stack((input_a, input_b, interaction, matrix[:, 4:]))


def decode_to_response_magnitude(values: np.ndarray, *, decoder: str) -> np.ndarray:
    """Decode a model target into the eight state-summary channels."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or not np.isfinite(matrix).all():
        raise ValueError("decoded model predictions must be a finite non-empty matrix.")
    if decoder == "identity_response_magnitude":
        return _validated_response_magnitude(matrix, expected_rows=None, context="identity prediction")
    if decoder != "factorial_contrast7" or matrix.shape[1] != 7:
        raise ValueError(f"unsupported label decoder {decoder!r} for shape {matrix.shape}.")
    input_a, input_b, interaction = matrix[:, 0], matrix[:, 1], matrix[:, 2]
    response = np.column_stack(
        (
            -input_a - input_b + interaction,
            input_a - input_b - interaction,
            -input_a + input_b - interaction,
            input_a + input_b + interaction,
        )
    )
    return np.column_stack((response, matrix[:, 3:]))


def aligned_response_magnitude(frame: pd.DataFrame, *, ids: Sequence[str], reduction_id: str) -> np.ndarray:
    """Align one reduced label table to candidate order."""

    required = {"id", *(f"r{state}" for state in STRESS_STATE_IDS), *(f"b{state}" for state in STRESS_STATE_IDS)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{reduction_id}: response summary missing columns: {missing}")
    work = frame.copy()
    work["id"] = work["id"].astype(str)
    if work["id"].duplicated().any() or set(work["id"]) != set(ids):
        raise ValueError(f"{reduction_id}: response summary candidate ids do not match labels.")
    ordered = work.set_index("id").loc[list(ids)]
    columns = [f"r{state}" for state in STRESS_STATE_IDS] + [f"b{state}" for state in STRESS_STATE_IDS]
    return _validated_response_magnitude(
        ordered.loc[:, columns].to_numpy(dtype=float),
        expected_rows=len(ids),
        context=reduction_id,
    )


def _validated_response_magnitude(
    values: np.ndarray,
    *,
    expected_rows: int | None,
    context: str,
) -> np.ndarray:
    """Validate an aligned state-summary matrix for model screening."""

    try:
        matrix = _validated_rmf_input(values, state_count=len(STRESS_STATE_IDS))
    except ValueError as exc:
        raise ValueError(f"{context}: invalid response/magnitude matrix: {exc}") from exc
    if expected_rows is not None and len(matrix) != expected_rows:
        raise ValueError(f"{context}: expected {expected_rows} rows; found {len(matrix)}.")
    return matrix


__all__ = [
    "LabelRepresentation",
    "aligned_response_magnitude",
    "build_label_representations",
    "decode_to_response_magnitude",
    "response_magnitude_to_factorial_contrast7",
]
