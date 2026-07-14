"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/candidate_identity.py

Resolve response-metastudy rows through the study candidate-binding artifact.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    READER_ALIAS_NAMESPACE,
    PromoterCandidateBindingsError,
    load_promoter_candidate_bindings,
    verify_promoter_candidate_bindings,
)

_MEASUREMENT_COLUMNS = ("design_id", "reader_experiment_id")


@dataclass(frozen=True)
class ResponseCandidateIdentityBindings:
    """Exact response rows plus provenance for one verified study binding bundle."""

    rows: pd.DataFrame
    bundle_root: Path
    manifest_path: Path
    records_path: Path
    binding_count: int
    candidate_count: int


def load_response_candidate_identity_bindings(
    *,
    measurement_selection: pd.DataFrame,
    bundle_root: Path,
) -> ResponseCandidateIdentityBindings:
    """Resolve every selected Reader design alias to study-owned candidate identity."""

    root = Path(bundle_root).expanduser().resolve()
    verification = verify_promoter_candidate_bindings(root, allowed_root=root)
    binding_rows = load_promoter_candidate_bindings(root, allowed_root=root)
    measurements = _validated_measurements(measurement_selection)
    reader_bindings = binding_rows.loc[
        binding_rows["alias_namespace"].astype(str).eq(READER_ALIAS_NAMESPACE),
        ["alias", "candidate_id"],
    ].rename(columns={"candidate_id": "resolved_candidate_id"})
    resolved = measurements.merge(
        reader_bindings,
        left_on="design_id",
        right_on="alias",
        how="left",
        validate="many_to_one",
    )
    missing = sorted(resolved.loc[resolved["resolved_candidate_id"].isna(), "design_id"].astype(str).unique())
    if missing:
        raise PromoterCandidateBindingsError(
            f"Response metastudy has unresolved Reader design aliases in {READER_ALIAS_NAMESPACE!r}: {missing[:10]}"
        )
    rows = resolved.loc[:, ["resolved_candidate_id", "design_id", "reader_experiment_id"]].rename(
        columns={"resolved_candidate_id": "id"}
    )
    if rows["id"].duplicated().any():
        duplicates = sorted(rows.loc[rows["id"].duplicated(keep=False), "id"].astype(str).unique())
        raise PromoterCandidateBindingsError(
            f"Response metastudy label rows resolve to duplicate candidate IDs: {duplicates[:10]}"
        )
    return ResponseCandidateIdentityBindings(
        rows=rows.reset_index(drop=True),
        bundle_root=root,
        manifest_path=verification.manifest_json,
        records_path=verification.bindings_parquet,
        binding_count=int(verification.binding_count),
        candidate_count=int(verification.candidate_count),
    )


def _validated_measurements(measurement_selection: pd.DataFrame) -> pd.DataFrame:
    missing_columns = sorted(set(_MEASUREMENT_COLUMNS) - set(measurement_selection.columns))
    if missing_columns:
        raise PromoterCandidateBindingsError(
            f"Response measurement selection lacks Reader identity columns: {missing_columns}"
        )
    measurements = measurement_selection.loc[:, _MEASUREMENT_COLUMNS].copy()
    for column in _MEASUREMENT_COLUMNS:
        missing_value = measurements[column].isna() | measurements[column].astype(str).str.strip().eq("")
        if missing_value.any():
            raise PromoterCandidateBindingsError(
                f"Response measurement selection column {column!r} contains empty values."
            )
        measurements[column] = measurements[column].astype(str)
    if measurements.duplicated().any():
        raise PromoterCandidateBindingsError("Response measurement selection contains duplicate Reader pairs.")
    return measurements


__all__ = ["ResponseCandidateIdentityBindings", "load_response_candidate_identity_bindings"]
