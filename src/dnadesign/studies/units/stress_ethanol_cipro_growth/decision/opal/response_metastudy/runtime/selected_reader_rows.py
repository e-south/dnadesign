"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/runtime/selected_reader_rows.py

Resolve the retrospective model screen's exact Reader row selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_bundle import (
    ReaderResponseBundle,
)

CANDIDATE_IDENTITY_COLUMNS = ("id", "design_id", "reader_experiment_id")


def build_selected_response_labels(
    bundle: ReaderResponseBundle,
    *,
    candidate_identity_bindings: pd.DataFrame,
) -> pd.DataFrame:
    labels = _validated_bindings(candidate_identity_bindings, require_unique_candidate=True)
    designs = bundle.designs.loc[~bundle.designs["is_reference"].astype(bool)].copy()
    designs = designs.rename(
        columns={
            "experiment_id": "reader_experiment_id",
            "reduction_role": "screen_role",
        }
    )
    selected = labels.merge(
        designs,
        on=["reader_experiment_id", "design_id"],
        how="left",
        validate="one_to_many",
    )
    if selected["reduction_id"].isna().any():
        missing_rows = selected.loc[selected["reduction_id"].isna(), ["id", "reader_experiment_id", "design_id"]]
        raise ValueError(f"Reader response-window bundle lacks selected labels: {missing_rows.to_dict('records')}")
    expected_rows = len(labels) * bundle.designs["reduction_id"].nunique()
    if len(selected) != expected_rows:
        raise ValueError(f"selected Reader label rows expected {expected_rows}; observed {len(selected)}.")
    if selected.duplicated(subset=["reduction_id", "id"]).any():
        raise ValueError("selected Reader labels contain duplicate reduction/candidate identities.")
    return selected.sort_values(["reduction_id", "id"], kind="mergesort").reset_index(drop=True)


def build_selected_bootstrap_draws(
    bundle: ReaderResponseBundle,
    *,
    candidate_identity_bindings: pd.DataFrame,
) -> pd.DataFrame:
    labels = _validated_bindings(candidate_identity_bindings, require_unique_candidate=False)
    draws = bundle.bootstrap_draws.loc[~bundle.bootstrap_draws["is_reference"].astype(bool)].rename(
        columns={"experiment_id": "reader_experiment_id"}
    )
    selected = labels.merge(
        draws,
        on=["reader_experiment_id", "design_id"],
        how="left",
        validate="one_to_many",
    )
    if selected["draw_index"].isna().any():
        raise ValueError("Reader bootstrap bundle lacks one or more selected labels.")
    key = ["id", "reduction_id", "draw_index"]
    if selected.duplicated(subset=key).any():
        raise ValueError("selected Reader bootstrap draws contain duplicate identities.")
    return selected.sort_values(key, kind="mergesort").reset_index(drop=True)


def _validated_bindings(frame: pd.DataFrame, *, require_unique_candidate: bool) -> pd.DataFrame:
    if missing := sorted(set(CANDIDATE_IDENTITY_COLUMNS) - set(frame.columns)):
        raise ValueError(f"candidate identity bindings lack Reader identity fields: {missing}")
    labels = frame.loc[:, CANDIDATE_IDENTITY_COLUMNS].copy()
    for column in CANDIDATE_IDENTITY_COLUMNS:
        labels[column] = labels[column].astype(str)
    if require_unique_candidate and labels["id"].duplicated().any():
        raise ValueError("candidate identity bindings contain duplicate candidate ids.")
    return labels


__all__ = ["build_selected_bootstrap_draws", "build_selected_response_labels"]
